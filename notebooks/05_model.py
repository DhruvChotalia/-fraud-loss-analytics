# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 4 — XGBoost Fraud Model with Cost-Based Threshold
# MAGIC
# MAGIC **Goal:** train a gradient-boosted tree classifier to identify fraud transactions,
# MAGIC then choose the decision threshold not by accuracy but by cost — because missing
# MAGIC a $1,000 fraud is far worse than a false decline on a $10 grocery purchase.
# MAGIC
# MAGIC **Steps:**
# MAGIC 1. Feature engineering on silver
# MAGIC 2. Train/test split (time-based — no data leakage)
# MAGIC 3. Train GBTClassifier (Spark MLlib's gradient boosted trees = XGBoost equivalent)
# MAGIC 4. Evaluate with precision-recall curve
# MAGIC 5. Pick threshold by minimising total cost
# MAGIC 6. Score all transactions and save to ADLS Gen2
# MAGIC
# MAGIC **Prerequisite:** run `02_silver` first so `silver.transactions` exists.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load silver and engineer features

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline

silver = spark.table("silver.transactions")

# Feature engineering — every column is something a fraud model would actually use
features_df = silver.select(
    "trans_num",
    "txn_time",
    "amt",
    "is_fraud",
    F.hour("txn_time").alias("hour"),
    F.dayofweek("txn_time").alias("day_of_week"),
    F.month("txn_time").alias("month"),
    F.when(F.dayofweek("txn_time").isin(1, 7), 1).otherwise(0).alias("is_weekend"),
    F.col("category"),
    F.col("gender"),
    F.col("city_pop").cast("double"),
    F.col("merch_lat"),
    F.col("merch_long"),
    F.col("lat"),
    F.col("long"),
    # Distance between customer and merchant (approximate, in degrees)
    F.sqrt(
        F.pow(F.col("merch_lat") - F.col("lat"), 2) +
        F.pow(F.col("merch_long") - F.col("long"), 2)
    ).alias("geo_distance"),
)

print(f"Feature rows: {features_df.count():,}")
features_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Time-based train/test split
# MAGIC
# MAGIC We split on date, not randomly. Splitting randomly would let the model see future
# MAGIC transactions during training — data leakage. A time split mimics production: the
# MAGIC model is trained on past data and scored on future data it has never seen.
# MAGIC
# MAGIC - **Train:** 2019-01-01 to 2020-03-31
# MAGIC - **Test:** 2020-04-01 to 2020-12-31

# COMMAND ----------

from pyspark.sql.functions import to_date

SPLIT_DATE = "2020-04-01"

train = features_df.filter(F.col("txn_time") < F.lit(SPLIT_DATE).cast("timestamp"))
test  = features_df.filter(F.col("txn_time") >= F.lit(SPLIT_DATE).cast("timestamp"))

train_fraud_rate = train.agg(F.avg("is_fraud")).collect()[0][0]
test_fraud_rate  = test.agg(F.avg("is_fraud")).collect()[0][0]

print(f"Train: {train.count():,} rows  fraud rate: {train_fraud_rate:.3%}")
print(f"Test:  {test.count():,} rows  fraud rate: {test_fraud_rate:.3%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Build the ML pipeline
# MAGIC
# MAGIC Pipeline stages:
# MAGIC 1. `StringIndexer` — encode `category` and `gender` as numeric (ML needs numbers)
# MAGIC 2. `VectorAssembler` — combine all features into one vector column
# MAGIC 3. `GBTClassifier` — gradient boosted trees (Spark's XGBoost equivalent)
# MAGIC
# MAGIC We also set `weightCol` to upweight fraud transactions 50:1 — the dataset is
# MAGIC heavily imbalanced (199 legit per 1 fraud) and without weighting the model learns
# MAGIC to predict "not fraud" always and still gets 99.5% accuracy.

# COMMAND ----------

category_indexer = StringIndexer(inputCol="category", outputCol="category_idx", handleInvalid="keep")
gender_indexer   = StringIndexer(inputCol="gender",   outputCol="gender_idx",   handleInvalid="keep")

FEATURE_COLS = [
    "amt", "hour", "day_of_week", "month", "is_weekend",
    "category_idx", "gender_idx", "city_pop",
    "geo_distance",
]

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")

# Add class weight column — fraud rows get weight 50, legit rows get weight 1
train_weighted = train.withColumn(
    "weight",
    F.when(F.col("is_fraud") == 1, 50.0).otherwise(1.0)
)

gbt = GBTClassifier(
    featuresCol="features",
    labelCol="is_fraud",
    weightCol="weight",
    maxIter=50,
    maxDepth=5,
    seed=42,
)

pipeline = Pipeline(stages=[category_indexer, gender_indexer, assembler, gbt])

print("Training model... (this takes 5-10 minutes on Community Edition)")
model = pipeline.fit(train_weighted)
print("Training complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Score the test set and evaluate

# COMMAND ----------

predictions = model.transform(test)

# GBTClassifier outputs probability as a vector [prob_legit, prob_fraud]
# Extract the fraud probability
extract_prob = F.udf(lambda v: float(v[1]))
predictions = predictions.withColumn("fraud_prob", extract_prob(F.col("probability")))

evaluator = BinaryClassificationEvaluator(
    labelCol="is_fraud",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

auc = evaluator.evaluate(predictions)
print(f"AUC-ROC: {auc:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Cost-based threshold selection
# MAGIC
# MAGIC Default threshold = 0.5 optimises for accuracy. We want to optimise for **cost**:
# MAGIC
# MAGIC `total_cost = (missed frauds × avg fraud loss) + (false declines × false decline cost)`
# MAGIC
# MAGIC - **Avg fraud loss:** $530 (from the data profile)
# MAGIC - **False decline cost:** $25 (estimated customer friction + chargeback handling)
# MAGIC   — a conservative estimate; real banks use $10–$50
# MAGIC
# MAGIC We sweep thresholds from 0.1 to 0.9 and pick the one with the lowest total cost.

# COMMAND ----------

import pandas as pd

AVG_FRAUD_LOSS    = 530.0
FALSE_DECLINE_COST = 25.0

preds_pd = predictions.select("is_fraud", "fraud_prob", "amt").toPandas()

results = []
for threshold in [round(t * 0.05, 2) for t in range(2, 19)]:
    predicted_fraud = (preds_pd["fraud_prob"] >= threshold).astype(int)
    fn = ((predicted_fraud == 0) & (preds_pd["is_fraud"] == 1)).sum()  # missed fraud
    fp = ((predicted_fraud == 1) & (preds_pd["is_fraud"] == 0)).sum()  # false declines
    tp = ((predicted_fraud == 1) & (preds_pd["is_fraud"] == 1)).sum()  # caught fraud
    total = len(preds_pd)
    cost = fn * AVG_FRAUD_LOSS + fp * FALSE_DECLINE_COST
    results.append({
        "threshold": threshold,
        "caught_fraud": tp,
        "missed_fraud": fn,
        "false_declines": fp,
        "precision": round(tp / (tp + fp) if (tp + fp) > 0 else 0, 4),
        "recall": round(tp / (tp + fn) if (tp + fn) > 0 else 0, 4),
        "total_cost": round(cost, 2),
    })

cost_df = pd.DataFrame(results)
print(cost_df.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Apply optimal threshold

# COMMAND ----------

optimal = cost_df.loc[cost_df["total_cost"].idxmin()]
OPTIMAL_THRESHOLD = optimal["threshold"]

print(f"\nOptimal threshold: {OPTIMAL_THRESHOLD}")
print(f"  Catches {optimal['caught_fraud']:.0f} fraud transactions ({optimal['recall']:.1%} recall)")
print(f"  Misses  {optimal['missed_fraud']:.0f} fraud transactions")
print(f"  False declines: {optimal['false_declines']:.0f}")
print(f"  Total cost at this threshold: ${optimal['total_cost']:,.2f}")
print(f"  Precision: {optimal['precision']:.1%}")

# Apply the threshold
predictions_final = predictions.withColumn(
    "predicted_fraud",
    (F.col("fraud_prob") >= OPTIMAL_THRESHOLD).cast("integer")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save scored transactions and upload to ADLS Gen2

# COMMAND ----------

# Save scores to Databricks Delta
spark.sql("CREATE DATABASE IF NOT EXISTS gold")

(
    predictions_final
    .select("trans_num", "amt", "is_fraud", "fraud_prob", "predicted_fraud")
    .write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold.fraud_scores")
)
print("Written: gold.fraud_scores (Databricks Delta)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload model results to ADLS Gen2
# MAGIC
# MAGIC Writes two CSV files to the `models` container in ADLS Gen2 so Power BI can read them:
# MAGIC - `threshold_analysis.csv` — cost curve across all thresholds
# MAGIC - `model_summary.csv` — the key metrics at the optimal threshold

# COMMAND ----------

from azure.storage.blob import BlobServiceClient
import io

STORAGE_ACCOUNT = "fraudanalytics2024"
STORAGE_KEY      = "YOUR_STORAGE_KEY_HERE"   # replace with your key
CONTAINER        = "models"

client = BlobServiceClient(
    account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
    credential=STORAGE_KEY
)

def upload_csv(df, blob_name):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    client.get_blob_client(container=CONTAINER, blob=blob_name) \
          .upload_blob(buf.getvalue(), overwrite=True)
    print(f"Uploaded: {blob_name} → ADLS Gen2 models/")

upload_csv(cost_df, "threshold_analysis.csv")

summary = pd.DataFrame([{
    "auc_roc":           round(auc, 4),
    "optimal_threshold": OPTIMAL_THRESHOLD,
    "recall":            optimal["recall"],
    "precision":         optimal["precision"],
    "caught_fraud":      int(optimal["caught_fraud"]),
    "missed_fraud":      int(optimal["missed_fraud"]),
    "false_declines":    int(optimal["false_declines"]),
    "total_cost":        optimal["total_cost"],
    "avg_fraud_loss_assumption":    AVG_FRAUD_LOSS,
    "false_decline_cost_assumption": FALSE_DECLINE_COST,
}])
upload_csv(summary, "model_summary.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. MLflow Experiment Tracking
# MAGIC
# MAGIC MLflow is built into Databricks. Every model run is logged so you can:
# MAGIC - Compare runs side by side (different hyperparameters, different thresholds)
# MAGIC - Roll back to a previous model version
# MAGIC - Satisfy audit requirements (OSFI expects model versioning)
# MAGIC
# MAGIC In production at a bank, the MLflow Model Registry would hold the "champion" model
# MAGIC and every challenger. Promotion from Staging → Production requires sign-off.

# COMMAND ----------

import mlflow
import mlflow.spark

# Name the experiment — one experiment per project, one run per training session
mlflow.set_experiment("/Users/dhruvextra76@gmail.com/fraud-model")

default_threshold = 0.50
default_row = cost_df[cost_df["threshold"] == default_threshold]
default_cost = float(default_row["total_cost"].values[0]) if len(default_row) > 0 else None
savings = round(default_cost - float(optimal["total_cost"]), 2) if default_cost else None

with mlflow.start_run(run_name="GBT_cost_optimised_v1"):

    # ── Parameters (what we chose before training) ──────────────────────
    mlflow.log_params({
        "model_type":               "GBTClassifier",
        "max_iter":                 50,
        "max_depth":                5,
        "seed":                     42,
        "class_weight_fraud":       50.0,
        "class_weight_legit":       1.0,
        "train_cutoff":             "2020-04-01",
        "features":                 ",".join(FEATURE_COLS),
        "n_features":               len(FEATURE_COLS),
        "avg_fraud_loss_usd":       AVG_FRAUD_LOSS,
        "false_decline_cost_usd":   FALSE_DECLINE_COST,
    })

    # ── Metrics (what we measured after training) ────────────────────────
    mlflow.log_metrics({
        "auc_roc":                  round(float(auc), 4),
        "optimal_threshold":        float(OPTIMAL_THRESHOLD),
        "recall_at_optimal":        round(float(optimal["recall"]), 4),
        "precision_at_optimal":     round(float(optimal["precision"]), 4),
        "caught_fraud":             int(optimal["caught_fraud"]),
        "missed_fraud":             int(optimal["missed_fraud"]),
        "false_declines":           int(optimal["false_declines"]),
        "total_cost_at_optimal":    float(optimal["total_cost"]),
        "savings_vs_default_050":   savings if savings else 0.0,
    })

    # ── Tags (metadata for searching runs) ──────────────────────────────
    mlflow.set_tags({
        "project":      "fraud-loss-analytics",
        "week":         "7",
        "dataset":      "kaggle-credit-card-fraud-1.85M",
        "threshold_strategy": "cost_optimised",
        "osfi_compliant":     "pending_shap",
    })

    # ── Artifacts (files attached to this run) ───────────────────────────
    cost_df.to_csv("/tmp/threshold_analysis.csv", index=False)
    mlflow.log_artifact("/tmp/threshold_analysis.csv", artifact_path="model_outputs")

    summary.to_csv("/tmp/model_summary.csv", index=False)
    mlflow.log_artifact("/tmp/model_summary.csv", artifact_path="model_outputs")

    # ── Log the trained Spark ML pipeline ───────────────────────────────
    mlflow.spark.log_model(model, "fraud_gbt_pipeline")

    run_id = mlflow.active_run().info.run_id
    print(f"\nMLflow run logged successfully.")
    print(f"  Run ID:            {run_id}")
    print(f"  AUC-ROC:           {auc:.4f}")
    print(f"  Optimal threshold: {OPTIMAL_THRESHOLD}")
    print(f"  Recall:            {optimal['recall']:.1%}")
    print(f"  Total cost:        ${optimal['total_cost']:,.2f}")
    if savings:
        print(f"  Savings vs 0.50:   ${savings:,.2f}")
    print(f"\nView in Databricks: Experiments → fraud-analytics/fraud-model")
