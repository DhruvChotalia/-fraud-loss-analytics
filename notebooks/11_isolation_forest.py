# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 11 — Isolation Forest Anomaly Detection
# MAGIC
# MAGIC **What this notebook builds:**
# MAGIC An unsupervised anomaly detection layer that runs alongside the supervised
# MAGIC GBT fraud model. Where the GBT catches known fraud patterns it learned from
# MAGIC labelled data, Isolation Forest catches statistically unusual transactions
# MAGIC regardless of whether they match any known fraud pattern.
# MAGIC
# MAGIC **How Isolation Forest works:**
# MAGIC It builds random decision trees and measures how many splits it takes to
# MAGIC isolate a transaction. Normal transactions are similar to many others —
# MAGIC they take many splits to isolate. Anomalies are unusual — they get isolated
# MAGIC quickly, in very few splits. A low anomaly score = isolated quickly = unusual.
# MAGIC
# MAGIC **Three outputs:**
# MAGIC 1. Anomaly scores for every transaction
# MAGIC 2. Overlap analysis: do anomalies align with actual fraud?
# MAGIC 3. Blind spot analysis: transactions Isolation Forest flags that the GBT missed
# MAGIC    — these are potential unknown fraud patterns worth investigating
# MAGIC
# MAGIC **Prerequisite:** Notebooks 02 and 05 complete (silver.transactions + gold.fraud_scores).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data and engineer features

# COMMAND ----------

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load silver — same features as the GBT model
silver_pd = spark.sql("""
    SELECT
        trans_num,
        txn_time,
        amt,
        is_fraud,
        category,
        gender,
        city_pop,
        lat, long,
        merch_lat, merch_long
    FROM silver.transactions
""").toPandas()

print(f"Loaded {len(silver_pd):,} rows")

# Feature engineering — mirrors notebook 05 and 07
silver_pd["txn_time"]    = pd.to_datetime(silver_pd["txn_time"])
silver_pd["hour"]        = silver_pd["txn_time"].dt.hour
silver_pd["day_of_week"] = silver_pd["txn_time"].dt.dayofweek + 1
silver_pd["month"]       = silver_pd["txn_time"].dt.month
silver_pd["is_weekend"]  = silver_pd["day_of_week"].isin([1, 7]).astype(int)
silver_pd["geo_distance"] = np.sqrt(
    (silver_pd["merch_lat"] - silver_pd["lat"]) ** 2 +
    (silver_pd["merch_long"] - silver_pd["long"]) ** 2
)

le_cat = LabelEncoder()
le_gen = LabelEncoder()
silver_pd["category_idx"] = le_cat.fit_transform(silver_pd["category"].astype(str))
silver_pd["gender_idx"]   = le_gen.fit_transform(silver_pd["gender"].astype(str))

FEATURE_COLS = [
    "amt", "hour", "day_of_week", "month", "is_weekend",
    "category_idx", "gender_idx", "city_pop", "geo_distance",
]

FEATURE_NAMES = [
    "Transaction Amount ($)", "Hour of Day", "Day of Week", "Month",
    "Is Weekend", "Merchant Category", "Gender",
    "City Population", "Customer–Merchant Distance",
]

print("Features engineered.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train Isolation Forest
# MAGIC
# MAGIC Key parameters:
# MAGIC - `contamination=0.005` — we tell the model roughly 0.5% of transactions
# MAGIC   are anomalies (matches our observed fraud rate of 0.52%)
# MAGIC - `n_estimators=200` — more trees = more stable scores
# MAGIC - `random_state=42` — reproducible results
# MAGIC
# MAGIC **Important:** we do NOT use the `is_fraud` label anywhere in training.
# MAGIC This is fully unsupervised.

# COMMAND ----------

SAMPLE_SIZE  = 50_000   # train on a sample for speed; score all transactions
CONTAMINATION = 0.005   # ~0.5% anomaly rate, matching observed fraud rate

# Sample for training
train_sample = silver_pd.sample(n=min(SAMPLE_SIZE, len(silver_pd)), random_state=42)
X_train = train_sample[FEATURE_COLS].values

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=42,
    n_jobs=-1,
)
iso_forest.fit(X_train)
print(f"Isolation Forest trained on {len(X_train):,} samples.")
print(f"Contamination parameter: {CONTAMINATION} ({CONTAMINATION*100:.1f}%)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Score all transactions
# MAGIC
# MAGIC `decision_function()` returns a score for each transaction:
# MAGIC - **Negative score (< 0)** → anomaly (unusual transaction)
# MAGIC - **Positive score (> 0)** → normal transaction
# MAGIC - The more negative, the more anomalous
# MAGIC
# MAGIC `predict()` returns -1 (anomaly) or 1 (normal) based on the contamination threshold.

# COMMAND ----------

X_all = silver_pd[FEATURE_COLS].values

# Score in batches to avoid memory issues on large datasets
BATCH_SIZE = 100_000
scores_list = []
predictions_list = []

for i in range(0, len(X_all), BATCH_SIZE):
    batch = X_all[i:i + BATCH_SIZE]
    scores_list.append(iso_forest.decision_function(batch))
    predictions_list.append(iso_forest.predict(batch))
    print(f"  Scored {min(i + BATCH_SIZE, len(X_all)):,} / {len(X_all):,}")

silver_pd["anomaly_score"]      = np.concatenate(scores_list)
silver_pd["is_anomaly"]         = (np.concatenate(predictions_list) == -1).astype(int)

total_anomalies = silver_pd["is_anomaly"].sum()
print(f"\nTotal anomalies flagged: {total_anomalies:,} ({total_anomalies/len(silver_pd):.2%})")
print(f"Actual fraud count:      {silver_pd['is_fraud'].sum():,} ({silver_pd['is_fraud'].mean():.2%})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Overlap analysis
# MAGIC
# MAGIC How well does Isolation Forest (unsupervised) align with actual fraud labels?
# MAGIC We're not optimising for this — an unsupervised model won't match labelled
# MAGIC fraud perfectly. But meaningful overlap validates the approach.

# COMMAND ----------

# Confusion matrix between anomaly flags and actual fraud labels
tp = ((silver_pd["is_anomaly"] == 1) & (silver_pd["is_fraud"] == 1)).sum()
fp = ((silver_pd["is_anomaly"] == 1) & (silver_pd["is_fraud"] == 0)).sum()
fn = ((silver_pd["is_anomaly"] == 0) & (silver_pd["is_fraud"] == 1)).sum()
tn = ((silver_pd["is_anomaly"] == 0) & (silver_pd["is_fraud"] == 0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

print("ISOLATION FOREST vs ACTUAL FRAUD LABELS")
print("=" * 55)
print(f"Flagged as anomaly AND actual fraud:    {tp:>7,}  (true positives)")
print(f"Flagged as anomaly but NOT fraud:       {fp:>7,}  (false positives — unusual but legit)")
print(f"NOT flagged but actual fraud:           {fn:>7,}  (missed fraud)")
print(f"NOT flagged AND not fraud:              {tn:>7,}  (true negatives)")
print()
print(f"Precision (of anomalies, how many are fraud): {precision:.1%}")
print(f"Recall    (of fraud, how many flagged):       {recall:.1%}")
print()
print("Note: precision ~0.5-2% is expected for unsupervised detection")
print("on a 0.5% fraud rate dataset — the value is in finding unknowns.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Blind spot analysis — what did the GBT miss?
# MAGIC
# MAGIC This is the most valuable output. We join the Isolation Forest anomaly scores
# MAGIC with the GBT fraud scores from Week 4/10. Transactions that:
# MAGIC - Isolation Forest flags as anomalous (is_anomaly = 1)
# MAGIC - GBT did NOT flag as fraud (predicted_fraud = 0)
# MAGIC - Are actually fraud (is_fraud = 1)
# MAGIC
# MAGIC ...are the GBT's blind spots — fraud the supervised model missed but the
# MAGIC unsupervised model caught. These are worth investigating for new fraud patterns.

# COMMAND ----------

# Load GBT scores from Week 5
gbt_scores = spark.table("gold.fraud_scores").toPandas()
gbt_scores = gbt_scores.rename(columns={"predicted_fraud": "gbt_predicted"})

# Join on transaction ID
combined = silver_pd.merge(
    gbt_scores[["trans_num", "fraud_prob", "gbt_predicted"]],
    on="trans_num",
    how="inner"
)

print(f"Joined {len(combined):,} transactions with GBT scores")

# Find blind spots: IF flagged, GBT missed, actually fraud
blind_spots = combined[
    (combined["is_anomaly"]     == 1) &   # IF says anomaly
    (combined["gbt_predicted"]  == 0) &   # GBT said legitimate
    (combined["is_fraud"]       == 1)     # actually fraud
].copy()

# Find new signals: IF flagged, actually fraud (regardless of GBT)
if_only_fraud = combined[
    (combined["is_anomaly"] == 1) &
    (combined["is_fraud"]   == 1)
].copy()

print(f"\nBLIND SPOT ANALYSIS")
print("=" * 55)
print(f"Fraud caught by GBT only (GBT=1, IF=0):      {((combined['gbt_predicted']==1) & (combined['is_anomaly']==0) & (combined['is_fraud']==1)).sum():,}")
print(f"Fraud caught by both GBT and IF:              {((combined['gbt_predicted']==1) & (combined['is_anomaly']==1) & (combined['is_fraud']==1)).sum():,}")
print(f"Fraud caught by IF only (GBT=0, IF=1):        {len(blind_spots):,}  ← GBT blind spots")
print(f"Fraud missed by both:                         {((combined['gbt_predicted']==0) & (combined['is_anomaly']==0) & (combined['is_fraud']==1)).sum():,}")

if len(blind_spots) > 0:
    print(f"\nTop 10 GBT blind spots (highest anomaly suspicion):")
    print(blind_spots.nsmallest(10, "anomaly_score")[
        ["trans_num", "amt", "hour", "category", "geo_distance", "anomaly_score", "fraud_prob"]
    ].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Anomaly score distribution plot

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: anomaly score distribution by fraud label
fraud_scores  = combined.loc[combined["is_fraud"] == 1, "anomaly_score"]
legit_scores  = combined.loc[combined["is_fraud"] == 0, "anomaly_score"].sample(
    n=min(50_000, (combined["is_fraud"] == 0).sum()), random_state=42
)

axes[0].hist(legit_scores,  bins=60, alpha=0.6, color="#2980B9", label="Legitimate", density=True)
axes[0].hist(fraud_scores,  bins=60, alpha=0.7, color="#C0392B", label="Fraud",      density=True)
axes[0].axvline(0, color="black", linestyle="--", linewidth=1, label="Decision boundary")
axes[0].set_xlabel("Anomaly Score  (negative = more anomalous)")
axes[0].set_ylabel("Density")
axes[0].set_title("Isolation Forest Score Distribution\nFraud vs Legitimate Transactions")
axes[0].legend()
axes[0].spines["top"].set_visible(False)
axes[0].spines["right"].set_visible(False)

# Plot 2: GBT fraud_prob vs IF anomaly score (scatter — sample)
sample_plot = combined.sample(n=min(5_000, len(combined)), random_state=42)
colors = sample_plot["is_fraud"].map({0: "#2980B9", 1: "#C0392B"})
axes[1].scatter(
    sample_plot["fraud_prob"],
    sample_plot["anomaly_score"],
    c=colors, alpha=0.3, s=5
)
axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8, label="IF boundary")
axes[1].axvline(0.60, color="orange", linestyle="--", linewidth=0.8, label="GBT threshold (0.60)")
axes[1].set_xlabel("GBT Fraud Probability")
axes[1].set_ylabel("Isolation Forest Anomaly Score")
axes[1].set_title("GBT vs Isolation Forest\n(red = actual fraud, blue = legitimate)")
axes[1].legend(fontsize=8)
axes[1].spines["top"].set_visible(False)
axes[1].spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("/tmp/isolation_forest_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print("Plot saved: /tmp/isolation_forest_analysis.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save results to ADLS and log to MLflow

# COMMAND ----------

import mlflow
from azure.storage.blob import BlobServiceClient
import io

STORAGE_ACCOUNT = "fraudanalytics2024"
STORAGE_KEY      = "YOUR_STORAGE_KEY_HERE"
CONTAINER        = "models"

blob_svc = BlobServiceClient(
    account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
    credential=STORAGE_KEY
)

def upload_csv(df, blob_name):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    blob_svc.get_blob_client(container=CONTAINER, blob=blob_name) \
            .upload_blob(buf.getvalue(), overwrite=True)
    print(f"Uploaded: {blob_name}")

def upload_file(local_path, blob_name):
    with open(local_path, "rb") as f:
        blob_svc.get_blob_client(container=CONTAINER, blob=blob_name) \
                .upload_blob(f, overwrite=True)
    print(f"Uploaded: {blob_name}")

# Save anomaly scores for all transactions
anomaly_output = combined[[
    "trans_num", "amt", "is_fraud", "anomaly_score", "is_anomaly",
    "fraud_prob", "gbt_predicted", "hour", "category", "geo_distance"
]].copy()
upload_csv(anomaly_output, "isolation_forest/anomaly_scores.csv")

# Save blind spots
if len(blind_spots) > 0:
    upload_csv(blind_spots, "isolation_forest/gbt_blind_spots.csv")

# Save plot
upload_file("/tmp/isolation_forest_analysis.png", "isolation_forest/analysis_plot.png")

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow.set_experiment("/Users/dhruvextra76@gmail.com/isolation-forest")

with mlflow.start_run(run_name="isolation_forest_v1"):

    mlflow.log_params({
        "n_estimators":        200,
        "contamination":       CONTAMINATION,
        "random_state":        42,
        "train_sample_size":   SAMPLE_SIZE,
        "features":            ",".join(FEATURE_COLS),
    })

    mlflow.log_metrics({
        "total_anomalies_flagged":  int(total_anomalies),
        "anomaly_rate_pct":         round(total_anomalies / len(silver_pd) * 100, 3),
        "overlap_precision":        round(precision, 4),
        "overlap_recall":           round(recall, 4),
        "gbt_blind_spots":          int(len(blind_spots)),
        "fraud_caught_by_both":     int(((combined["gbt_predicted"]==1) & (combined["is_anomaly"]==1) & (combined["is_fraud"]==1)).sum()),
        "fraud_gbt_only":           int(((combined["gbt_predicted"]==1) & (combined["is_anomaly"]==0) & (combined["is_fraud"]==1)).sum()),
        "fraud_if_only":            int(len(blind_spots)),
    })

    mlflow.set_tags({
        "project":      "fraud-loss-analytics",
        "week":         "11",
        "model_type":   "IsolationForest",
        "approach":     "unsupervised_anomaly_detection",
    })

    mlflow.log_artifact("/tmp/isolation_forest_analysis.png", "plots")

    anomaly_output.to_csv("/tmp/anomaly_scores.csv", index=False)
    mlflow.log_artifact("/tmp/anomaly_scores.csv", "outputs")

    print("Isolation Forest results logged to MLflow.")
    print("View: Experiments → isolation-forest")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary — What Week 11 adds
# MAGIC
# MAGIC | GBT Fraud Model (supervised) | Isolation Forest (unsupervised) |
# MAGIC |---|---|
# MAGIC | Trained on labelled fraud data | No labels needed |
# MAGIC | Catches known fraud patterns | Catches statistically unusual transactions |
# MAGIC | High precision, optimised threshold | Broad net — catches things GBT misses |
# MAGIC | Production scoring | Early warning layer |
# MAGIC
# MAGIC **The key insight from this notebook:**
# MAGIC The GBT blind spots — fraud transactions Isolation Forest flagged but GBT missed —
# MAGIC are a signal worth investigating. They may represent a new fraud pattern that wasn't
# MAGIC in the training data. In production, these would be routed to a human analyst for
# MAGIC manual review rather than being auto-approved.
# MAGIC
# MAGIC **Interview answer:**
# MAGIC "We run Isolation Forest as an unsupervised layer alongside the supervised GBT model.
# MAGIC Where the GBT catches fraud that looks like historical fraud patterns, Isolation Forest
# MAGIC flags anything statistically unusual — regardless of whether it matches a known pattern.
# MAGIC The blind spot analysis shows transactions the GBT approved but Isolation Forest flagged
# MAGIC as anomalous — these are routed to manual review as potential unknown fraud schemes.
# MAGIC In practice, a new synthetic identity fraud scheme would likely show up here before
# MAGIC the supervised model catches it."
