# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 3 — Fraud KPIs & Merchant Clustering (K-Means)
# MAGIC
# MAGIC Two deliverables in this notebook:
# MAGIC 1. **Fraud KPIs** — the key business metrics an analyst or dashboard would surface
# MAGIC 2. **K-Means merchant clustering** — group the 693 merchants by their fraud behaviour
# MAGIC    using PySpark MLlib, the same library used on production Databricks clusters
# MAGIC
# MAGIC **Prerequisite:** run `02_silver` first so `silver.transactions` exists.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1 — Fraud KPIs

# COMMAND ----------

# MAGIC %md
# MAGIC ### KPI 1 — Portfolio overview

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   count(*)                                                        AS total_transactions,
# MAGIC   sum(is_fraud)                                                   AS fraud_transactions,
# MAGIC   round(100.0 * avg(is_fraud), 3)                                 AS fraud_rate_pct,
# MAGIC   round(sum(amt), 2)                                              AS total_portfolio_value,
# MAGIC   round(sum(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2)       AS total_fraud_losses,
# MAGIC   round(
# MAGIC     100.0 * sum(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END) / sum(amt),
# MAGIC     3
# MAGIC   )                                                               AS loss_rate_pct
# MAGIC FROM silver.transactions

# COMMAND ----------

# MAGIC %md
# MAGIC ### KPI 2 — Fraud by category (highest risk first)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   category,
# MAGIC   count(*)                                                          AS transactions,
# MAGIC   sum(is_fraud)                                                     AS fraud_txns,
# MAGIC   round(100.0 * avg(is_fraud), 3)                                   AS fraud_rate_pct,
# MAGIC   round(sum(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2)         AS fraud_losses,
# MAGIC   round(avg(CASE WHEN is_fraud = 1 THEN amt END), 2)                AS avg_fraud_ticket,
# MAGIC   round(avg(CASE WHEN is_fraud = 0 THEN amt END), 2)                AS avg_legit_ticket,
# MAGIC   round(avg(CASE WHEN is_fraud = 1 THEN amt END) /
# MAGIC         avg(CASE WHEN is_fraud = 0 THEN amt END), 1)                AS ticket_ratio
# MAGIC FROM silver.transactions
# MAGIC GROUP BY category
# MAGIC ORDER BY fraud_rate_pct DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### KPI 3 — Fraud by hour of day

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   hour(txn_time)                                                   AS hour,
# MAGIC   count(*)                                                         AS transactions,
# MAGIC   sum(is_fraud)                                                    AS fraud_txns,
# MAGIC   round(100.0 * avg(is_fraud), 3)                                  AS fraud_rate_pct
# MAGIC FROM silver.transactions
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### KPI 4 — Fraud by US state (top 15 by losses)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   state,
# MAGIC   count(*)                                                          AS transactions,
# MAGIC   sum(is_fraud)                                                     AS fraud_txns,
# MAGIC   round(100.0 * avg(is_fraud), 3)                                   AS fraud_rate_pct,
# MAGIC   round(sum(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2)         AS fraud_losses
# MAGIC FROM silver.transactions
# MAGIC GROUP BY state
# MAGIC ORDER BY fraud_losses DESC
# MAGIC LIMIT 15

# COMMAND ----------

# MAGIC %md
# MAGIC ### KPI 5 — Monthly fraud loss trend

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   date_format(txn_time, 'yyyy-MM')                                  AS month,
# MAGIC   count(*)                                                          AS transactions,
# MAGIC   sum(is_fraud)                                                     AS fraud_txns,
# MAGIC   round(100.0 * avg(is_fraud), 3)                                   AS fraud_rate_pct,
# MAGIC   round(sum(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2)         AS fraud_losses
# MAGIC FROM silver.transactions
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1

# COMMAND ----------

# MAGIC %md
# MAGIC ### KPI 6 — Weekend vs weekday fraud

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   CASE WHEN dayofweek(txn_time) IN (1, 7) THEN 'Weekend' ELSE 'Weekday' END AS day_type,
# MAGIC   count(*)                                                          AS transactions,
# MAGIC   sum(is_fraud)                                                     AS fraud_txns,
# MAGIC   round(100.0 * avg(is_fraud), 3)                                   AS fraud_rate_pct,
# MAGIC   round(sum(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2)         AS fraud_losses
# MAGIC FROM silver.transactions
# MAGIC GROUP BY 1
# MAGIC ORDER BY 1

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Part 2 — K-Means Merchant Clustering
# MAGIC
# MAGIC **Goal:** group the 693 merchants into clusters based on their fraud behaviour.
# MAGIC Each merchant gets 5 features:
# MAGIC
# MAGIC | Feature | What it measures |
# MAGIC |---|---|
# MAGIC | `fraud_rate_pct` | How risky this merchant is |
# MAGIC | `avg_fraud_ticket` | How much a fraudster spends here |
# MAGIC | `avg_legit_ticket` | Normal transaction size |
# MAGIC | `txn_volume` | How busy this merchant is |
# MAGIC | `fraud_loss_share` | This merchant's share of total portfolio fraud losses |
# MAGIC
# MAGIC We scale all features to mean=0, std=1 before clustering so no single feature
# MAGIC dominates just because of its units.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 1 — Build merchant feature table

# COMMAND ----------

from pyspark.sql import functions as F

merchant_features = spark.sql("""
    SELECT
        merchant,
        category,
        count(*)                                                          AS txn_volume,
        round(100.0 * avg(is_fraud), 4)                                   AS fraud_rate_pct,
        round(avg(CASE WHEN is_fraud = 1 THEN amt END), 4)                AS avg_fraud_ticket,
        round(avg(CASE WHEN is_fraud = 0 THEN amt END), 4)                AS avg_legit_ticket,
        round(sum(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 4)         AS fraud_losses
    FROM silver.transactions
    GROUP BY merchant, category
    HAVING count(*) > 0
""")

total_fraud = merchant_features.agg(F.sum("fraud_losses")).collect()[0][0]

merchant_features = merchant_features.withColumn(
    "fraud_loss_share",
    F.round(F.col("fraud_losses") / total_fraud * 100, 4)
)

print(f"Merchants to cluster: {merchant_features.count()}")
merchant_features.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2 — Assemble feature vector and scale

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

FEATURE_COLS = ["fraud_rate_pct", "avg_fraud_ticket", "avg_legit_ticket",
                "txn_volume", "fraud_loss_share"]

# Replace nulls — a merchant with zero fraud transactions has no avg_fraud_ticket
merchant_clean = merchant_features.fillna(0.0, subset=FEATURE_COLS)

assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features_raw")
scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                           withMean=True, withStd=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3 — Choose k with the elbow method
# MAGIC
# MAGIC We train K-Means for k = 2 to 8 and plot the within-cluster sum of squares (WCSS).
# MAGIC The "elbow" — where adding another cluster stops giving a big improvement — is our k.

# COMMAND ----------

from pyspark.ml.clustering import KMeans

prep_pipeline = Pipeline(stages=[assembler, scaler])
prep_model    = prep_pipeline.fit(merchant_clean)
scaled_data   = prep_model.transform(merchant_clean)

wcss = []
for k in range(2, 9):
    km = KMeans(featuresCol="features", k=k, seed=42)
    m  = km.fit(scaled_data)
    wcss.append((k, round(m.summary.trainingCost, 2)))
    print(f"  k={k}  WCSS={wcss[-1][1]:,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4 — Fit final model (k = 4)
# MAGIC
# MAGIC k=4 is the standard choice for a first merchant segmentation:
# MAGIC high-risk online, high-risk in-person, low-risk high-volume, low-risk low-volume.
# MAGIC Adjust after reviewing the elbow curve above.

# COMMAND ----------

K = 4

kmeans       = KMeans(featuresCol="features", k=K, seed=42)
final_model  = kmeans.fit(scaled_data)
clustered    = final_model.transform(scaled_data)

print(f"Cluster sizes:")
clustered.groupBy("prediction").count().orderBy("prediction").show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5 — Interpret the clusters
# MAGIC
# MAGIC Average each original (unscaled) feature per cluster so the numbers are human-readable.

# COMMAND ----------

cluster_profiles = (
    clustered
    .groupBy("prediction")
    .agg(
        F.count("*").alias("merchant_count"),
        F.round(F.avg("fraud_rate_pct"),    3).alias("avg_fraud_rate_pct"),
        F.round(F.avg("avg_fraud_ticket"),  2).alias("avg_fraud_ticket"),
        F.round(F.avg("avg_legit_ticket"),  2).alias("avg_legit_ticket"),
        F.round(F.avg("txn_volume"),        0).alias("avg_txn_volume"),
        F.round(F.avg("fraud_loss_share"),  4).alias("avg_loss_share_pct"),
    )
    .orderBy("avg_fraud_rate_pct", ascending=False)
)

cluster_profiles.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6 — Label and save clusters

# COMMAND ----------

# Assign human-readable labels based on fraud rate ranking
# (run Step 5 first, then adjust these labels to match your actual cluster numbers)
CLUSTER_LABELS = {
    # prediction → label  (update these after reading the cluster_profiles output above)
    0: "High-risk online",
    1: "High-risk in-person",
    2: "Low-risk high-volume",
    3: "Low-risk low-volume",
}

label_map = (
    spark.createDataFrame(
        [(k, v) for k, v in CLUSTER_LABELS.items()],
        ["prediction", "cluster_label"]
    )
)

merchant_clusters = (
    clustered
    .join(label_map, on="prediction", how="left")
    .select("merchant", "category", "prediction", "cluster_label",
            "txn_volume", "fraud_rate_pct", "avg_fraud_ticket",
            "avg_legit_ticket", "fraud_losses", "fraud_loss_share")
    .orderBy("prediction", "merchant")
)

spark.sql("CREATE DATABASE IF NOT EXISTS gold")

(
    merchant_clusters.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("gold.merchant_clusters")
)

print("Written: gold.merchant_clusters")
merchant_clusters.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7 — Top fraud merchants per cluster

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   cluster_label,
# MAGIC   merchant,
# MAGIC   category,
# MAGIC   txn_volume,
# MAGIC   fraud_rate_pct,
# MAGIC   avg_fraud_ticket,
# MAGIC   fraud_losses
# MAGIC FROM (
# MAGIC   SELECT *,
# MAGIC     row_number() OVER (PARTITION BY cluster_label ORDER BY fraud_losses DESC) AS rn
# MAGIC   FROM gold.merchant_clusters
# MAGIC ) t
# MAGIC WHERE rn <= 3
# MAGIC ORDER BY cluster_label, fraud_losses DESC
