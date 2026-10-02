# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 2 — Gold Layer (Star Schema)
# MAGIC
# MAGIC Reads `silver.transactions` and builds a star schema:
# MAGIC
# MAGIC ```
# MAGIC                     ┌─────────────────┐
# MAGIC                     │  dim_datetime   │
# MAGIC                     │  (datetime_key) │
# MAGIC                     └────────┬────────┘
# MAGIC                              │
# MAGIC ┌──────────────┐   ┌────────┴─────────┐   ┌──────────────┐
# MAGIC │ dim_merchant ├───┤ fact_transactions ├───┤ dim_customer │
# MAGIC │(merchant_key)│   │                  │   │(customer_key)│
# MAGIC └──────────────┘   └──────────────────┘   └──────────────┘
# MAGIC ```
# MAGIC
# MAGIC **Prerequisite:** run `02_silver` first so `silver.transactions` exists.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read silver

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

silver = spark.table("silver.transactions")
print(f"Silver rows: {silver.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. dim_datetime
# MAGIC
# MAGIC One row per unique hour bucket.  The surrogate key is the hour expressed as
# MAGIC `yyyyMMddHH` (e.g., `2019010214`) — a compact integer that is also human-readable
# MAGIC and fully deterministic without a sequence generator.

# COMMAND ----------

dim_datetime = (
    silver
    .select(
        F.date_format("txn_time", "yyyyMMddHH").cast("long").alias("datetime_key"),
        F.to_date("txn_time").alias("txn_date"),
        F.year("txn_time").alias("year"),
        F.month("txn_time").alias("month"),
        F.dayofmonth("txn_time").alias("day"),
        F.hour("txn_time").alias("hour"),
        F.dayofweek("txn_time").alias("day_of_week"),   # 1 = Sunday … 7 = Saturday
        F.when(F.dayofweek("txn_time").isin(1, 7), 1)
         .otherwise(0)
         .alias("is_weekend"),
    )
    .distinct()
)

print(f"dim_datetime rows: {dim_datetime.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. dim_merchant
# MAGIC
# MAGIC One row per unique merchant.  `dense_rank()` over merchant name gives a clean,
# MAGIC deterministic integer surrogate key — the same merchant always gets the same key
# MAGIC on every full-overwrite run.

# COMMAND ----------

dim_merchant = (
    silver
    .select("merchant", "category", "merch_lat", "merch_long")
    .distinct()
    .withColumn(
        "merchant_key",
        F.dense_rank().over(Window.orderBy("merchant"))
    )
)

print(f"dim_merchant rows: {dim_merchant.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. dim_customer
# MAGIC
# MAGIC One row per masked card number.  PII was already stripped in silver — the
# MAGIC customer dimension contains only analytical attributes.

# COMMAND ----------

dim_customer = (
    silver
    .select("cc_num_masked", "gender", "city", "state", "zip",
            "lat", "long", "city_pop", "job", "birth_year")
    .distinct()
    .withColumn(
        "customer_key",
        F.dense_rank().over(Window.orderBy("cc_num_masked"))
    )
)

print(f"dim_customer rows: {dim_customer.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. fact_transactions
# MAGIC
# MAGIC Joins silver to the three dimensions to replace natural keys with surrogate keys,
# MAGIC then keeps only the measurable facts (amount, fraud flag) plus the foreign keys.

# COMMAND ----------

fact_transactions = (
    silver.alias("s")
    .join(
        dim_merchant.select("merchant", "merchant_key").alias("m"),
        on="merchant",
        how="inner",
    )
    .join(
        dim_customer.select("cc_num_masked", "customer_key").alias("c"),
        on="cc_num_masked",
        how="inner",
    )
    .select(
        F.col("s.trans_num"),
        F.date_format("s.txn_time", "yyyyMMddHH").cast("long").alias("datetime_key"),
        F.col("m.merchant_key"),
        F.col("c.customer_key"),
        F.col("s.amt"),
        F.col("s.is_fraud"),
        F.to_date("s.txn_time").alias("txn_date"),
    )
)

print(f"fact_transactions rows: {fact_transactions.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Write all four tables as Delta

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS gold")

tables = {
    "gold.dim_datetime":     dim_datetime,
    "gold.dim_merchant":     dim_merchant,
    "gold.dim_customer":     dim_customer,
    "gold.fact_transactions": fact_transactions,
}

for name, df in tables.items():
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(name)
    )
    print(f"Written: {name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Fraud KPIs — querying the star schema
# MAGIC
# MAGIC These are the same analytical questions answered in week 1's `profile_data.py`,
# MAGIC but now written as Spark SQL against the star schema instead of PostgreSQL CTEs.
# MAGIC See `docs/sql_migration_notes.md` for a side-by-side comparison.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Overall fraud summary
# MAGIC SELECT
# MAGIC   count(*)                                                AS total_txns,
# MAGIC   sum(is_fraud)                                          AS fraud_txns,
# MAGIC   round(100.0 * avg(is_fraud), 3)                        AS fraud_rate_pct,
# MAGIC   round(sum(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2) AS fraud_losses
# MAGIC FROM gold.fact_transactions

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Fraud by category (highest rate first)
# MAGIC SELECT
# MAGIC   m.category,
# MAGIC   count(*)                                                    AS transactions,
# MAGIC   sum(f.is_fraud)                                             AS fraud_txns,
# MAGIC   round(100.0 * avg(f.is_fraud), 3)                           AS fraud_rate_pct,
# MAGIC   round(sum(CASE WHEN f.is_fraud = 1 THEN f.amt ELSE 0 END), 2) AS fraud_amount,
# MAGIC   round(avg(CASE WHEN f.is_fraud = 1 THEN f.amt END), 2)      AS avg_fraud_ticket,
# MAGIC   round(avg(CASE WHEN f.is_fraud = 0 THEN f.amt END), 2)      AS avg_legit_ticket
# MAGIC FROM gold.fact_transactions f
# MAGIC JOIN gold.dim_merchant m ON m.merchant_key = f.merchant_key
# MAGIC GROUP BY m.category
# MAGIC ORDER BY fraud_rate_pct DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Fraud by hour of day
# MAGIC SELECT
# MAGIC   d.hour,
# MAGIC   count(*)                                                    AS transactions,
# MAGIC   sum(f.is_fraud)                                             AS fraud_txns,
# MAGIC   round(100.0 * avg(f.is_fraud), 3)                           AS fraud_rate_pct
# MAGIC FROM gold.fact_transactions f
# MAGIC JOIN gold.dim_datetime d ON d.datetime_key = f.datetime_key
# MAGIC GROUP BY d.hour
# MAGIC ORDER BY d.hour

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Top 10 merchants by fraud loss
# MAGIC SELECT
# MAGIC   m.merchant,
# MAGIC   count(*)                                                      AS transactions,
# MAGIC   sum(f.is_fraud)                                               AS fraud_txns,
# MAGIC   round(sum(CASE WHEN f.is_fraud = 1 THEN f.amt ELSE 0 END), 2) AS fraud_amount
# MAGIC FROM gold.fact_transactions f
# MAGIC JOIN gold.dim_merchant m ON m.merchant_key = f.merchant_key
# MAGIC GROUP BY m.merchant
# MAGIC HAVING sum(f.is_fraud) > 0
# MAGIC ORDER BY fraud_amount DESC
# MAGIC LIMIT 10
