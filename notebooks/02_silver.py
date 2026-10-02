# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 2 — Silver Layer
# MAGIC
# MAGIC Reads the raw Kaggle CSVs from DBFS, casts every column to its correct type,
# MAGIC masks PII, and writes `silver.transactions` as a Delta table.
# MAGIC
# MAGIC **One-time setup before running this notebook:**
# MAGIC 1. In the Databricks left sidebar click **Data → Add Data → DBFS → FileStore**
# MAGIC 2. Upload `fraudTrain.csv` and `fraudTest.csv` into `/FileStore/fraud/`
# MAGIC 3. Attach this notebook to a running cluster (DBR 13.x or later)
# MAGIC 4. Run all cells top to bottom (**Run All**)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Read raw CSVs from DBFS
# MAGIC
# MAGIC `inferSchema=false` keeps every field as a string — the same principle as the
# MAGIC PostgreSQL bronze layer. We cast explicitly in the next step so nothing is
# MAGIC silently converted or dropped.

# COMMAND ----------

from pyspark.sql import functions as F

RAW_PATH = "/Volumes/workspace/default/fraud/"

df_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .csv([RAW_PATH + "fraudTrain.csv", RAW_PATH + "fraudTest.csv"])
)

# The first column has a blank header in the Kaggle CSV; Spark names it _c0
df_raw = df_raw.withColumnRenamed(df_raw.columns[0], "src_index")

print(f"Raw rows loaded: {df_raw.count():,}")
df_raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Cast types and mask PII
# MAGIC
# MAGIC | Column | Treatment | Reason |
# MAGIC |---|---|---|
# MAGIC | `cc_num` | Last 4 digits only (`xxxx-xxxx-xxxx-1234`) | PCI-DSS: full card number must not sit in an analytical store |
# MAGIC | `first`, `last` | **Dropped** | Names add no analytical value; removing them is simpler than hashing |
# MAGIC | `street` | **Dropped** | City, state, zip are sufficient for geo analysis |
# MAGIC | `dob` | Birth year only | Age band is useful; exact date of birth is PII |

# COMMAND ----------

silver = df_raw.select(
    F.col("src_index").cast("long"),
    F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss").alias("txn_time"),
    F.concat(
        F.lit("xxxx-xxxx-xxxx-"),
        F.substring("cc_num", -4, 4)
    ).alias("cc_num_masked"),
    F.col("merchant"),
    F.col("category"),
    F.col("amt").cast("decimal(12,2)"),
    # first, last → dropped (PII)
    F.col("gender"),
    # street → dropped (PII)
    F.col("city"),
    F.col("state"),
    F.col("zip"),
    F.col("lat").cast("double"),
    F.col("long").cast("double"),
    F.col("city_pop").cast("integer"),
    F.col("job"),
    F.year(
        F.to_date("dob", "yyyy-MM-dd")
    ).alias("birth_year"),
    F.col("trans_num"),
    F.col("unix_time").cast("long"),
    F.col("merch_lat").cast("double"),
    F.col("merch_long").cast("double"),
    F.col("is_fraud").cast("integer"),
    F.current_timestamp().alias("_loaded_at"),
)

silver.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reconciliation check
# MAGIC
# MAGIC Same principle as the week 1 PostgreSQL controls: verify nothing was lost or
# MAGIC gained in the transformation before writing.  The expected count comes from
# MAGIC `audit.etl_run_log` in the Postgres bronze layer.

# COMMAND ----------

EXPECTED_ROWS = 1_852_394  # fraudTrain (1,296,675) + fraudTest (555,719)

actual_rows = silver.count()
status = "PASS" if actual_rows == EXPECTED_ROWS else "FAIL"
print(f"[{status}] row count  expected={EXPECTED_ROWS:,}  actual={actual_rows:,}")

if status == "FAIL":
    raise ValueError("Row count mismatch — do not write silver until this is resolved.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Write as Delta table

# COMMAND ----------

spark.sql("CREATE DATABASE IF NOT EXISTS silver")

(
    silver.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable("silver.transactions")
)

print("Written: silver.transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify the written table

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   count(*)                                          AS total_rows,
# MAGIC   count(DISTINCT trans_num)                         AS distinct_txns,
# MAGIC   sum(is_fraud)                                     AS fraud_txns,
# MAGIC   round(sum(amt), 2)                                AS total_amount,
# MAGIC   round(100.0 * avg(is_fraud), 3)                   AS fraud_rate_pct,
# MAGIC   min(txn_time)                                     AS earliest_txn,
# MAGIC   max(txn_time)                                     AS latest_txn
# MAGIC FROM silver.transactions

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Confirm PII is gone

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Should show xxxx-xxxx-xxxx-NNNN format only; no real card numbers
# MAGIC SELECT cc_num_masked, birth_year
# MAGIC FROM silver.transactions
# MAGIC LIMIT 5
