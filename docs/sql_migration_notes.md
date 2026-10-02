# SQL → Spark SQL Migration Notes

Week 1 used PostgreSQL for the bronze/controls layer.
Week 2 moved the analytical layers (silver, gold) to Databricks PySpark.
This document shows the key SQL patterns side-by-side so the translation is clear.

---

## 1. Schema / table creation

**PostgreSQL (week 1)**
```sql
CREATE SCHEMA IF NOT EXISTS bronze;

CREATE TABLE IF NOT EXISTS bronze.transactions_raw (
    trans_num  TEXT,
    amt        TEXT,
    is_fraud   TEXT,
    _run_id    UUID NOT NULL,
    _loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
    -- ... all columns as TEXT
);
```

**PySpark / Databricks (week 2)**
```python
spark.sql("CREATE DATABASE IF NOT EXISTS silver")

df.write \
  .format("delta") \
  .mode("overwrite") \
  .option("overwriteSchema", "true") \
  .saveAsTable("silver.transactions")
```

**What changed and why:**
PostgreSQL needs explicit DDL before you can insert. Spark/Delta infers the schema
from the DataFrame and creates the table in one step. Delta also adds ACID transactions,
time-travel (versioned history), and `OPTIMIZE` / `VACUUM` commands for production use.

---

## 2. Bulk data load

**PostgreSQL (week 1)**
```python
# psycopg3 COPY — the fastest bulk-insert path for Postgres
with cur.copy("COPY bronze.transactions_raw (...) FROM STDIN WITH (FORMAT csv)") as cp:
    cp.write(buf.getvalue())
```

**PySpark (week 2)**
```python
df = spark.read \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .csv("/Volumes/workspace/default/fraud/fraudTrain.csv")
```

**What changed and why:**
PostgreSQL COPY streams a buffer into a single-node database. Spark reads CSVs in
parallel partitions across a distributed cluster — each executor reads a slice of
the file simultaneously, which is what makes Spark fast on large datasets.

---

## 3. Explicit type casting

**PostgreSQL (week 1)**
```sql
-- Cast inside a query
SELECT amt::numeric, trans_date_trans_time::timestamp
FROM bronze.transactions_raw
WHERE amt ~ '^-?[0-9]+(\.[0-9]+)?$'
```

**PySpark (week 2)**
```python
from pyspark.sql import functions as F

df.select(
    F.col("amt").cast("decimal(12,2)"),
    F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss").alias("txn_time"),
)
```

**What changed and why:**
PostgreSQL uses `::type` cast syntax or `CAST(col AS type)`. PySpark uses `.cast()`
on Column objects or `F.to_timestamp()` / `F.to_date()` for date parsing.
The `inferSchema=false` on read keeps everything as string first — same philosophy
as the bronze layer: cast explicitly, never silently.

---

## 4. String regex validation

**PostgreSQL (week 1) — DQ rule predicate**
```sql
-- Identifies BAD rows where amount is not numeric
amt IS NULL OR amt !~ '^-?[0-9]+(\.[0-9]+)?$'
```

**PySpark equivalent**
```python
# Flag bad rows the same way
from pyspark.sql import functions as F

bad_amt = df.filter(
    F.col("amt").isNull() |
    ~F.col("amt").rlike(r"^-?[0-9]+(\.[0-9]+)?$")
)
```

**What changed and why:**
PostgreSQL uses `~` for regex match and `!~` for "does not match".
PySpark uses `.rlike()` (regex like) and `~` is the Python bitwise NOT used to negate
a Column expression. The regex syntax itself is identical.

---

## 5. Window function — duplicate detection

**PostgreSQL (week 1)**
```sql
SELECT _source_row, trans_num
FROM (
    SELECT _source_row, trans_num,
           count(*) OVER (PARTITION BY trans_num) AS n
    FROM bronze.transactions_raw
    WHERE _run_id = :run_id AND trans_num IS NOT NULL
) d
WHERE n > 1
```

**PySpark equivalent**
```python
from pyspark.sql.window import Window

w = Window.partitionBy("trans_num")

duplicates = df.withColumn("n", F.count("*").over(w)) \
               .filter(F.col("n") > 1)
```

**What changed and why:**
Window functions work identically in concept. The syntax difference:
- PostgreSQL: `count(*) OVER (PARTITION BY col)`
- PySpark: `F.count("*").over(Window.partitionBy("col"))`

The `Window` object in PySpark is built separately and passed to the aggregate function.

---

## 6. Conditional aggregation (fraud amount)

**PostgreSQL (week 1 — profile_data.py)**
```sql
sum(amt) FILTER (WHERE is_fraud = 1)  AS fraud_amount
```

**Spark SQL (week 2 — gold notebook)**
```sql
sum(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END) AS fraud_amount
```

**PySpark DataFrame API**
```python
F.sum(F.when(F.col("is_fraud") == 1, F.col("amt")).otherwise(0)).alias("fraud_amount")
```

**What changed and why:**
PostgreSQL's `FILTER (WHERE ...)` clause is elegant but not part of the SQL standard
that Spark follows. Spark SQL uses the standard `CASE WHEN` form. The PySpark API
uses `F.when().otherwise()` which compiles to the same plan.

---

## 7. Latest run per source file (DISTINCT ON)

**PostgreSQL (week 1 — profile_data.py)**
```sql
WITH latest AS (
    SELECT DISTINCT ON (source_file) run_id, source_file
    FROM audit.etl_run_log
    WHERE status = 'SUCCEEDED'
    ORDER BY source_file, started_at DESC
)
```

**PySpark equivalent**
```python
from pyspark.sql.window import Window

w = Window.partitionBy("source_file").orderBy(F.desc("started_at"))

latest = (
    etl_run_log
    .filter(F.col("status") == "SUCCEEDED")
    .withColumn("rn", F.row_number().over(w))
    .filter(F.col("rn") == 1)
    .drop("rn")
)
```

**What changed and why:**
`DISTINCT ON` is a PostgreSQL extension that picks one row per partition based on
ORDER BY. It has no direct equivalent in standard SQL or Spark SQL. The idiomatic
Spark equivalent is `row_number()` over a partition window, then filter to `rn = 1`.
This is also the standard pattern in most other SQL dialects (BigQuery, Redshift, etc.).

---

## Summary table

| Pattern | PostgreSQL | PySpark / Spark SQL |
|---|---|---|
| Bulk load | `COPY FROM STDIN` | `spark.read.csv()` |
| Type cast | `col::type` | `.cast("type")` or `F.to_timestamp()` |
| Regex match | `col ~ 'pattern'` | `col.rlike("pattern")` |
| Window function | `fn() OVER (PARTITION BY …)` | `fn().over(Window.partitionBy(…))` |
| Conditional sum | `sum(x) FILTER (WHERE …)` | `sum(CASE WHEN … END)` |
| First per group | `DISTINCT ON (col) ORDER BY …` | `row_number().over(w) = 1` |
| Create table | `CREATE TABLE …` + `INSERT` | `.write.format("delta").saveAsTable()` |
