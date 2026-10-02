# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 4 — Loss Forecast: SARIMA vs Holt-Winters
# MAGIC
# MAGIC **Goal:** forecast monthly fraud losses for the next 6 months using two classical
# MAGIC time-series models, then compare their accuracy on a held-out test period.
# MAGIC
# MAGIC | Model | What it captures |
# MAGIC |---|---|
# MAGIC | SARIMA | Trend + seasonality + autocorrelation (AR terms) |
# MAGIC | Holt-Winters | Trend + seasonality via exponential smoothing |
# MAGIC
# MAGIC **Data:** monthly fraud losses from `silver.transactions` (Jan 2019 – Dec 2020, 24 months)
# MAGIC
# MAGIC **Split:** train on Jan 2019 – Jun 2020 (18 months), test on Jul–Dec 2020 (6 months)
# MAGIC
# MAGIC **Prerequisite:** run `02_silver` first so `silver.transactions` exists.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Build monthly fraud loss series

# COMMAND ----------

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

monthly_spark = spark.sql("""
    SELECT
        date_trunc('month', txn_time)                                     AS month,
        count(*)                                                          AS transactions,
        sum(is_fraud)                                                     AS fraud_txns,
        round(sum(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2)        AS fraud_losses
    FROM silver.transactions
    GROUP BY 1
    ORDER BY 1
""")

monthly = monthly_spark.toPandas()
monthly["month"] = pd.to_datetime(monthly["month"])
monthly = monthly.set_index("month")
monthly.index = monthly.index.to_period("M")

print(f"Monthly series: {len(monthly)} periods")
print(monthly[["fraud_losses"]].to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train/test split

# COMMAND ----------

TRAIN_END = "2020-06"
TEST_START = "2020-07"

train_ts = monthly.loc[:TRAIN_END, "fraud_losses"]
test_ts  = monthly.loc[TEST_START:, "fraud_losses"]

print(f"Train: {len(train_ts)} months ({train_ts.index[0]} to {train_ts.index[-1]})")
print(f"Test:  {len(test_ts)} months ({test_ts.index[0]} to {test_ts.index[-1]})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. SARIMA model
# MAGIC
# MAGIC SARIMA(p,d,q)(P,D,Q,s) where s=12 for monthly seasonality.
# MAGIC We use (1,1,1)(1,1,0,12) — a standard starting point for monthly financial data.
# MAGIC In production you'd use auto_arima to search the parameter space automatically.

# COMMAND ----------

from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(
    train_ts,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 0, 12),
    enforce_stationarity=False,
    enforce_invertibility=False,
)
sarima_fit = sarima_model.fit(disp=False)

sarima_forecast = sarima_fit.forecast(steps=len(test_ts))
sarima_forecast.index = test_ts.index

print("SARIMA forecast vs actual:")
comparison = pd.DataFrame({
    "actual":   test_ts,
    "sarima":   sarima_forecast.round(2),
    "sarima_err": (sarima_forecast - test_ts).round(2),
})
print(comparison.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Holt-Winters model
# MAGIC
# MAGIC Exponential smoothing with additive trend and additive seasonality.
# MAGIC Simpler than SARIMA — fewer parameters, easier to explain to non-technical stakeholders.

# COMMAND ----------

from statsmodels.tsa.holtwinters import ExponentialSmoothing

hw_model = ExponentialSmoothing(
    train_ts,
    trend="add",
    seasonal="add",
    seasonal_periods=12,
)
hw_fit = hw_model.fit(optimized=True)

hw_forecast = hw_fit.forecast(steps=len(test_ts))
hw_forecast.index = test_ts.index

print("Holt-Winters forecast vs actual:")
comparison["holt_winters"]     = hw_forecast.round(2)
comparison["holt_winters_err"] = (hw_forecast - test_ts).round(2)
print(comparison.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare accuracy (MAE and MAPE)
# MAGIC
# MAGIC - **MAE** (Mean Absolute Error) — average dollar error per month
# MAGIC - **MAPE** (Mean Absolute Percentage Error) — average % error, scale-independent

# COMMAND ----------

def mae(actual, forecast):
    return round(np.mean(np.abs(actual - forecast)), 2)

def mape(actual, forecast):
    return round(np.mean(np.abs((actual - forecast) / actual)) * 100, 2)

sarima_mae  = mae(test_ts, sarima_forecast)
sarima_mape = mape(test_ts, sarima_forecast)
hw_mae      = mae(test_ts, hw_forecast)
hw_mape     = mape(test_ts, hw_forecast)

print("=" * 45)
print(f"{'Model':<15} {'MAE ($)':>12} {'MAPE (%)':>10}")
print("-" * 45)
print(f"{'SARIMA':<15} {sarima_mae:>12,.2f} {sarima_mape:>10.2f}")
print(f"{'Holt-Winters':<15} {hw_mae:>12,.2f} {hw_mape:>10.2f}")
print("=" * 45)

winner = "SARIMA" if sarima_mape < hw_mape else "Holt-Winters"
print(f"\nLower MAPE wins → {winner} is the better model on this data.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Forecast next 6 months beyond the data
# MAGIC
# MAGIC Refit the winning model on the full 24-month series, then forecast Jan–Jun 2021.

# COMMAND ----------

full_series = monthly["fraud_losses"]

if winner == "SARIMA":
    final_model = SARIMAX(
        full_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 0, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    future_forecast = final_model.forecast(steps=6)
else:
    final_model = ExponentialSmoothing(
        full_series, trend="add", seasonal="add", seasonal_periods=12
    ).fit(optimized=True)
    future_forecast = final_model.forecast(steps=6)

future_index = pd.period_range(start="2021-01", periods=6, freq="M")
future_forecast.index = future_index

print(f"\n{winner} forecast — Jan 2021 to Jun 2021:")
print(future_forecast.round(2).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Upload results to ADLS Gen2

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
    df.to_csv(buf, index=True)
    client.get_blob_client(container=CONTAINER, blob=blob_name) \
          .upload_blob(buf.getvalue(), overwrite=True)
    print(f"Uploaded: {blob_name} → ADLS Gen2 models/")

# Full comparison table
upload_csv(comparison, "forecast_comparison.csv")

# Future forecast
upload_csv(future_forecast.rename("forecast_fraud_losses").to_frame(), "forecast_future_6m.csv")

# Accuracy summary
accuracy_summary = pd.DataFrame([
    {"model": "SARIMA",       "mae": sarima_mae, "mape_pct": sarima_mape},
    {"model": "Holt-Winters", "mae": hw_mae,     "mape_pct": hw_mape},
])
upload_csv(accuracy_summary, "forecast_accuracy.csv")

print(f"\nWinner: {winner}  MAPE={min(sarima_mape, hw_mape)}%")
