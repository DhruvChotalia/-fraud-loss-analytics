# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 14 — Real-Time Streaming Simulation
# MAGIC
# MAGIC **What this notebook builds:**
# MAGIC A simulation of a production fraud scoring pipeline operating in real time.
# MAGIC Transactions are processed in chronological order in micro-batches, scored
# MAGIC against a fraud model, and monitored for SLA compliance and fraud spikes —
# MAGIC exactly how a bank's transaction monitoring system would operate.
# MAGIC
# MAGIC **Why this matters:**
# MAGIC All previous weeks scored transactions in batch (all 1.85M at once). In
# MAGIC production, transactions arrive one at a time and must be scored in milliseconds.
# MAGIC This notebook answers: does the model hold up under streaming conditions?
# MAGIC What is the end-to-end latency? When does a fraud spike trigger an alert?
# MAGIC
# MAGIC **Pipeline architecture simulated:**
# MAGIC ```
# MAGIC  Transactions (time-ordered)
# MAGIC       │
# MAGIC       ▼
# MAGIC  Micro-batch ingestion (2,000 txns/batch)
# MAGIC       │
# MAGIC       ▼
# MAGIC  Feature engineering (per batch)
# MAGIC       │
# MAGIC       ▼
# MAGIC  Fraud scoring (sklearn GBT, threshold 0.60)
# MAGIC       │
# MAGIC       ├──► Alert engine (spike detection, rolling rate)
# MAGIC       │
# MAGIC       └──► Metrics store (latency, precision, recall, cost)
# MAGIC ```
# MAGIC
# MAGIC **Six outputs:**
# MAGIC 1. Per-batch metrics: latency, alerts triggered, fraud caught, cost saved
# MAGIC 2. SLA report: P50 / P95 / P99 batch latency
# MAGIC 3. Spike detection: batches where fraud rate > 2× rolling average
# MAGIC 4. Cumulative cost saved over the streaming run
# MAGIC 5. 4-panel streaming dashboard plot
# MAGIC 6. MLflow experiment with all metrics
# MAGIC
# MAGIC **Prerequisite:** Notebook 02 complete (silver.transactions exists).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Train scoring model on historical sample

# COMMAND ----------

import pandas as pd
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from pyspark.sql import functions as F

mlflow.set_experiment('/Users/dhruvextra76@gmail.com/streaming-simulation')

# ── Load and prepare data ─────────────────────────────────────────────────────
print('Loading silver transactions...')
silver = spark.table('silver.transactions')

df_all = (
    silver
    .withColumn('hour',     F.hour('txn_time'))
    .withColumn('dow',      F.dayofweek('txn_time'))
    .withColumn('month',    F.month('txn_time'))
    .withColumn('geo_dist', F.sqrt(
        F.pow(F.col('lat') - F.col('merch_lat'), 2) +
        F.pow(F.col('long') - F.col('merch_long'), 2)
    ))
    .select('txn_time', 'category', 'amt', 'city_pop',
            'geo_dist', 'hour', 'dow', 'month', 'is_fraud')
    .orderBy('txn_time')   # chronological order for streaming
    .toPandas()
)
df_all.attrs.clear()

# Cast Decimal columns
for col in ['amt', 'city_pop', 'geo_dist']:
    df_all[col] = df_all[col].astype(float)

print(f'Total transactions: {len(df_all):,}')
print(f'Date range: {df_all["txn_time"].min()} → {df_all["txn_time"].max()}')

# ── Feature engineering ───────────────────────────────────────────────────────
le = LabelEncoder()
df_all['category_enc'] = le.fit_transform(df_all['category'].astype(str))

FEATURES = ['amt', 'city_pop', 'geo_dist', 'hour', 'dow', 'month', 'category_enc']

# Train on first 70% (historical), stream the remaining 30%
split = int(len(df_all) * 0.70)
train = df_all.iloc[:split]
stream_data = df_all.iloc[split:].reset_index(drop=True)

X_train = train[FEATURES].values
y_train = train['is_fraud'].values

print(f'\nTraining on {len(train):,} historical transactions...')
t0 = time.time()
model = GradientBoostingClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.8, random_state=42
)
model.fit(X_train, y_train)
train_time = time.time() - t0
print(f'Model trained in {train_time:.1f}s')

# Validate on training set
train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
print(f'Training AUC: {train_auc:.4f}')
print(f'\nStreaming {len(stream_data):,} transactions in micro-batches...')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Stream transactions in micro-batches

# COMMAND ----------

BATCH_SIZE      = 2_000    # transactions per micro-batch
THRESHOLD       = 0.60     # fraud score threshold (cost-optimised from week 4)
FN_COST         = 530      # cost of missed fraud ($)
FP_COST         = 25       # cost of false decline ($)
SPIKE_MULTIPLIER = 2.0     # alert if batch fraud rate > 2× rolling avg
ROLLING_WINDOW  = 10       # batches for rolling average

batch_records = []
n_batches = len(stream_data) // BATCH_SIZE

print(f'Processing {n_batches} batches of {BATCH_SIZE:,} transactions each...')

for b in range(n_batches):
    t_start = time.time()

    batch = stream_data.iloc[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
    X_batch = batch[FEATURES].values
    y_batch = batch['is_fraud'].values

    # Score
    proba   = model.predict_proba(X_batch)[:, 1]
    y_pred  = (proba >= THRESHOLD).astype(int)

    # Metrics
    latency_ms  = (time.time() - t_start) * 1000
    tp = int(((y_pred == 1) & (y_batch == 1)).sum())
    fp = int(((y_pred == 1) & (y_batch == 0)).sum())
    fn = int(((y_pred == 0) & (y_batch == 1)).sum())
    actual_fraud = int(y_batch.sum())
    fraud_rate   = float(y_batch.mean())
    alert_rate   = float(y_pred.mean())
    cost_saved   = float(tp * FN_COST - fp * FP_COST)   # vs catching nothing

    batch_records.append({
        'batch':        b + 1,
        'latency_ms':   latency_ms,
        'actual_fraud': actual_fraud,
        'tp': tp, 'fp': fp, 'fn': fn,
        'fraud_rate':   fraud_rate,
        'alert_rate':   alert_rate,
        'cost_saved':   cost_saved,
    })

metrics = pd.DataFrame(batch_records)

# Rolling fraud rate and spike detection
metrics['rolling_fraud_rate'] = (
    metrics['fraud_rate'].rolling(ROLLING_WINDOW, min_periods=1).mean()
)
metrics['spike'] = (
    metrics['fraud_rate'] > SPIKE_MULTIPLIER * metrics['rolling_fraud_rate'].shift(1).fillna(metrics['fraud_rate'].mean())
)
metrics['cum_cost_saved'] = metrics['cost_saved'].cumsum()

print(f'Done. {n_batches} batches processed.')
print(f'Spikes detected: {metrics["spike"].sum()}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. SLA and performance report

# COMMAND ----------

total_tp  = metrics['tp'].sum()
total_fp  = metrics['fp'].sum()
total_fn  = metrics['fn'].sum()
total_fraud = metrics['actual_fraud'].sum()
overall_recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
total_cost_saved  = metrics['cum_cost_saved'].iloc[-1]

p50 = np.percentile(metrics['latency_ms'], 50)
p95 = np.percentile(metrics['latency_ms'], 95)
p99 = np.percentile(metrics['latency_ms'], 99)

print('=' * 60)
print('STREAMING PIPELINE — SLA REPORT')
print('=' * 60)
print(f'Batches processed     : {n_batches:,}')
print(f'Transactions streamed : {n_batches * BATCH_SIZE:,}')
print(f'Batch size            : {BATCH_SIZE:,} transactions')
print()
print('LATENCY (per batch)')
print(f'  P50 : {p50:.1f} ms')
print(f'  P95 : {p95:.1f} ms')
print(f'  P99 : {p99:.1f} ms')
print()
print('FRAUD DETECTION')
print(f'  Recall    : {overall_recall*100:.2f}%')
print(f'  Precision : {overall_precision*100:.2f}%')
print(f'  Spikes detected : {int(metrics["spike"].sum())}')
print()
print('COST')
print(f'  Total cost saved  : ${total_cost_saved:,.0f}')
print(f'  Avg per batch     : ${metrics["cost_saved"].mean():,.0f}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Streaming dashboard

# COMMAND ----------

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Real-Time Fraud Detection — Streaming Dashboard', fontsize=14, fontweight='bold')

# ── Plot 1: Batch latency over time ──────────────────────────────────────────
ax1 = axes[0, 0]
ax1.plot(metrics['batch'], metrics['latency_ms'], color='#4575b4', linewidth=0.8, alpha=0.7)
ax1.axhline(p95, color='#fdae61', linestyle='--', linewidth=1.5, label=f'P95 = {p95:.0f}ms')
ax1.axhline(p99, color='#d73027', linestyle='--', linewidth=1.5, label=f'P99 = {p99:.0f}ms')
ax1.set_xlabel('Batch #')
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Batch Processing Latency')
ax1.legend()

# ── Plot 2: Rolling fraud rate with spike markers ─────────────────────────────
ax2 = axes[0, 1]
ax2.plot(metrics['batch'], metrics['fraud_rate'] * 100,
         color='#4575b4', linewidth=0.8, alpha=0.5, label='Batch fraud rate')
ax2.plot(metrics['batch'], metrics['rolling_fraud_rate'] * 100,
         color='#1a9641', linewidth=2, label=f'Rolling avg ({ROLLING_WINDOW} batches)')
spikes = metrics[metrics['spike']]
ax2.scatter(spikes['batch'], spikes['fraud_rate'] * 100,
            color='#d73027', s=60, zorder=5, label=f'Spike alert ({len(spikes)})')
ax2.set_xlabel('Batch #')
ax2.set_ylabel('Fraud Rate (%)')
ax2.set_title('Rolling Fraud Rate & Spike Alerts')
ax2.legend(fontsize=8)

# ── Plot 3: Cumulative cost saved ─────────────────────────────────────────────
ax3 = axes[1, 0]
ax3.fill_between(metrics['batch'], metrics['cum_cost_saved'] / 1000,
                 alpha=0.3, color='#1a9641')
ax3.plot(metrics['batch'], metrics['cum_cost_saved'] / 1000,
         color='#1a9641', linewidth=2)
ax3.set_xlabel('Batch #')
ax3.set_ylabel('Cumulative Cost Saved ($K)')
ax3.set_title(f'Cumulative Cost Saved\n(Total: ${total_cost_saved/1000:,.0f}K)')
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}K'))

# ── Plot 4: Latency distribution (histogram) ─────────────────────────────────
ax4 = axes[1, 1]
ax4.hist(metrics['latency_ms'], bins=40, color='#4575b4', edgecolor='white', alpha=0.8)
ax4.axvline(p50, color='#1a9641', linewidth=2, linestyle='--', label=f'P50={p50:.0f}ms')
ax4.axvline(p95, color='#fdae61', linewidth=2, linestyle='--', label=f'P95={p95:.0f}ms')
ax4.axvline(p99, color='#d73027', linewidth=2, linestyle='--', label=f'P99={p99:.0f}ms')
ax4.set_xlabel('Latency (ms)')
ax4.set_ylabel('Batch Count')
ax4.set_title('Latency Distribution')
ax4.legend()

plt.tight_layout()
plot_path = '/tmp/streaming_dashboard.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved: {plot_path}')
display(plt.imread(plot_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Upload to ADLS and log to MLflow

# COMMAND ----------

from azure.storage.blob import BlobServiceClient

STORAGE_ACCOUNT = 'fraudanalytics2024'
try:
    STORAGE_KEY = dbutils.secrets.get(scope='fraud-scope', key='adls-key')
except Exception:
    STORAGE_KEY = 'YOUR_STORAGE_KEY_HERE'

blob_client = BlobServiceClient(
    account_url=f'https://{STORAGE_ACCOUNT}.blob.core.windows.net',
    credential=STORAGE_KEY
)

# Upload plot
with open(plot_path, 'rb') as f:
    blob_client.get_blob_client('reports', 'streaming_dashboard.png').upload_blob(f, overwrite=True)
print('Uploaded: reports/streaming_dashboard.png')

# Upload metrics parquet
metrics_path = '/tmp/streaming_metrics.parquet'
metrics.attrs.clear()
metrics.to_parquet(metrics_path, index=False)
with open(metrics_path, 'rb') as f:
    blob_client.get_blob_client('gold', 'streaming/streaming_metrics.parquet').upload_blob(f, overwrite=True)
print('Uploaded: gold/streaming/streaming_metrics.parquet')

# MLflow
with mlflow.start_run(run_name='streaming_simulation_v1'):
    mlflow.log_param('batch_size',       BATCH_SIZE)
    mlflow.log_param('threshold',        THRESHOLD)
    mlflow.log_param('n_batches',        n_batches)
    mlflow.log_param('spike_multiplier', SPIKE_MULTIPLIER)
    mlflow.log_param('rolling_window',   ROLLING_WINDOW)

    mlflow.log_metric('latency_p50_ms',      round(float(p50), 2))
    mlflow.log_metric('latency_p95_ms',      round(float(p95), 2))
    mlflow.log_metric('latency_p99_ms',      round(float(p99), 2))
    mlflow.log_metric('streaming_recall',    round(float(overall_recall), 4))
    mlflow.log_metric('streaming_precision', round(float(overall_precision), 4))
    mlflow.log_metric('spikes_detected',     int(metrics['spike'].sum()))
    mlflow.log_metric('total_cost_saved',    round(float(total_cost_saved), 2))
    mlflow.log_metric('training_auc',        round(float(train_auc), 4))

    mlflow.log_artifact(plot_path)
    run_id = mlflow.active_run().info.run_id
    print(f'MLflow run: {run_id}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Final summary

# COMMAND ----------

print('=' * 60)
print('STREAMING SIMULATION — FINAL SUMMARY')
print('=' * 60)
print(f'Transactions streamed  : {n_batches * BATCH_SIZE:,}')
print(f'Batches processed      : {n_batches:,}  ({BATCH_SIZE:,} txns each)')
print(f'Scoring threshold      : {THRESHOLD}  (cost-optimised)')
print()
print('LATENCY SLA')
print(f'  P50 latency : {p50:.1f} ms  ← median batch time')
print(f'  P95 latency : {p95:.1f} ms  ← 95th percentile')
print(f'  P99 latency : {p99:.1f} ms  ← worst-case tail')
print()
print('DETECTION QUALITY')
print(f'  Recall      : {overall_recall*100:.2f}%')
print(f'  Precision   : {overall_precision*100:.2f}%')
print(f'  Fraud spikes detected : {int(metrics["spike"].sum())}')
print()
print('BUSINESS IMPACT')
print(f'  Total cost saved      : ${total_cost_saved:,.0f}')
print(f'  Avg cost saved/batch  : ${metrics["cost_saved"].mean():,.0f}')
print()
print('Outputs:')
print('  ADLS  reports/streaming_dashboard.png')
print('  ADLS  gold/streaming/streaming_metrics.parquet')
print('  MLflow /Users/dhruvextra76@gmail.com/streaming-simulation')
