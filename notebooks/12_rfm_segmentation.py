# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 12 — RFM Cardholder Segmentation
# MAGIC
# MAGIC **What this notebook builds:**
# MAGIC RFM (Recency, Frequency, Monetary) segmentation of every cardholder in the
# MAGIC dataset, with fraud rate overlaid on each segment. This answers a question the
# MAGIC GBT model cannot: *which type of cardholder is structurally at the highest fraud
# MAGIC risk*, regardless of any single transaction's features.
# MAGIC
# MAGIC **Three RFM dimensions:**
# MAGIC - **Recency (R):** Days since the cardholder's last transaction. A cardholder who
# MAGIC   has not transacted recently may have a dormant or compromised account.
# MAGIC - **Frequency (F):** Total number of transactions in the observation window.
# MAGIC   High-frequency cardholders generate more exposure.
# MAGIC - **Monetary (M):** Total spend across the window. High-spend cardholders
# MAGIC   represent higher per-incident losses.
# MAGIC
# MAGIC **Scoring:** Each dimension is split into quintiles (1–5). Recency is
# MAGIC reverse-scored: fewer days since last transaction = score 5 (more active).
# MAGIC
# MAGIC **Fraud overlay:** After segmenting, fraud rate (% of transactions flagged) is
# MAGIC computed per segment to identify which RFM profiles carry the most risk.
# MAGIC
# MAGIC **Five outputs:**
# MAGIC 1. Per-cardholder RFM scores and segment labels
# MAGIC 2. Fraud rate by RFM segment (the key risk insight)
# MAGIC 3. Three plots: segment distribution, fraud rate by segment, R-F-M scatter
# MAGIC 4. Parquet file uploaded to ADLS gold/rfm_segments/
# MAGIC 5. MLflow experiment with metrics and artefacts
# MAGIC
# MAGIC **Prerequisite:** Notebook 02 complete (silver.transactions exists).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and data load

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import mlflow
import mlflow.spark
import os
from datetime import timedelta
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# MLflow experiment — absolute path required on Community Edition
mlflow.set_experiment('/Users/dhruvextra76@gmail.com/rfm-segmentation')

print('Loading silver transactions...')
silver = spark.table('silver.transactions')
print(f'  Rows: {silver.count():,}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Compute RFM metrics per cardholder

# COMMAND ----------

# Reference date = day after the latest transaction in the dataset
# silver uses txn_time (timestamp) — cast to date for recency arithmetic
max_date = silver.agg(F.max(F.to_date('txn_time'))).collect()[0][0]
ref_date  = max_date + timedelta(days=1)
print(f'Reference date: {ref_date}  (day after last transaction)')

rfm_spark = (
    silver
    .withColumn('txn_date', F.to_date('txn_time'))
    .groupBy('cc_num_masked')
    .agg(
        F.max('txn_date').alias('last_trans_date'),
        F.count('*').alias('frequency'),
        F.sum('amt').alias('monetary'),
        F.sum('is_fraud').alias('fraud_txns'),
        F.count('*').alias('total_txns'),
    )
    .withColumn('recency_days', F.datediff(F.lit(ref_date), F.col('last_trans_date')))
    .withColumn('fraud_rate', F.round(F.col('fraud_txns') / F.col('total_txns') * 100, 4))
    .drop('last_trans_date')
)

rfm = rfm_spark.toPandas()
print(f'Cardholders: {len(rfm):,}')
print(rfm[['recency_days', 'frequency', 'monetary', 'fraud_rate']].describe().round(2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Score each dimension into quintiles (1–5)

# COMMAND ----------

def quintile_score(series, reverse=False):
    """
    Assign 1–5 quintile scores.
    Uses rank(pct=True) to convert to uniform [0,1] percentiles first,
    which eliminates duplicate-bin errors from pd.qcut.
    reverse=True gives score 5 to the lowest values (e.g. Recency).
    """
    pct = series.rank(method='first', pct=True)
    score = pd.cut(pct, bins=5, labels=[1, 2, 3, 4, 5]).astype(int)
    if reverse:
        score = 6 - score
    return score

rfm['R'] = quintile_score(rfm['recency_days'], reverse=True)   # lower days = more recent = better
rfm['F'] = quintile_score(rfm['frequency'],    reverse=False)
rfm['M'] = quintile_score(rfm['monetary'],     reverse=False)
rfm['RFM_score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
rfm['RFM_total'] = rfm['R'] + rfm['F'] + rfm['M']

print('Score distributions:')
for dim in ['R', 'F', 'M']:
    print(f'  {dim}: {dict(rfm[dim].value_counts().sort_index())}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Assign segment labels

# COMMAND ----------

def assign_segment(row):
    r, f, m = row['R'], row['F'], row['M']
    if r >= 4 and f >= 4 and m >= 4:
        return 'Champions'
    elif r >= 3 and f >= 4:
        return 'Loyal'
    elif r >= 3 and f >= 2:
        return 'Potential Loyalist'
    elif r >= 4 and f <= 1:
        return 'New / Infrequent'
    elif r <= 2 and f >= 3 and m >= 3:
        return 'At Risk – High Value'
    elif r <= 2 and f >= 3:
        return 'At Risk'
    elif r <= 1 and f >= 4:
        return 'Cannot Lose'
    elif r <= 2 and f <= 2:
        return 'Dormant'
    else:
        return 'Mid-Tier'

rfm['segment'] = rfm.apply(assign_segment, axis=1)

seg_counts = rfm['segment'].value_counts()
print('Segment distribution:')
print(seg_counts.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Fraud rate by segment (the key risk insight)

# COMMAND ----------

seg_fraud = (
    rfm.groupby('segment')
    .agg(
        cardholders=('cc_num_masked', 'count'),
        total_txns=('total_txns', 'sum'),
        fraud_txns=('fraud_txns', 'sum'),
    )
    .assign(fraud_rate_pct=lambda d: (d['fraud_txns'] / d['total_txns'] * 100).round(4))
    .sort_values('fraud_rate_pct', ascending=False)
)

print('\nFraud rate by RFM segment:')
print(seg_fraud.to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Visualisations

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('RFM Cardholder Segmentation — Fraud Risk Analysis', fontsize=14, fontweight='bold')

# ── Plot 1: cardholder count by segment ──────────────────────────────────────
ax1 = axes[0]
colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(seg_counts)))
bars = ax1.barh(seg_counts.index, seg_counts.values, color=colors)
ax1.set_xlabel('Cardholders')
ax1.set_title('Cardholder Count by Segment')
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
for bar, val in zip(bars, seg_counts.values):
    ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
             f'{val:,}', va='center', fontsize=8)

# ── Plot 2: fraud rate by segment ────────────────────────────────────────────
ax2 = axes[1]
fraud_sorted = seg_fraud.sort_values('fraud_rate_pct', ascending=True)
bar_colors = ['#d73027' if x > fraud_sorted['fraud_rate_pct'].median() else '#4575b4'
              for x in fraud_sorted['fraud_rate_pct']]
bars2 = ax2.barh(fraud_sorted.index, fraud_sorted['fraud_rate_pct'], color=bar_colors)
ax2.set_xlabel('Fraud Rate (%)')
ax2.set_title('Fraud Rate by Segment\n(red = above median)')
for bar, val in zip(bars2, fraud_sorted['fraud_rate_pct']):
    ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
             f'{val:.3f}%', va='center', fontsize=8)

# ── Plot 3: RFM total score vs fraud rate scatter ────────────────────────────
ax3 = axes[2]
sample = rfm.sample(min(10000, len(rfm)), random_state=42)
sc = ax3.scatter(
    sample['RFM_total'], sample['fraud_rate'],
    c=sample['fraud_rate'], cmap='RdYlGn_r',
    alpha=0.3, s=8, vmin=0, vmax=sample['fraud_rate'].quantile(0.99)
)
plt.colorbar(sc, ax=ax3, label='Fraud Rate (%)')
ax3.set_xlabel('RFM Total Score (3–15)')
ax3.set_ylabel('Cardholder Fraud Rate (%)')
ax3.set_title('RFM Score vs Fraud Rate\n(sample of 10K cardholders)')

plt.tight_layout()
plot_path = '/tmp/rfm_segmentation.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Plot saved: {plot_path}')
display(plt.imread(plot_path))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Upload results to ADLS and log to MLflow

# COMMAND ----------

from azure.storage.blob import BlobServiceClient
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

STORAGE_ACCOUNT = 'fraudanalytics2024'
try:
    STORAGE_KEY = dbutils.secrets.get(scope='fraud-scope', key='adls-key')
except Exception:
    STORAGE_KEY = 'YOUR_STORAGE_KEY_HERE'  # paste your ADLS key here if secret scope not configured
CONTAINER = 'gold'

blob_client = BlobServiceClient(
    account_url=f'https://{STORAGE_ACCOUNT}.blob.core.windows.net',
    credential=STORAGE_KEY
)

# Clear Databricks/Spark execution metadata from .attrs before any serialization.
# .toPandas() stores PlanMetrics in df.attrs; parquet/json serialization fails on it.
rfm.attrs.clear()
seg_fraud.attrs.clear()

# Upload parquet
rfm_path = '/tmp/rfm_segments.parquet'
rfm.to_parquet(rfm_path, index=False)

with open(rfm_path, 'rb') as f:
    blob_client.get_blob_client(CONTAINER, 'rfm_segments/rfm_segments.parquet').upload_blob(f, overwrite=True)
print('Uploaded: gold/rfm_segments/rfm_segments.parquet')

# Upload segment fraud summary as JSON
# Manually cast every value to plain Python primitives to avoid pandas PlanMetrics
# serialization errors in newer pandas versions
seg_records = [
    {
        'segment':       str(row['segment']),
        'cardholders':   int(row['cardholders']),
        'total_txns':    int(row['total_txns']),
        'fraud_txns':    int(row['fraud_txns']),
        'fraud_rate_pct': float(row['fraud_rate_pct']),
    }
    for _, row in seg_fraud.reset_index().iterrows()
]
blob_client.get_blob_client(CONTAINER, 'rfm_segments/segment_fraud_summary.json').upload_blob(
    json.dumps(seg_records, indent=2), overwrite=True
)
print('Uploaded: gold/rfm_segments/segment_fraud_summary.json')

# Upload plot
with open(plot_path, 'rb') as f:
    blob_client.get_blob_client('reports', 'rfm_segmentation.png').upload_blob(f, overwrite=True)
print('Uploaded: reports/rfm_segmentation.png')

# COMMAND ----------

# MLflow logging
with mlflow.start_run(run_name='rfm_segmentation_v1'):
    # Params
    mlflow.log_param('n_cardholders',   int(len(rfm)))
    mlflow.log_param('n_segments',      rfm['segment'].nunique())
    mlflow.log_param('scoring_method',  'quintiles_1_to_5')
    mlflow.log_param('reference_date',  str(ref_date))

    # Metrics — top 3 highest-risk segments
    top3 = seg_fraud.head(3)
    for i, (seg, row) in enumerate(top3.iterrows()):
        safe = seg.lower().replace(' ', '_').replace('/', '_').replace('–', '')
        mlflow.log_metric(f'fraud_rate_pct_{safe}', round(row['fraud_rate_pct'], 4))

    # Overall metrics
    mlflow.log_metric('overall_fraud_rate_pct', round(rfm['fraud_txns'].sum() / rfm['total_txns'].sum() * 100, 4))
    mlflow.log_metric('pct_cardholders_at_risk', round(
        rfm[rfm['segment'].str.contains('At Risk|Cannot Lose|Dormant')].shape[0] / len(rfm) * 100, 2
    ))

    # Artefacts
    mlflow.log_artifact(plot_path)
    mlflow.log_artifact(rfm_path)

    run_id = mlflow.active_run().info.run_id
    print(f'MLflow run: {run_id}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary

# COMMAND ----------

print('=' * 60)
print('RFM SEGMENTATION SUMMARY')
print('=' * 60)
print(f'Total cardholders analysed : {len(rfm):,}')
print(f'Segments identified        : {rfm["segment"].nunique()}')
print()
print('TOP 3 HIGHEST-RISK SEGMENTS:')
print('-' * 60)
for seg, row in seg_fraud.head(3).iterrows():
    print(f'  {seg:<25}  fraud rate: {row["fraud_rate_pct"]:.3f}%  '
          f'({int(row["fraud_txns"]):,} fraud txns / {int(row["total_txns"]):,} total)')
print()
print('LARGEST SEGMENTS BY CARDHOLDER COUNT:')
print('-' * 60)
for seg, row in seg_fraud.sort_values('cardholders', ascending=False).head(3).iterrows():
    print(f'  {seg:<25}  {int(row["cardholders"]):,} cardholders')
print()
print('Outputs saved to:')
print('  ADLS  gold/rfm_segments/rfm_segments.parquet')
print('  ADLS  gold/rfm_segments/segment_fraud_summary.json')
print('  ADLS  reports/rfm_segmentation.png')
print('  MLflow /Users/dhruvextra76@gmail.com/rfm-segmentation')
