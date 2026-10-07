# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 13 — Causal Inference: Does Night-Time Transacting Cause Fraud?
# MAGIC
# MAGIC **The question:**
# MAGIC Week 1 found that transactions at 22–23h have a 2.5–2.6% fraud rate vs ~0.09%
# MAGIC during business hours. But correlation is not causation. Maybe night transactions
# MAGIC happen to cluster in high-risk merchant categories (shopping_net, grocery_net)
# MAGIC or involve larger amounts — and *those* are the real drivers of fraud.
# MAGIC
# MAGIC **Causal inference answers:** after controlling for every observable confounder,
# MAGIC does transacting at night still *cause* higher fraud, or does the effect disappear?
# MAGIC
# MAGIC **The causal DAG (Directed Acyclic Graph):**
# MAGIC ```
# MAGIC   category ──┐
# MAGIC   amt        ├──► is_night ──► is_fraud
# MAGIC   state      │        ▲
# MAGIC   city_pop ──┘        │
# MAGIC              └── (confounders also affect fraud directly)
# MAGIC ```
# MAGIC Confounders (category, amt, state, city_pop) affect both the treatment
# MAGIC (is_night) and the outcome (is_fraud). We must control for them to isolate
# MAGIC the causal effect of night.
# MAGIC
# MAGIC **Method: Propensity Score Matching (PSM)**
# MAGIC 1. Estimate P(night | confounders) via logistic regression — the propensity score
# MAGIC 2. For each night transaction, find a daytime transaction with a similar score
# MAGIC 3. Compare fraud rates in the matched sample — this is the causal estimate
# MAGIC 4. Check covariate balance: did matching remove confounder bias?
# MAGIC 5. Bootstrap confidence interval on the Average Treatment Effect (ATE)
# MAGIC
# MAGIC **Prerequisite:** Notebook 02 complete (silver.transactions exists).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load data and engineer features

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from pyspark.sql import functions as F

mlflow.set_experiment('/Users/dhruvextra76@gmail.com/causal-inference')

print('Loading silver transactions...')
silver = spark.table('silver.transactions')

# Sample for tractability — Community Edition has 15GB RAM; keep it small
SAMPLE_FRAC = 0.08   # ~150K from 1.85M
df = (
    silver
    .withColumn('hour', F.hour('txn_time'))
    .withColumn('geo_dist', F.sqrt(
        F.pow(F.col('lat') - F.col('merch_lat'), 2) +
        F.pow(F.col('long') - F.col('merch_long'), 2)
    ))
    .select('hour', 'category', 'amt', 'state', 'city_pop',
            'geo_dist', 'is_fraud')
    .sample(fraction=SAMPLE_FRAC, seed=42)
    .toPandas()
)
df.attrs.clear()

print(f'Sample rows: {len(df):,}')
print(f'Fraud rate in sample: {df["is_fraud"].mean()*100:.3f}%')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define treatment, outcome and confounders

# COMMAND ----------

# Treatment: transaction during peak fraud hours (22–23h)
df['is_night'] = df['hour'].isin([22, 23]).astype(int)

print(f'Night transactions : {df["is_night"].sum():,}  ({df["is_night"].mean()*100:.1f}%)')
print(f'Day transactions   : {(1-df["is_night"]).sum():,}')

# Naive (unadjusted) ATE — what a naive analyst would report
naive_night = df[df['is_night']==1]['is_fraud'].mean()
naive_day   = df[df['is_night']==0]['is_fraud'].mean()
naive_ate   = naive_night - naive_day

print(f'\nNaive fraud rate — night: {naive_night*100:.3f}%  day: {naive_day*100:.3f}%')
print(f'Naive ATE (unadjusted)  : +{naive_ate*100:.3f}pp')
print('(This conflates causal effect with confounder bias)')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Estimate propensity scores

# COMMAND ----------

# One-hot encode categorical confounders.
# Drop 'state' — 50 states × 150K rows blows Community Edition RAM.
# category (14 values → 13 dummies) + 3 numerics is sufficient.
confounder_cols = ['category']
numeric_cols    = ['amt', 'city_pop', 'geo_dist']

# Cast to float first — amt arrives as Decimal from Spark, which pd.get_dummies
# treats as categorical and one-hot encodes, destroying the column.
for col in numeric_cols:
    df[col] = df[col].astype(float)

df_enc = pd.get_dummies(df[confounder_cols + numeric_cols], drop_first=True)
df_enc[numeric_cols] = df_enc[numeric_cols].fillna(df_enc[numeric_cols].median())

scaler  = StandardScaler()
X       = scaler.fit_transform(df_enc)
y_treat = df['is_night'].values

lr = LogisticRegression(max_iter=500, C=1.0, random_state=42)
lr.fit(X, y_treat)

df['pscore'] = lr.predict_proba(X)[:, 1]

ps_auc = roc_auc_score(y_treat, df['pscore'])
print(f'Propensity score model AUC: {ps_auc:.4f}')
print('(AUC near 0.5 = confounders barely predict night → low confounding)')
print('(AUC near 1.0 = confounders strongly predict night → high confounding)')

print(f'\nPropensity score stats:')
print(df.groupby('is_night')['pscore'].describe().round(4).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Nearest-neighbour matching with caliper

# COMMAND ----------

# Caliper = 0.2 × std of propensity scores (standard rule of thumb)
caliper = 0.2 * df['pscore'].std()
print(f'Caliper: {caliper:.6f}')

night_df = df[df['is_night'] == 1].reset_index(drop=True)
day_df   = df[df['is_night'] == 0].reset_index(drop=True)

day_pscores  = day_df['pscore'].values
matched_night_idx = []
matched_day_idx   = []
used_day          = set()

# Greedy 1:1 nearest-neighbour matching
for i, row in night_df.iterrows():
    diffs = np.abs(day_pscores - row['pscore'])
    # Mask already-used controls
    diffs[list(used_day)] = np.inf
    best = np.argmin(diffs)
    if diffs[best] <= caliper:
        matched_night_idx.append(i)
        matched_day_idx.append(best)
        used_day.add(best)

matched_night = night_df.loc[matched_night_idx].copy()
matched_day   = day_df.loc[matched_day_idx].copy()
matched        = pd.concat([matched_night, matched_day], ignore_index=True)
matched.attrs.clear()

print(f'Matched pairs       : {len(matched_night_idx):,}')
print(f'Night unmatched     : {len(night_df) - len(matched_night_idx):,}')
print(f'Match rate          : {len(matched_night_idx)/len(night_df)*100:.1f}%')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Estimate causal ATE with bootstrap confidence interval

# COMMAND ----------

def compute_ate(df_matched):
    night_fraud = df_matched[df_matched['is_night']==1]['is_fraud'].mean()
    day_fraud   = df_matched[df_matched['is_night']==0]['is_fraud'].mean()
    return night_fraud - day_fraud

ate = compute_ate(matched)

# Bootstrap 95% CI
N_BOOT = 1000
rng    = np.random.default_rng(42)
boot_ates = []
n_pairs = len(matched_night_idx)

for _ in range(N_BOOT):
    idx     = rng.integers(0, n_pairs, size=n_pairs)
    boot_n  = matched_night.iloc[idx]['is_fraud'].mean()
    boot_d  = matched_day.iloc[idx]['is_fraud'].mean()
    boot_ates.append(boot_n - boot_d)

ci_lo, ci_hi = np.percentile(boot_ates, [2.5, 97.5])

print('=' * 55)
print('CAUSAL AVERAGE TREATMENT EFFECT (PSM)')
print('=' * 55)
print(f'  Naive ATE (unadjusted)    : +{naive_ate*100:.3f}pp')
print(f'  Causal ATE (PSM-adjusted) : +{ate*100:.3f}pp')
print(f'  95% Bootstrap CI          :  [{ci_lo*100:.3f}pp, {ci_hi*100:.3f}pp]')
print(f'  Confounding bias removed  :  {(naive_ate - ate)*100:.3f}pp')
significant = ci_lo > 0
print(f'  Statistically significant :  {"YES" if significant else "NO"} (CI excludes zero: {significant})')
print()
if significant:
    print('  Conclusion: Night-time transacting has a genuine causal effect')
    print('  on fraud probability, even after controlling for merchant')
    print('  category, transaction amount, location, and city size.')
else:
    print('  Conclusion: The night-time effect is largely explained by')
    print('  confounders. Merchant category / amount drive most of the risk.')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Covariate balance check

# COMMAND ----------

# Standardised Mean Difference (SMD) before and after matching
# SMD < 0.1 = good balance; SMD < 0.25 = acceptable
balance_vars = ['amt', 'city_pop', 'geo_dist']

def smd(group1, group2):
    diff = group1.mean() - group2.mean()
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    return abs(diff / pooled_std) if pooled_std > 0 else 0

balance_rows = []
for var in balance_vars:
    before_smd = smd(df[df['is_night']==1][var], df[df['is_night']==0][var])
    after_smd  = smd(matched_night[var], matched_day[var])
    balance_rows.append({'variable': var, 'SMD_before': before_smd, 'SMD_after': after_smd})

balance_df = pd.DataFrame(balance_rows)
print('Covariate balance (SMD < 0.1 = well balanced):')
print(balance_df.round(4).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualisations

# COMMAND ----------

fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle('Causal Inference: Does Night-Time Transacting Cause Fraud?',
             fontsize=14, fontweight='bold')

# ── Plot 1: Propensity score overlap (common support) ────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist(df[df['is_night']==0]['pscore'], bins=50, alpha=0.6,
         color='#4575b4', label='Day', density=True)
ax1.hist(df[df['is_night']==1]['pscore'], bins=50, alpha=0.6,
         color='#d73027', label='Night', density=True)
ax1.set_xlabel('Propensity Score')
ax1.set_ylabel('Density')
ax1.set_title('Propensity Score Distribution\n(Overlap = Valid Comparison)')
ax1.legend()

# ── Plot 2: Propensity score after matching ───────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(matched_day['pscore'],   bins=40, alpha=0.6,
         color='#4575b4', label='Matched Day', density=True)
ax2.hist(matched_night['pscore'], bins=40, alpha=0.6,
         color='#d73027', label='Matched Night', density=True)
ax2.set_xlabel('Propensity Score')
ax2.set_title('Propensity Score After Matching\n(Distributions Should Overlap)')
ax2.legend()

# ── Plot 3: ATE with CI ───────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
labels  = ['Naive\n(Unadjusted)', 'Causal\n(PSM-Adjusted)']
values  = [naive_ate * 100, ate * 100]
colors  = ['#fdae61', '#d73027' if significant else '#74add1']
bars    = ax3.bar(labels, values, color=colors, width=0.4)
ax3.errorbar(1, ate * 100,
             yerr=[[( ate - ci_lo) * 100], [(ci_hi - ate) * 100]],
             fmt='none', color='black', capsize=6, linewidth=2)
ax3.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax3.set_ylabel('Effect on Fraud Rate (percentage points)')
ax3.set_title('Naive vs Causal ATE\n(error bar = 95% Bootstrap CI)')
for bar, val in zip(bars, values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'+{val:.3f}pp', ha='center', va='bottom', fontsize=9)

# ── Plot 4: Covariate balance (SMD lollipop) ─────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
y_pos = range(len(balance_df))
ax4.hlines(y_pos, 0, balance_df['SMD_before'], color='#fdae61', linewidth=3, label='Before matching')
ax4.scatter(balance_df['SMD_before'], y_pos, color='#fdae61', s=80, zorder=5)
ax4.hlines(y_pos, 0, balance_df['SMD_after'],  color='#1a9641', linewidth=3, label='After matching')
ax4.scatter(balance_df['SMD_after'],  y_pos, color='#1a9641', s=80, zorder=5)
ax4.axvline(0.1, color='gray', linestyle='--', linewidth=1, label='SMD=0.1 threshold')
ax4.set_yticks(list(y_pos))
ax4.set_yticklabels(balance_df['variable'])
ax4.set_xlabel('Standardised Mean Difference')
ax4.set_title('Covariate Balance\n(green <0.1 = well balanced)')
ax4.legend(fontsize=8)

# ── Plot 5: Bootstrap ATE distribution ───────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
boot_pct = [x * 100 for x in boot_ates]
ax5.hist(boot_pct, bins=50, color='#4575b4', edgecolor='white', alpha=0.8)
ax5.axvline(ate * 100,  color='#d73027', linewidth=2, label=f'ATE = +{ate*100:.3f}pp')
ax5.axvline(ci_lo * 100, color='gray', linewidth=1.5, linestyle='--', label=f'95% CI lower')
ax5.axvline(ci_hi * 100, color='gray', linewidth=1.5, linestyle='--', label=f'95% CI upper')
ax5.axvline(0, color='black', linewidth=1, linestyle=':')
ax5.set_xlabel('ATE (percentage points)')
ax5.set_ylabel('Bootstrap Frequency')
ax5.set_title(f'Bootstrap Distribution of ATE\n(n={N_BOOT:,} resamples)')
ax5.legend(fontsize=8)

# ── Plot 6: Fraud rate by hour — causal framing ───────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
hourly = df.groupby('hour').agg(
    fraud_rate=('is_fraud', 'mean'),
    n=('is_fraud', 'count')
).reset_index()
colors_h = ['#d73027' if h in [22, 23] else '#4575b4' for h in hourly['hour']]
ax6.bar(hourly['hour'], hourly['fraud_rate'] * 100, color=colors_h, edgecolor='white')
ax6.set_xlabel('Hour of Day')
ax6.set_ylabel('Fraud Rate (%)')
ax6.set_title('Fraud Rate by Hour\n(red = treatment hours 22–23)')
ax6.set_xticks(range(0, 24, 2))

plt.savefig('/tmp/causal_inference.png', dpi=150, bbox_inches='tight')
plt.close()
print('Plot saved: /tmp/causal_inference.png')
display(plt.imread('/tmp/causal_inference.png'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Upload to ADLS and log to MLflow

# COMMAND ----------

from azure.storage.blob import BlobServiceClient
import json

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
with open('/tmp/causal_inference.png', 'rb') as f:
    blob_client.get_blob_client('reports', 'causal_inference.png').upload_blob(f, overwrite=True)
print('Uploaded: reports/causal_inference.png')

# Upload matched dataset (plain primitives to avoid PlanMetrics)
matched.attrs.clear()
matched.to_parquet('/tmp/psm_matched.parquet', index=False)
with open('/tmp/psm_matched.parquet', 'rb') as f:
    blob_client.get_blob_client('gold', 'causal/psm_matched.parquet').upload_blob(f, overwrite=True)
print('Uploaded: gold/causal/psm_matched.parquet')

# MLflow
with mlflow.start_run(run_name='psm_causal_inference_v1'):
    mlflow.log_param('treatment',         'is_night (hours 22-23)')
    mlflow.log_param('outcome',           'is_fraud')
    mlflow.log_param('matching_method',   'nearest_neighbour_1to1')
    mlflow.log_param('caliper',           round(float(caliper), 6))
    mlflow.log_param('n_bootstrap',       N_BOOT)
    mlflow.log_param('sample_size',       len(df))

    mlflow.log_metric('propensity_auc',   round(float(ps_auc), 4))
    mlflow.log_metric('matched_pairs',    int(len(matched_night_idx)))
    mlflow.log_metric('naive_ate_pp',     round(float(naive_ate * 100), 4))
    mlflow.log_metric('causal_ate_pp',    round(float(ate * 100), 4))
    mlflow.log_metric('ci_lower_pp',      round(float(ci_lo * 100), 4))
    mlflow.log_metric('ci_upper_pp',      round(float(ci_hi * 100), 4))
    mlflow.log_metric('confounding_bias_pp', round(float((naive_ate - ate) * 100), 4))
    mlflow.log_metric('statistically_significant', int(significant))

    mlflow.log_artifact('/tmp/causal_inference.png')
    run_id = mlflow.active_run().info.run_id
    print(f'MLflow run: {run_id}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Summary

# COMMAND ----------

print('=' * 60)
print('CAUSAL INFERENCE SUMMARY — PSM')
print('=' * 60)
print(f'Treatment          : Night transaction (hour 22–23)')
print(f'Sample size        : {len(df):,} transactions')
print(f'Matched pairs      : {len(matched_night_idx):,}')
print(f'Propensity AUC     : {ps_auc:.4f}')
print()
print(f'Naive ATE          : +{naive_ate*100:.3f}pp  (unadjusted)')
print(f'Causal ATE         : +{ate*100:.3f}pp  (PSM-adjusted)')
print(f'95% CI             : [{ci_lo*100:.3f}pp, {ci_hi*100:.3f}pp]')
print(f'Confounding bias   : {(naive_ate-ate)*100:.3f}pp removed by matching')
print(f'Significant        : {"YES — night causally increases fraud" if significant else "NO — effect explained by confounders"}')
print()
print('Covariate balance after matching:')
for _, row in balance_df.iterrows():
    status = 'GOOD' if row['SMD_after'] < 0.1 else 'ACCEPTABLE' if row['SMD_after'] < 0.25 else 'POOR'
    print(f'  {row["variable"]:<12} SMD before={row["SMD_before"]:.3f}  after={row["SMD_after"]:.3f}  [{status}]')
