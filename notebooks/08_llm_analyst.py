# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 8 — LLM Integration: AI Fraud Analyst
# MAGIC
# MAGIC **What this notebook builds:**
# MAGIC An AI copilot for fraud analysts. Instead of a person manually reading dashboards,
# MAGIC SHAP plots, and threshold tables — they ask a question and get a clear answer.
# MAGIC
# MAGIC **Three capabilities:**
# MAGIC
# MAGIC | Capability | What it does | Who uses it |
# MAGIC |---|---|---|
# MAGIC | Weekly fraud brief | Reads KPIs + patterns, writes executive summary | Head of Fraud Risk |
# MAGIC | Transaction explainer | Reads SHAP values, explains why a transaction was flagged | Fraud analyst |
# MAGIC | Anomaly alert | Compares current vs historical KPIs, flags deviations | Risk monitoring team |
# MAGIC
# MAGIC **Why this matters at a bank:**
# MAGIC Banks are actively building AI copilots to reduce fraud investigation time. The current
# MAGIC average time to investigate a flagged transaction is 15–20 minutes. With an LLM that
# MAGIC can read the model's outputs and explain them in plain English, that drops to 2–3 minutes.
# MAGIC OSFI has also issued guidance on AI use in financial services — having a human-readable
# MAGIC audit trail of every AI-generated explanation is part of responsible AI governance.
# MAGIC
# MAGIC **Prerequisite:** Weeks 2–7 complete (silver layer, model scores, SHAP outputs in ADLS).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install and configure

# COMMAND ----------

%pip install anthropic>=0.25.0

# COMMAND ----------

import anthropic
import pandas as pd
import json
from datetime import datetime, date
from io import StringIO
from azure.storage.blob import BlobServiceClient

# ── Storage config ────────────────────────────────────────────────────────────
STORAGE_ACCOUNT = "fraudanalytics2024"
STORAGE_KEY      = "YOUR_STORAGE_KEY_HERE"   # replace at runtime — never commit
CONTAINER_MODELS = "models"
CONTAINER_DATA   = "frauddata"
CONTAINER_REPORTS = "reports"

# ── LLM config ────────────────────────────────────────────────────────────────
# API key stored in Databricks Secrets — never hardcode it.
# To create: databricks secrets create-scope llm-keys
#            databricks secrets put --scope llm-keys --key anthropic-api-key
#
# If you don't have the secret set up yet, uncomment the fallback line below
# and paste your key temporarily — but NEVER commit that line to GitHub.

try:
    ANTHROPIC_API_KEY = dbutils.secrets.get(scope="llm-keys", key="anthropic-api-key")
    print("API key loaded from Databricks Secrets.")
except Exception:
    ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE"   # replace at runtime
    print("WARNING: using placeholder key — set up Databricks Secrets for production.")

llm = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
MODEL = "claude-haiku-4-5-20251001"   # fast + cheap for structured data tasks; swap to claude-sonnet-5-5 for richer prose

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load model outputs from ADLS Gen2
# MAGIC
# MAGIC Everything the LLM needs was already uploaded to ADLS in earlier weeks:
# MAGIC - `models/model_summary.csv` — AUC, threshold, recall, cost savings (Week 7)
# MAGIC - `models/threshold_analysis.csv` — cost curve across thresholds (Week 7)
# MAGIC - `models/shap/shap_feature_importance.csv` — mean |SHAP| per feature (Week 7)
# MAGIC
# MAGIC We also pull live KPIs directly from the gold Delta table.

# COMMAND ----------

blob_client = BlobServiceClient(
    account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
    credential=STORAGE_KEY
)

def read_csv_from_adls(container: str, blob_name: str) -> pd.DataFrame:
    data = blob_client.get_blob_client(container=container, blob=blob_name) \
                      .download_blob().readall().decode("utf-8")
    return pd.read_csv(StringIO(data))

def upload_text_to_adls(text: str, container: str, blob_name: str):
    blob_client.get_blob_client(container=container, blob=blob_name) \
               .upload_blob(text.encode("utf-8"), overwrite=True)
    print(f"Saved: {blob_name}")

# Load pre-computed outputs
model_summary  = read_csv_from_adls(CONTAINER_MODELS, "model_summary.csv")
threshold_df   = read_csv_from_adls(CONTAINER_MODELS, "threshold_analysis.csv")
shap_df        = read_csv_from_adls(CONTAINER_MODELS, "shap/shap_feature_importance.csv")

print("model_summary.csv:")
print(model_summary.to_string(index=False))
print("\nshap_feature_importance.csv (top 5):")
print(shap_df.head().to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Pull live KPIs from the gold layer

# COMMAND ----------

# Fraud KPIs by category
category_kpis = spark.sql("""
    SELECT
        category,
        COUNT(*)                                    AS total_txns,
        SUM(is_fraud)                               AS fraud_count,
        ROUND(AVG(is_fraud) * 100, 2)               AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2) AS fraud_losses_usd
    FROM silver.transactions
    GROUP BY category
    ORDER BY fraud_losses_usd DESC
""").toPandas()

# Fraud KPIs by hour (top fraud hours)
hourly_kpis = spark.sql("""
    SELECT
        HOUR(txn_time)                              AS hour,
        COUNT(*)                                    AS total_txns,
        SUM(is_fraud)                               AS fraud_count,
        ROUND(AVG(is_fraud) * 100, 2)               AS fraud_rate_pct
    FROM silver.transactions
    GROUP BY HOUR(txn_time)
    ORDER BY fraud_rate_pct DESC
    LIMIT 5
""").toPandas()

# Overall portfolio KPIs
overall_kpis = spark.sql("""
    SELECT
        COUNT(*)                                    AS total_transactions,
        SUM(is_fraud)                               AS total_fraud_count,
        ROUND(AVG(is_fraud) * 100, 3)               AS overall_fraud_rate_pct,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN amt ELSE 0 END), 2) AS total_fraud_losses_usd,
        ROUND(AVG(CASE WHEN is_fraud = 1 THEN amt END), 2)        AS avg_fraud_txn_usd
    FROM silver.transactions
""").toPandas()

print("Overall KPIs:")
print(overall_kpis.to_string(index=False))
print("\nTop categories by fraud loss:")
print(category_kpis.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Capability 1 — Weekly Fraud Brief
# MAGIC
# MAGIC Reads all KPIs and model outputs, produces a 3-paragraph executive summary
# MAGIC suitable for the Head of Fraud Risk Management. No manual writing required.

# COMMAND ----------

def build_weekly_brief_prompt(overall: pd.DataFrame, categories: pd.DataFrame,
                               hourly: pd.DataFrame, model: pd.DataFrame,
                               shap: pd.DataFrame) -> str:
    top_categories = categories.head(3).to_dict(orient="records")
    top_hours      = hourly.head(3).to_dict(orient="records")
    top_features   = shap.head(4)["feature"].tolist()
    m = model.iloc[0]

    return f"""You are a senior fraud risk analyst at a Canadian bank preparing a weekly briefing
for the Head of Fraud Risk Management. Write a concise, professional 3-paragraph executive summary
based on the data below. Use specific numbers. No bullet points. No headers. Write in the tone
of a bank internal memo.

PORTFOLIO OVERVIEW (period: full dataset, 2019–2020):
- Total transactions: {overall['total_transactions'].iloc[0]:,}
- Total fraud count: {overall['total_fraud_count'].iloc[0]:,}
- Overall fraud rate: {overall['overall_fraud_rate_pct'].iloc[0]}%
- Total fraud losses: ${overall['total_fraud_losses_usd'].iloc[0]:,.2f}
- Average fraud transaction: ${overall['avg_fraud_txn_usd'].iloc[0]:,.2f}

TOP RISK CATEGORIES (by fraud loss):
{json.dumps(top_categories, indent=2)}

HIGHEST FRAUD-RATE HOURS:
{json.dumps(top_hours, indent=2)}

DETECTION MODEL PERFORMANCE:
- AUC-ROC: {m['auc_roc']}
- Operating threshold: {m['optimal_threshold']} (cost-optimised)
- Recall (fraud caught): {float(m['recall']) * 100:.1f}%
- False declines: {int(m['false_declines']):,}
- Total cost at optimal threshold: ${float(m['total_cost']):,.2f}

TOP SHAP FEATURES DRIVING MODEL DECISIONS:
{', '.join(top_features)}

Write paragraph 1: overall fraud picture and losses.
Write paragraph 2: key risk concentrations (category, hour).
Write paragraph 3: model performance and recommended actions.
"""

prompt = build_weekly_brief_prompt(overall_kpis, category_kpis, hourly_kpis,
                                   model_summary, shap_df)

response = llm.messages.create(
    model=MODEL,
    max_tokens=600,
    messages=[{"role": "user", "content": prompt}]
)

weekly_brief = response.content[0].text
print("=" * 70)
print("WEEKLY FRAUD BRIEF")
print("=" * 70)
print(weekly_brief)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Capability 2 — Transaction Explainer
# MAGIC
# MAGIC A fraud analyst gets a call: "My card was declined — why?"
# MAGIC This takes the transaction details and the SHAP feature contributions
# MAGIC and produces a clear plain-English explanation.

# COMMAND ----------

def explain_transaction(transaction: dict, shap_contributions: dict) -> str:
    """
    transaction: dict with keys amt, hour, category, geo_distance, fraud_prob
    shap_contributions: dict mapping feature name → SHAP value (positive = toward fraud)
    """
    # Sort features by absolute SHAP value — most influential first
    sorted_features = sorted(shap_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    top_drivers     = sorted_features[:4]

    fraud_factors  = [(f, v) for f, v in top_drivers if v > 0]
    legit_factors  = [(f, v) for f, v in top_drivers if v < 0]

    prompt = f"""You are a fraud analyst at a Canadian bank explaining a transaction decision
to a colleague in plain, professional English. Be specific about which factors mattered.
Keep it to 3–4 sentences. Do not use bullet points.

TRANSACTION DETAILS:
- Amount: ${transaction['amt']:.2f}
- Hour: {transaction['hour']}:00
- Merchant category: {transaction['category']}
- Customer-merchant distance: {transaction['geo_distance']:.1f} km
- Model fraud probability: {transaction['fraud_prob']:.1%}
- Decision: {"FLAGGED FOR REVIEW" if transaction['fraud_prob'] >= 0.60 else "APPROVED"}

FACTORS THAT INCREASED FRAUD SCORE:
{json.dumps([{"feature": f, "shap_contribution": round(v, 4)} for f, v in fraud_factors], indent=2)}

FACTORS THAT DECREASED FRAUD SCORE (toward legitimate):
{json.dumps([{"feature": f, "shap_contribution": round(v, 4)} for f, v in legit_factors], indent=2)}

Explain in 3–4 sentences why this transaction received this score.
"""

    response = llm.messages.create(
        model=MODEL,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


# Demo — explain a high-risk transaction
sample_transaction = {
    "trans_num":    "t1-0012345",
    "amt":          2847.50,
    "hour":         23,
    "category":     "shopping_net",
    "geo_distance": 8.3,
    "fraud_prob":   0.91,
}

sample_shap = {
    "Transaction Amount ($)":         +0.312,
    "Hour of Day":                    +0.187,
    "Merchant Category":              +0.143,
    "Customer–Merchant Distance":     -0.041,
    "Is Weekend":                     +0.029,
    "City Population":                -0.018,
    "Day of Week":                    +0.011,
    "Month":                          -0.008,
    "Gender":                         +0.003,
}

explanation = explain_transaction(sample_transaction, sample_shap)
print("=" * 70)
print(f"TRANSACTION EXPLANATION — {sample_transaction['trans_num']}")
print("=" * 70)
print(explanation)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Capability 3 — Anomaly Alert
# MAGIC
# MAGIC Compares current-period KPIs against a historical baseline and flags
# MAGIC anything unusual. In production this would run on a daily schedule.
# MAGIC Here we simulate it by comparing two halves of the dataset.

# COMMAND ----------

# Simulate: split dataset into "historical" (2019) and "current" (2020)
historical_kpis = spark.sql("""
    SELECT
        ROUND(AVG(is_fraud) * 100, 3)               AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN is_fraud=1 THEN amt END) / COUNT(*), 4) AS fraud_loss_per_txn,
        ROUND(AVG(CASE WHEN is_fraud=1 THEN amt END), 2)            AS avg_fraud_amt
    FROM silver.transactions
    WHERE YEAR(txn_time) = 2019
""").toPandas()

current_kpis = spark.sql("""
    SELECT
        ROUND(AVG(is_fraud) * 100, 3)               AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN is_fraud=1 THEN amt END) / COUNT(*), 4) AS fraud_loss_per_txn,
        ROUND(AVG(CASE WHEN is_fraud=1 THEN amt END), 2)            AS avg_fraud_amt
    FROM silver.transactions
    WHERE YEAR(txn_time) = 2020
""").toPandas()

def build_anomaly_prompt(historical: pd.DataFrame, current: pd.DataFrame) -> str:
    h = historical.iloc[0]
    c = current.iloc[0]

    # Compute % changes
    rate_change   = ((float(c['fraud_rate_pct']) - float(h['fraud_rate_pct']))
                     / float(h['fraud_rate_pct'])) * 100
    loss_change   = ((float(c['fraud_loss_per_txn']) - float(h['fraud_loss_per_txn']))
                     / float(h['fraud_loss_per_txn'])) * 100
    amt_change    = ((float(c['avg_fraud_amt']) - float(h['avg_fraud_amt']))
                     / float(h['avg_fraud_amt'])) * 100

    return f"""You are a fraud risk monitoring system at a Canadian bank. Analyse the KPI changes
below and write a 2-paragraph alert memo. Flag any metric that changed by more than 5%.
Be direct and specific. Recommend one concrete action if warranted.

HISTORICAL BASELINE (2019):
- Fraud rate: {h['fraud_rate_pct']}%
- Fraud loss per transaction: ${h['fraud_loss_per_txn']}
- Average fraud transaction amount: ${h['avg_fraud_amt']}

CURRENT PERIOD (2020):
- Fraud rate: {c['fraud_rate_pct']}% ({rate_change:+.1f}% vs baseline)
- Fraud loss per transaction: ${c['fraud_loss_per_txn']} ({loss_change:+.1f}% vs baseline)
- Average fraud transaction amount: ${c['avg_fraud_amt']} ({amt_change:+.1f}% vs baseline)

Write paragraph 1: summarise what changed and by how much.
Write paragraph 2: assess severity and recommend one action.
"""

anomaly_prompt = build_anomaly_prompt(historical_kpis, current_kpis)

response = llm.messages.create(
    model=MODEL,
    max_tokens=300,
    messages=[{"role": "user", "content": anomaly_prompt}]
)

anomaly_alert = response.content[0].text
print("=" * 70)
print("ANOMALY ALERT — 2019 vs 2020 KPI COMPARISON")
print("=" * 70)
print(anomaly_alert)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Save all outputs to ADLS Gen2

# COMMAND ----------

today = date.today().isoformat()

# Combine all three outputs into one report file
full_report = f"""FRAUD ANALYTICS AI REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
Model: {MODEL}
{"=" * 70}

WEEKLY FRAUD BRIEF
{"=" * 70}
{weekly_brief}

{"=" * 70}
TRANSACTION EXPLANATION — {sample_transaction['trans_num']}
Amount: ${sample_transaction['amt']:,.2f} | Fraud prob: {sample_transaction['fraud_prob']:.1%}
{"=" * 70}
{explanation}

{"=" * 70}
ANOMALY ALERT — KPI MONITORING
{"=" * 70}
{anomaly_alert}
"""

upload_text_to_adls(full_report,    "reports", f"fraud_ai_report_{today}.txt")
upload_text_to_adls(weekly_brief,   "reports", "weekly_brief_latest.txt")
upload_text_to_adls(anomaly_alert,  "reports", "anomaly_alert_latest.txt")

print(f"\nAll reports saved to ADLS Gen2 reports/ container.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Log to MLflow

# COMMAND ----------

import mlflow

mlflow.set_experiment("/Users/dhruvextra76@gmail.com/llm-analyst")

with mlflow.start_run(run_name="LLM_fraud_analyst_v1"):

    mlflow.log_params({
        "llm_model":          MODEL,
        "capabilities":       "weekly_brief,transaction_explainer,anomaly_alert",
        "api_provider":       "anthropic",
        "threshold_used":     float(model_summary["optimal_threshold"].iloc[0]),
    })

    mlflow.log_metrics({
        "total_fraud_losses_usd": float(overall_kpis["total_fraud_losses_usd"].iloc[0]),
        "overall_fraud_rate_pct": float(overall_kpis["overall_fraud_rate_pct"].iloc[0]),
        "model_auc_roc":          float(model_summary["auc_roc"].iloc[0]),
    })

    mlflow.set_tags({
        "project": "fraud-loss-analytics",
        "week":    "8",
        "type":    "llm-integration",
    })

    # Log the full report as an artifact
    with open("/tmp/fraud_ai_report.txt", "w") as f:
        f.write(full_report)
    mlflow.log_artifact("/tmp/fraud_ai_report.txt", "llm_outputs")

    print("Week 8 logged to MLflow: Experiments → fraud-analytics/llm-analyst")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary — What Week 8 adds
# MAGIC
# MAGIC | Before Week 8 | After Week 8 |
# MAGIC |---|---|
# MAGIC | Analyst reads dashboards manually | AI reads dashboards and writes the summary |
# MAGIC | "Why was this flagged?" takes 20 minutes | Transaction explainer answers in seconds |
# MAGIC | KPI changes noticed days later | Anomaly alert fires immediately on new data |
# MAGIC | Model outputs are numbers | Model outputs are plain English a non-technical person can read |
# MAGIC
# MAGIC **Interview answer when asked about LLM integration:**
# MAGIC "We use the Claude API to build three capabilities on top of the fraud model outputs.
# MAGIC First, a weekly brief generator that reads the KPIs and SHAP importances and produces
# MAGIC an executive summary — no manual writing. Second, a transaction explainer that takes
# MAGIC the SHAP values for any individual transaction and explains in plain English why it
# MAGIC was flagged — this directly addresses the fraud analyst's workflow. Third, an anomaly
# MAGIC alert that compares current KPIs against a historical baseline and flags deviations.
# MAGIC The API key is stored in Databricks Secrets, never in code, and every LLM call is
# MAGIC logged in MLflow with the outputs saved to ADLS for audit purposes."
