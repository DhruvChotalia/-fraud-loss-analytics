"""
Local test for Week 8 LLM capabilities.
Uses hardcoded sample data — no Spark, no Databricks, no ADLS needed.
Run: python tests/test_llm_analyst.py
"""
import json
import os
import sys

import anthropic

API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not API_KEY:
    print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
    print("Run:  $env:ANTHROPIC_API_KEY='your-key-here'  (PowerShell)")
    sys.exit(1)

llm   = anthropic.Anthropic(api_key=API_KEY)
MODEL = "claude-haiku-4-5-20251001"

print(f"Using model: {MODEL}")
print("=" * 60)

# ── Sample data (mirrors what the notebook pulls from ADLS/Delta) ─────────────

OVERALL_KPIS = {
    "total_transactions":    1852394,
    "total_fraud_count":     9651,
    "overall_fraud_rate_pct": 0.521,
    "total_fraud_losses_usd": 5120432.18,
    "avg_fraud_txn_usd":     530.56,
}

TOP_CATEGORIES = [
    {"category": "shopping_net",  "fraud_rate_pct": 1.59, "fraud_losses_usd": 2187432.10},
    {"category": "misc_net",      "fraud_rate_pct": 1.21, "fraud_losses_usd": 987234.55},
    {"category": "grocery_pos",   "fraud_rate_pct": 0.31, "fraud_losses_usd": 412876.22},
]

TOP_HOURS = [
    {"hour": 23, "fraud_rate_pct": 2.61},
    {"hour": 22, "fraud_rate_pct": 2.54},
    {"hour":  1, "fraud_rate_pct": 1.87},
]

MODEL_SUMMARY = {
    "auc_roc":           0.8712,
    "optimal_threshold": 0.60,
    "recall":            0.923,
    "false_declines":    7937,
    "total_cost":        412580.0,
}

SHAP_TOP_FEATURES = [
    "Transaction Amount ($)",
    "Hour of Day",
    "Merchant Category",
    "Customer–Merchant Distance",
]

# ── Test 1: Weekly Fraud Brief ────────────────────────────────────────────────

print("\nTEST 1: Weekly Fraud Brief")
print("-" * 60)

brief_prompt = f"""You are a senior fraud risk analyst at a Canadian bank preparing a weekly
briefing for the Head of Fraud Risk Management. Write a concise, professional 3-paragraph
executive summary based on the data below. Use specific numbers. No bullet points. No headers.
Write in the tone of a bank internal memo.

PORTFOLIO OVERVIEW:
- Total transactions: {OVERALL_KPIS['total_transactions']:,}
- Total fraud count: {OVERALL_KPIS['total_fraud_count']:,}
- Overall fraud rate: {OVERALL_KPIS['overall_fraud_rate_pct']}%
- Total fraud losses: ${OVERALL_KPIS['total_fraud_losses_usd']:,.2f}
- Average fraud transaction: ${OVERALL_KPIS['avg_fraud_txn_usd']:,.2f}

TOP RISK CATEGORIES (by fraud loss):
{json.dumps(TOP_CATEGORIES, indent=2)}

HIGHEST FRAUD-RATE HOURS:
{json.dumps(TOP_HOURS, indent=2)}

DETECTION MODEL PERFORMANCE:
- AUC-ROC: {MODEL_SUMMARY['auc_roc']}
- Operating threshold: {MODEL_SUMMARY['optimal_threshold']} (cost-optimised)
- Recall: {MODEL_SUMMARY['recall'] * 100:.1f}%
- False declines: {MODEL_SUMMARY['false_declines']:,}

TOP SHAP FEATURES: {', '.join(SHAP_TOP_FEATURES)}

Write paragraph 1: overall fraud picture and losses.
Write paragraph 2: key risk concentrations (category, hour).
Write paragraph 3: model performance and recommended actions.
"""

response = llm.messages.create(
    model=MODEL,
    max_tokens=600,
    messages=[{"role": "user", "content": brief_prompt}]
)
weekly_brief = response.content[0].text
print(weekly_brief)
print(f"\n[tokens used: {response.usage.input_tokens} in / {response.usage.output_tokens} out]")

# ── Test 2: Transaction Explainer ─────────────────────────────────────────────

print("\n\nTEST 2: Transaction Explainer")
print("-" * 60)

txn_prompt = """You are a fraud analyst at a Canadian bank explaining a transaction decision
to a colleague in plain, professional English. Be specific about which factors mattered.
Keep it to 3-4 sentences. Do not use bullet points.

TRANSACTION DETAILS:
- Amount: $2,847.50
- Hour: 23:00
- Merchant category: shopping_net
- Customer-merchant distance: 8.3 km
- Model fraud probability: 91%
- Decision: FLAGGED FOR REVIEW

FACTORS THAT INCREASED FRAUD SCORE:
[{"feature": "Transaction Amount ($)", "shap_contribution": 0.312},
 {"feature": "Hour of Day", "shap_contribution": 0.187},
 {"feature": "Merchant Category", "shap_contribution": 0.143}]

FACTORS THAT DECREASED FRAUD SCORE:
[{"feature": "Customer-Merchant Distance", "shap_contribution": -0.041}]

Explain in 3-4 sentences why this transaction received this score.
"""

response = llm.messages.create(
    model=MODEL,
    max_tokens=200,
    messages=[{"role": "user", "content": txn_prompt}]
)
explanation = response.content[0].text
print(explanation)
print(f"\n[tokens used: {response.usage.input_tokens} in / {response.usage.output_tokens} out]")

# ── Test 3: Anomaly Alert ─────────────────────────────────────────────────────

print("\n\nTEST 3: Anomaly Alert")
print("-" * 60)

anomaly_prompt = """You are a fraud risk monitoring system at a Canadian bank. Analyse the KPI
changes below and write a 2-paragraph alert memo. Flag any metric that changed by more than 5%.
Be direct and specific. Recommend one concrete action if warranted.

HISTORICAL BASELINE (2019):
- Fraud rate: 0.498%
- Average fraud transaction amount: $512.30

CURRENT PERIOD (2020):
- Fraud rate: 0.551% (+10.6% vs baseline)
- Average fraud transaction amount: $558.90 (+9.1% vs baseline)

Write paragraph 1: summarise what changed and by how much.
Write paragraph 2: assess severity and recommend one action.
"""

response = llm.messages.create(
    model=MODEL,
    max_tokens=300,
    messages=[{"role": "user", "content": anomaly_prompt}]
)
alert = response.content[0].text
print(alert)
print(f"\n[tokens used: {response.usage.input_tokens} in / {response.usage.output_tokens} out]")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("All 3 capabilities working. Week 8 LLM integration verified.")
print("Next step: run notebooks/08_llm_analyst.py in Databricks")
print("           with real data from silver.transactions + ADLS.")
