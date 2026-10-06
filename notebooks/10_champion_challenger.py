# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Week 10 — Champion / Challenger A/B Framework
# MAGIC
# MAGIC **What this notebook builds:**
# MAGIC A production-style model governance framework. Before any new fraud model
# MAGIC goes live at a bank, it must be tested against the current production model
# MAGIC on real traffic. This is called Champion/Challenger testing.
# MAGIC
# MAGIC **Why OSFI requires this:**
# MAGIC OSFI Guideline E-23 (Model Risk Management) requires financial institutions to:
# MAGIC 1. Document the process for promoting a model from development to production
# MAGIC 2. Test challenger models on live or representative traffic before promotion
# MAGIC 3. Keep an audit trail of every model comparison decision
# MAGIC
# MAGIC **Setup:**
# MAGIC - **Champion:** GBT model, threshold = 0.60 (current production, from Week 4/7)
# MAGIC - **Challenger A:** GBT model, threshold = 0.55 (higher recall, more aggressive)
# MAGIC - **Challenger B:** GBT model, threshold = 0.65 (higher precision, more conservative)
# MAGIC - **Split:** 60% champion / 20% challenger A / 20% challenger B
# MAGIC - **Decision metric:** total cost = (missed fraud × $530) + (false declines × $25)
# MAGIC
# MAGIC **Prerequisite:** Run notebooks 02 and 05 first so silver.transactions and
# MAGIC gold.fraud_scores exist.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load scored transactions from the champion model

# COMMAND ----------

import pandas as pd
import numpy as np
from pyspark.sql import functions as F

# Load the full scored dataset from notebook 05
# gold.fraud_scores has: trans_num, amt, is_fraud, fraud_prob, predicted_fraud
scores = spark.table("gold.fraud_scores")
print(f"Total scored transactions: {scores.count():,}")
scores.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Define the A/B split
# MAGIC
# MAGIC In production, the split happens at the point of transaction — each incoming
# MAGIC transaction is randomly routed to either the champion or challenger model.
# MAGIC
# MAGIC We simulate this by assigning a random bucket to each scored transaction.
# MAGIC The fraud_prob is already computed — we just apply different thresholds to
# MAGIC the same probability score to simulate different model variants.
# MAGIC
# MAGIC **Why this is valid:** All three variants use the same underlying GBT model.
# MAGIC The only difference is the decision threshold. This is the most common form
# MAGIC of champion/challenger in production — same model, different operating points.

# COMMAND ----------

# Reproducible random split
np.random.seed(42)

scores_pd = scores.toPandas()
n = len(scores_pd)

# Assign each transaction to a model variant
buckets = np.random.choice(
    ["champion", "challenger_a", "challenger_b"],
    size=n,
    p=[0.60, 0.20, 0.20]   # 60/20/20 split
)
scores_pd["model_variant"] = buckets

print("Split distribution:")
print(scores_pd["model_variant"].value_counts().to_string())
print(f"\nFraud rate by variant (should be similar — random split):")
for variant in ["champion", "challenger_a", "challenger_b"]:
    subset = scores_pd[scores_pd["model_variant"] == variant]
    print(f"  {variant}: {subset['is_fraud'].mean():.3%} fraud rate ({len(subset):,} transactions)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Apply each model's threshold
# MAGIC
# MAGIC Champion:     threshold = 0.60 (cost-optimised in Week 4)
# MAGIC Challenger A: threshold = 0.55 (more aggressive — catches more fraud, more false declines)
# MAGIC Challenger B: threshold = 0.65 (more conservative — fewer false declines, misses more fraud)

# COMMAND ----------

THRESHOLDS = {
    "champion":     0.60,
    "challenger_a": 0.55,
    "challenger_b": 0.65,
}

AVG_FRAUD_LOSS    = 530.0
FALSE_DECLINE_COST = 25.0

def evaluate_variant(df: pd.DataFrame, threshold: float, name: str) -> dict:
    predicted = (df["fraud_prob"] >= threshold).astype(int)
    actual    = df["is_fraud"]

    tp = ((predicted == 1) & (actual == 1)).sum()   # caught fraud
    fn = ((predicted == 0) & (actual == 1)).sum()   # missed fraud
    fp = ((predicted == 1) & (actual == 0)).sum()   # false declines
    tn = ((predicted == 0) & (actual == 0)).sum()   # correct approvals

    precision  = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall     = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1         = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    total_cost = fn * AVG_FRAUD_LOSS + fp * FALSE_DECLINE_COST

    fraud_loss_caught   = df.loc[(predicted == 1) & (actual == 1), "amt"].sum()
    fraud_loss_missed   = df.loc[(predicted == 0) & (actual == 1), "amt"].sum()

    return {
        "variant":            name,
        "threshold":          threshold,
        "transactions":       len(df),
        "caught_fraud":       int(tp),
        "missed_fraud":       int(fn),
        "false_declines":     int(fp),
        "true_negatives":     int(tn),
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "f1_score":           round(f1, 4),
        "total_cost":         round(total_cost, 2),
        "fraud_loss_caught":  round(fraud_loss_caught, 2),
        "fraud_loss_missed":  round(fraud_loss_missed, 2),
    }

results = []
for variant, threshold in THRESHOLDS.items():
    subset = scores_pd[scores_pd["model_variant"] == variant]
    result = evaluate_variant(subset, threshold, variant)
    results.append(result)

results_df = pd.DataFrame(results)
print("\nChampion / Challenger Results:")
print("=" * 90)
print(results_df[[
    "variant", "threshold", "transactions",
    "caught_fraud", "missed_fraud", "false_declines",
    "recall", "precision", "total_cost"
]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Normalise to equal traffic for fair comparison
# MAGIC
# MAGIC The three variants processed different numbers of transactions (60/20/20 split).
# MAGIC To compare apples-to-apples, we project each variant's costs to what they would
# MAGIC look like on the same volume — 10,000 transactions.

# COMMAND ----------

NORMALISE_TO = 10_000

normalised = []
for r in results:
    scale = NORMALISE_TO / r["transactions"]
    normalised.append({
        "variant":                  r["variant"],
        "threshold":                r["threshold"],
        "normalised_transactions":  NORMALISE_TO,
        "normalised_caught_fraud":  round(r["caught_fraud"]   * scale),
        "normalised_missed_fraud":  round(r["missed_fraud"]   * scale),
        "normalised_false_declines": round(r["false_declines"] * scale),
        "normalised_total_cost":    round(r["total_cost"]     * scale, 2),
        "recall":                   r["recall"],
        "precision":                r["precision"],
        "f1_score":                 r["f1_score"],
    })

norm_df = pd.DataFrame(normalised)
print(f"\nNormalised to {NORMALISE_TO:,} transactions (fair comparison):")
print("=" * 90)
print(norm_df[[
    "variant", "threshold",
    "normalised_caught_fraud", "normalised_missed_fraud",
    "normalised_false_declines", "normalised_total_cost",
    "recall", "precision"
]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Decision — promote, keep, or reject?
# MAGIC
# MAGIC Decision rule: promote a challenger if it reduces total cost by more than 5%
# MAGIC vs the champion on normalised traffic. 5% is a common bank threshold —
# MAGIC small improvements don't justify the operational cost of switching models.

# COMMAND ----------

PROMOTION_THRESHOLD_PCT = 5.0   # must beat champion by at least 5% on total cost

champion_cost = norm_df[norm_df["variant"] == "champion"]["normalised_total_cost"].values[0]

print("CHAMPION / CHALLENGER DECISION")
print("=" * 60)
print(f"Champion total cost (per {NORMALISE_TO:,} txns): ${champion_cost:,.2f}")
print()

decisions = []
for _, row in norm_df[norm_df["variant"] != "champion"].iterrows():
    challenger_cost = row["normalised_total_cost"]
    improvement_pct = (champion_cost - challenger_cost) / champion_cost * 100
    beats_threshold = improvement_pct > PROMOTION_THRESHOLD_PCT

    decision = "PROMOTE" if beats_threshold else "REJECT"
    reason = (
        f"reduces cost by {improvement_pct:.1f}% > {PROMOTION_THRESHOLD_PCT}% threshold"
        if beats_threshold else
        f"only {improvement_pct:.1f}% improvement — below {PROMOTION_THRESHOLD_PCT}% threshold"
        if improvement_pct > 0 else
        f"increases cost by {abs(improvement_pct):.1f}% — worse than champion"
    )

    decisions.append({
        "variant":          row["variant"],
        "threshold":        row["threshold"],
        "challenger_cost":  challenger_cost,
        "vs_champion":      f"{improvement_pct:+.1f}%",
        "decision":         decision,
        "reason":           reason,
    })

    print(f"{row['variant'].upper()} (threshold={row['threshold']}):")
    print(f"  Cost:      ${challenger_cost:,.2f}")
    print(f"  vs Champion: {improvement_pct:+.1f}%")
    print(f"  Decision:  {decision} — {reason}")
    print(f"  Recall:    {row['recall']:.1%}  |  Precision: {row['precision']:.1%}")
    print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Save results and log to MLflow

# COMMAND ----------

import mlflow
from azure.storage.blob import BlobServiceClient
import io

STORAGE_ACCOUNT = "fraudanalytics2024"
STORAGE_KEY      = "YOUR_STORAGE_KEY_HERE"
CONTAINER        = "models"

blob_svc = BlobServiceClient(
    account_url=f"https://{STORAGE_ACCOUNT}.blob.core.windows.net",
    credential=STORAGE_KEY
)

def upload_csv(df, blob_name):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    blob_svc.get_blob_client(container=CONTAINER, blob=blob_name) \
            .upload_blob(buf.getvalue(), overwrite=True)
    print(f"Uploaded: {blob_name}")

upload_csv(results_df, "champion_challenger/raw_results.csv")
upload_csv(norm_df,    "champion_challenger/normalised_results.csv")
upload_csv(pd.DataFrame(decisions), "champion_challenger/decisions.csv")

# ── MLflow ────────────────────────────────────────────────────────────────────
mlflow.set_experiment("/Users/dhruvextra76@gmail.com/champion-challenger")

with mlflow.start_run(run_name="champion_challenger_v1"):

    for r in results:
        prefix = r["variant"]
        mlflow.log_metrics({
            f"{prefix}_recall":        r["recall"],
            f"{prefix}_precision":     r["precision"],
            f"{prefix}_f1":            r["f1_score"],
            f"{prefix}_total_cost":    r["total_cost"],
            f"{prefix}_caught_fraud":  r["caught_fraud"],
            f"{prefix}_false_declines": r["false_declines"],
        })

    mlflow.log_params({
        "champion_threshold":     THRESHOLDS["champion"],
        "challenger_a_threshold": THRESHOLDS["challenger_a"],
        "challenger_b_threshold": THRESHOLDS["challenger_b"],
        "split_champion_pct":     0.60,
        "split_challenger_pct":   0.20,
        "promotion_threshold_pct": PROMOTION_THRESHOLD_PCT,
        "avg_fraud_loss_usd":     AVG_FRAUD_LOSS,
        "false_decline_cost_usd": FALSE_DECLINE_COST,
        "normalised_to_n_txns":   NORMALISE_TO,
    })

    mlflow.set_tags({
        "project":       "fraud-loss-analytics",
        "week":          "10",
        "framework":     "champion_challenger",
        "osfi_e23":      "model_governance",
    })

    results_df.to_csv("/tmp/cc_raw_results.csv", index=False)
    norm_df.to_csv("/tmp/cc_normalised.csv", index=False)
    mlflow.log_artifact("/tmp/cc_raw_results.csv",   "champion_challenger")
    mlflow.log_artifact("/tmp/cc_normalised.csv",    "champion_challenger")

    print("Champion/Challenger results logged to MLflow.")
    print("View: Experiments → champion-challenger")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary — What Week 10 adds
# MAGIC
# MAGIC | Before Week 10 | After Week 10 |
# MAGIC |---|---|
# MAGIC | One model, deployed once | Champion and challengers running simultaneously |
# MAGIC | No governance process | Formal promotion/rejection decision with audit trail |
# MAGIC | Can't prove new model is better | Side-by-side cost comparison on equal traffic |
# MAGIC | OSFI E-23 gap | Champion/challenger framework satisfies OSFI model governance |
# MAGIC
# MAGIC **Interview answer when asked about model deployment:**
# MAGIC "We use a champion/challenger framework. The current production model is the
# MAGIC champion — GBT at threshold 0.60. Any new variant is a challenger that runs on
# MAGIC 20% of traffic while the champion handles 60%. After the test period, we compare
# MAGIC total cost normalised to equal transaction volume. A challenger must beat the
# MAGIC champion by at least 5% on total cost to be promoted — small improvements don't
# MAGIC justify the operational cost of switching. Every comparison is logged in MLflow
# MAGIC with the decision and the reason. This satisfies OSFI E-23's requirement for
# MAGIC documented model promotion processes."
