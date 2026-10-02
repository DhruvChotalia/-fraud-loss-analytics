# Executive Memo — Credit Card Fraud Loss Analysis

**To:** Head of Fraud Risk Management  
**From:** Fraud Analytics  
**Date:** October 1, 2026  
**Re:** 2019–2020 Portfolio Fraud Review — Findings and Recommended Actions  
**Classification:** Internal — Restricted

---

## Summary

A full review of 1,852,394 credit card transactions across the 2019–2020 portfolio identified
**$5.12 million in fraud losses** at an overall fraud rate of **0.53%**. Three risk concentrations
account for the majority of losses: a single merchant category, a two-hour overnight window, and a
subset of high-risk merchants. A machine-learning model is ready to deploy that would have caught
92.3% of fraud at a lower cost than the current rule-based threshold.

---

## Key Findings

### 1. Online shopping is the primary loss driver
`shopping_net` (online retail transactions) accounts for **$2.2 million** — 43% of total fraud losses
— at a fraud rate of **1.59%**, three times the portfolio average. The next highest category is
`grocery_pos` at 0.97%. All other categories are below the portfolio average.

**Action:** Prioritise enhanced authentication (step-up challenges, device fingerprinting) for
online retail transactions above a defined amount threshold.

### 2. Two hours account for disproportionate overnight fraud
Fraud rates spike to **2.5–2.6%** between 22:00 and 23:59 — **15× the midday rate** of ~0.09%.
This pattern is consistent with automated card-testing scripts that run after business hours when
real-time intervention is lowest.

**Action:** Apply tighter velocity controls and lower automatic-approval thresholds during the
22:00–00:00 window. Consider an overnight review queue for flagged transactions above $200.

### 3. Merchant clustering reveals four distinct risk profiles
K-Means clustering across 1,852 merchants identified four segments:

| Cluster | Profile | Avg Fraud Rate | Avg Ticket |
|---|---|---|---|
| High-risk online | Small ticket, high frequency | Highest | Low |
| High-value targets | Large ticket, lower volume | Elevated | High |
| Standard retail | Consistent with portfolio average | Average | Medium |
| Low-risk recurring | Subscriptions, utilities | Lowest | Low |

Merchants in the high-risk online cluster warrant enhanced transaction monitoring and
accelerated dispute resolution SLAs.

---

## Model Recommendation

A gradient-boosted tree model trained on nine features (transaction amount, hour, day of week,
merchant category, customer–merchant distance, and demographics) achieves an **AUC-ROC of 0.87**.

Cost-based threshold optimisation (using $530 per missed fraud vs. $25 per false decline) selects
**threshold 0.60** as optimal:

| Metric | Default (0.50) | Recommended (0.60) |
|---|---|---|
| Fraud caught | — | 92.3% (recall) |
| False declines | Higher | 7,937 |
| Total cost | Baseline | **$23K lower** |

The model uses a time-based train/test split (no data leakage) and is re-trainable on a rolling
12-month window as new transaction data arrives.

---

## Loss Forecast

Holt-Winters exponential smoothing (MAPE 36.9%) outperformed SARIMA (MAPE 91.4%) on this 24-month
series. The winning model forecasts fraud losses for the next six months and is recalibrated
quarterly. The high MAPE reflects limited seasonal history (24 months); forecast accuracy will
improve as the model accumulates a longer series.

---

## Data Infrastructure

All findings are derived from a fully reconciled pipeline:

- Every load is **reconciled back to source** — row count, total dollar amount (±$0.01 tolerance),
  fraud count, and distinct transaction IDs are verified before any data enters the analytical layer.
- **10 data-quality rules** gate each load; critical failures block the pipeline and create a
  full audit trail.
- Every run is logged in `audit.etl_run_log` with timestamps, row counts, and control results,
  making findings fully reproducible and traceable to source files.

This means every number in this memo can be traced back to a specific pipeline run and source file.

---

## Recommended Next Steps

1. **Immediate:** Deploy enhanced authentication for `shopping_net` transactions above $100.
2. **30 days:** Implement tightened velocity controls for the 22:00–23:59 window.
3. **60 days:** Deploy the GBT model at threshold 0.60 in a shadow-scoring mode; compare decisions
   against the existing rule set for 30 days before cutover.
4. **90 days:** Schedule quarterly forecast recalibration and model retraining as standard
   operational cadence.

---

*Analysis based on the Kaggle Credit Card Fraud Detection dataset (simulated, 2019–2020).
All figures are derived directly from the data — none are estimated or adjusted.*
