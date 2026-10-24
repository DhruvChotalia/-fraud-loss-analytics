# Credit Card Fraud & Loss Analytics Platform

[![DQ Controls Gate](https://github.com/DhruvChotalia/-fraud-loss-analytics/actions/workflows/controls.yml/badge.svg)](https://github.com/DhruvChotalia/-fraud-loss-analytics/actions/workflows/controls.yml)

> **14-week end-to-end fraud analytics platform** built to bank-grade standards.  
> Every load is reconciled back to source · every record passes data-quality controls · every model decision is logged and explainable · every analytical finding is causally validated.

---

## Key Findings

| Metric | Value |
|---|---|
| Portfolio size | 1,852,394 transactions · $142M total spend |
| Total fraud losses | **$5,121,413** across 9,651 fraud transactions |
| Fraud rate | **0.521%** portfolio average |
| Highest-risk channel | **shopping_net** — 1.59% fraud rate · $2.21M losses |
| Night-time spike | **22:00–23:00 hits 2.6% fraud rate — 5× the portfolio average** |
| GBT model AUC-ROC | **0.9952** · Recall 92.3% at cost-optimised threshold 0.60 |
| Cost saved vs default threshold | **$23K** (threshold 0.60 vs default 0.50) |
| Causal ATE (night effect) | **+2.127pp** (95% CI [1.883, 2.378], p < 0.05) |
| Dormant cardholder fraud rate | **1.61%** — 3.1× portfolio average |
| Streaming P95 latency | **2.7ms** per 2,000-transaction batch |
| dbt gold layer tests | **14/14 passing** |

---

## Architecture

```
Kaggle CSVs (1.85M rows)
      │
      ▼
┌─────────────────────────────────────────────────┐
│  BRONZE  —  PostgreSQL 17                       │
│  All 22 columns stored as TEXT · 4 lineage cols │
│  Reconciliation: row count, $total, fraud count │
│  DQ rules → audit.dq_exceptions                 │
│  CI/CD gate: exit 1 on CRITICAL failure         │
└─────────────────────┬───────────────────────────┘
                      │  Azure Data Factory (design)
                      ▼
┌─────────────────────────────────────────────────┐
│  SILVER  —  Databricks / Delta Lake             │
│  PySpark: type cast · PII mask · geo_dist       │
│  silver.transactions (1.85M rows, typed)        │
└─────────────────────┬───────────────────────────┘
                      ▼
┌─────────────────────────────────────────────────┐
│  GOLD  —  Star Schema + dbt                     │
│  fact_transactions · dim_merchant               │
│  dim_customer · dim_datetime                    │
│  gold.fraud_kpis · gold.merchant_clusters       │
│  14 dbt tests gate every deployment             │
└──────┬──────────────┬──────────────┬────────────┘
       │              │              │
       ▼              ▼              ▼
  ML Models      Advanced        Output
  ─────────      Analytics       ──────
  GBTClassifier  Isolation       Power BI (10 visuals)
  AUC 0.9952     Forest          Claude API fraud agent
  SHAP + MLflow  RFM segments    Streaming simulation
  Champion /     PSM causal      ADLS Gen2
  Challenger     inference
```

---

## What Was Built (14 Weeks)

### Weeks 1–2 · Data Pipeline & Bronze/Silver/Gold

- **Bronze landing** (PostgreSQL): all 22 columns stored as TEXT with 4 lineage columns; every load logged in `audit.etl_run_log`
- **Reconciliation**: source CSV re-read independently; row count, total amount (±$0.01), fraud count, distinct IDs compared against bronze — any mismatch is CRITICAL
- **5 CRITICAL + 2 WARNING DQ rules**: null/duplicate IDs, non-numeric amounts, invalid timestamps, bad fraud flags, missing card numbers
- **GitHub Actions CI/CD**: pipeline runs on every push against sample data; exits with code 1 on any CRITICAL failure
- **Silver layer** (Databricks/PySpark): type casting, PII masking (`cc_num` → last 4 digits), `geo_dist` feature, Delta Lake storage
- **Gold star schema**: `fact_transactions` + `dim_merchant`, `dim_customer`, `dim_datetime` — surrogate keys via `dense_rank()`

### Weeks 3–4 · Analytics, ML Model & Forecast

- **Fraud KPIs**: 6 SQL aggregations (by category, hour, state, month, day-type, merchant)
- **K-Means merchant clustering** (PySpark MLlib, k=4): 693 merchants segmented by fraud rate, avg ticket, volume — 4 risk tiers for targeted controls
- **Azure Data Factory pipeline design**: JSON-based ADF pipeline orchestrating Bronze→Silver→Gold with error handling
- **GBTClassifier**: AUC-ROC 0.9952 · cost-based threshold optimisation (FN×$530 + FP×$25) → threshold 0.60, recall 92.3%
- **SARIMA vs Holt-Winters** loss forecast: 18-month train / 6-month test; Holt-Winters MAPE 36.9% vs SARIMA 91.4%

### Weeks 5–6 · Power BI & Executive Reporting

- **Power BI dashboard**: 10 visuals, DirectQuery to gold tables, DAX measures for fraud rate, loss rate, MoM trend
- **Architecture document** + executive memo summarising findings and recommendations for a C-suite audience

### Weeks 7–8 · Explainability & LLM Integration

- **SHAP explainability** (TreeExplainer): global beeswarm plot (top features: amt, hour, geo_dist, category) + per-transaction waterfall; logged to MLflow; satisfies OSFI E-23
- **MLflow tracking**: AUC, recall, precision, cost logged on every run; tagged with OSFI E-23 compliance status
- **GenAI fraud analyst** (Claude API): weekly risk brief generator + transaction explainer; generates human-readable summaries from structured KPI inputs

### Weeks 9–10 · Data Governance & Model Management

- **dbt gold layer**: 4 models (`stg_transactions`, `fraud_by_category`, `fraud_by_hour`, `fraud_monthly_trend`) — 14 data tests (not_null, unique, accepted_values), all passing
- **Champion / Challenger A/B framework**: 60/20/20 traffic split; normalised cost comparison per 10,000 transactions; promotion rule (>5% cost improvement); full MLflow audit trail — satisfies OSFI E-23 model governance

### Weeks 11–12 · Anomaly Detection & Segmentation

- **Isolation Forest**: unsupervised anomaly scoring; cross-referenced with GBT to identify blind spots (4 transactions flagged by IF, missed by GBT)
- **RFM cardholder segmentation**: 9 segments; **Dormant cardholders at 1.61% fraud rate — 3.1× portfolio average**; actionable: step-up auth on first dormant-account use

### Weeks 13–14 · Causal Inference & Streaming

- **Propensity Score Matching** (causal inference): 15,138 matched night/day pairs; covariate balance (all SMD < 0.1 post-match); bootstrap 95% CI; **causal ATE +2.127pp confirms the late-night spike is a genuine causal effect, not a confounder artefact**
- **Streaming simulation**: 554,000 transactions in 277 micro-batches; P50=2.3ms, P95=2.7ms, P99=3.5ms latency; 37 fraud spikes detected; $711,885 cost saved

---

## Controls (What Makes This Different from Typical Portfolio Projects)

| Control | Type | Severity | What it proves |
|---|---|---|---|
| Row count, total $ amount, fraud count, distinct IDs | Reconciliation | CRITICAL | Nothing lost or duplicated in the load |
| Missing / duplicate transaction ID | Data quality | CRITICAL | Every transaction is uniquely identifiable |
| Amount not numeric or ≤ 0 | Data quality | CRITICAL | Loss calculations can be trusted |
| Invalid timestamp | Data quality | CRITICAL | Time-series analysis needs clean dates |
| Invalid fraud flag (not 0 or 1) | Data quality | CRITICAL | Label integrity for model training |
| Missing card number | Data quality | CRITICAL | Cardholder linkage required |
| Coordinates out of range | Data quality | WARNING | Flagged, does not block |
| Missing merchant category | Data quality | WARNING | Clustering degrades without it |

Every run is logged in `audit.etl_run_log`; every control result in `audit.control_results`; every failing record in `audit.dq_exceptions`.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Ingestion & Storage | PostgreSQL 17 · Azure ADLS Gen2 · Delta Lake |
| Orchestration | Azure Data Factory (pipeline design) |
| Processing | PySpark · Databricks Community Edition |
| Data Quality | Python · SQLAlchemy · dbt Core 1.12 |
| ML & Explainability | PySpark MLlib (GBT, K-Means) · SHAP · MLflow |
| Statistics | statsmodels (SARIMA, Holt-Winters) · scipy (PSM bootstrap) |
| GenAI | Claude API (Anthropic) |
| Visualisation | Power BI · matplotlib |
| CI/CD | GitHub Actions |
| Language | Python 3.13 |

---

## Portfolio Documentation

Seven bi-weekly portfolio documents (`docs/portfolio/`) walk through every stage with screenshots, code explanations, and interview-ready talking points:

| File | Weeks | Contents |
|---|---|---|
| `Week01_02_Data_Pipeline_Architecture.docx` | 1–2 | Bronze schema, DQ controls, CI/CD gate, silver schema |
| `Week03_04_Azure_Orchestration_ML.docx` | 3–4 | K-Means clusters, cost curve, confusion matrix, forecast |
| `Week05_06_Visualization_Executive_Reporting.docx` | 5–6 | Power BI dashboard, architecture diagram |
| `Week07_08_Explainability_LLM.docx` | 7–8 | SHAP plots, MLflow run, LLM fraud brief |
| `Week09_10_Data_Governance_Model_Management.docx` | 9–10 | dbt tests, champion/challenger comparison |
| `Week11_12_Anomaly_RFM_Segmentation.docx` | 11–12 | Isolation Forest, RFM segments |
| `Week13_14_Causal_Inference_Streaming.docx` | 13–14 | PSM causal analysis, streaming dashboard |

---

## Data

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) — simulated, ~1.85M transactions, 2019–2020. CSVs are not committed; place `fraudTrain.csv` and `fraudTest.csv` in `data/`.

## Run It

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # set DATABASE_URL=postgresql+psycopg://postgres:PASSWORD@localhost:5432/fraud
python run_week1.py               # load → reconcile → data-quality → profile
```

Test the controls without the real data:
```bash
python tests/make_sample.py
python run_week1.py --files data/sample_clean.csv data/sample_dirty.csv
```
`sample_dirty.csv` has planted defects covering every CRITICAL rule — all must be caught.
