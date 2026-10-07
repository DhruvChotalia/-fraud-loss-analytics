# Credit Card Fraud & Loss Analytics Platform

[![DQ Controls Gate](https://github.com/DhruvChotalia/-fraud-loss-analytics/actions/workflows/controls.yml/badge.svg)](https://github.com/DhruvChotalia/-fraud-loss-analytics/actions/workflows/controls.yml)

An end-to-end fraud analytics pipeline built the way a bank would run it: every load is
**reconciled back to source**, every record is checked against **data-quality controls**, and every
run leaves an **audit trail**, before any analysis or model touches the data.

> **Key findings** *(updated week 1 — model findings to follow)*
> - Fraud rate: 0.53% of 1.85M transactions; $5.12M in total fraud losses across both files
> - Highest-risk category: shopping_net at 1.59% fraud rate (3× the portfolio average); $2.2M in losses
> - Fraud spikes sharply at night: 22:00–23:59 hit 2.5–2.6% fraud rate vs. ~0.09% during business hours
> - Model: catches 92.3% of fraud at threshold 0.60 (cost-optimised); 7,937 false declines; saves $23K vs default 0.50 threshold
> - Forecast: Holt-Winters beats SARIMA on 24-month series (MAPE 36.9% vs 91.4%); SARIMA underperforms due to insufficient seasonal history

## Architecture

```
Kaggle CSVs ──► BRONZE (raw, as received, with lineage)
                  │   ├─ Reconciliation: row count, $ total, fraud count, distinct IDs vs source
                  │   └─ Data-quality rules → audit.dq_exceptions + audit.control_results
                  ▼
               SILVER (typed, cleaned, PII masked)          ← week 2 (Databricks / PySpark)
                  ▼
               GOLD (star schema: fact_transactions + dims) ← week 2
                  ▼
     Analytics · fraud model · loss forecast · Power BI    ← weeks 3–5
```

## Controls (what makes this different)

| Control | Type | Severity | What it proves |
|---|---|---|---|
| Row count, total $ amount, fraud count, distinct transaction IDs | Reconciliation (source → bronze) | Critical | Nothing was lost or duplicated in the load |
| Missing / duplicate transaction ID | Data quality | Critical | Every transaction is uniquely identifiable |
| Amount not numeric or not positive | Data quality | Critical | Loss figures can be trusted |
| Invalid timestamp, invalid fraud flag, missing card | Data quality | Critical | Records are usable for analysis |
| Coordinates out of range, missing category | Data quality | Warning | Flagged for investigation, does not block |

Every run is logged in `audit.etl_run_log`; each control result in `audit.control_results`;
each failing record in `audit.dq_exceptions`. The pipeline exits with code 1 on any critical
failure, so it can gate CI.

## Data

[Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
(simulated, ~1.85M transactions, 2019–2020). The CSVs are not committed; place `fraudTrain.csv`
and `fraudTest.csv` in `data/`.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then edit DATABASE_URL (macOS/Linux: cp)
python run_week1.py               # load → reconcile → data-quality → profile
```

Test the controls without the real data:

```bash
python tests/make_sample.py
python run_week1.py --files data/sample_clean.csv data/sample_dirty.csv
```
`sample_dirty.csv` contains planted defects; every one must be caught.

## Project layout

```
sql/        schema definitions (bronze + audit)
pipeline/   loading code
controls/   reconciliation + data-quality rules
analysis/   profiling, clustering, forecasting, model
dashboards/ Power BI file + screenshots
docs/       data profile, migration notes, memo
tests/      sample-data generator for testing the controls
```

## Roadmap

- [x] Week 1 — bronze landing, reconciliation, data-quality controls, audit log, data profile
- [x] Week 2 — Databricks: silver/gold in PySpark, SQL → Spark SQL migration notes
- [x] Week 3 — Azure Data Factory ingestion pipeline design, fraud KPIs, K-Means merchant clustering (PySpark MLlib)
- [x] Week 4 — XGBoost fraud model (GBT + cost-based threshold), SARIMA vs Holt-Winters loss forecast
- [x] Week 5 — Power BI dashboard (10 visuals: fraud by category/hour/month/state/cluster, model threshold, scatter plot)
- [x] Week 6 — executive memo (docs/executive_memo.md), architecture document (docs/), GitHub Actions DQ gate
- [x] Week 7 — SHAP explainability (notebooks/07_explainability.py): global feature importance, beeswarm summary, single-transaction waterfall; MLflow experiment tracking on all model runs
- [x] Week 8 — LLM integration (notebooks/08_llm_analyst.py): AI fraud analyst with weekly brief generator, transaction explainer, and anomaly alert using Claude API
- [x] Week 9 — dbt gold layer (dbt/fraud_analytics/): staging + 3 mart models, 14 data tests passing (not_null, unique, accepted_values)
- [x] Week 10 — Champion/Challenger A/B framework (notebooks/10_champion_challenger.py): 60/20/20 split, normalised cost comparison, promotion decision with MLflow audit trail
- [x] Week 11 — Isolation Forest anomaly detection (notebooks/11_isolation_forest.py): unsupervised anomaly layer, GBT blind spot analysis, score distribution plots
- [x] Week 12 — RFM cardholder segmentation (notebooks/12_rfm_segmentation.py): Recency/Frequency/Monetary quintile scoring, 9 cardholder segments, fraud rate overlay by segment, MLflow tracking
- [x] Week 13 — Causal inference (notebooks/13_causal_inference.py): Propensity Score Matching isolates causal effect of night-time transactions on fraud, bootstrap 95% CI on ATE, covariate balance diagnostics
- [x] Week 14 — Streaming simulation (notebooks/14_streaming_simulation.py): micro-batch fraud scoring pipeline, P50/P95/P99 latency SLA, spike detection, cumulative cost saved dashboard
