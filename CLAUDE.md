# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal
Portfolio project for bank data roles (TD, RBC, BMO, Scotiabank). It must show bank-grade
practices: reconciliation, data-quality controls, audit trails, governance. These controls
are the differentiator, so never remove or weaken them.

## Rules
- I must be able to explain every line in an interview. Explain what you change and why,
  briefly, in plain language.
- Never commit .env or anything in data/. Never hardcode passwords.
- Keep the README "Key findings" section up to date with real numbers only. Never invent figures.
- Windows machine, PostgreSQL 17 local, Python venv in .venv.

## Roadmap
Week 1: bronze load + controls ✓ → Week 2: Databricks silver/gold in PySpark ✓ → Week 3: Azure Data Factory, fraud KPIs, K-Means merchants ✓ → Week 4: XGBoost with cost-based threshold, SARIMA vs Holt-Winters forecast ✓ → Week 5: Power BI dashboard (10 visuals) ✓ → Week 6: executive memo + architecture diagram ✓ → Week 7: SHAP explainability + MLflow tracking ✓ → Week 8: LLM integration (AI fraud analyst) ✓ → Week 9: dbt gold layer ✓ → Week 10: Champion/Challenger A/B ✓ → Week 11: Isolation Forest anomaly detection ✓ → Week 12: RFM cardholder segmentation ✓ → Week 13: causal inference (PSM) ✓ → Week 14: streaming simulation (next).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env            # then set DATABASE_URL in .env
```

`DATABASE_URL` format: `postgresql+psycopg://postgres:PASSWORD@localhost:5432/fraud`

## Running the pipeline

```bash
# Full run with Kaggle CSVs (data/fraudTrain.csv + data/fraudTest.csv)
python run_week1.py

# Against a specific file
python run_week1.py --files data/sample_clean.csv

# Skip the profiling step
python run_week1.py --skip-profile

# Test controls without real data
python tests/make_sample.py
python run_week1.py --files data/sample_clean.csv data/sample_dirty.csv
```

Exit code 1 on any CRITICAL control failure — designed to gate CI.

## Architecture

The pipeline follows a medallion pattern: **source CSV → BRONZE → (Silver/Gold in week 2)**

Each run goes through four steps in sequence:
1. **`pipeline/load_bronze.py`** — streams the CSV in 100k-row chunks via PostgreSQL `COPY`, storing every value as TEXT. Each row gets `_run_id`, `_source_file`, `_source_row` lineage columns. Records the run in `audit.etl_run_log`.
2. **`controls/reconcile.py`** — re-reads the source CSV independently and compares row count, total amount (±0.01 tolerance), fraud count, and distinct transaction IDs against what landed in bronze. Any mismatch is CRITICAL.
3. **`controls/dq_rules.py`** — runs SQL predicates against `bronze.transactions_raw` for the run's `_run_id`. Bad records are written to `audit.dq_exceptions`; a summary row goes to `audit.control_results`. CRITICAL failures return `False`; WARN failures do not.
4. **`analysis/profile_data.py`** — queries the latest successful run per source file and writes `docs/data_profile.md`.

## Database schema

Two PostgreSQL schemas are created by `sql/01_create_schemas.sql`:

- **`bronze.transactions_raw`** — all source columns as TEXT, plus 4 lineage columns (`_run_id`, `_source_file`, `_source_row`, `_loaded_at`).
- **`audit.etl_run_log`** — one row per pipeline run (RUNNING → SUCCEEDED/FAILED).
- **`audit.control_results`** — one row per control executed (both RECONCILIATION and DATA_QUALITY types).
- **`audit.dq_exceptions`** — one row per record that broke a DQ rule.

## Adding a data-quality rule

Append a tuple to `RULES` in `controls/dq_rules.py`:
```python
("rule_name", "CRITICAL"|"WARNING", "SQL predicate for BAD rows", "column_expression_to_show")
```
No other changes needed. The predicate runs against `bronze.transactions_raw` filtered by `_run_id`.

## Data files

The Kaggle CSVs (`fraudTrain.csv`, `fraudTest.csv`) are not committed — place them in `data/`. The `tests/make_sample.py` generator creates `data/sample_clean.csv` and `data/sample_dirty.csv` with planted defects covering all CRITICAL rules plus one WARNING.

## Roadmap

Week 2 adds Databricks/PySpark silver and gold layers. Weeks 3–5 add Azure Data Factory, fraud KPIs, XGBoost model, SARIMA forecast, and Power BI dashboard.
