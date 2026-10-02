"""
Profile the landed data and write docs/data_profile.md.

Uses only the latest SUCCEEDED run for each source file, and only rows whose
values are valid, so the profile reflects data that passed the controls.
"""
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data_profile.md"

LATEST_RUNS = """
WITH latest AS (
    SELECT DISTINCT ON (source_file) run_id, source_file
    FROM audit.etl_run_log
    WHERE pipeline = 'load_bronze' AND status = 'SUCCEEDED'
    ORDER BY source_file, started_at DESC
)
"""

CLEAN = f"""{LATEST_RUNS},
clean AS (
    SELECT b._source_file                         AS source_file,
           b.trans_date_trans_time::timestamp     AS txn_time,
           b.cc_num, b.merchant, b.category,
           b.amt::numeric                         AS amt,
           b.is_fraud::int                        AS is_fraud
    FROM bronze.transactions_raw b
    JOIN latest l ON l.run_id = b._run_id
    WHERE b.amt ~ '^-?[0-9]+(\\.[0-9]+)?$'
      AND b.is_fraud IN ('0','1')
      AND b.trans_date_trans_time ~ '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}} [0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}$'
)
"""

QUERIES = {
    "Overview by source file": CLEAN + """
        SELECT source_file,
               count(*)                                  AS transactions,
               count(DISTINCT cc_num)                    AS cards,
               count(DISTINCT merchant)                  AS merchants,
               min(txn_time)::date                       AS first_day,
               max(txn_time)::date                       AS last_day,
               round(sum(amt), 2)                        AS total_amount,
               sum(is_fraud)                             AS fraud_txns,
               round(100.0 * avg(is_fraud), 3)           AS fraud_rate_pct,
               round(sum(amt) FILTER (WHERE is_fraud = 1), 2) AS fraud_amount
        FROM clean GROUP BY source_file ORDER BY source_file""",
    "Fraud by category (highest fraud rate first)": CLEAN + """
        SELECT category,
               count(*)                                        AS transactions,
               sum(is_fraud)                                   AS fraud_txns,
               round(100.0 * avg(is_fraud), 3)                 AS fraud_rate_pct,
               round(sum(amt) FILTER (WHERE is_fraud = 1), 2)  AS fraud_amount,
               round(avg(amt) FILTER (WHERE is_fraud = 1), 2)  AS avg_fraud_ticket,
               round(avg(amt) FILTER (WHERE is_fraud = 0), 2)  AS avg_legit_ticket
        FROM clean GROUP BY category ORDER BY fraud_rate_pct DESC""",
    "Top 10 merchants by fraud loss": CLEAN + """
        SELECT merchant,
               count(*)                                        AS transactions,
               sum(is_fraud)                                   AS fraud_txns,
               round(sum(amt) FILTER (WHERE is_fraud = 1), 2)  AS fraud_amount
        FROM clean GROUP BY merchant
        HAVING sum(is_fraud) > 0
        ORDER BY fraud_amount DESC LIMIT 10""",
    "Fraud by hour of day": CLEAN + """
        SELECT extract(hour FROM txn_time)::int                AS hour,
               count(*)                                        AS transactions,
               sum(is_fraud)                                   AS fraud_txns,
               round(100.0 * avg(is_fraud), 3)                 AS fraud_rate_pct
        FROM clean GROUP BY 1 ORDER BY 1""",
    "Monthly fraud loss": CLEAN + """
        SELECT to_char(date_trunc('month', txn_time), 'YYYY-MM') AS month,
               count(*)                                          AS transactions,
               sum(is_fraud)                                     AS fraud_txns,
               round(sum(amt) FILTER (WHERE is_fraud = 1), 2)    AS fraud_amount
        FROM clean GROUP BY 1 ORDER BY 1""",
}

CONTROLS = LATEST_RUNS + """
SELECT l.source_file, c.control_type, c.control_name, c.severity, c.status, c.actual_value, c.details
FROM audit.control_results c
JOIN latest l ON l.run_id = c.run_id
ORDER BY l.source_file, c.control_type DESC, c.control_name
"""


def build_profile(engine) -> Path:
    parts = [f"# Data profile\n\nGenerated {date.today():%Y-%m-%d} from the latest successful load "
             "of each source file. Only records with valid amount, fraud flag and timestamp are profiled.\n"]
    with engine.connect() as conn:
        for title, sql in QUERIES.items():
            df = pd.read_sql(sql, conn)
            parts.append(f"## {title}\n\n{df.to_markdown(index=False)}\n")
        ctl = pd.read_sql(CONTROLS, conn)
    parts.append("## Control results for these loads\n\n" + ctl.to_markdown(index=False) + "\n")
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(parts), encoding="utf-8")
    return OUT
