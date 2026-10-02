"""
Data-quality rules on the bronze layer.

Each rule is a SQL predicate that identifies BAD records. For every rule:
  - the offending records are written to audit.dq_exceptions (for investigation), and
  - a summary row is written to audit.control_results (PASS / FAIL / WARN).
CRITICAL rules fail the run; WARNING rules are reported but do not block it.
Add a rule by appending to RULES; nothing else needs to change.
"""
from sqlalchemy import text

NUM = r"'^-?[0-9]+(\.[0-9]+)?$'"
TS = r"'^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$'"

RULES = [
    # name, severity, predicate for a BAD row, column shown as offending value
    ("trans_num_missing", "CRITICAL", "trans_num IS NULL", "trans_num"),
    ("amt_not_numeric", "CRITICAL", f"amt IS NULL OR amt !~ {NUM}", "amt"),
    ("amt_not_positive", "CRITICAL", f"amt ~ {NUM} AND amt::numeric <= 0", "amt"),
    ("timestamp_invalid", "CRITICAL",
     f"trans_date_trans_time IS NULL OR trans_date_trans_time !~ {TS}", "trans_date_trans_time"),
    ("is_fraud_invalid", "CRITICAL", "is_fraud IS NULL OR is_fraud NOT IN ('0','1')", "is_fraud"),
    ("cc_num_missing", "CRITICAL", "cc_num IS NULL", "cc_num"),
    ("customer_coords_out_of_range", "WARNING",
     f"NOT (lat ~ {NUM} AND long ~ {NUM} AND lat::numeric BETWEEN -90 AND 90 "
     f"AND long::numeric BETWEEN -180 AND 180)", "lat || ',' || long"),
    ("merchant_coords_out_of_range", "WARNING",
     f"NOT (merch_lat ~ {NUM} AND merch_long ~ {NUM} AND merch_lat::numeric BETWEEN -90 AND 90 "
     f"AND merch_long::numeric BETWEEN -180 AND 180)", "merch_lat || ',' || merch_long"),
    ("category_missing", "WARNING", "category IS NULL", "category"),
]

# Duplicates need a window, so they are handled separately.
DUPLICATE_SQL = """
SELECT _source_row, trans_num
FROM (SELECT _source_row, trans_num,
             count(*) OVER (PARTITION BY trans_num) AS n
      FROM bronze.transactions_raw
      WHERE _run_id = :run_id AND trans_num IS NOT NULL) d
WHERE n > 1
"""


def _record(conn, run_id, name, severity, bad_count, total):
    status = "PASS" if bad_count == 0 else ("FAIL" if severity == "CRITICAL" else "WARN")
    conn.execute(text("""
        INSERT INTO audit.control_results
          (run_id, control_type, control_name, layer, severity, expected_value, actual_value, status, details)
        VALUES (:r, 'DATA_QUALITY', :n, 'bronze', :s, 0, :bad, :st, :d)"""),
        {"r": run_id, "n": name, "s": severity, "bad": bad_count, "st": status,
         "d": f"{bad_count:,} of {total:,} records broke this rule"})
    print(f"  [{status}] dq {name:<30} bad records: {bad_count:,}")
    return status


def run_dq(engine, run_id: str) -> bool:
    all_passed = True
    with engine.begin() as conn:
        total = conn.execute(text(
            "SELECT count(*) FROM bronze.transactions_raw WHERE _run_id = :r"), {"r": run_id}).scalar()

        for name, severity, predicate, shown in RULES:
            inserted = conn.execute(text(f"""
                INSERT INTO audit.dq_exceptions (run_id, rule_name, severity, source_row, trans_num, offending_value)
                SELECT :r, :n, :s, _source_row, trans_num, {shown}
                FROM bronze.transactions_raw
                WHERE _run_id = :r AND ({predicate})"""), {"r": run_id, "n": name, "s": severity}).rowcount
            if _record(conn, run_id, name, severity, inserted, total) == "FAIL":
                all_passed = False

        dup_rows = conn.execute(text(DUPLICATE_SQL), {"r": run_id, "run_id": run_id}).all()
        if dup_rows:
            conn.execute(text("""
                INSERT INTO audit.dq_exceptions (run_id, rule_name, severity, source_row, trans_num, offending_value)
                VALUES (:r, 'trans_num_duplicate', 'CRITICAL', :row, :t, :t)"""),
                [{"r": run_id, "row": row, "t": t} for row, t in dup_rows])
        if _record(conn, run_id, "trans_num_duplicate", "CRITICAL", len(dup_rows), total) == "FAIL":
            all_passed = False
    return all_passed
