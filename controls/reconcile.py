"""
Source-to-bronze reconciliation.

Re-reads the source file independently of the loader and compares control
totals with what actually landed in bronze for that run. Any break is a
CRITICAL failure: the load must not be trusted downstream.
"""
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd
from sqlalchemy import text

AMOUNT_TOLERANCE = Decimal("0.01")


def _to_decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except (InvalidOperation, TypeError):
        return Decimal("0")


def source_totals(csv_path: Path) -> dict:
    """Control totals computed straight from the file."""
    rows = fraud = 0
    amount = Decimal("0")
    txn_ids = set()
    for chunk in pd.read_csv(csv_path, dtype=str, keep_default_na=False, chunksize=200_000,
                             usecols=["amt", "is_fraud", "trans_num"]):
        rows += len(chunk)
        amount += sum((_to_decimal(v) for v in chunk["amt"]), Decimal("0"))
        fraud += int((chunk["is_fraud"] == "1").sum())
        txn_ids.update(v for v in chunk["trans_num"] if v)
    return {"row_count": rows, "total_amount": amount, "fraud_count": fraud,
            "distinct_trans_num": len(txn_ids)}


BRONZE_TOTALS_SQL = """
SELECT count(*)                                                        AS row_count,
       coalesce(sum(CASE WHEN amt ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN amt::numeric ELSE 0 END), 0)
                                                                       AS total_amount,
       count(*) FILTER (WHERE is_fraud = '1')                          AS fraud_count,
       count(DISTINCT trans_num)                                       AS distinct_trans_num
FROM bronze.transactions_raw
WHERE _run_id = :run_id
"""


def reconcile(engine, run_id: str, csv_path: Path) -> bool:
    expected = source_totals(csv_path)
    with engine.connect() as conn:
        actual = dict(conn.execute(text(BRONZE_TOTALS_SQL), {"run_id": run_id}).mappings().one())

    results, all_passed = [], True
    for name, exp in expected.items():
        act = Decimal(str(actual[name]))
        tolerance = AMOUNT_TOLERANCE if name == "total_amount" else Decimal("0")
        passed = abs(Decimal(str(exp)) - act) <= tolerance
        all_passed &= passed
        results.append({
            "run_id": run_id, "control_type": "RECONCILIATION", "control_name": name,
            "layer": "source->bronze", "severity": "CRITICAL",
            "expected_value": exp, "actual_value": act,
            "status": "PASS" if passed else "FAIL",
            "details": f"{csv_path.name}: source={exp} bronze={act}",
        })

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO audit.control_results
              (run_id, control_type, control_name, layer, severity, expected_value, actual_value, status, details)
            VALUES (:run_id, :control_type, :control_name, :layer, :severity, :expected_value,
                    :actual_value, :status, :details)"""), results)

    for r in results:
        print(f"  [{r['status']}] reconcile {r['control_name']:<20} "
              f"source={r['expected_value']}  bronze={r['actual_value']}")
    return all_passed
