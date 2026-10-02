"""
Week 1 pipeline: create schemas -> land each CSV in bronze -> reconcile -> data-quality rules -> profile.

Usage:
    python run_week1.py                       # loads data/fraudTrain.csv and data/fraudTest.csv
    python run_week1.py --files data/sample_train.csv
Exit code is 1 if any CRITICAL control fails, so this can gate CI later.
"""
import argparse
import sys
from pathlib import Path

from analysis.profile_data import build_profile
from controls.dq_rules import run_dq
from controls.reconcile import reconcile
from pipeline.db import get_engine, run_sql_file
from pipeline.load_bronze import load_file

ROOT = Path(__file__).resolve().parent
DEFAULT_FILES = [ROOT / "data" / "fraudTrain.csv", ROOT / "data" / "fraudTest.csv"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", type=Path, default=DEFAULT_FILES)
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    engine = get_engine()
    run_sql_file(engine, ROOT / "sql" / "01_create_schemas.sql")

    all_ok = True
    for path in args.files:
        if not path.exists():
            print(f"Missing file: {path}. Put the Kaggle CSVs in the data/ folder.")
            return 1
        print(f"\n=== {path.name} ===")
        run_id = load_file(engine, path)
        print(f"  run_id: {run_id}")
        recon_ok = reconcile(engine, run_id, path)
        dq_ok = run_dq(engine, run_id)
        all_ok &= recon_ok and dq_ok

    if not args.skip_profile:
        out = build_profile(engine)
        print(f"\nProfile written to {out.relative_to(ROOT)}")

    print("\nRESULT:", "ALL CRITICAL CONTROLS PASSED" if all_ok else "CRITICAL CONTROL FAILURE — see audit.control_results")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
