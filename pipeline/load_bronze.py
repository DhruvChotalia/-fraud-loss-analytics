"""
Land a raw CSV into bronze.transactions_raw exactly as received.

- Every value is kept as text (no silent type conversion, no dropped rows).
- Each row gets lineage: run id, source file, source row number, load time.
- Every run is recorded in audit.etl_run_log (RUNNING -> SUCCEEDED / FAILED).
- Loads with PostgreSQL COPY in chunks, so ~1.3M rows take a minute or two.
"""
import io
import uuid
from pathlib import Path

import pandas as pd
from sqlalchemy import text

BRONZE_COLUMNS = [
    "src_index", "trans_date_trans_time", "cc_num", "merchant", "category", "amt",
    "first", "last", "gender", "street", "city", "state", "zip", "lat", "long",
    "city_pop", "job", "dob", "trans_num", "unix_time", "merch_lat", "merch_long",
    "is_fraud",
]
CHUNK_ROWS = 100_000


def _start_run(engine, run_id, source_file):
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO audit.etl_run_log (run_id, pipeline, source_file, status)
                    VALUES (:r, 'load_bronze', :f, 'RUNNING')"""),
            {"r": run_id, "f": source_file},
        )


def _finish_run(engine, run_id, status, rows_read, rows_loaded, error=None):
    with engine.begin() as conn:
        conn.execute(
            text("""UPDATE audit.etl_run_log
                    SET status=:s, finished_at=now(), rows_read=:rr, rows_loaded=:rl, error_message=:e
                    WHERE run_id=:r"""),
            {"s": status, "rr": rows_read, "rl": rows_loaded, "e": error, "r": run_id},
        )


def _copy_chunk(engine, df: pd.DataFrame) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False)
    buf.seek(0)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            with cur.copy(f"COPY bronze.transactions_raw ({cols}) FROM STDIN WITH (FORMAT csv)") as cp:
                cp.write(buf.getvalue())
        raw.commit()
    finally:
        raw.close()


def load_file(engine, csv_path: Path) -> str:
    """Load one CSV into bronze. Returns the run_id."""
    run_id = str(uuid.uuid4())
    source_file = csv_path.name
    _start_run(engine, run_id, source_file)
    rows_read = rows_loaded = 0
    try:
        reader = pd.read_csv(
            csv_path, dtype=str, keep_default_na=False, chunksize=CHUNK_ROWS
        )
        for chunk in reader:
            chunk = chunk.rename(columns={"Unnamed: 0": "src_index"})
            missing = set(BRONZE_COLUMNS) - set(chunk.columns)
            if missing:
                raise ValueError(f"Source file is missing expected columns: {sorted(missing)}")
            chunk = chunk[BRONZE_COLUMNS].copy()
            # Blank strings become NULL so missing values are visible to the controls.
            chunk = chunk.replace({"": None})
            chunk["_run_id"] = run_id
            chunk["_source_file"] = source_file
            chunk["_source_row"] = range(rows_read + 1, rows_read + len(chunk) + 1)
            rows_read += len(chunk)
            _copy_chunk(engine, chunk)
            rows_loaded += len(chunk)
            print(f"  {source_file}: {rows_loaded:,} rows landed", end="\r")
        print()
        _finish_run(engine, run_id, "SUCCEEDED", rows_read, rows_loaded)
    except Exception as exc:  # record the failure, then re-raise
        _finish_run(engine, run_id, "FAILED", rows_read, rows_loaded, str(exc)[:1000])
        raise
    return run_id
