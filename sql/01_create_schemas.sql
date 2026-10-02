-- =====================================================================
-- Week 1: landing (bronze) layer + audit/controls layer
-- Bronze stores every source value exactly as received (TEXT), plus
-- lineage columns, so nothing is lost or silently converted on load.
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS audit;

-- One row per source record, raw and untyped ---------------------------
CREATE TABLE IF NOT EXISTS bronze.transactions_raw (
    src_index              TEXT,
    trans_date_trans_time  TEXT,
    cc_num                 TEXT,
    merchant               TEXT,
    category               TEXT,
    amt                    TEXT,
    first                  TEXT,
    last                   TEXT,
    gender                 TEXT,
    street                 TEXT,
    city                   TEXT,
    state                  TEXT,
    zip                    TEXT,
    lat                    TEXT,
    long                   TEXT,
    city_pop               TEXT,
    job                    TEXT,
    dob                    TEXT,
    trans_num              TEXT,
    unix_time              TEXT,
    merch_lat              TEXT,
    merch_long             TEXT,
    is_fraud               TEXT,
    -- lineage
    _run_id                UUID        NOT NULL,
    _source_file           TEXT        NOT NULL,
    _source_row            INTEGER     NOT NULL,
    _loaded_at             TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_bronze_run ON bronze.transactions_raw (_run_id);

-- One row per pipeline run ----------------------------------------------
CREATE TABLE IF NOT EXISTS audit.etl_run_log (
    run_id         UUID PRIMARY KEY,
    pipeline       TEXT        NOT NULL,
    source_file    TEXT        NOT NULL,
    status         TEXT        NOT NULL,          -- RUNNING / SUCCEEDED / FAILED
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at    TIMESTAMPTZ,
    rows_read      BIGINT,
    rows_loaded    BIGINT,
    error_message  TEXT
);

-- One row per control executed (reconciliation + data-quality rules) ----
CREATE TABLE IF NOT EXISTS audit.control_results (
    id             BIGSERIAL PRIMARY KEY,
    run_id         UUID        NOT NULL REFERENCES audit.etl_run_log(run_id),
    control_type   TEXT        NOT NULL,          -- RECONCILIATION / DATA_QUALITY
    control_name   TEXT        NOT NULL,
    layer          TEXT        NOT NULL,          -- source->bronze, bronze, ...
    severity       TEXT        NOT NULL,          -- CRITICAL / WARNING
    expected_value NUMERIC,
    actual_value   NUMERIC,
    status         TEXT        NOT NULL,          -- PASS / FAIL / WARN
    details        TEXT,
    checked_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One row per record that broke a data-quality rule ---------------------
CREATE TABLE IF NOT EXISTS audit.dq_exceptions (
    id             BIGSERIAL PRIMARY KEY,
    run_id         UUID        NOT NULL REFERENCES audit.etl_run_log(run_id),
    rule_name      TEXT        NOT NULL,
    severity       TEXT        NOT NULL,
    source_row     INTEGER,
    trans_num      TEXT,
    offending_value TEXT,
    logged_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_dq_run ON audit.dq_exceptions (run_id, rule_name);
