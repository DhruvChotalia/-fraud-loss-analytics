# Azure Data Factory — Ingestion Pipeline Design

## Deployment status

**Pipeline deployed and verified** — `pl_fraud_ingest` ran successfully on 2026-09-30.
- Storage account: `fraudanalytics2024` (Canada Central, ADLS Gen2, LRS)
- Source container: `landing` (fraudTrain.csv + fraudTest.csv uploaded)
- Sink container: `bronze` (both files confirmed after pipeline run)
- ADF instance: `fraud-adf-2024` (Canada Central)
- Resource group: `fraud-analytics-rg`

## What this pipeline does

`pl_fraud_ingest` moves raw fraud CSVs from the `landing` container into the `bronze`
container of ADLS Gen2 (`fraudanalytics2024`). In a production setup this would run
at 02:00 UTC daily and trigger the Databricks silver notebook on success.

The pipeline definition is in `adf/pipeline_fraud_ingest.json`.

---

## Architecture

```
┌─────────────────────┐     ┌──────────────────────────────────┐
│  Source             │     │  Azure                           │
│  Blob Storage       │     │                                  │
│  /fraud/            │     │  ┌─────────────────────────┐     │
│  fraudTrain.csv     │     │  │  ADF Pipeline           │     │
│  fraudTest.csv      │     │  │                         │     │
└──────────┬──────────┘     │  │  1. Validate file       │     │
           │                │  │       ↓                 │     │
           │ triggers       │  │  2. Copy to ADLS        │     │
           └───────────────►│  │       ↓                 │     │
                            │  │  3. Log to audit DB     │     │
                            │  │       ↓                 │     │
                            │  │  4. Trigger Databricks  │     │
                            │  └─────────────────────────┘     │
                            │           │                       │
                            │           ▼                       │
                            │  ADLS Gen2 (bronze zone)         │
                            │  abfss://bronze@.../fraud/       │
                            │           │                       │
                            │           ▼                       │
                            │  Databricks silver notebook       │
                            │  → silver.transactions (Delta)   │
                            └──────────────────────────────────┘
```

---

## The four activities

### 1. Validate source file (`act_validate_source_file`)
Waits up to 30 minutes for the file to appear in Blob Storage before doing anything.
Retries every 60 seconds. If the file never arrives, the pipeline fails here and
nothing downstream runs.

**Why this matters:** upstream systems are unreliable. Without validation, a missing
file would either cause the copy to fail mid-run (messy) or copy an empty/stale file
silently (dangerous). Failing fast with a clear error is better than silent corruption.

### 2. Copy to ADLS Gen2 (`act_copy_to_adls`)
Copies the file byte-for-byte into the bronze container. No transformation — same
principle as the PostgreSQL bronze layer: land the data exactly as received, validate
and transform in the next step.

ADF captures `rowsRead` and `rowsCopied` in the activity output. These feed the next
activity as the expected control totals.

### 3. Log to audit table (`act_log_control_result`)
Writes one row to `audit.etl_run_log` in Azure SQL Database:
- `run_id` — ADF pipeline run ID (unique per execution)
- `source_file` — file name
- `rows_read` — from ADF copy output
- `rows_copied` — from ADF copy output
- `loaded_at` — UTC timestamp
- `status` — SUCCEEDED

This mirrors the `audit.etl_run_log` table from the PostgreSQL week 1 layer.
Every load is traceable. If something goes wrong downstream, you can look up
exactly when the file arrived and how many rows it had.

### 4. Trigger Databricks silver notebook (`act_trigger_databricks_silver`)
Calls the Databricks REST API to run `02_silver` and passes two parameters:
- `adls_path` — the full ADLS path of the file that was just copied
- `run_id` — the ADF pipeline run ID, so silver can stamp it on the rows it writes

**Why ADF triggers Databricks and not the other way around:**
ADF is the orchestrator — it owns the schedule, the dependencies, and the retry
logic. Databricks is the compute engine — it does the transformation. Keeping
orchestration and compute separate is a standard pattern in Azure data platforms.

---

## Linked services (connections)

| Name | Type | Purpose |
|---|---|---|
| `ls_blob_source` | Azure Blob Storage | Where the upstream system drops CSV files |
| `ls_adls_bronze` | ADLS Gen2 | Bronze landing zone; Databricks reads from here |
| `ls_azure_sql_audit` | Azure SQL Database | Audit log — same schema as PostgreSQL audit layer |
| `ls_databricks` | Azure Databricks | Workspace where the silver notebook runs |

In ADF, linked services store connection strings and credentials using
**Azure Key Vault** — passwords are never hardcoded in the pipeline JSON.
This is the same principle as the `.env` file in the local PostgreSQL setup.

---

## Trigger

`tr_daily_0200_utc` — Schedule trigger, runs every day at 02:00 UTC.

In production you would use a **Storage Event trigger** instead: ADF fires
the moment the file lands in Blob Storage, rather than on a fixed schedule.
The schedule trigger is simpler to explain and sufficient for a batch pipeline.

---

## What this would look like in a TD/RBC/BMO environment

- Source Blob would be replaced by an SFTP drop from a card processor (Visa/MC)
  or an internal mainframe extract
- ADLS Gen2 would be inside a VNet with private endpoints (no public internet access)
- The Databricks cluster would use a managed identity — no passwords, no service principals
- Azure Monitor alerts would fire on pipeline failure and page the on-call engineer
- The audit SQL table would feed a Power BI ops dashboard showing pipeline health

---

## Interview talking points

**"Why ADF and not just a cron job or Airflow?"**
ADF is the standard orchestrator on Azure. It has a visual designer so non-engineers
can read the pipeline, native connectors to every Azure service, built-in retry and
dependency logic, and it integrates with Azure Monitor for alerting. Airflow is better
for complex Python-heavy DAGs; ADF is better when the primary job is moving data
between Azure services.

**"Why validate before copying?"**
Because a missing file should be a loud failure, not a silent one. If the copy runs on
a zero-byte file, downstream jobs get empty tables and the fraud dashboard shows
zero transactions — which looks like no fraud, not like a broken pipeline. Failing at
the validation step makes the problem immediately visible.

**"How does ADF connect to Databricks?"**
Through a Databricks-type linked service that authenticates via a personal access token
stored in Key Vault. ADF passes parameters to the notebook (ADLS path, run ID) so the
notebook knows which file to process. ADF waits for the notebook job to complete and
marks the pipeline as failed if the notebook fails.
