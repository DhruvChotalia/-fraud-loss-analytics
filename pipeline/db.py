"""Database connection helpers. The connection string comes from .env, never from code."""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def get_engine():
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    return create_engine(url, future=True)


def run_sql_file(engine, path: Path) -> None:
    """Execute a .sql file (used to create schemas/tables)."""
    with engine.begin() as conn:
        conn.exec_driver_sql(path.read_text(encoding="utf-8"))


def scalar(engine, sql: str, **params):
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar()
