"""Real idempotent CSV-to-Postgres ingestion for Cashflow IQ.

Replaces the Phase A ingest_minimal.py. Key improvements:
  - Pydantic validation of every row before insertion
  - Idempotent upsert (ON CONFLICT) — safe to re-run without duplication
  - Ingestion log table tracking accepted/rejected rows per file
  - CLI interface with --customers, --transactions, --labels args

Usage:
    python -m apps.pipelines.ingest.run \
        --customers path/to/customers.csv \
        --transactions path/to/transactions.csv \
        --labels path/to/labels.csv
"""

import argparse
import csv
import hashlib
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import psycopg2
from pydantic import ValidationError

from packages.schemas.customer import Customer
from packages.schemas.label import Label
from packages.schemas.transaction import Transaction

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_connection() -> psycopg2.extensions.connection:
    """Create a Postgres connection from DATABASE_URL env var."""
    database_url = os.environ["DATABASE_URL"]
    return psycopg2.connect(database_url)


def _ensure_tables(conn: psycopg2.extensions.connection) -> None:
    """Create or migrate all required tables.

    Uses CREATE TABLE IF NOT EXISTS for new installs, and ALTER TABLE ADD
    COLUMN IF NOT EXISTS for tables that already exist from Phase A.
    """
    with conn.cursor() as cur:
        # --- raw_customers (expanded from Phase A) ---
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT,
                occupation TEXT,
                employer_tenure_months INTEGER,
                city TEXT,
                account_vintage_months INTEGER,
                persona_type TEXT
            );
        """)
        # Add columns that may be missing from Phase A tables
        for col, coltype in [
            ("occupation", "TEXT"),
            ("employer_tenure_months", "INTEGER"),
            ("city", "TEXT"),
            ("account_vintage_months", "INTEGER"),
            ("persona_type", "TEXT"),
        ]:
            cur.execute(f"""
                DO $$ BEGIN
                    ALTER TABLE raw_customers ADD COLUMN {col} {coltype};
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$;
            """)

        # --- raw_transactions (expanded from Phase A) ---
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT,
                date DATE,
                amount NUMERIC,
                category TEXT,
                type TEXT,
                merchant_category TEXT,
                recurring_group_id TEXT
            );
        """)
        # Migrate Phase A table: old table had SERIAL id, no transaction_id.
        # If the old 'id' column (SERIAL) exists but 'transaction_id' doesn't,
        # we need to drop and recreate. Since this is dev data, that's safe.
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'raw_transactions' AND column_name = 'transaction_id';
        """)
        has_txn_id = cur.fetchone() is not None
        if not has_txn_id:
            # Phase A table shape — drop and recreate
            logger.info("Migrating raw_transactions from Phase A shape (SERIAL id → transaction_id PK)")
            cur.execute("DROP TABLE IF EXISTS raw_transactions;")
            cur.execute("""
                CREATE TABLE raw_transactions (
                    transaction_id TEXT PRIMARY KEY,
                    customer_id TEXT,
                    date DATE,
                    amount NUMERIC,
                    category TEXT,
                    type TEXT,
                    merchant_category TEXT,
                    recurring_group_id TEXT
                );
            """)

        # Add columns that may be missing
        for col, coltype in [
            ("merchant_category", "TEXT"),
            ("recurring_group_id", "TEXT"),
        ]:
            cur.execute(f"""
                DO $$ BEGIN
                    ALTER TABLE raw_transactions ADD COLUMN {col} {coltype};
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$;
            """)

        # --- raw_labels (new in Phase B) ---
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_labels (
                customer_id TEXT,
                product TEXT,
                converted BOOLEAN,
                PRIMARY KEY (customer_id, product)
            );
        """)

        # --- ingestion_log ---
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_log (
                id SERIAL PRIMARY KEY,
                filename TEXT,
                row_count INTEGER,
                rows_accepted INTEGER,
                rows_rejected INTEGER,
                rejection_sample TEXT,
                ingested_at TIMESTAMP DEFAULT NOW()
            );
        """)

    conn.commit()
    logger.info("Tables created/migrated successfully.")


# ---------------------------------------------------------------------------
# Transaction ID hashing
# ---------------------------------------------------------------------------

def _compute_transaction_id(row: dict) -> str:
    """Compute a deterministic transaction ID from row content.

    NOTE: hash-based ID assumes no true duplicate transactions same-day/same-amount
    — revisit if this becomes a real risk with production data.
    """
    raw = (
        f"{row.get('customer_id', '')}|"
        f"{row.get('date', '')}|"
        f"{float(row.get('amount', 0)):.2f}|"
        f"{row.get('category', '')}|"
        f"{row.get('type', '')}|"
        f"{row.get('merchant_category', '')}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Ingest functions
# ---------------------------------------------------------------------------

def _log_ingestion(conn: psycopg2.extensions.connection,
                   filename: str, row_count: int,
                   accepted: int, rejected: int,
                   rejection_reasons: list[str]) -> None:
    """Write a row to the ingestion_log table."""
    sample = "; ".join(rejection_reasons[:5]) if rejection_reasons else ""
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO ingestion_log
               (filename, row_count, rows_accepted, rows_rejected, rejection_sample)
               VALUES (%s, %s, %s, %s, %s)""",
            (filename, row_count, accepted, rejected, sample[:500]),
        )
    conn.commit()


def ingest_customers(conn: psycopg2.extensions.connection,
                     filepath: Path) -> tuple[int, int]:
    """Validate and upsert customers from CSV.

    Returns (accepted, rejected) counts.
    """
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    accepted = 0
    rejected = 0
    rejection_reasons: list[str] = []

    for row in rows:
        try:
            customer = Customer(**row)
        except ValidationError as e:
            rejected += 1
            rejection_reasons.append(
                f"Row customer_id={row.get('customer_id', '?')}: {e.error_count()} validation errors"
            )
            logger.warning("Rejected customer row: %s", row.get("customer_id", "?"))
            continue

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO raw_customers
                   (customer_id, name, occupation, employer_tenure_months,
                    city, account_vintage_months, persona_type)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (customer_id) DO UPDATE SET
                       name = EXCLUDED.name,
                       occupation = EXCLUDED.occupation,
                       employer_tenure_months = EXCLUDED.employer_tenure_months,
                       city = EXCLUDED.city,
                       account_vintage_months = EXCLUDED.account_vintage_months,
                       persona_type = EXCLUDED.persona_type""",
                (customer.customer_id, customer.name, customer.occupation,
                 customer.employer_tenure_months, customer.city,
                 customer.account_vintage_months, customer.persona_type),
            )
        accepted += 1

    conn.commit()
    _log_ingestion(conn, str(filepath), len(rows), accepted, rejected, rejection_reasons)
    return accepted, rejected


def ingest_transactions(conn: psycopg2.extensions.connection,
                        filepath: Path) -> tuple[int, int]:
    """Validate and upsert transactions from CSV.

    Returns (accepted, rejected) counts.
    """
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    accepted = 0
    rejected = 0
    rejection_reasons: list[str] = []

    for row in rows:
        try:
            txn = Transaction(**row)
        except ValidationError as e:
            rejected += 1
            rejection_reasons.append(
                f"Row cid={row.get('customer_id', '?')} date={row.get('date', '?')}: "
                f"{e.error_count()} validation errors"
            )
            logger.warning("Rejected transaction row: cid=%s date=%s",
                           row.get("customer_id", "?"), row.get("date", "?"))
            continue

        txn_id = _compute_transaction_id(row)

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO raw_transactions
                   (transaction_id, customer_id, date, amount, category, type,
                    merchant_category, recurring_group_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (transaction_id) DO NOTHING""",
                (txn_id, txn.customer_id, txn.date.isoformat(), txn.amount,
                 txn.category, txn.type, txn.merchant_category,
                 txn.recurring_group_id),
            )
        accepted += 1

    conn.commit()
    _log_ingestion(conn, str(filepath), len(rows), accepted, rejected, rejection_reasons)
    return accepted, rejected


def ingest_labels(conn: psycopg2.extensions.connection,
                  filepath: Path) -> tuple[int, int]:
    """Validate and upsert labels from CSV.

    Returns (accepted, rejected) counts.
    """
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    accepted = 0
    rejected = 0
    rejection_reasons: list[str] = []

    for row in rows:
        # CSV writes booleans as strings — normalize before validation
        if "converted" in row and isinstance(row["converted"], str):
            row["converted"] = row["converted"].strip().lower() in ("true", "1", "yes")

        try:
            label = Label(**row)
        except ValidationError as e:
            rejected += 1
            rejection_reasons.append(
                f"Row cid={row.get('customer_id', '?')} product={row.get('product', '?')}: "
                f"{e.error_count()} validation errors"
            )
            logger.warning("Rejected label row: cid=%s product=%s",
                           row.get("customer_id", "?"), row.get("product", "?"))
            continue

        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO raw_labels (customer_id, product, converted)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (customer_id, product) DO UPDATE SET
                       converted = EXCLUDED.converted""",
                (label.customer_id, label.product, label.converted),
            )
        accepted += 1

    conn.commit()
    _log_ingestion(conn, str(filepath), len(rows), accepted, rejected, rejection_reasons)
    return accepted, rejected


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: validate and ingest CSVs into Postgres."""
    parser = argparse.ArgumentParser(
        description="Ingest synthetic data into Cashflow IQ Postgres",
    )
    parser.add_argument("--customers", type=str, required=True,
                        help="Path to customers.csv")
    parser.add_argument("--transactions", type=str, required=True,
                        help="Path to transactions.csv")
    parser.add_argument("--labels", type=str, default=None,
                        help="Path to labels.csv (optional)")
    args = parser.parse_args()

    conn = _get_connection()
    try:
        _ensure_tables(conn)

        ca, cr = ingest_customers(conn, Path(args.customers))
        logger.info("Customers: %d accepted, %d rejected", ca, cr)

        ta, tr = ingest_transactions(conn, Path(args.transactions))
        logger.info("Transactions: %d accepted, %d rejected", ta, tr)

        if args.labels:
            la, lr = ingest_labels(conn, Path(args.labels))
            logger.info("Labels: %d accepted, %d rejected", la, lr)

        logger.info("Ingestion complete.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
