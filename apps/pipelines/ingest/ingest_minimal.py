"""Minimal CSV-to-Postgres ingestion for the walking skeleton.

Reads customers.csv and transactions.csv from the synth_data output directory
and inserts them into raw_customers and raw_transactions tables.

Phase A only — no validation, upsert, or error handling. Will be replaced in Phase B.
"""

import csv
import os
from pathlib import Path

import psycopg2


def get_connection() -> psycopg2.extensions.connection:
    """Create a Postgres connection from DATABASE_URL env var."""
    database_url = os.environ["DATABASE_URL"]
    return psycopg2.connect(database_url)


def create_tables(conn: psycopg2.extensions.connection) -> None:
    """Create raw_customers and raw_transactions tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_customers (
                customer_id TEXT PRIMARY KEY,
                name TEXT
            );
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS raw_transactions (
                id SERIAL PRIMARY KEY,
                customer_id TEXT,
                date DATE,
                amount NUMERIC,
                category TEXT,
                type TEXT
            );
        """)
    conn.commit()
    print("Tables created (if not existing).")


def ingest_customers(conn: psycopg2.extensions.connection, filepath: Path) -> int:
    """Insert all rows from customers.csv into raw_customers."""
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                "INSERT INTO raw_customers (customer_id, name) VALUES (%s, %s)",
                (row["customer_id"], row["name"]),
            )
    conn.commit()
    return len(rows)


def ingest_transactions(conn: psycopg2.extensions.connection, filepath: Path) -> int:
    """Insert all rows from transactions.csv into raw_transactions."""
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    with conn.cursor() as cur:
        for row in rows:
            cur.execute(
                """INSERT INTO raw_transactions (customer_id, date, amount, category, type)
                   VALUES (%s, %s, %s, %s, %s)""",
                (row["customer_id"], row["date"], row["amount"], row["category"], row["type"]),
            )
    conn.commit()
    return len(rows)


def main() -> None:
    """Entry point: ingest CSVs into Postgres."""
    data_dir = Path("apps/pipelines/synth_data/output")

    conn = get_connection()
    try:
        create_tables(conn)

        n_customers = ingest_customers(conn, data_dir / "customers.csv")
        n_transactions = ingest_transactions(conn, data_dir / "transactions.csv")

        print(f"Ingested {n_customers} customers, {n_transactions} transactions.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
