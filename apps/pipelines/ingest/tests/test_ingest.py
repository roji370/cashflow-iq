"""Unit tests for the idempotent ingestion module.

Tests cover:
  1. Valid rows load correctly
  2. Malformed rows are rejected and logged
  3. Re-running the same file doesn't duplicate rows
"""

import csv
import os
import tempfile
from pathlib import Path

import psycopg2
import pytest

from apps.pipelines.ingest.run import (
    _ensure_tables,
    _get_connection,
    ingest_customers,
    ingest_labels,
    ingest_transactions,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def db_conn():
    """Shared database connection for the test module.

    Requires DATABASE_URL to be set (runs inside the pipelines container).
    """
    conn = _get_connection()
    _ensure_tables(conn)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def _clean_tables(db_conn):
    """Truncate all raw tables before each test for isolation."""
    with db_conn.cursor() as cur:
        cur.execute("TRUNCATE raw_customers, raw_transactions, raw_labels, ingestion_log CASCADE;")
    db_conn.commit()


def _write_csv_fixture(filepath: Path, fieldnames: list[str],
                       rows: list[dict]) -> Path:
    """Write a fixture CSV to a temp file and return the path."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return filepath


# ---------------------------------------------------------------------------
# Customer tests
# ---------------------------------------------------------------------------

class TestIngestCustomers:
    """Tests for customer ingestion."""

    def test_valid_rows_load(self, db_conn: psycopg2.extensions.connection,
                             tmp_path: Path) -> None:
        """Valid customer rows should be inserted into raw_customers."""
        filepath = _write_csv_fixture(
            tmp_path / "customers.csv",
            ["customer_id", "name", "occupation", "employer_tenure_months",
             "city", "account_vintage_months", "persona_type"],
            [
                {"customer_id": "c1", "name": "Alice", "occupation": "Eng",
                 "employer_tenure_months": "24", "city": "Mumbai",
                 "account_vintage_months": "36", "persona_type": "stable_salaried"},
                {"customer_id": "c2", "name": "Bob", "occupation": "",
                 "employer_tenure_months": "", "city": "",
                 "account_vintage_months": "", "persona_type": ""},
            ],
        )
        accepted, rejected = ingest_customers(db_conn, filepath)
        assert accepted == 2
        assert rejected == 0

        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM raw_customers;")
            assert cur.fetchone()[0] == 2

    def test_malformed_rows_rejected(self, db_conn: psycopg2.extensions.connection,
                                     tmp_path: Path) -> None:
        """Rows with invalid types should be rejected, not crash the batch."""
        filepath = _write_csv_fixture(
            tmp_path / "customers.csv",
            ["customer_id", "name", "employer_tenure_months"],
            [
                {"customer_id": "c1", "name": "Good", "employer_tenure_months": "12"},
                # employer_tenure_months is not a valid int
                {"customer_id": "c2", "name": "Bad", "employer_tenure_months": "not_a_number"},
            ],
        )
        accepted, rejected = ingest_customers(db_conn, filepath)
        assert accepted == 1
        assert rejected == 1

    def test_idempotent_rerun(self, db_conn: psycopg2.extensions.connection,
                              tmp_path: Path) -> None:
        """Running the same file twice should not duplicate rows."""
        filepath = _write_csv_fixture(
            tmp_path / "customers.csv",
            ["customer_id", "name"],
            [{"customer_id": "c1", "name": "Alice"}],
        )
        ingest_customers(db_conn, filepath)
        ingest_customers(db_conn, filepath)

        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM raw_customers;")
            assert cur.fetchone()[0] == 1  # NOT 2


# ---------------------------------------------------------------------------
# Transaction tests
# ---------------------------------------------------------------------------

class TestIngestTransactions:
    """Tests for transaction ingestion."""

    def test_valid_rows_load(self, db_conn: psycopg2.extensions.connection,
                             tmp_path: Path) -> None:
        """Valid transaction rows should be inserted."""
        filepath = _write_csv_fixture(
            tmp_path / "txns.csv",
            ["customer_id", "date", "amount", "category", "type",
             "merchant_category", "recurring_group_id"],
            [
                {"customer_id": "c1", "date": "2024-01-15", "amount": "5000",
                 "category": "salary", "type": "credit",
                 "merchant_category": "salary", "recurring_group_id": ""},
            ],
        )
        accepted, rejected = ingest_transactions(db_conn, filepath)
        assert accepted == 1
        assert rejected == 0

    def test_malformed_date_rejected(self, db_conn: psycopg2.extensions.connection,
                                     tmp_path: Path) -> None:
        """A row with an invalid date should be rejected."""
        filepath = _write_csv_fixture(
            tmp_path / "txns.csv",
            ["customer_id", "date", "amount", "category", "type",
             "merchant_category", "recurring_group_id"],
            [
                {"customer_id": "c1", "date": "not-a-date", "amount": "5000",
                 "category": "salary", "type": "credit",
                 "merchant_category": "salary", "recurring_group_id": ""},
            ],
        )
        accepted, rejected = ingest_transactions(db_conn, filepath)
        assert accepted == 0
        assert rejected == 1

    def test_idempotent_rerun(self, db_conn: psycopg2.extensions.connection,
                              tmp_path: Path) -> None:
        """Running the same transaction file twice must not duplicate rows."""
        filepath = _write_csv_fixture(
            tmp_path / "txns.csv",
            ["customer_id", "date", "amount", "category", "type",
             "merchant_category", "recurring_group_id"],
            [
                {"customer_id": "c1", "date": "2024-01-15", "amount": "5000.00",
                 "category": "salary", "type": "credit",
                 "merchant_category": "salary", "recurring_group_id": ""},
                {"customer_id": "c1", "date": "2024-01-20", "amount": "2000.00",
                 "category": "grocery", "type": "debit",
                 "merchant_category": "grocery", "recurring_group_id": ""},
            ],
        )
        ingest_transactions(db_conn, filepath)
        ingest_transactions(db_conn, filepath)

        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM raw_transactions;")
            count = cur.fetchone()[0]
            assert count == 2, f"Expected 2 rows after double-run, got {count}"


# ---------------------------------------------------------------------------
# Label tests
# ---------------------------------------------------------------------------

class TestIngestLabels:
    """Tests for label ingestion."""

    def test_valid_rows_load(self, db_conn: psycopg2.extensions.connection,
                             tmp_path: Path) -> None:
        """Valid label rows should be inserted."""
        filepath = _write_csv_fixture(
            tmp_path / "labels.csv",
            ["customer_id", "product", "converted"],
            [
                {"customer_id": "c1", "product": "home_loan", "converted": "True"},
                {"customer_id": "c1", "product": "auto_loan", "converted": "False"},
            ],
        )
        accepted, rejected = ingest_labels(db_conn, filepath)
        assert accepted == 2
        assert rejected == 0

    def test_invalid_product_rejected(self, db_conn: psycopg2.extensions.connection,
                                      tmp_path: Path) -> None:
        """A label with an invalid product literal should be rejected."""
        filepath = _write_csv_fixture(
            tmp_path / "labels.csv",
            ["customer_id", "product", "converted"],
            [
                {"customer_id": "c1", "product": "spaceship_loan", "converted": "True"},
            ],
        )
        accepted, rejected = ingest_labels(db_conn, filepath)
        assert accepted == 0
        assert rejected == 1

    def test_idempotent_rerun(self, db_conn: psycopg2.extensions.connection,
                              tmp_path: Path) -> None:
        """Running labels twice should not duplicate rows."""
        filepath = _write_csv_fixture(
            tmp_path / "labels.csv",
            ["customer_id", "product", "converted"],
            [{"customer_id": "c1", "product": "home_loan", "converted": "True"}],
        )
        ingest_labels(db_conn, filepath)
        ingest_labels(db_conn, filepath)

        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM raw_labels;")
            assert cur.fetchone()[0] == 1
