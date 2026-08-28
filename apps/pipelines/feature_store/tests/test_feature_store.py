"""Integration test for the feature store.

Tests the full pipeline: ingest → compute features → verify get_feature_vector()
returns expected features for a known customer.

Requires DATABASE_URL to be set (runs inside the pipelines container).
"""

import csv
from datetime import date
from pathlib import Path

import psycopg2
import pytest

from apps.pipelines.feature_store.run_nightly_features import (
    _get_connection,
    get_feature_vector,
    migrate_features_table,
    run_nightly,
)
from apps.pipelines.ingest.run import (
    _ensure_tables,
    ingest_customers,
    ingest_transactions,
)


@pytest.fixture(scope="module")
def db_conn():
    """Shared database connection for the test module."""
    conn = _get_connection()
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def _clean_tables(db_conn):
    """Clean all tables before each test."""
    with db_conn.cursor() as cur:
        for table in ["features", "raw_transactions", "raw_customers",
                       "raw_labels", "ingestion_log"]:
            cur.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
    db_conn.commit()
    _ensure_tables(db_conn)
    migrate_features_table(db_conn)


def _write_fixture_csv(filepath: Path, fieldnames: list[str],
                       rows: list[dict]) -> Path:
    """Write test fixture CSV."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return filepath


class TestFeatureStoreIntegration:
    """Integration test: ingest → compute → verify feature vector."""

    def test_ingest_compute_read(self, db_conn: psycopg2.extensions.connection,
                                  tmp_path: Path) -> None:
        """Full pipeline: ingest customer + transactions, compute features,
        verify get_feature_vector returns expected keys.
        """
        # 1. Write fixture CSVs
        cust_path = _write_fixture_csv(
            tmp_path / "customers.csv",
            ["customer_id", "name"],
            [{"customer_id": "test_001", "name": "Test User"}],
        )
        txn_path = _write_fixture_csv(
            tmp_path / "transactions.csv",
            ["customer_id", "date", "amount", "category", "type",
             "merchant_category", "recurring_group_id"],
            [
                # 4 months of salary + rent to get enough data for features
                {"customer_id": "test_001", "date": "2024-01-01", "amount": "70000",
                 "category": "salary", "type": "credit",
                 "merchant_category": "salary", "recurring_group_id": "sal_1"},
                {"customer_id": "test_001", "date": "2024-02-01", "amount": "71000",
                 "category": "salary", "type": "credit",
                 "merchant_category": "salary", "recurring_group_id": "sal_1"},
                {"customer_id": "test_001", "date": "2024-03-01", "amount": "69000",
                 "category": "salary", "type": "credit",
                 "merchant_category": "salary", "recurring_group_id": "sal_1"},
                {"customer_id": "test_001", "date": "2024-04-01", "amount": "72000",
                 "category": "salary", "type": "credit",
                 "merchant_category": "salary", "recurring_group_id": "sal_1"},
                {"customer_id": "test_001", "date": "2024-01-05", "amount": "20000",
                 "category": "rent", "type": "debit",
                 "merchant_category": "rent", "recurring_group_id": "rent_1"},
                {"customer_id": "test_001", "date": "2024-02-05", "amount": "20000",
                 "category": "rent", "type": "debit",
                 "merchant_category": "rent", "recurring_group_id": "rent_1"},
                {"customer_id": "test_001", "date": "2024-03-05", "amount": "20000",
                 "category": "rent", "type": "debit",
                 "merchant_category": "rent", "recurring_group_id": "rent_1"},
                {"customer_id": "test_001", "date": "2024-04-05", "amount": "20000",
                 "category": "rent", "type": "debit",
                 "merchant_category": "rent", "recurring_group_id": "rent_1"},
            ],
        )

        # 2. Ingest
        ingest_customers(db_conn, cust_path)
        ingest_transactions(db_conn, txn_path)

        # 3. Compute features
        run_nightly(computed_at=date(2024, 4, 15))

        # 4. Verify feature vector
        fv = get_feature_vector("test_001")

        # Should have key cash flow features
        assert "avg_monthly_income" in fv
        assert "avg_monthly_outflow" in fv
        assert fv["avg_monthly_income"] > 60000  # ~70k salary
        assert fv["avg_monthly_outflow"] > 15000  # ~20k rent

        # Should have income features
        assert "salary_amount_median" in fv
        assert fv["salary_amount_median"] > 60000

    def test_idempotent_rerun(self, db_conn: psycopg2.extensions.connection,
                              tmp_path: Path) -> None:
        """Running nightly features twice on same day should not duplicate rows."""
        cust_path = _write_fixture_csv(
            tmp_path / "customers.csv",
            ["customer_id", "name"],
            [{"customer_id": "test_002", "name": "Idem User"}],
        )
        txn_path = _write_fixture_csv(
            tmp_path / "transactions.csv",
            ["customer_id", "date", "amount", "category", "type",
             "merchant_category", "recurring_group_id"],
            [
                {"customer_id": "test_002", "date": "2024-01-01", "amount": "50000",
                 "category": "salary", "type": "credit",
                 "merchant_category": "salary", "recurring_group_id": ""},
                {"customer_id": "test_002", "date": "2024-01-10", "amount": "10000",
                 "category": "rent", "type": "debit",
                 "merchant_category": "rent", "recurring_group_id": ""},
            ],
        )

        ingest_customers(db_conn, cust_path)
        ingest_transactions(db_conn, txn_path)

        today = date(2024, 4, 15)
        run_nightly(computed_at=today)

        # Count features after first run
        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM features WHERE customer_id = 'test_002';")
            count_1 = cur.fetchone()[0]

        # Run again
        run_nightly(computed_at=today)

        with db_conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM features WHERE customer_id = 'test_002';")
            count_2 = cur.fetchone()[0]

        assert count_1 == count_2, (
            f"Feature count changed after re-run: {count_1} → {count_2}"
        )
