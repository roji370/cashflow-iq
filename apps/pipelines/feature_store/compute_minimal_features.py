"""Compute minimal behavioral features from raw transactions.

DEPRECATED: This module is superseded by:
  - apps.pipelines.feature_store.behavioral_features (income, cash flow, debt features)
  - apps.pipelines.feature_store.anomaly_detectors (subscription cleansing, liquidity pooling, bill shift)
  - apps.pipelines.feature_store.run_nightly_features (orchestration + feature store)
The three features here (avg_monthly_income, avg_monthly_outflow, savings_ratio) are now a
subset of what behavioral_features.py produces.

Phase A walking skeleton — computes three hand-picked features for a single customer:
  - avg_monthly_income: average of monthly credit totals
  - avg_monthly_outflow: average of monthly debit totals
  - savings_ratio: (income - outflow) / income

Usage: python -m apps.pipelines.feature_store.compute_minimal_features <customer_id>
"""

import os
import sys
from collections import defaultdict

import psycopg2


def get_connection() -> psycopg2.extensions.connection:
    """Create a Postgres connection from DATABASE_URL env var."""
    database_url = os.environ["DATABASE_URL"]
    return psycopg2.connect(database_url)


def compute_features(customer_id: str) -> dict[str, float]:
    """Compute three behavioral features from raw_transactions for a given customer.

    Returns a dict with keys: avg_monthly_income, avg_monthly_outflow, savings_ratio.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT date, amount, type FROM raw_transactions
                   WHERE customer_id = %s""",
                (customer_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"No transactions found for customer_id={customer_id}")

    # Aggregate by month
    monthly_credits: dict[str, float] = defaultdict(float)
    monthly_debits: dict[str, float] = defaultdict(float)

    for txn_date, amount, txn_type in rows:
        month_key = txn_date.strftime("%Y-%m")
        if txn_type == "credit":
            monthly_credits[month_key] += float(amount)
        else:
            monthly_debits[month_key] += float(amount)

    # Use all months that have any transaction
    all_months = set(monthly_credits.keys()) | set(monthly_debits.keys())
    num_months = len(all_months)

    total_income = sum(monthly_credits.values())
    total_outflow = sum(monthly_debits.values())

    avg_monthly_income = total_income / num_months
    avg_monthly_outflow = total_outflow / num_months
    savings_ratio = (avg_monthly_income - avg_monthly_outflow) / avg_monthly_income

    return {
        "avg_monthly_income": round(avg_monthly_income, 2),
        "avg_monthly_outflow": round(avg_monthly_outflow, 2),
        "savings_ratio": round(savings_ratio, 4),
    }


def write_features_to_db(customer_id: str, features: dict[str, float]) -> None:
    """Write computed features to the features table in Postgres."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    customer_id TEXT,
                    feature_name TEXT,
                    feature_value NUMERIC,
                    PRIMARY KEY (customer_id, feature_name)
                );
            """)
            for name, value in features.items():
                cur.execute(
                    """INSERT INTO features (customer_id, feature_name, feature_value)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (customer_id, feature_name)
                       DO UPDATE SET feature_value = EXCLUDED.feature_value""",
                    (customer_id, name, value),
                )
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    """Entry point: compute and store features for a given customer_id."""
    if len(sys.argv) < 2:
        print("Usage: python -m apps.pipelines.feature_store.compute_minimal_features <customer_id>")
        sys.exit(1)

    customer_id = sys.argv[1]
    features = compute_features(customer_id)

    print(f"Features for {customer_id}:")
    for name, value in features.items():
        print(f"  {name}: {value}")

    write_features_to_db(customer_id, features)
    print("Written to features table.")


if __name__ == "__main__":
    main()
