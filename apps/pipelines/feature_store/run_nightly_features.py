"""Real feature store for Cashflow IQ — nightly batch job.

Replaces the ad-hoc 3-row features table from Phase A with a proper long-format
design. Orchestrates all feature functions (behavioral + anomaly detectors)
through a FEATURE_FUNCTIONS registry.

Key components:
  - migrate_features_table(): DROP and recreate the features table in long-format
  - FEATURE_FUNCTIONS: registry dict of feature function name → callable
  - run_nightly(): batch job computing all features for all customers
  - get_feature_vector(): read-path contract pivoting long → wide dict

Usage:
    python -m apps.pipelines.feature_store.run_nightly_features
"""

import logging
import os
from datetime import date
from typing import Any, Callable, Optional

import pandas as pd
import psycopg2

from apps.pipelines.feature_store.anomaly_detectors import (
    detect_bill_shift,
    detect_liquidity_pooling,
    detect_subscription_cleansing,
)
from apps.pipelines.feature_store.behavioral_features import (
    compute_cashflow_features,
    compute_debt_features,
    compute_income_features,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

FEATURE_SCHEMA_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _get_connection() -> psycopg2.extensions.connection:
    """Create a Postgres connection from DATABASE_URL env var."""
    database_url = os.environ["DATABASE_URL"]
    return psycopg2.connect(database_url)


def migrate_features_table(conn: psycopg2.extensions.connection) -> None:
    """DROP and recreate the features table in long-format.

    The Phase A features table had shape (customer_id, feature_name, feature_value)
    with PK (customer_id, feature_name). The new long-format adds
    feature_schema_version, computed_at, data_completeness_score and changes the
    PK to (customer_id, feature_name, computed_at).

    Since this is synthetic dev data, we DROP and recreate rather than trying to
    migrate old rows that don't fit the new schema.
    """
    with conn.cursor() as cur:
        # Check if old-shape table exists
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'features'
            ORDER BY ordinal_position;
        """)
        existing_cols = [row[0] for row in cur.fetchall()]

        if existing_cols and "computed_at" not in existing_cols:
            logger.info(
                "Dropping Phase A features table (columns: %s) — "
                "recreating with long-format schema.",
                existing_cols,
            )
            cur.execute("DROP TABLE IF EXISTS features;")
        elif existing_cols:
            logger.info("Features table already in long-format — no migration needed.")
            conn.commit()
            return

        cur.execute("""
            CREATE TABLE IF NOT EXISTS features (
                customer_id TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                feature_value NUMERIC,
                feature_schema_version TEXT DEFAULT '1.0.0',
                computed_at DATE NOT NULL,
                data_completeness_score NUMERIC,
                PRIMARY KEY (customer_id, feature_name, computed_at)
            );
        """)
    conn.commit()
    logger.info("Features table created with long-format schema.")


# ---------------------------------------------------------------------------
# FEATURE_FUNCTIONS registry
# ---------------------------------------------------------------------------
# Each entry: name → callable(txns_df, customer_id) → dict[str, Any]
# The callable must return a dict where keys are feature names and values are
# either float or None (None = skip, don't write a row).
# A special key "_completeness" (float 0–1) sets data_completeness_score for
# all features in that group.

def _wrap_anomaly_detector(
    detector_fn: Callable[[pd.DataFrame], dict[str, Any]],
) -> Callable[[pd.DataFrame, str], dict[str, Any]]:
    """Wrap an anomaly detector (takes df only) to match the registry signature."""
    def wrapped(txns_df: pd.DataFrame, customer_id: str) -> dict[str, Any]:
        """Wrapped anomaly detector with customer_id parameter."""
        result = detector_fn(txns_df)
        # Convert bool flags to float (1.0 / 0.0) for storage
        return {
            k: (float(v) if isinstance(v, bool) else v)
            for k, v in result.items()
        }
    return wrapped


FEATURE_FUNCTIONS: dict[str, Callable[[pd.DataFrame, str], dict[str, Any]]] = {
    # Behavioral features (B5)
    "income": compute_income_features,
    "cashflow": compute_cashflow_features,
    "debt": compute_debt_features,
    # Anomaly detectors (B4) — wrapped to accept customer_id
    "subscription_cleansing": _wrap_anomaly_detector(detect_subscription_cleansing),
    "liquidity_pooling": _wrap_anomaly_detector(detect_liquidity_pooling),
    "bill_shift": _wrap_anomaly_detector(detect_bill_shift),
}


# ---------------------------------------------------------------------------
# Feature computation + writing
# ---------------------------------------------------------------------------

def _write_features(
    conn: psycopg2.extensions.connection,
    customer_id: str,
    features: dict[str, Any],
    computed_at: date,
    completeness: float = 1.0,
) -> int:
    """Write feature dict to the features table. Idempotent per (customer_id, feature_name, computed_at).

    Returns number of features written.
    """
    written = 0
    with conn.cursor() as cur:
        for name, value in features.items():
            if name.startswith("_"):
                continue  # skip metadata keys like _completeness
            if value is None:
                continue  # skip features that couldn't be computed

            cur.execute(
                """INSERT INTO features
                   (customer_id, feature_name, feature_value,
                    feature_schema_version, computed_at, data_completeness_score)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (customer_id, feature_name, computed_at)
                   DO UPDATE SET
                       feature_value = EXCLUDED.feature_value,
                       feature_schema_version = EXCLUDED.feature_schema_version,
                       data_completeness_score = EXCLUDED.data_completeness_score""",
                (customer_id, name, value, FEATURE_SCHEMA_VERSION,
                 computed_at.isoformat(), completeness),
            )
            written += 1
    return written


def _validate_features(features: dict[str, Any]) -> list[str]:
    """Lightweight validation — flag unexpected nulls or out-of-range values.

    Returns list of warning messages (empty = all good).
    """
    warnings: list[str] = []
    ratio_features = {
        "savings_ratio_6m", "emi_punctuality_score", "salary_variance_12m",
        "expense_volatility", "data_completeness_score",
        "subscription_cleansing_score", "liquidity_pooling_score", "bill_shift_score",
    }
    for name, value in features.items():
        if name.startswith("_"):
            continue
        if value is None:
            continue  # None is valid for sparse data
        if name in ratio_features and not (0.0 <= float(value) <= 1.5):
            warnings.append(f"  {name}={value} outside expected range [0, 1.5]")
    return warnings


# ---------------------------------------------------------------------------
# Batch job
# ---------------------------------------------------------------------------

def run_nightly(computed_at: Optional[date] = None) -> None:
    """Run the nightly feature computation batch job.

    For each customer with transaction data, compute all registered features
    and write to the features table. Idempotent — safe to re-run same day.
    """
    if computed_at is None:
        computed_at = date.today()

    conn = _get_connection()
    try:
        migrate_features_table(conn)

        # Get all customer IDs with transaction data
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT customer_id FROM raw_transactions;")
            customer_ids = [row[0] for row in cur.fetchall()]

        logger.info("Computing features for %d customers...", len(customer_ids))
        total_features = 0
        total_warnings = 0

        for cid in customer_ids:
            # Load transactions for this customer
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT customer_id, date, amount, category, type,
                              merchant_category, recurring_group_id
                       FROM raw_transactions WHERE customer_id = %s""",
                    (cid,),
                )
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()

            if not rows:
                continue

            txns_df = pd.DataFrame(rows, columns=columns)
            txns_df["amount"] = txns_df["amount"].astype(float)

            # Run all registered feature functions
            for group_name, fn in FEATURE_FUNCTIONS.items():
                try:
                    features = fn(txns_df, cid)
                except Exception as e:
                    logger.error("Error computing %s for %s: %s", group_name, cid, e)
                    continue

                completeness = features.pop("_completeness", 1.0)

                # Validate
                warnings = _validate_features(features)
                if warnings:
                    total_warnings += len(warnings)
                    for w in warnings:
                        logger.warning("Validation warning [%s/%s]: %s",
                                       cid, group_name, w)

                # Write
                n = _write_features(conn, cid, features, computed_at, completeness)
                total_features += n

            conn.commit()

        logger.info(
            "Nightly features complete: %d features for %d customers "
            "(%d validation warnings).",
            total_features, len(customer_ids), total_warnings,
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read path: get_feature_vector
# ---------------------------------------------------------------------------

def get_feature_vector(
    customer_id: str,
    as_of_date: Optional[date] = None,
) -> dict[str, float]:
    """Pivot the long-format features table into a wide dict for downstream use.

    This is the read-path contract that trivial_scoring.py and future models
    depend on.

    Args:
        customer_id: The customer to look up.
        as_of_date: If given, use features computed on or before this date.
                    If None, use the most recent features.

    Returns:
        Dict mapping feature_name → feature_value.

    Raises:
        ValueError: If no features found for the customer.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            if as_of_date:
                cur.execute(
                    """SELECT DISTINCT ON (feature_name)
                              feature_name, feature_value
                       FROM features
                       WHERE customer_id = %s AND computed_at <= %s
                       ORDER BY feature_name, computed_at DESC""",
                    (customer_id, as_of_date.isoformat()),
                )
            else:
                cur.execute(
                    """SELECT DISTINCT ON (feature_name)
                              feature_name, feature_value
                       FROM features
                       WHERE customer_id = %s
                       ORDER BY feature_name, computed_at DESC""",
                    (customer_id,),
                )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"No features found for customer_id={customer_id}")

    return {name: float(value) for name, value in rows}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: run nightly feature computation."""
    run_nightly()


if __name__ == "__main__":
    main()
