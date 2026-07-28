"""Trivial capacity and intent scoring for the walking skeleton.

NOT a real model — just arithmetic on features and a hardcoded placeholder.
Real LightGBM models come in Phase B.

Phase A only — will be replaced with actual trained models.
"""

import os

import psycopg2

from packages.schemas.score import CapacityScoreResult, IntentScoreResult


def _get_connection() -> psycopg2.extensions.connection:
    """Create a Postgres connection from DATABASE_URL env var."""
    database_url = os.environ["DATABASE_URL"]
    return psycopg2.connect(database_url)


def _read_features(customer_id: str) -> dict[str, float]:
    """Read all features for a customer from the features table.

    Returns a dict mapping feature_name → feature_value.
    Raises ValueError if no features found.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT feature_name, feature_value FROM features WHERE customer_id = %s",
                (customer_id,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    if not rows:
        raise ValueError(f"No features found for customer_id={customer_id}")

    return {name: float(value) for name, value in rows}


def get_capacity_score(customer_id: str) -> CapacityScoreResult:
    """Estimate repayment capacity from features.

    Trivial implementation: estimated_income = avg_monthly_income - avg_monthly_outflow.
    Confidence is hardcoded to 0.5.
    """
    features = _read_features(customer_id)
    estimated_income = features["avg_monthly_income"] - features["avg_monthly_outflow"]

    return CapacityScoreResult(
        customer_id=customer_id,
        estimated_income=round(estimated_income, 2),
        confidence=0.5,
    )


def get_intent_score(customer_id: str, product: str) -> IntentScoreResult:
    """Return a hardcoded intent score placeholder.

    Trivial implementation: always returns 50.0 regardless of input.
    Real model comes in Phase B.
    """
    # Verify customer exists in features (raises ValueError if not)
    _read_features(customer_id)

    return IntentScoreResult(
        customer_id=customer_id,
        product=product,
        intent_score=50.0,
    )
