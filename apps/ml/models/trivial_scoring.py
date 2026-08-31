"""Trivial capacity and intent scoring for the walking skeleton.

DEPRECATED — Phase C (2026-08-18).
Real scoring is now handled by:
  - apps.ml.models.capacity_model (LightGBM quantile regressor)
  - apps.ml.models.intent_model (LightGBM binary classifier + isotonic calibration)
This file is kept for reference only. It is no longer called by the API.

NOT a real model — just arithmetic on features and a hardcoded placeholder.
Phase B patch: updated to read from the long-format feature store via
get_feature_vector() instead of querying the old 3-column table directly.
"""

from apps.pipelines.feature_store.run_nightly_features import get_feature_vector
from packages.schemas.score import CapacityScoreResult, IntentScoreResult


def get_capacity_score(customer_id: str) -> CapacityScoreResult:
    """Estimate repayment capacity from features.

    Trivial implementation: estimated_income = avg_monthly_income - avg_monthly_outflow.
    Confidence is hardcoded to 0.5.

    Reads features via get_feature_vector() from the long-format feature store.
    """
    features = get_feature_vector(customer_id)
    estimated_income = features["avg_monthly_income"] - features["avg_monthly_outflow"]

    return CapacityScoreResult(
        customer_id=customer_id,
        estimated_income=round(estimated_income, 2),
        confidence=0.5,
    )


def get_intent_score(customer_id: str, product: str) -> IntentScoreResult:
    """Return a hardcoded intent score placeholder.

    Trivial implementation: always returns 50.0 regardless of input.
    Real model comes in Phase C.
    """
    # Verify customer exists in features (raises ValueError if not)
    get_feature_vector(customer_id)

    return IntentScoreResult(
        customer_id=customer_id,
        product=product,
        intent_score=50.0,
    )
