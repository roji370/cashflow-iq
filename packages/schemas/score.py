"""Score result schemas for Cashflow IQ.

Phase C expanded schemas — adds quantile capacity estimates, eligibility
gating results, SHAP-based reason codes, and a unified ScoreResponse envelope.
"""

from typing import Optional

from pydantic import BaseModel


class CapacityScoreResult(BaseModel):
    """Repayment capacity estimate with uncertainty quantiles.

    The capacity model produces three quantile predictions (q10/q50/q90) to
    express uncertainty. estimated_income is the median (q50) estimate.
    """

    customer_id: str
    estimated_income: float
    estimated_income_q10: float = 0.0
    estimated_income_q90: float = 0.0
    confidence: float = 0.5
    dti: float = 0.0


class IntentScoreResult(BaseModel):
    """Loan intent/propensity score with calibrated probability.

    intent_score is a calibrated probability (0–100 scale) from the
    isotonic-calibrated LightGBM classifier. confidence reflects the
    model's certainty in the prediction.
    """

    customer_id: str
    product: str
    intent_score: float
    confidence: float = 0.5


class ReasonCode(BaseModel):
    """Single SHAP-based reason code for explainability.

    Each reason code maps a feature's SHAP contribution to a human-readable
    explanation of why the model scored this customer higher or lower.
    """

    feature_name: str
    direction: str  # "↑" or "↓"
    human_label: str
    shap_value: float


class EligibilityResult(BaseModel):
    """Result of the eligibility gating rules check.

    Eligibility gating is a separate, human-auditable rules module that runs
    independently of the ML models. It checks DTI, account vintage, and model
    confidence against configurable thresholds.
    """

    passed: bool
    rules_checked: list[str]
    rules_failed: list[str]


class ScoreResponse(BaseModel):
    """Unified scoring response envelope.

    Wraps capacity, intent, eligibility, and explainability into a single
    API response. This is the contract between the API and the dashboard.
    """

    customer_id: str
    product: str
    capacity: CapacityScoreResult
    intent: IntentScoreResult
    eligibility: EligibilityResult
    reason_codes: list[ReasonCode] = []
    model_version: Optional[str] = None
