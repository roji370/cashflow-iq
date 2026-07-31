"""Score result schemas for Cashflow IQ.

Phase A minimal schema — will be expanded in Phase B/C.
"""

from pydantic import BaseModel


class CapacityScoreResult(BaseModel):
    """Repayment capacity estimate. Phase A minimal schema — will be expanded in Phase B/C."""

    customer_id: str
    estimated_income: float
    confidence: float


class IntentScoreResult(BaseModel):
    """Loan intent/propensity score. Phase A minimal schema — will be expanded in Phase B/C."""

    customer_id: str
    product: str
    intent_score: float
