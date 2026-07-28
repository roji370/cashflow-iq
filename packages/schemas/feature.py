"""Feature record schema for Cashflow IQ.

Phase A minimal schema — will be expanded in Phase B/C.
"""

from pydantic import BaseModel


class FeatureRecord(BaseModel):
    """Single computed feature for a customer. Phase A minimal schema — will be expanded in Phase B/C."""

    customer_id: str
    feature_name: str
    feature_value: float
