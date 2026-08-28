"""Cashflow IQ shared schemas — source of truth for all data contracts.

Re-exports all schema classes for convenient imports:
    from packages.schemas import Customer, Transaction, FeatureRecord, ...
"""

from packages.schemas.customer import Customer
from packages.schemas.feature import FeatureRecord
from packages.schemas.label import Label
from packages.schemas.score import (
    CapacityScoreResult,
    EligibilityResult,
    IntentScoreResult,
    ReasonCode,
    ScoreResponse,
)
from packages.schemas.transaction import Transaction

__all__ = [
    "Customer",
    "Transaction",
    "Label",
    "FeatureRecord",
    "CapacityScoreResult",
    "IntentScoreResult",
    "ReasonCode",
    "EligibilityResult",
    "ScoreResponse",
]
