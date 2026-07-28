"""Transaction schema for Cashflow IQ.

Phase A minimal schema — will be expanded in Phase B/C.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel


class Transaction(BaseModel):
    """Single financial transaction. Phase A minimal schema — will be expanded in Phase B/C."""

    customer_id: str
    date: date
    amount: float
    category: str
    type: Literal["credit", "debit"]
