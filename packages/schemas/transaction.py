"""Transaction schema for Cashflow IQ.

Phase B expanded schema — adds merchant_category and recurring_group_id
used by behavioral feature engineering and anomaly detection.
"""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel


class Transaction(BaseModel):
    """Single financial transaction.

    Phase B adds merchant_category (for income source detection) and
    recurring_group_id (for EMI/recurring payment identification).
    Both are Optional so Phase A CSV data still validates.
    """

    customer_id: str
    date: date
    amount: float
    category: str
    type: Literal["credit", "debit"]
    merchant_category: Optional[str] = None
    recurring_group_id: Optional[str] = None

