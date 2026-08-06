"""Transaction schema for Cashflow IQ.

Expanded in Phase B — added merchant_category and recurring_group_id fields.
All new fields are Optional so existing Phase A code paths remain unbroken.
"""

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel


class Transaction(BaseModel):
    """Single financial transaction.

    Core fields (customer_id, date, amount, category, type) are required.
    Phase B additions (merchant_category, recurring_group_id) are Optional.
    """

    customer_id: str
    date: date
    amount: float
    category: str
    type: Literal["credit", "debit"]
    # merchant_category: MCC-like tag for the merchant. Expected values:
    # grocery, dining, travel, entertainment, subscriptions, fuel, jewellery,
    # education, medical, real_estate, auto_dealer, utility, rent, salary,
    # investment
    merchant_category: Optional[str] = None
    # recurring_group_id: nullable identifier linking transactions that belong
    # to the same recurring payment series (e.g. same subscription, same EMI).
    # Populated by recurring-payment detection logic.
    recurring_group_id: Optional[str] = None
