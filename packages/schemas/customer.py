"""Customer schema for Cashflow IQ.

Expanded in Phase B — added occupation, employer tenure, city, account vintage,
and persona_type fields. All new fields are Optional so existing Phase A code
paths remain unbroken.
"""

from typing import Optional

from pydantic import BaseModel


class Customer(BaseModel):
    """Customer record for Cashflow IQ.

    Core fields (customer_id, name) are required. All Phase B additions are
    Optional so that existing code that only supplies the minimal fields keeps
    working without changes.
    """

    customer_id: str
    name: str
    occupation: Optional[str] = None
    employer_tenure_months: Optional[int] = None
    city: Optional[str] = None
    account_vintage_months: Optional[int] = None
    # persona_type is synthetic-data-only — used to tag which persona template
    # generated this customer. A real pipeline would NOT populate this field.
    persona_type: Optional[str] = None
