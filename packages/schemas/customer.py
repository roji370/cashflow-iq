"""Customer schema for Cashflow IQ.

Phase B expanded schema — adds demographic and tenure fields used by
the multi-persona synthetic data generator and behavioral feature pipeline.
"""

from typing import Optional

from pydantic import BaseModel


class Customer(BaseModel):
    """Customer record with optional demographic fields.

    All Phase B fields are Optional so Phase A data (customer_id + name only)
    still validates without breaking existing ingestion paths.
    """

    customer_id: str
    name: str
    occupation: Optional[str] = None
    employer_tenure_months: Optional[int] = None
    city: Optional[str] = None
    account_vintage_months: Optional[int] = None
    persona_type: Optional[str] = None

