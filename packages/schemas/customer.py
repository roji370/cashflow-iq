"""Customer schema for Cashflow IQ.

Phase A minimal schema — will be expanded in Phase B/C.
"""

from pydantic import BaseModel


class Customer(BaseModel):
    """Minimal customer record. Phase A minimal schema — will be expanded in Phase B/C."""

    customer_id: str
    name: str
