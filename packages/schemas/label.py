"""Label schema for Cashflow IQ.

Conversion labels per customer-product pair, used as the training target
for intent models. Ingested from synthetic label CSVs via the ingest pipeline.
"""

from pydantic import BaseModel


class Label(BaseModel):
    """Conversion label for a customer-product pair.

    Attributes:
        customer_id: The customer identifier.
        product: Loan product type (e.g., 'home_loan', 'auto_loan').
        converted: Whether the customer converted (accepted a loan offer).
    """

    customer_id: str
    product: str
    converted: bool
