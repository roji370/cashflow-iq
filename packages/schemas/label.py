"""Label schema for Cashflow IQ.

Ground truth for model training — tracks whether a customer converted on a
specific loan product. Added in Phase B.
"""

from typing import Literal

from pydantic import BaseModel


class Label(BaseModel):
    """Per-customer, per-product conversion ground truth.

    Used for supervised model training in Phase C. Generated synthetically
    in Phase B with noise to target AUC ~0.75–0.85.
    """

    customer_id: str
    product: Literal["personal_loan", "home_loan", "mortgage_loan", "auto_loan"]
    converted: bool
