"""Label schema for Cashflow IQ.

Ground truth for model training — tracks whether a customer converted on a
specific loan product. Added in Phase B.
"""

from pydantic import BaseModel


class Label(BaseModel):
    """Per-customer, per-product conversion ground truth.

    Used for supervised model training in Phase C. Generated synthetically
    in Phase B with noise to target AUC ~0.75–0.85.
    """

    customer_id: str
    product: str
    converted: bool

