"""Feature record schema for Cashflow IQ.

Expanded in Phase B to the long-format design: each row is one feature for one
customer on one date, with versioning and data-completeness metadata.
"""

from datetime import date

from pydantic import BaseModel, Field


class FeatureRecord(BaseModel):
    """Single computed feature for a customer in long-format.

    The feature store writes one row per (customer_id, feature_name, computed_at)
    tuple. Downstream consumers pivot this into a wide dict via
    get_feature_vector().
    """

    customer_id: str
    feature_name: str
    feature_value: float
    feature_schema_version: str = "1.0.0"
    computed_at: date
    data_completeness_score: float = Field(
        ge=0.0, le=1.0,
        description=(
            "Fraction of the ideal data window that was available when this "
            "feature was computed. 1.0 = full 12-month history; lower values "
            "indicate sparse data and reduced confidence."
        ),
    )
