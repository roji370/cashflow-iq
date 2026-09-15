"""SHAP TreeExplainer wrapper for Cashflow IQ models.

Provides human-readable reason codes by mapping SHAP feature contributions
to explanatory labels. Works with both capacity and intent LightGBM models.

Per AGENTS.md: every model that outputs a score must have a corresponding
SHAP explainability path. No predict() without a paired explain().
"""

import logging
from typing import Any, Optional

import numpy as np
import shap

from packages.schemas.score import ReasonCode

logger = logging.getLogger(__name__)

# Human-readable labels for feature names.
# Keys are feature_name from the feature store, values are plain-language labels.
FEATURE_LABELS: dict[str, str] = {
    # Income features
    "salary_variance_12m": "Income stability",
    "income_growth_pct": "Income growth trend",
    "num_income_sources": "Number of income sources",
    "income_source_tenure_months": "Income source tenure",
    "salary_amount_median": "Median salary amount",
    # Cash flow features
    "avg_monthly_income": "Average monthly income",
    "avg_monthly_outflow": "Average monthly spending",
    "savings_ratio_6m": "Savings ratio",
    "liquidity_buffer": "Liquidity buffer",
    "expense_volatility": "Spending consistency",
    # Debt features
    "emi_punctuality_score": "EMI payment punctuality",
    "current_dti": "Debt-to-income ratio",
    "debt_trend_direction": "Debt trend",
    "num_active_emis": "Active EMI count",
    # Anomaly features
    "subscription_cleansing_score": "Subscription cleanup activity",
    "subscription_cleansing_flag": "Subscription cleanup detected",
    "liquidity_pooling_score": "Liquidity pooling activity",
    "liquidity_pooling_flag": "Liquidity pooling detected",
    "bill_shift_score": "Bill payment shifting",
    "bill_shift_flag": "Bill shifting detected",
    # Data quality
    "data_completeness_score": "Data completeness",
}


def explain(
    model: Any,
    features: dict[str, float],
    feature_names: list[str],
    top_n: int = 5,
    model_type: str = "capacity",
) -> list[ReasonCode]:
    """Generate SHAP-based reason codes for a prediction.

    Args:
        model: A trained LightGBM model (LGBMRegressor or LGBMClassifier).
            For intent models, pass the BASE model (not calibrated wrapper).
        features: Feature vector for a single customer.
        feature_names: Ordered list of feature names matching training order.
        top_n: Number of top reason codes to return.
        model_type: 'capacity' or 'intent' — affects SHAP output indexing.

    Returns:
        List of ReasonCode sorted by |SHAP value| descending.
    """
    # Build input array in correct feature order
    X = np.array([[features.get(f, 0.0) for f in feature_names]])

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    except Exception as e:
        logger.error("SHAP explanation failed: %s", e)
        return []

    # Handle different SHAP output shapes
    if isinstance(shap_values, list):
        # Binary classifier returns [class_0_shap, class_1_shap]
        # We want class_1 (positive class) contributions
        values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
    elif shap_values.ndim == 1:
        values = shap_values
    else:
        values = shap_values[0]

    # Check for all-zero SHAP (stump model)
    if np.allclose(values, 0.0):
        logger.warning(
            "All SHAP values are zero — model likely collapsed to a stump "
            "(single leaf, no splits). Reason codes will be empty."
        )
        return []

    # Build reason codes sorted by absolute contribution
    indexed_values = list(zip(feature_names, values))
    indexed_values.sort(key=lambda x: abs(x[1]), reverse=True)

    reason_codes: list[ReasonCode] = []
    for fname, sv in indexed_values[:top_n]:
        if abs(sv) < 1e-8:
            continue  # Skip negligible contributions

        direction = "↑" if sv > 0 else "↓"
        human_label = FEATURE_LABELS.get(fname, fname.replace("_", " ").title())

        reason_codes.append(ReasonCode(
            feature_name=fname,
            direction=direction,
            human_label=human_label,
            shap_value=round(float(sv), 6),
        ))

    return reason_codes
