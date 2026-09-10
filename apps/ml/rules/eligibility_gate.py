"""Eligibility gating rules for Cashflow IQ.

This is a separate, explicit, human-auditable rules module. Per AGENTS.md:
eligibility/underwriting decision logic must NEVER be buried inside a model's
internals. Each rule is a named, inspectable function.

Rules check customer features and model outputs against configurable
thresholds loaded from gating_thresholds.yaml.

No fully automated adverse action — this gate only flags customers who
should NOT be surfaced as leads. A human always signs off.
"""

import logging
import os
from pathlib import Path
from typing import Any

import yaml

from packages.schemas.score import EligibilityResult

logger = logging.getLogger(__name__)

# Default thresholds (used if YAML not found — should not happen in production)
_DEFAULT_THRESHOLDS: dict[str, Any] = {
    "max_dti": 0.50,
    "min_vintage_months": 6,
    "min_confidence": 0.30,
}


def _load_thresholds(product: str) -> dict[str, Any]:
    """Load gating thresholds from YAML config.

    Args:
        product: Loan product type (e.g., 'home_loan').

    Returns:
        Dict of threshold_name → value for the product.
    """
    config_path = Path(__file__).parent.parent / "config" / "gating_thresholds.yaml"
    if not config_path.exists():
        logger.warning(
            "Gating config not found at %s — using defaults.", config_path,
        )
        return _DEFAULT_THRESHOLDS.copy()

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if product not in config:
        logger.warning(
            "No thresholds for product '%s' — using defaults.", product,
        )
        return _DEFAULT_THRESHOLDS.copy()

    return config[product]


def check_dti(
    dti: float,
    max_dti: float,
) -> tuple[bool, str]:
    """Check if DTI is within acceptable range.

    Args:
        dti: Customer's debt-to-income ratio.
        max_dti: Maximum allowed DTI threshold.

    Returns:
        Tuple of (passed, rule_name).
    """
    rule_name = f"dti_below_{max_dti}"
    passed = dti <= max_dti
    if not passed:
        logger.info(
            "DTI gate FAILED: %.4f > %.2f", dti, max_dti,
        )
    return passed, rule_name


def check_vintage(
    account_vintage_months: int,
    min_vintage_months: int,
) -> tuple[bool, str]:
    """Check if account vintage meets minimum requirement.

    Args:
        account_vintage_months: Months since account opening.
        min_vintage_months: Minimum required vintage.

    Returns:
        Tuple of (passed, rule_name).
    """
    rule_name = f"vintage_above_{min_vintage_months}m"
    passed = account_vintage_months >= min_vintage_months
    if not passed:
        logger.info(
            "Vintage gate FAILED: %d < %d months",
            account_vintage_months, min_vintage_months,
        )
    return passed, rule_name


def check_confidence(
    confidence: float,
    min_confidence: float,
) -> tuple[bool, str]:
    """Check if model confidence meets minimum threshold.

    Args:
        confidence: Model's confidence in its prediction.
        min_confidence: Minimum required confidence.

    Returns:
        Tuple of (passed, rule_name).
    """
    rule_name = f"confidence_above_{min_confidence}"
    passed = confidence >= min_confidence
    if not passed:
        logger.info(
            "Confidence gate FAILED: %.4f < %.2f", confidence, min_confidence,
        )
    return passed, rule_name


def check_eligibility(
    dti: float,
    account_vintage_months: int,
    model_confidence: float,
    product: str = "home_loan",
    thresholds: dict[str, Any] | None = None,
) -> EligibilityResult:
    """Run all eligibility gating rules for a customer.

    This is the main entry point. It checks DTI, vintage, and confidence
    against configurable thresholds and returns a structured result with
    a three-valued gate_result:

    - "pass": all rules satisfied.
    - "fail": hard disqualification (DTI exceeds maximum).
    - "manual_review": data-quality or confidence concern (low confidence,
      insufficient vintage) — needs human review, never an automatic rejection.

    Args:
        dti: Customer's debt-to-income ratio (from capacity model or features).
        account_vintage_months: Months since account opening.
        model_confidence: Combined model confidence score.
        product: Loan product type.
        thresholds: Override thresholds (for testing). If None, loads from YAML.

    Returns:
        EligibilityResult with gate_result and rule details.
    """
    if thresholds is None:
        thresholds = _load_thresholds(product)

    max_dti = thresholds.get("max_dti", _DEFAULT_THRESHOLDS["max_dti"])
    min_vintage = thresholds.get(
        "min_vintage_months", _DEFAULT_THRESHOLDS["min_vintage_months"],
    )
    min_conf = thresholds.get(
        "min_confidence", _DEFAULT_THRESHOLDS["min_confidence"],
    )

    rules_checked: list[str] = []
    rules_failed: list[str] = []

    # Track which category of rule failed — DTI is a hard fail,
    # confidence/vintage are soft fails (manual_review).
    hard_fail = False
    soft_fail = False

    # Run each rule
    dti_passed, dti_rule = check_dti(dti, max_dti)
    rules_checked.append(dti_rule)
    if not dti_passed:
        rules_failed.append(f"DTI {dti:.4f} exceeds {max_dti:.2f}")
        hard_fail = True

    vintage_passed, vintage_rule = check_vintage(
        account_vintage_months, min_vintage,
    )
    rules_checked.append(vintage_rule)
    if not vintage_passed:
        rules_failed.append(
            f"Account vintage {account_vintage_months}m below {min_vintage}m minimum",
        )
        soft_fail = True

    conf_passed, conf_rule = check_confidence(model_confidence, min_conf)
    rules_checked.append(conf_rule)
    if not conf_passed:
        rules_failed.append(
            f"Model confidence {model_confidence:.4f} below {min_conf:.2f} minimum",
        )
        soft_fail = True

    # Determine gate_result:
    # - DTI failure is a hard disqualification → "fail"
    # - Confidence/vintage failures are data-quality concerns → "manual_review"
    # - All pass → "pass"
    if hard_fail:
        gate_result = "fail"
    elif soft_fail:
        gate_result = "manual_review"
    else:
        gate_result = "pass"

    return EligibilityResult(
        gate_result=gate_result,
        rules_checked=rules_checked,
        rules_failed=rules_failed,
    )
