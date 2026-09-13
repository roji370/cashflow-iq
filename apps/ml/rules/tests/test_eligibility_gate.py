"""Unit tests for eligibility gating rules.

15 tests covering:
  - DTI pass/fail/boundary
  - Vintage pass/fail/boundary
  - Confidence pass/fail/boundary
  - Multiple simultaneous failures
  - All-pass happy path
  - Custom threshold overrides
  - Edge cases (zero DTI, zero vintage, exact boundary values)
  - Three-valued gate_result: pass / fail / manual_review
"""

import pytest

from apps.ml.rules.eligibility_gate import (
    check_confidence,
    check_dti,
    check_eligibility,
    check_vintage,
)


# -----------------------------------------------------------------------
# DTI rule tests
# -----------------------------------------------------------------------

class TestCheckDTI:
    """Tests for the DTI gating rule."""

    def test_dti_below_threshold_passes(self) -> None:
        """DTI below max should pass."""
        passed, rule = check_dti(dti=0.35, max_dti=0.50)
        assert passed is True
        assert "dti" in rule.lower() or "0.5" in rule

    def test_dti_above_threshold_fails(self) -> None:
        """DTI above max should fail."""
        passed, _ = check_dti(dti=0.65, max_dti=0.50)
        assert passed is False

    def test_dti_exactly_at_threshold_passes(self) -> None:
        """DTI exactly at threshold is acceptable (<=)."""
        passed, _ = check_dti(dti=0.50, max_dti=0.50)
        assert passed is True

    def test_dti_zero_passes(self) -> None:
        """Zero DTI (no debt) should always pass."""
        passed, _ = check_dti(dti=0.0, max_dti=0.50)
        assert passed is True


# -----------------------------------------------------------------------
# Vintage rule tests
# -----------------------------------------------------------------------

class TestCheckVintage:
    """Tests for the account vintage gating rule."""

    def test_vintage_above_minimum_passes(self) -> None:
        """Vintage above minimum should pass."""
        passed, _ = check_vintage(account_vintage_months=12, min_vintage_months=6)
        assert passed is True

    def test_vintage_below_minimum_fails(self) -> None:
        """Vintage below minimum should fail."""
        passed, _ = check_vintage(account_vintage_months=3, min_vintage_months=6)
        assert passed is False

    def test_vintage_exactly_at_minimum_passes(self) -> None:
        """Vintage exactly at minimum is acceptable (>=)."""
        passed, _ = check_vintage(account_vintage_months=6, min_vintage_months=6)
        assert passed is True

    def test_vintage_zero_fails(self) -> None:
        """Zero vintage (brand new account) should fail."""
        passed, _ = check_vintage(account_vintage_months=0, min_vintage_months=6)
        assert passed is False


# -----------------------------------------------------------------------
# Confidence rule tests
# -----------------------------------------------------------------------

class TestCheckConfidence:
    """Tests for the model confidence gating rule."""

    def test_confidence_above_threshold_passes(self) -> None:
        """Confidence above minimum should pass."""
        passed, _ = check_confidence(confidence=0.75, min_confidence=0.30)
        assert passed is True

    def test_confidence_below_threshold_fails(self) -> None:
        """Confidence below minimum should fail."""
        passed, _ = check_confidence(confidence=0.10, min_confidence=0.30)
        assert passed is False

    def test_confidence_exactly_at_threshold_passes(self) -> None:
        """Confidence exactly at threshold is acceptable (>=)."""
        passed, _ = check_confidence(confidence=0.30, min_confidence=0.30)
        assert passed is True


# -----------------------------------------------------------------------
# Integrated check_eligibility tests
# -----------------------------------------------------------------------

class TestCheckEligibility:
    """Tests for the combined eligibility check with three-valued gate_result."""

    _THRESHOLDS = {
        "max_dti": 0.50,
        "min_vintage_months": 6,
        "min_confidence": 0.30,
    }

    def test_all_rules_pass(self) -> None:
        """Customer meeting all criteria should get gate_result='pass'."""
        result = check_eligibility(
            dti=0.35,
            account_vintage_months=12,
            model_confidence=0.75,
            thresholds=self._THRESHOLDS,
        )
        assert result.gate_result == "pass"
        assert len(result.rules_checked) == 3
        assert len(result.rules_failed) == 0

    def test_dti_only_fails_hard(self) -> None:
        """DTI failure is a hard disqualification — gate_result='fail'."""
        result = check_eligibility(
            dti=0.80,
            account_vintage_months=12,
            model_confidence=0.75,
            thresholds=self._THRESHOLDS,
        )
        assert result.gate_result == "fail"
        assert len(result.rules_failed) == 1
        assert "DTI" in result.rules_failed[0]

    def test_confidence_only_fails_manual_review(self) -> None:
        """Low confidence alone → gate_result='manual_review', not 'fail'."""
        result = check_eligibility(
            dti=0.35,
            account_vintage_months=12,
            model_confidence=0.10,
            thresholds=self._THRESHOLDS,
        )
        assert result.gate_result == "manual_review"
        assert len(result.rules_failed) == 1
        assert "confidence" in result.rules_failed[0].lower()

    def test_vintage_only_fails_manual_review(self) -> None:
        """Low vintage alone → gate_result='manual_review', not 'fail'."""
        result = check_eligibility(
            dti=0.35,
            account_vintage_months=2,
            model_confidence=0.75,
            thresholds=self._THRESHOLDS,
        )
        assert result.gate_result == "manual_review"
        assert len(result.rules_failed) == 1
        assert "vintage" in result.rules_failed[0].lower()

    def test_multiple_soft_failures_manual_review(self) -> None:
        """Vintage + confidence fail (no DTI) → 'manual_review'."""
        result = check_eligibility(
            dti=0.35,
            account_vintage_months=2,
            model_confidence=0.10,
            thresholds=self._THRESHOLDS,
        )
        assert result.gate_result == "manual_review"
        assert len(result.rules_failed) == 2

    def test_dti_plus_others_fails_hard(self) -> None:
        """DTI failure + any other failures → 'fail' (hard gate dominates)."""
        result = check_eligibility(
            dti=0.80,
            account_vintage_months=2,
            model_confidence=0.10,
            thresholds=self._THRESHOLDS,
        )
        assert result.gate_result == "fail"
        assert len(result.rules_failed) == 3
        assert len(result.rules_checked) == 3

