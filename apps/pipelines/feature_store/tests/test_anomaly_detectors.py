"""Unit tests for behavioral anomaly detectors.

Uses specific personas from the synthetic data generator to validate that:
  - Subscription cleansing fires for persona 1 (stable_salaried), not persona 2 (gig_worker)
  - Liquidity pooling fires for persona 5 (hni), not persona 1
  - Bill shift fires for persona 4 (over_leveraged), not persona 1
"""

import pandas as pd
import pytest

from apps.pipelines.feature_store.anomaly_detectors import (
    detect_bill_shift,
    detect_liquidity_pooling,
    detect_subscription_cleansing,
)
from apps.pipelines.synth_data.generate import generate


# ---------------------------------------------------------------------------
# Fixture: generate all persona data once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def persona_data() -> dict[str, pd.DataFrame]:
    """Generate synthetic data and return per-customer transaction DataFrames."""
    _, all_txns, _ = generate(num_personas=8, num_months=12, seed=42)
    df = pd.DataFrame(all_txns)
    return {cid: grp for cid, grp in df.groupby("customer_id")}


# ---------------------------------------------------------------------------
# Subscription cleansing tests
# ---------------------------------------------------------------------------

class TestSubscriptionCleansing:
    """Tests for the subscription cleansing detector."""

    def test_triggers_for_stable_salaried(self, persona_data: dict) -> None:
        """Persona 1 (stable_salaried) has 3 subscriptions cancelled in months 10–11.

        The detector should flag this.
        """
        txns = persona_data["cust_001"]
        result = detect_subscription_cleansing(txns)
        assert result["subscription_cleansing_flag"] is True
        assert 0.0 < result["subscription_cleansing_score"] <= 1.0

    def test_does_not_trigger_for_gig_worker(self, persona_data: dict) -> None:
        """Persona 2 (gig_worker) has no subscription cleansing pattern.

        The detector should NOT flag this (false-positive check).
        """
        txns = persona_data["cust_002"]
        result = detect_subscription_cleansing(txns)
        assert result["subscription_cleansing_flag"] is False
        assert result["subscription_cleansing_score"] == 0.0


# ---------------------------------------------------------------------------
# Liquidity pooling tests
# ---------------------------------------------------------------------------

class TestLiquidityPooling:
    """Tests for the liquidity pooling detector."""

    def test_triggers_for_hni(self, persona_data: dict) -> None:
        """Persona 5 (hni) has an FD maturity in month 8, not reinvested.

        The detector should flag this.
        """
        txns = persona_data["cust_005"]
        result = detect_liquidity_pooling(txns)
        assert result["liquidity_pooling_flag"] is True
        assert 0.0 < result["liquidity_pooling_score"] <= 1.0

    def test_does_not_trigger_for_stable_salaried(self, persona_data: dict) -> None:
        """Persona 1 (stable_salaried) has no investment maturity events.

        The detector should NOT flag this.
        """
        txns = persona_data["cust_001"]
        result = detect_liquidity_pooling(txns)
        assert result["liquidity_pooling_flag"] is False


# ---------------------------------------------------------------------------
# Bill shift tests
# ---------------------------------------------------------------------------

class TestBillShift:
    """Tests for the out-of-cycle bill shift detector."""

    def test_triggers_for_over_leveraged(self, persona_data: dict) -> None:
        """Persona 4 (over_leveraged) has utility bill payment day drifting
        from day 5 to day 25 over months 7–11.

        The detector should flag this.
        """
        txns = persona_data["cust_004"]
        result = detect_bill_shift(txns)
        assert result["bill_shift_flag"] is True
        assert 0.0 < result["bill_shift_score"] <= 1.0

    def test_does_not_trigger_for_stable_salaried(self, persona_data: dict) -> None:
        """Persona 1 (stable_salaried) pays utility on day 10 consistently.

        The detector should NOT flag this.
        """
        txns = persona_data["cust_001"]
        result = detect_bill_shift(txns)
        assert result["bill_shift_flag"] is False
