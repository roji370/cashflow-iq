"""Integration tests for SHAP explainer.

Tests verify that:
  1. SHAP values are non-zero for models with actual splits (Bug #2 regression guard)
  2. Reason codes have correct structure and human-readable labels
  3. Top-N filtering works correctly
"""

import numpy as np
import pytest

try:
    import lightgbm as lgb
    import shap
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from apps.ml.explain.shap_explainer import FEATURE_LABELS, explain


@pytest.mark.skipif(not HAS_DEPS, reason="lightgbm/shap not installed")
class TestSHAPExplainer:
    """Tests for the SHAP explainer using a synthetic model."""

    @pytest.fixture()
    def trained_regressor(self) -> tuple[lgb.LGBMRegressor, list[str]]:
        """Create a small LightGBM regressor with guaranteed splits.

        Uses min_data_in_leaf=1 to ensure the model actually splits
        even with a tiny dataset. This is the regression guard for Bug #2:
        if the model has splits, SHAP values MUST be non-zero.
        """
        np.random.seed(42)
        n = 30
        feature_names = [
            "savings_ratio_6m",
            "income_growth_pct",
            "current_dti",
            "num_active_emis",
            "liquidity_buffer",
        ]
        X = np.random.randn(n, len(feature_names))
        # Target that actually depends on features (not random)
        y = X[:, 0] * 1000 + X[:, 2] * -500 + np.random.randn(n) * 100

        model = lgb.LGBMRegressor(
            n_estimators=50,
            num_leaves=8,
            min_data_in_leaf=1,
            verbose=-1,
            seed=42,
        )
        model.fit(X, y)

        # Sanity: confirm model has splits
        trees = model.booster_.trees_to_dataframe()
        n_splits = int((trees["split_feature"].notna()).sum())
        assert n_splits > 0, "Test fixture model has no splits — test is invalid"

        return model, feature_names

    @pytest.fixture()
    def trained_classifier(self) -> tuple[lgb.LGBMClassifier, list[str]]:
        """Create a small LightGBM classifier with guaranteed splits."""
        np.random.seed(42)
        n = 40
        feature_names = [
            "savings_ratio_6m",
            "income_growth_pct",
            "current_dti",
            "num_active_emis",
            "liquidity_buffer",
        ]
        X = np.random.randn(n, len(feature_names))
        # Binary target correlated with features
        y = (X[:, 0] + X[:, 4] > 0).astype(int)

        model = lgb.LGBMClassifier(
            n_estimators=50,
            num_leaves=8,
            min_data_in_leaf=1,
            verbose=-1,
            seed=42,
        )
        model.fit(X, y)

        return model, feature_names

    def test_regressor_shap_values_nonzero(
        self,
        trained_regressor: tuple[lgb.LGBMRegressor, list[str]],
    ) -> None:
        """SHAP values must be non-zero for a model with actual splits.

        This is the primary regression guard for Bug #2. If this test fails,
        the capacity model has collapsed to a stump.
        """
        model, feature_names = trained_regressor
        features = {f: float(np.random.randn()) for f in feature_names}

        reason_codes = explain(
            model, features, feature_names,
            top_n=5, model_type="capacity",
        )

        assert len(reason_codes) > 0, (
            "SHAP returned zero reason codes — model may be a stump"
        )
        # At least one SHAP value should be meaningfully non-zero
        max_shap = max(abs(rc.shap_value) for rc in reason_codes)
        assert max_shap > 1e-6, (
            f"All SHAP values near zero (max={max_shap}). "
            f"Model likely has no real splits."
        )

    def test_classifier_shap_values_nonzero(
        self,
        trained_classifier: tuple[lgb.LGBMClassifier, list[str]],
    ) -> None:
        """SHAP values must be non-zero for intent classifier."""
        model, feature_names = trained_classifier
        features = {f: float(np.random.randn()) for f in feature_names}

        reason_codes = explain(
            model, features, feature_names,
            top_n=5, model_type="intent",
        )

        assert len(reason_codes) > 0

    def test_reason_code_structure(
        self,
        trained_regressor: tuple[lgb.LGBMRegressor, list[str]],
    ) -> None:
        """Reason codes should have valid structure."""
        model, feature_names = trained_regressor
        features = {f: float(np.random.randn()) for f in feature_names}

        reason_codes = explain(model, features, feature_names, top_n=3)

        for rc in reason_codes:
            assert rc.feature_name in feature_names
            assert rc.direction in ("↑", "↓")
            assert isinstance(rc.human_label, str)
            assert len(rc.human_label) > 0
            assert isinstance(rc.shap_value, float)

    def test_human_labels_mapped(
        self,
        trained_regressor: tuple[lgb.LGBMRegressor, list[str]],
    ) -> None:
        """Known features should get human-readable labels from FEATURE_LABELS."""
        model, feature_names = trained_regressor
        features = {f: float(np.random.randn()) for f in feature_names}

        reason_codes = explain(model, features, feature_names, top_n=5)

        for rc in reason_codes:
            if rc.feature_name in FEATURE_LABELS:
                assert rc.human_label == FEATURE_LABELS[rc.feature_name]

    def test_top_n_limits_output(
        self,
        trained_regressor: tuple[lgb.LGBMRegressor, list[str]],
    ) -> None:
        """top_n should limit the number of reason codes returned."""
        model, feature_names = trained_regressor
        features = {f: float(np.random.randn()) for f in feature_names}

        reason_codes = explain(model, features, feature_names, top_n=2)

        assert len(reason_codes) <= 2

    def test_sorted_by_abs_shap(
        self,
        trained_regressor: tuple[lgb.LGBMRegressor, list[str]],
    ) -> None:
        """Reason codes should be sorted by |SHAP value| descending."""
        model, feature_names = trained_regressor
        features = {f: float(np.random.randn()) for f in feature_names}

        reason_codes = explain(model, features, feature_names, top_n=5)

        if len(reason_codes) >= 2:
            for i in range(len(reason_codes) - 1):
                assert abs(reason_codes[i].shap_value) >= abs(
                    reason_codes[i + 1].shap_value
                )
