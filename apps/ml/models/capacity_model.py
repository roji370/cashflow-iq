"""LightGBM quantile regressor for disposable income (capacity).

Trains three LightGBM models at quantiles 0.1, 0.5, 0.9 to produce a
median estimate with uncertainty bounds. The q10/q90 spread drives the
confidence score: tighter interval → higher confidence.

IMPORTANT — Target leakage exclusion:
  The synthetic training target is `salary_amount_median - avg_monthly_outflow`.
  These two features MUST be excluded from training input, since they directly
  construct the target. See CAPACITY_EXCLUDED_FEATURES below.

IMPORTANT — Small-data hyperparameters:
  With ~36–80 synthetic training customers, LightGBM's default
  min_data_in_leaf=20 causes the model to collapse to a single-leaf stump
  (constant prediction, zero SHAP values). We set min_data_in_leaf=3 and
  num_leaves=8 to allow actual splits at this scale.
"""

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Features that construct the synthetic target — MUST NOT be used as inputs.
# salary_amount_median - avg_monthly_outflow = synthetic disposable income target.
CAPACITY_EXCLUDED_FEATURES: set[str] = {
    "salary_amount_median",
    "avg_monthly_outflow",
}

# Hyperparameters tuned for small synthetic datasets (36–80 samples).
# These MUST be revisited when training on real production data.
CAPACITY_PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "metric": "quantile",
    "boosting_type": "gbdt",
    "num_leaves": 8,
    "min_data_in_leaf": 3,
    "n_estimators": 100,
    "learning_rate": 0.05,
    "verbose": -1,
    "seed": 42,
}

MODEL_VERSION = "0.1.0"
FEATURE_SCHEMA_VERSION = "1.0.0"


def _get_artifact_dir() -> Path:
    """Return the model artifact directory from env var."""
    artifact_path = os.environ.get("MODEL_ARTIFACT_PATH", "apps/ml/artifacts")
    return Path(artifact_path)


def _build_feature_matrix(
    feature_vectors: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, list[str]]:
    """Convert {customer_id: {feature: value}} to a DataFrame.

    Excludes CAPACITY_EXCLUDED_FEATURES from the feature columns.

    Returns:
        Tuple of (feature DataFrame, ordered list of feature names).
    """
    rows = []
    for cid, fv in feature_vectors.items():
        row = {
            k: v for k, v in fv.items()
            if k not in CAPACITY_EXCLUDED_FEATURES
        }
        row["_customer_id"] = cid
        rows.append(row)

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c != "_customer_id"]
    feature_cols.sort()  # deterministic ordering
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df, feature_cols


def train(
    feature_vectors: dict[str, dict[str, float]],
    targets: dict[str, float],
) -> dict[str, Any]:
    """Train three quantile regressors (q=0.1, 0.5, 0.9).

    Args:
        feature_vectors: {customer_id: {feature_name: value}} for all customers.
        targets: {customer_id: synthetic_disposable_income} training labels.

    Returns:
        Dict with training metrics: mae_median, coverage_80, n_train,
        n_features, n_trees_per_model.
    """
    df, feature_cols = _build_feature_matrix(feature_vectors)

    # Align targets with feature matrix
    cids = df["_customer_id"].tolist()
    y = np.array([targets.get(cid, 0.0) for cid in cids])
    X = df[feature_cols].values

    logger.info(
        "Capacity training: %d samples, %d features (excluded: %s)",
        len(y), len(feature_cols), CAPACITY_EXCLUDED_FEATURES,
    )

    models: dict[str, lgb.LGBMRegressor] = {}
    for alpha in [0.1, 0.5, 0.9]:
        params = {**CAPACITY_PARAMS, "alpha": alpha}
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        models[f"q{alpha}"] = model

    # --- Metrics (honestly logged, not meaningful at small scale) ---
    preds_median = models["q0.5"].predict(X)
    mae = float(np.mean(np.abs(y - preds_median)))

    preds_q10 = models["q0.1"].predict(X)
    preds_q90 = models["q0.9"].predict(X)
    in_interval = np.sum((y >= preds_q10) & (y <= preds_q90))
    coverage = float(in_interval / len(y)) if len(y) > 0 else 0.0

    # Check for stump (single-leaf) models
    trees_df = models["q0.5"].booster_.trees_to_dataframe()
    n_splits = int((trees_df["split_feature"].notna()).sum())

    metrics = {
        "mae_median": round(mae, 2),
        "coverage_80": round(coverage, 4),
        "n_train": len(y),
        "n_features": len(feature_cols),
        "n_trees_per_model": CAPACITY_PARAMS["n_estimators"],
        "n_splits_median_model": n_splits,
        "is_stump": n_splits == 0,
    }

    if metrics["is_stump"]:
        logger.warning(
            "CAPACITY MODEL IS A STUMP (0 splits). All predictions will be "
            "constant and SHAP values will be zero. This usually means "
            "min_data_in_leaf is too large for the training set size."
        )
    else:
        logger.info("Capacity model has %d splits — non-trivial tree structure.", n_splits)

    logger.info("Capacity training metrics: %s", metrics)

    # --- Save artifacts ---
    artifact_dir = _get_artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    training_date = date.today().isoformat()
    for name, model in models.items():
        artifact_path = artifact_dir / f"capacity_{name}.joblib"
        joblib.dump(model, artifact_path)

    # Save manifest
    manifest = {
        "model_type": "capacity",
        "model_version": MODEL_VERSION,
        "training_date": training_date,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": feature_cols,
        "excluded_features": sorted(CAPACITY_EXCLUDED_FEATURES),
        "hyperparameters": CAPACITY_PARAMS,
        "metrics": metrics,
    }
    manifest_path = artifact_dir / "capacity_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("Capacity artifacts saved to %s", artifact_dir)
    return metrics


def load_models() -> Optional[dict[str, lgb.LGBMRegressor]]:
    """Load trained capacity models from artifact directory.

    Returns:
        Dict with keys 'q0.1', 'q0.5', 'q0.9' mapping to LGBMRegressor,
        or None if models haven't been trained yet.
    """
    artifact_dir = _get_artifact_dir()
    models = {}
    for alpha in [0.1, 0.5, 0.9]:
        path = artifact_dir / f"capacity_q{alpha}.joblib"
        if not path.exists():
            return None
        models[f"q{alpha}"] = joblib.load(path)
    return models


def load_manifest() -> Optional[dict[str, Any]]:
    """Load the capacity model manifest.

    Returns:
        Manifest dict or None if not found.
    """
    manifest_path = _get_artifact_dir() / "capacity_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path) as f:
        return json.load(f)


def predict(
    features: dict[str, float],
    models: Optional[dict[str, lgb.LGBMRegressor]] = None,
    manifest: Optional[dict[str, Any]] = None,
) -> tuple[float, float, float, float]:
    """Predict disposable income with uncertainty bounds.

    Args:
        features: Feature vector for a single customer.
        models: Pre-loaded models (or None to load from disk).
        manifest: Pre-loaded manifest (or None to load from disk).

    Returns:
        Tuple of (q10, q50, q90, confidence).

    Raises:
        RuntimeError: If models haven't been trained.
    """
    if models is None:
        models = load_models()
    if models is None:
        raise RuntimeError("Capacity models not trained. Run train.py first.")

    if manifest is None:
        manifest = load_manifest()

    # Build input vector in the correct feature order
    feature_names = manifest["feature_names"] if manifest else sorted(
        k for k in features.keys() if k not in CAPACITY_EXCLUDED_FEATURES
    )
    X = np.array([[features.get(f, 0.0) for f in feature_names]])

    q10 = float(models["q0.1"].predict(X)[0])
    q50 = float(models["q0.5"].predict(X)[0])
    q90 = float(models["q0.9"].predict(X)[0])

    # Confidence: inverse of relative interval width
    interval_width = q90 - q10
    if q50 != 0:
        relative_width = abs(interval_width / q50)
        confidence = max(0.0, min(1.0, 1.0 - relative_width))
    else:
        confidence = 0.0

    return q10, q50, q90, round(confidence, 4)
