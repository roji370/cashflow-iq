"""LightGBM binary classifier for loan intent/propensity scoring.

Per-product model (initially home_loan only). Uses isotonic calibration
on the predicted probabilities so the output can be interpreted as a true
probability rather than an arbitrary ranking score.

Artifact versioning: every saved model includes model_version, training_date,
and feature_schema_version in its manifest.
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# Hyperparameters tuned for small synthetic datasets (36–80 samples).
INTENT_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "binary_logloss",
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


def train(
    feature_vectors: dict[str, dict[str, float]],
    labels: dict[str, bool],
    product: str,
) -> dict[str, Any]:
    """Train a LightGBM classifier with isotonic calibration.

    Args:
        feature_vectors: {customer_id: {feature_name: value}} for all customers.
        labels: {customer_id: converted (True/False)} for the given product.
        product: Loan product type (e.g., 'home_loan').

    Returns:
        Dict with training metrics: auc, precision_top20, n_train,
        n_positive, n_features.
    """
    # Align features and labels
    common_cids = sorted(set(feature_vectors.keys()) & set(labels.keys()))
    if len(common_cids) < 5:
        raise ValueError(
            f"Need at least 5 customers with both features and labels for "
            f"product '{product}', got {len(common_cids)}."
        )

    # Build feature matrix
    all_feature_names: set[str] = set()
    for cid in common_cids:
        all_feature_names.update(feature_vectors[cid].keys())
    feature_cols = sorted(all_feature_names)

    rows = []
    y_list = []
    for cid in common_cids:
        fv = feature_vectors[cid]
        rows.append([fv.get(f, 0.0) for f in feature_cols])
        y_list.append(1 if labels[cid] else 0)

    X = np.array(rows)
    y = np.array(y_list)

    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    logger.info(
        "Intent training [%s]: %d samples (%d pos, %d neg), %d features",
        product, len(y), n_positive, n_negative, len(feature_cols),
    )

    if n_positive == 0 or n_negative == 0:
        logger.warning(
            "All labels are %s — model will be trivial.",
            "positive" if n_positive > 0 else "negative",
        )

    # Train base model
    base_model = lgb.LGBMClassifier(**INTENT_PARAMS)
    base_model.fit(X, y)

    # Isotonic calibration (cv=3 for small data)
    cv_folds = min(3, n_positive, n_negative)
    if cv_folds >= 2:
        calibrated = CalibratedClassifierCV(
            base_model, method="isotonic", cv=cv_folds,
        )
        calibrated.fit(X, y)
    else:
        # Not enough samples per class for CV — use base model directly
        logger.warning(
            "Too few samples per class for calibration CV (need ≥2, got %d). "
            "Using uncalibrated model.",
            cv_folds,
        )
        calibrated = base_model

    # --- Metrics ---
    probs = (
        calibrated.predict_proba(X)[:, 1]
        if hasattr(calibrated, "predict_proba")
        else base_model.predict_proba(X)[:, 1]
    )

    try:
        auc = float(roc_auc_score(y, probs))
    except ValueError:
        auc = 0.0  # All one class

    # Precision at top 20%
    top_k = max(1, int(len(y) * 0.2))
    top_indices = np.argsort(probs)[-top_k:]
    precision_top20 = float(y[top_indices].mean())

    # Check for stump
    trees_df = base_model.booster_.trees_to_dataframe()
    n_splits = int((trees_df["split_feature"].notna()).sum())

    metrics = {
        "auc": round(auc, 4),
        "precision_top20": round(precision_top20, 4),
        "n_train": len(y),
        "n_positive": n_positive,
        "n_features": len(feature_cols),
        "n_splits": n_splits,
    }
    logger.info("Intent [%s] training metrics: %s", product, metrics)

    # --- Save artifacts ---
    artifact_dir = _get_artifact_dir()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    training_date = date.today().isoformat()

    # Save calibrated model (or base if calibration failed)
    model_path = artifact_dir / f"intent_{product}.joblib"
    joblib.dump(calibrated, model_path)

    # Save base model separately (for SHAP — TreeExplainer needs the raw LGB)
    base_path = artifact_dir / f"intent_{product}_base.joblib"
    joblib.dump(base_model, base_path)

    # Save manifest
    manifest = {
        "model_type": "intent",
        "product": product,
        "model_version": MODEL_VERSION,
        "training_date": training_date,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": feature_cols,
        "hyperparameters": INTENT_PARAMS,
        "metrics": metrics,
        "calibration_method": "isotonic" if cv_folds >= 2 else "none",
    }
    manifest_path = artifact_dir / f"intent_{product}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    logger.info("Intent [%s] artifacts saved to %s", product, artifact_dir)
    return metrics


def load_model(product: str) -> Optional[Any]:
    """Load the calibrated intent model for a product.

    Returns:
        Calibrated classifier, or None if not trained.
    """
    path = _get_artifact_dir() / f"intent_{product}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def load_base_model(product: str) -> Optional[lgb.LGBMClassifier]:
    """Load the raw LightGBM base model (for SHAP TreeExplainer).

    Returns:
        LGBMClassifier, or None if not trained.
    """
    path = _get_artifact_dir() / f"intent_{product}_base.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


def load_manifest(product: str) -> Optional[dict[str, Any]]:
    """Load the intent model manifest for a product.

    Returns:
        Manifest dict or None if not found.
    """
    path = _get_artifact_dir() / f"intent_{product}_manifest.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def predict(
    features: dict[str, float],
    product: str,
    model: Optional[Any] = None,
    manifest: Optional[dict[str, Any]] = None,
) -> tuple[float, float]:
    """Predict loan intent probability.

    Args:
        features: Feature vector for a single customer.
        product: Loan product type.
        model: Pre-loaded calibrated model (or None to load from disk).
        manifest: Pre-loaded manifest (or None to load from disk).

    Returns:
        Tuple of (intent_score 0–100, confidence 0–1).

    Raises:
        RuntimeError: If model hasn't been trained for this product.
    """
    if model is None:
        model = load_model(product)
    if model is None:
        raise RuntimeError(
            f"Intent model not trained for product '{product}'. "
            f"Run: python -m apps.ml.models.train --model intent --product {product}"
        )

    if manifest is None:
        manifest = load_manifest(product)

    # Build input vector in the correct feature order
    feature_names = manifest["feature_names"] if manifest else sorted(features.keys())
    X = np.array([[features.get(f, 0.0) for f in feature_names]])

    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
    else:
        prob = float(model.predict(X)[0])

    intent_score = round(prob * 100, 2)

    # Confidence: how far from 0.5 (uncertain) the prediction is
    confidence = round(abs(prob - 0.5) * 2, 4)

    return intent_score, confidence
