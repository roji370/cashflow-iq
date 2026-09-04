"""Training CLI for Cashflow IQ models.

Usage:
    python -m apps.ml.models.train --model capacity
    python -m apps.ml.models.train --model intent --product home_loan

Loads feature vectors and labels from the Postgres feature store, trains the
requested model, and saves versioned artifacts to MODEL_ARTIFACT_PATH.
"""

import argparse
import logging
import os
from typing import Any

import psycopg2

from apps.ml.models import capacity_model, intent_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _get_connection() -> psycopg2.extensions.connection:
    """Create a Postgres connection from DATABASE_URL env var."""
    database_url = os.environ["DATABASE_URL"]
    return psycopg2.connect(database_url)


def _load_feature_vectors() -> dict[str, dict[str, float]]:
    """Load all customer feature vectors from the features table.

    Returns:
        Dict mapping customer_id → {feature_name: feature_value}.
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT ON (customer_id, feature_name)
                       customer_id, feature_name, feature_value
                FROM features
                ORDER BY customer_id, feature_name, computed_at DESC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    vectors: dict[str, dict[str, float]] = {}
    for cid, fname, fvalue in rows:
        if cid not in vectors:
            vectors[cid] = {}
        vectors[cid][fname] = float(fvalue)

    logger.info("Loaded feature vectors for %d customers.", len(vectors))
    return vectors


def _load_labels(product: str) -> dict[str, bool]:
    """Load conversion labels for a product from the raw_labels table.

    Args:
        product: Loan product type (e.g., 'home_loan').

    Returns:
        Dict mapping customer_id → converted (True/False).
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT customer_id, converted FROM raw_labels WHERE product = %s",
                (product,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    labels = {cid: bool(converted) for cid, converted in rows}
    n_pos = sum(1 for v in labels.values() if v)
    logger.info(
        "Loaded %d labels for product '%s' (%d positive, %d negative).",
        len(labels), product, n_pos, len(labels) - n_pos,
    )
    return labels


def _compute_capacity_targets(
    feature_vectors: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Compute synthetic capacity targets from feature vectors.

    Target = salary_amount_median - avg_monthly_outflow.

    This is a placeholder synthetic target. The real value of the capacity
    model infrastructure is the quantile regression, confidence scoring,
    and SHAP pipeline — not this target's predictive accuracy.

    Args:
        feature_vectors: All customer feature vectors.

    Returns:
        Dict mapping customer_id → synthetic disposable income target.
    """
    targets: dict[str, float] = {}
    for cid, fv in feature_vectors.items():
        salary = fv.get("salary_amount_median", 0.0)
        outflow = fv.get("avg_monthly_outflow", 0.0)
        if salary > 0:
            targets[cid] = salary - outflow
    logger.info("Computed capacity targets for %d customers.", len(targets))
    return targets


def train_capacity() -> dict[str, Any]:
    """Train the capacity model end-to-end.

    Returns:
        Training metrics dict.
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Capacity model (quantile regressor)")
    logger.info("=" * 60)

    feature_vectors = _load_feature_vectors()
    targets = _compute_capacity_targets(feature_vectors)

    if len(targets) < 5:
        logger.error(
            "Need at least 5 customers with features, got %d. "
            "Run the pipeline first: make ingest",
            len(targets),
        )
        raise SystemExit(1)

    metrics = capacity_model.train(feature_vectors, targets)

    if metrics.get("is_stump"):
        logger.warning(
            "⚠ Capacity model collapsed to a stump. SHAP values will be zero. "
            "Check min_data_in_leaf vs training set size."
        )

    return metrics


def train_intent(product: str) -> dict[str, Any]:
    """Train the intent model for a specific product.

    Args:
        product: Loan product type (e.g., 'home_loan').

    Returns:
        Training metrics dict.
    """
    logger.info("=" * 60)
    logger.info("TRAINING: Intent model [%s] (binary classifier)", product)
    logger.info("=" * 60)

    feature_vectors = _load_feature_vectors()
    labels = _load_labels(product)

    if len(labels) == 0:
        logger.error(
            "No labels found for product '%s'. "
            "Make sure labels.csv has been ingested.",
            product,
        )
        raise SystemExit(1)

    metrics = intent_model.train(feature_vectors, labels, product)
    return metrics


def main() -> None:
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(
        description="Train Cashflow IQ scoring models.",
    )
    parser.add_argument(
        "--model",
        choices=["capacity", "intent"],
        required=True,
        help="Which model to train.",
    )
    parser.add_argument(
        "--product",
        type=str,
        default="home_loan",
        help="Product type for intent model (default: home_loan).",
    )
    args = parser.parse_args()

    if args.model == "capacity":
        metrics = train_capacity()
    elif args.model == "intent":
        metrics = train_intent(args.product)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    logger.info("Training complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
