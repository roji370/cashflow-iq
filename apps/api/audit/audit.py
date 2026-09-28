"""Audit logging for Cashflow IQ scoring requests.

Provides functions to create the ``audit_log`` table and to log each
scoring request with model versions, gate results, and a SHA-256 hash
of the input feature vector (never raw feature values — PII stays out
of the queryable audit trail).

Model versions are read from each model's ``manifest.json``, not
hardcoded.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional

import psycopg2

logger = logging.getLogger(__name__)


def _get_db_url() -> str:
    """Read the database connection URL from environment.

    Returns:
        The DATABASE_URL string.

    Raises:
        RuntimeError: If DATABASE_URL is not set.
    """
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set.")
    return url


def create_audit_table() -> None:
    """Create the audit_log table if it does not exist.

    Called from the API lifespan handler on startup. Uses the same DDL
    as ``infra/sql/init/002_audit_log.sql`` but is safe to call multiple
    times (IF NOT EXISTS).
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS audit_log (
        id                      SERIAL PRIMARY KEY,
        customer_id             TEXT NOT NULL,
        product                 TEXT NOT NULL,
        capacity_model_version  TEXT NOT NULL,
        intent_model_version    TEXT NOT NULL,
        gate_result             TEXT NOT NULL,
        requesting_role         TEXT NOT NULL,
        input_feature_hash      TEXT NOT NULL,
        created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_audit_log_customer_id
        ON audit_log (customer_id, created_at DESC);
    """
    try:
        conn = psycopg2.connect(_get_db_url())
        try:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
            logger.info("audit_log table ensured.")
        finally:
            conn.close()
    except Exception as exc:
        logger.error("Failed to create audit_log table: %s", exc)
        raise


def hash_features(features: dict[str, Any]) -> str:
    """Compute a SHA-256 hash of a feature vector.

    The feature dict is sorted by key and JSON-serialized before hashing
    to produce a deterministic, reproducible hash. Raw feature values
    are never stored — only this hash goes into the audit log.

    Args:
        features: The customer's feature vector dict.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    serialized = json.dumps(features, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def log_score_request(
    customer_id: str,
    product: str,
    capacity_model_version: str,
    intent_model_version: str,
    gate_result: str,
    requesting_role: str,
    input_feature_hash: str,
) -> None:
    """Insert one audit log row for a scoring request.

    Called from the ``/score/{customer_id}`` endpoint after a successful
    score is computed, before the response is returned.

    Args:
        customer_id: The scored customer's ID.
        product: Loan product type (e.g., 'home_loan').
        capacity_model_version: Version from capacity model manifest.
        intent_model_version: Version from intent model manifest.
        gate_result: Eligibility gate outcome ('pass', 'fail', or 'manual_review').
        requesting_role: The role of the caller (from JWT).
        input_feature_hash: SHA-256 hash of the feature vector.
    """
    sql = """
    INSERT INTO audit_log
        (customer_id, product, capacity_model_version, intent_model_version,
         gate_result, requesting_role, input_feature_hash)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    try:
        conn = psycopg2.connect(_get_db_url())
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    customer_id,
                    product,
                    capacity_model_version,
                    intent_model_version,
                    gate_result,
                    requesting_role,
                    input_feature_hash,
                ))
            conn.commit()
            logger.info(
                "Audit log: customer=%s product=%s gate=%s role=%s",
                customer_id, product, gate_result, requesting_role,
            )
        finally:
            conn.close()
    except Exception as exc:
        # Audit logging failure should not break the score response —
        # log the error but don't re-raise.
        logger.error(
            "Failed to write audit log for customer=%s: %s",
            customer_id, exc,
        )


def get_audit_trail(customer_id: str) -> list[dict[str, Any]]:
    """Retrieve the audit trail for a customer, most recent first.

    Args:
        customer_id: The customer to look up.

    Returns:
        List of audit log row dicts, ordered by created_at DESC.
    """
    sql = """
    SELECT id, customer_id, product, capacity_model_version,
           intent_model_version, gate_result, requesting_role,
           input_feature_hash, created_at
    FROM audit_log
    WHERE customer_id = %s
    ORDER BY created_at DESC
    """
    conn = psycopg2.connect(_get_db_url())
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (customer_id,))
            columns = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()

    results = []
    for row in rows:
        entry = dict(zip(columns, row))
        # Convert datetime to ISO string for JSON serialization
        if isinstance(entry.get("created_at"), datetime):
            entry["created_at"] = entry["created_at"].isoformat()
        results.append(entry)

    return results
