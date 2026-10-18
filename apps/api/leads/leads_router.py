"""Customer enumeration and ranked lead list endpoints.

Provides:
    GET /customers  — list all customers (rm role)
    GET /leads      — ranked, filtered lead list (rm role)

These endpoints were not part of the original Phase C/D API — they are new
in Phase E to support the dashboard's lead list view.

Performance note: the /leads endpoint scores every customer on every request
by calling the existing predict() functions in a loop. This is intentionally
simple and correct at prototype scale (dozens of customers). Before scaling
past a few hundred customers, this should move to a precomputed/cached
ranking — e.g. a nightly batch job that writes scored results to a leads
table, with the endpoint reading from that table instead of scoring live.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any

import psycopg2
from fastapi import APIRouter, Depends, HTTPException, Query

from apps.api.auth.auth import require_role
from apps.ml.models import capacity_model, intent_model
from apps.ml.rules.eligibility_gate import check_eligibility
from apps.pipelines.feature_store.run_nightly_features import get_feature_vector

logger = logging.getLogger(__name__)

router = APIRouter(tags=["leads"])

# Only products with trained models
SUPPORTED_PRODUCTS = {"home_loan"}


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


def _get_all_customers() -> list[dict[str, str]]:
    """Query raw_customers for all (customer_id, name) pairs.

    Returns:
        List of dicts with customer_id and name keys.
    """
    conn = psycopg2.connect(_get_db_url())
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT customer_id, name FROM raw_customers ORDER BY customer_id;")
            rows = cur.fetchall()
    finally:
        conn.close()

    return [{"customer_id": row[0], "name": row[1]} for row in rows]


def _log_leads_request(
    product: str,
    min_score: float,
    num_results: int,
    requesting_role: str,
) -> None:
    """Log a lightweight audit entry for a /leads batch call.

    Uses a separate, simple INSERT rather than reusing log_score_request
    (which logs per-customer detail). This avoids flooding the audit table
    when an RM refreshes the lead list — individual customer scores already
    have their own audit trail via the /score endpoint.

    Args:
        product: Loan product type requested.
        min_score: Minimum intent score filter applied.
        num_results: Number of leads returned.
        requesting_role: Role of the calling user.
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
                    "__leads_batch__",
                    product,
                    "n/a",
                    "n/a",
                    f"returned_{num_results}_leads",
                    requesting_role,
                    f"min_score={min_score}",
                ))
            conn.commit()
            logger.info(
                "Leads audit: product=%s min_score=%s results=%d role=%s",
                product, min_score, num_results, requesting_role,
            )
        finally:
            conn.close()
    except Exception as exc:
        # Audit failure should not break the response
        logger.error("Failed to log leads request: %s", exc)


@router.get("/customers")
def list_customers(
    user: dict = Depends(require_role("rm")),
) -> list[dict[str, str]]:
    """Return all customers (customer_id, name) from raw_customers.

    Requires at least 'rm' role. This is a simple read — no scoring involved.

    Args:
        user: Decoded JWT payload (injected by auth dependency).

    Returns:
        List of {customer_id, name} dicts.
    """
    return _get_all_customers()


@router.get("/leads")
def list_leads(
    product: str = Query(default="home_loan", description="Loan product type"),
    min_score: float = Query(default=0, description="Minimum intent score filter"),
    user: dict = Depends(require_role("rm")),
) -> list[dict[str, Any]]:
    """Return a ranked list of eligible leads, sorted by intent score descending.

    Loops over all customers, scores each using the existing capacity_model,
    intent_model, and eligibility_gate functions (same code path as /score),
    filters to gate_result == "pass" AND intent_score >= min_score, and
    returns sorted by intent_score descending.

    Performance: this scores every customer live on each request. Fine at
    prototype scale (dozens of customers). Should move to a precomputed/cached
    ranking (e.g. a nightly batch job writing to a leads table) before scaling
    past a few hundred customers.

    Args:
        product: Loan product type. Must be 'home_loan' (only supported product).
        min_score: Minimum intent score threshold (0–100). Leads below this
            are filtered out.
        user: Decoded JWT payload (injected by auth dependency).

    Returns:
        List of lead dicts sorted by intent_score descending.

    Raises:
        400: If product is not supported.
        503: If models are not trained.
    """
    # --- Product validation ---
    if product not in SUPPORTED_PRODUCTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Product '{product}' is not currently supported. "
                f"Only {', '.join(sorted(SUPPORTED_PRODUCTS))} "
                f"has trained models."
            ),
        )

    # --- Load models once (not per customer) ---
    cap_models = capacity_model.load_models()
    cap_manifest = capacity_model.load_manifest()
    if cap_models is None:
        raise HTTPException(
            status_code=503,
            detail="Capacity models not trained. Run: make train",
        )

    int_model = intent_model.load_model(product)
    int_manifest = intent_model.load_manifest(product)
    if int_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Intent model not trained for '{product}'. Run: make train",
        )

    # --- Get all customers ---
    customers = _get_all_customers()

    # --- Score each customer ---
    leads: list[dict[str, Any]] = []

    for cust in customers:
        cid = cust["customer_id"]
        name = cust["name"]

        try:
            features = get_feature_vector(cid)
        except ValueError:
            logger.debug("No features for %s — skipping.", cid)
            continue

        # Capacity scoring
        try:
            q10, q50, q90, cap_confidence = capacity_model.predict(
                features, models=cap_models, manifest=cap_manifest,
            )
        except Exception as e:
            logger.warning("Capacity prediction failed for %s: %s", cid, e)
            continue

        dti = features.get("current_dti", 0.0)

        # Intent scoring
        try:
            intent_score, int_confidence = intent_model.predict(
                features, product, model=int_model, manifest=int_manifest,
            )
        except Exception as e:
            logger.warning("Intent prediction failed for %s: %s", cid, e)
            continue

        # Eligibility gating
        account_vintage = int(features.get("income_source_tenure_months", 0))
        combined_confidence = min(cap_confidence, int_confidence)

        eligibility_result = check_eligibility(
            dti=dti,
            account_vintage_months=account_vintage,
            model_confidence=combined_confidence,
            product=product,
        )

        # Filter: only gate_result == "pass" AND intent_score >= min_score
        if eligibility_result.gate_result != "pass":
            continue
        if intent_score < min_score:
            continue

        leads.append({
            "customer_id": cid,
            "name": name,
            "intent_score": round(intent_score, 2),
            "estimated_income": round(q50, 2),
            "dti": round(dti, 4),
            "confidence": round(combined_confidence, 4),
            "gate_result": eligibility_result.gate_result,
        })

    # Sort by intent_score descending
    leads.sort(key=lambda x: x["intent_score"], reverse=True)

    # --- Audit log ---
    requesting_role = user.get("role", "unknown")
    _log_leads_request(product, min_score, len(leads), requesting_role)

    return leads
