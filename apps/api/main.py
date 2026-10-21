"""Cashflow IQ API — scoring with auth, audit logging, and admin config.

Phase E: adds customer enumeration and ranked lead list endpoints on top of
Phase D's auth, audit, and config infrastructure.

Endpoints:
    GET  /health                    → {"status": "ok"} (unauthenticated)
    POST /auth/dev-token?role=...   → {"token": "..."} (local env only)
    GET  /score/{customer_id}       → ScoreResponse (requires rm role)
    GET  /customers                 → customer list (requires rm role)
    GET  /leads?product=...&min_score=... → ranked leads (requires rm role)
    GET  /audit/{customer_id}       → audit trail (requires admin role)
    GET  /config/gating-thresholds  → current thresholds (requires admin role)
    PUT  /config/gating-thresholds  → update thresholds (requires admin role)
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from apps.api.audit.audit import (
    create_audit_table,
    hash_features,
    log_score_request,
)
from apps.api.audit.audit_router import router as audit_router
from apps.api.auth.auth import require_role
from apps.api.auth.dev_token import create_dev_token_router
from apps.api.config.config_router import (
    create_config_change_log_table,
    router as config_router,
)
from apps.api.leads.leads_router import router as leads_router
from apps.ml.explain.shap_explainer import explain
from apps.ml.models import capacity_model, intent_model
from apps.ml.rules.eligibility_gate import check_eligibility
from apps.pipelines.feature_store.run_nightly_features import get_feature_vector
from packages.schemas.score import (
    CapacityScoreResult,
    EligibilityResult,
    IntentScoreResult,
    ReasonCode,
    ScoreResponse,
)

logger = logging.getLogger(__name__)


# ── Lifespan: create tables on startup ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler — runs on startup and shutdown.

    On startup: ensures audit_log and config_change_log tables exist.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to the application.
    """
    logger.info("Cashflow IQ API starting up — ensuring audit tables...")
    try:
        create_audit_table()
        create_config_change_log_table()
        logger.info("Audit tables ready.")
    except Exception as exc:
        logger.error("Failed to create audit tables on startup: %s", exc)
        # Don't prevent startup — tables may already exist from SQL init
    yield
    logger.info("Cashflow IQ API shutting down.")


app = FastAPI(
    title="Cashflow IQ API",
    version="0.4.0-phase-e",
    lifespan=lifespan,
)

# Allow dashboard (localhost:5173) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ──────────────────────────────────────────────────

# Dev token endpoint (local environment only)
dev_router = create_dev_token_router()
if dev_router is not None:
    app.include_router(dev_router)

# Audit trail endpoint (admin only)
app.include_router(audit_router)

# Config endpoint (admin only)
app.include_router(config_router)

# Customer enumeration and leads (rm role)
app.include_router(leads_router)

# Supported products (only those with trained models)
SUPPORTED_PRODUCTS = {"home_loan"}


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness check — unauthenticated.

    Stays unprotected for load balancer / orchestration health checks.
    """
    return {"status": "ok"}


@app.get("/score/{customer_id}")
def score(
    customer_id: str,
    product: str = Query(default="home_loan", description="Loan product type"),
    user: dict = Depends(require_role("rm")),
) -> ScoreResponse:
    """Return combined capacity, intent, eligibility, and SHAP scores.

    Requires at least 'rm' role (both rm and admin can call this).
    Each successful call is logged to the audit_log table.

    Args:
        customer_id: The customer to score.
        product: Loan product type.
        user: Decoded JWT payload (injected by auth dependency).

    Returns:
        ScoreResponse envelope with all scoring components.

    Raises:
        400: If the product is not supported (no trained models).
        401: If no valid Bearer token is provided.
        404: If the customer has no features in the database.
        503: If models haven't been trained yet.
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

    # --- Load features ---
    try:
        features = get_feature_vector(customer_id)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{customer_id}' not found in features table.",
        )

    # --- Capacity scoring ---
    cap_models = capacity_model.load_models()
    cap_manifest = capacity_model.load_manifest()
    if cap_models is None:
        raise HTTPException(
            status_code=503,
            detail="Capacity models not trained. Run: make train",
        )

    try:
        q10, q50, q90, cap_confidence = capacity_model.predict(
            features, models=cap_models, manifest=cap_manifest,
        )
    except Exception as e:
        logger.error("Capacity prediction failed for %s: %s", customer_id, e)
        raise HTTPException(status_code=500, detail="Capacity scoring failed.")

    dti = features.get("current_dti", 0.0)

    capacity_result = CapacityScoreResult(
        customer_id=customer_id,
        estimated_income=round(q50, 2),
        estimated_income_q10=round(q10, 2),
        estimated_income_q90=round(q90, 2),
        confidence=cap_confidence,
        dti=round(dti, 4),
    )

    # --- Intent scoring ---
    int_model = intent_model.load_model(product)
    int_manifest = intent_model.load_manifest(product)
    if int_model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Intent model not trained for '{product}'. Run: make train",
        )

    try:
        intent_score, int_confidence = intent_model.predict(
            features, product, model=int_model, manifest=int_manifest,
        )
    except Exception as e:
        logger.error("Intent prediction failed for %s: %s", customer_id, e)
        raise HTTPException(status_code=500, detail="Intent scoring failed.")

    intent_result = IntentScoreResult(
        customer_id=customer_id,
        product=product,
        intent_score=intent_score,
        confidence=int_confidence,
    )

    # --- Eligibility gating ---
    account_vintage = int(features.get("income_source_tenure_months", 0))
    combined_confidence = min(cap_confidence, int_confidence)

    eligibility_result = check_eligibility(
        dti=dti,
        account_vintage_months=account_vintage,
        model_confidence=combined_confidence,
        product=product,
    )

    # --- SHAP explainability ---
    reason_codes: list[ReasonCode] = []

    # Intent SHAP (use base model for TreeExplainer)
    int_base = intent_model.load_base_model(product)
    if int_base is not None and int_manifest is not None:
        try:
            intent_reasons = explain(
                int_base, features, int_manifest["feature_names"],
                top_n=5, model_type="intent",
            )
            reason_codes.extend(intent_reasons)
        except Exception as e:
            logger.warning("Intent SHAP failed: %s", e)

    # Capacity SHAP
    if cap_manifest is not None:
        cap_q50_model = cap_models.get("q0.5")
        if cap_q50_model is not None:
            try:
                cap_reasons = explain(
                    cap_q50_model, features, cap_manifest["feature_names"],
                    top_n=3, model_type="capacity",
                )
                reason_codes.extend(cap_reasons)
            except Exception as e:
                logger.warning("Capacity SHAP failed: %s", e)

    # Deduplicate by feature name, keep highest |SHAP|
    seen: dict[str, ReasonCode] = {}
    for rc in reason_codes:
        if rc.feature_name not in seen or abs(rc.shap_value) > abs(seen[rc.feature_name].shap_value):
            seen[rc.feature_name] = rc
    reason_codes = sorted(seen.values(), key=lambda r: abs(r.shap_value), reverse=True)[:5]

    # --- Model version ---
    cap_version = cap_manifest.get("model_version", "unknown") if cap_manifest else "unknown"
    int_version = int_manifest.get("model_version", "unknown") if int_manifest else "unknown"

    # --- Audit logging (after score, before response) ---
    gate_result_str = eligibility_result.gate_result
    requesting_role = user.get("role", "unknown")
    feature_hash = hash_features(features)

    log_score_request(
        customer_id=customer_id,
        product=product,
        capacity_model_version=cap_version,
        intent_model_version=int_version,
        gate_result=gate_result_str,
        requesting_role=requesting_role,
        input_feature_hash=feature_hash,
    )

    return ScoreResponse(
        customer_id=customer_id,
        product=product,
        capacity=capacity_result,
        intent=intent_result,
        eligibility=eligibility_result,
        reason_codes=reason_codes,
        model_version=cap_version,
    )
