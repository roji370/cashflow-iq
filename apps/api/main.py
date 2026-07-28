"""Minimal FastAPI app for the walking skeleton.

Two endpoints:
  GET /health          → {"status": "ok"}
  GET /score/{cid}     → combined capacity + intent score

Phase A only — no auth, no registry, no SHAP.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from apps.ml.models.trivial_scoring import get_capacity_score, get_intent_score

app = FastAPI(title="Cashflow IQ API", version="0.1.0-skeleton")

# Allow dashboard (localhost:5173) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Basic liveness check."""
    return {"status": "ok"}


@app.get("/score/{customer_id}")
def score(customer_id: str, product: str = "home_loan") -> dict:
    """Return combined capacity and intent scores for a customer.

    Returns 404 if the customer has no features in the database.
    """
    try:
        capacity = get_capacity_score(customer_id)
        intent = get_intent_score(customer_id, product)
    except ValueError:
        raise HTTPException(
            status_code=404,
            detail=f"Customer '{customer_id}' not found in features table.",
        )

    return {
        "customer_id": customer_id,
        "capacity": capacity.model_dump(),
        "intent": intent.model_dump(),
    }
