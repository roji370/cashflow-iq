"""Tests for customer enumeration and ranked lead list endpoints.

Validates:
- GET /customers returns customer list, requires rm role.
- GET /leads returns only gate_result=="pass" customers, sorted by intent_score desc.
- GET /leads respects min_score filtering.
- GET /leads returns 400 for unsupported products.
- Role checks (rm can access, no-token gets 401).
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from apps.api.leads.leads_router import router as leads_router

# ── Fixtures ──────────────────────────────────────────────────────────

TEST_JWT_SECRET = "test-secret-do-not-use"


@pytest.fixture(autouse=True)
def _set_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set required environment variables for every test.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    """
    monkeypatch.setenv("JWT_SECRET", TEST_JWT_SECRET)
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost:5432/test")


def _make_token(role: str) -> str:
    """Mint a JWT for testing.

    Args:
        role: Role claim to embed.

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": f"test-{role}",
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=24),
    }
    return jwt.encode(payload, TEST_JWT_SECRET, algorithm="HS256")


@pytest.fixture()
def app() -> FastAPI:
    """Build a test FastAPI app with leads routes.

    Returns:
        FastAPI application with leads router included.
    """
    test_app = FastAPI()
    test_app.include_router(leads_router)
    return test_app


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    """Create a test client.

    Args:
        app: The test FastAPI application.

    Returns:
        TestClient instance.
    """
    return TestClient(app)


# ── Mock data ─────────────────────────────────────────────────────────

MOCK_CUSTOMERS = [
    {"customer_id": "cust_001", "name": "Alice"},
    {"customer_id": "cust_002", "name": "Bob"},
    {"customer_id": "cust_003", "name": "Charlie"},
]

MOCK_FEATURES = {
    "cust_001": {"current_dti": 0.30, "income_source_tenure_months": 12.0},
    "cust_002": {"current_dti": 0.80, "income_source_tenure_months": 12.0},
    "cust_003": {"current_dti": 0.25, "income_source_tenure_months": 12.0},
}


def _mock_get_feature_vector(customer_id: str) -> dict[str, float]:
    """Mock feature vector lookup.

    Args:
        customer_id: Customer to look up.

    Returns:
        Feature dict.

    Raises:
        ValueError: If customer not found.
    """
    if customer_id not in MOCK_FEATURES:
        raise ValueError(f"No features for {customer_id}")
    return MOCK_FEATURES[customer_id]


# ── GET /customers tests ─────────────────────────────────────────────

class TestListCustomers:
    """Tests for the GET /customers endpoint."""

    @patch("apps.api.leads.leads_router._get_all_customers", return_value=MOCK_CUSTOMERS)
    def test_rm_can_list_customers(
        self, mock_get: MagicMock, client: TestClient,
    ) -> None:
        """RM should get 200 and a customer list."""
        token = _make_token("rm")
        resp = client.get(
            "/customers",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["customer_id"] == "cust_001"
        assert data[0]["name"] == "Alice"

    def test_no_token_returns_401(self, client: TestClient) -> None:
        """Missing token should get 401."""
        resp = client.get("/customers")
        assert resp.status_code == 401


# ── GET /leads tests ──────────────────────────────────────────────────


class TestListLeads:
    """Tests for the GET /leads endpoint."""

    def test_leads_only_returns_passing_customers(self, client: TestClient) -> None:
        """Only gate_result=='pass' customers should appear in leads."""
        from packages.schemas.score import EligibilityResult

        def mock_cap_predict(features: dict, models: Any = None, manifest: Any = None) -> tuple:
            """Mock capacity prediction."""
            return (10000.0, 50000.0, 90000.0, 0.75)

        def mock_int_predict(features: dict, product: str, model: Any = None, manifest: Any = None) -> tuple:
            """Mock intent — score based on DTI to differentiate customers."""
            dti = features.get("current_dti", 0.0)
            score = 80.0 if dti == 0.30 else (60.0 if dti < 0.5 else 50.0)
            return (score, 0.75)

        def mock_check(dti: float, **kwargs: Any) -> EligibilityResult:
            """Mock eligibility — fail high DTI."""
            if dti > 0.5:
                return EligibilityResult(
                    gate_result="fail",
                    rules_checked=["dti"],
                    rules_failed=["DTI too high"],
                )
            return EligibilityResult(
                gate_result="pass",
                rules_checked=["dti", "vintage", "confidence"],
                rules_failed=[],
            )

        with patch("apps.api.leads.leads_router._get_all_customers", return_value=MOCK_CUSTOMERS), \
             patch("apps.api.leads.leads_router.get_feature_vector", side_effect=_mock_get_feature_vector), \
             patch("apps.api.leads.leads_router.capacity_model.load_models", return_value={"q0.5": MagicMock()}), \
             patch("apps.api.leads.leads_router.capacity_model.load_manifest", return_value={"model_version": "test", "feature_names": []}), \
             patch("apps.api.leads.leads_router.intent_model.load_model", return_value=MagicMock()), \
             patch("apps.api.leads.leads_router.intent_model.load_manifest", return_value={"model_version": "test", "feature_names": []}), \
             patch("apps.api.leads.leads_router.capacity_model.predict", side_effect=mock_cap_predict), \
             patch("apps.api.leads.leads_router.intent_model.predict", side_effect=mock_int_predict), \
             patch("apps.api.leads.leads_router.check_eligibility", side_effect=mock_check), \
             patch("apps.api.leads.leads_router._log_leads_request"):

            token = _make_token("rm")
            resp = client.get(
                "/leads?product=home_loan&min_score=0",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        data = resp.json()
        # cust_002 (high DTI, gate_result=fail) should be filtered out
        customer_ids = [d["customer_id"] for d in data]
        assert "cust_002" not in customer_ids
        assert "cust_001" in customer_ids
        assert "cust_003" in customer_ids
        # Sorted by intent_score desc: cust_001 (80) before cust_003 (60)
        assert data[0]["customer_id"] == "cust_001"
        assert data[0]["intent_score"] >= data[1]["intent_score"]

    def test_leads_respects_min_score(self, client: TestClient) -> None:
        """Leads below min_score should be filtered out."""
        from packages.schemas.score import EligibilityResult

        with patch("apps.api.leads.leads_router._get_all_customers", return_value=MOCK_CUSTOMERS), \
             patch("apps.api.leads.leads_router.get_feature_vector", side_effect=_mock_get_feature_vector), \
             patch("apps.api.leads.leads_router.capacity_model.load_models", return_value={"q0.5": MagicMock()}), \
             patch("apps.api.leads.leads_router.capacity_model.load_manifest", return_value={"model_version": "test"}), \
             patch("apps.api.leads.leads_router.intent_model.load_model", return_value=MagicMock()), \
             patch("apps.api.leads.leads_router.intent_model.load_manifest", return_value={"model_version": "test"}), \
             patch("apps.api.leads.leads_router.capacity_model.predict", return_value=(10000, 50000, 90000, 0.75)), \
             patch("apps.api.leads.leads_router.intent_model.predict", return_value=(40.0, 0.75)), \
             patch("apps.api.leads.leads_router.check_eligibility", return_value=EligibilityResult(
                 gate_result="pass", rules_checked=["dti"], rules_failed=[])), \
             patch("apps.api.leads.leads_router._log_leads_request"):

            token = _make_token("rm")
            resp = client.get(
                "/leads?product=home_loan&min_score=50",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200
        # All customers score 40, min_score=50 → empty
        assert len(resp.json()) == 0

    def test_unsupported_product_returns_400(self, client: TestClient) -> None:
        """Requesting a non-home_loan product should return 400."""
        token = _make_token("rm")
        resp = client.get(
            "/leads?product=auto_loan",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 400
        assert "not currently supported" in resp.json()["detail"]

    def test_no_token_returns_401(self, client: TestClient) -> None:
        """Missing token should get 401."""
        resp = client.get("/leads?product=home_loan")
        assert resp.status_code == 401
