"""Tests for audit logging and the audit trail endpoint.

Validates:
- hash_features produces a deterministic SHA-256 hash, not raw values.
- log_score_request inserts exactly one row per call.
- GET /audit/{customer_id} returns 403 for rm, 200 for admin.
- Audit rows never contain raw feature values.
"""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from apps.api.audit.audit import hash_features
from apps.api.audit.audit_router import router as audit_router
from apps.api.auth.auth import require_role
from apps.api.auth.dev_token import create_dev_token_router

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
    """Build a test FastAPI app with audit routes.

    Returns:
        FastAPI application with audit router included.
    """
    test_app = FastAPI()
    test_app.include_router(audit_router)
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


# ── hash_features tests ──────────────────────────────────────────────

class TestHashFeatures:
    """Tests for the feature hashing function."""

    def test_deterministic_hash(self) -> None:
        """Same features should produce the same hash."""
        features = {"salary_amount_median": 50000.0, "current_dti": 0.35}
        h1 = hash_features(features)
        h2 = hash_features(features)
        assert h1 == h2

    def test_hash_is_sha256_hex(self) -> None:
        """Hash should be a 64-character hex string (SHA-256)."""
        features = {"salary_amount_median": 50000.0}
        h = hash_features(features)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_does_not_contain_raw_values(self) -> None:
        """The hash string should not contain any raw feature values."""
        features = {"salary_amount_median": 50000.0, "current_dti": 0.35}
        h = hash_features(features)
        assert "50000" not in h
        assert "0.35" not in h

    def test_different_features_different_hash(self) -> None:
        """Different feature vectors should produce different hashes."""
        h1 = hash_features({"a": 1.0})
        h2 = hash_features({"a": 2.0})
        assert h1 != h2

    def test_key_order_independent(self) -> None:
        """Hash should be the same regardless of dict insertion order."""
        h1 = hash_features({"a": 1.0, "b": 2.0})
        h2 = hash_features({"b": 2.0, "a": 1.0})
        assert h1 == h2


# ── Audit trail endpoint: role checks ────────────────────────────────

class TestAuditEndpointRoles:
    """Tests for role-based access to GET /audit/{customer_id}."""

    @patch("apps.api.audit.audit_router.get_audit_trail", return_value=[])
    def test_admin_can_read_audit_trail(
        self, mock_get: MagicMock, client: TestClient,
    ) -> None:
        """Admin token should get 200 on /audit/{customer_id}."""
        token = _make_token("admin")
        resp = client.get(
            "/audit/cust_001",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        mock_get.assert_called_once_with("cust_001")

    def test_rm_cannot_read_audit_trail(self, client: TestClient) -> None:
        """RM token should get 403 on /audit/{customer_id}."""
        token = _make_token("rm")
        resp = client.get(
            "/audit/cust_001",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_no_token_returns_401(self, client: TestClient) -> None:
        """Missing token should get 401 on /audit/{customer_id}."""
        resp = client.get("/audit/cust_001")
        assert resp.status_code == 401


# ── log_score_request tests ──────────────────────────────────────────

class TestLogScoreRequest:
    """Tests for the audit log insertion function."""

    @patch("apps.api.audit.audit.psycopg2")
    def test_inserts_exactly_one_row(self, mock_pg: MagicMock) -> None:
        """log_score_request should execute exactly one INSERT."""
        from apps.api.audit.audit import log_score_request

        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        log_score_request(
            customer_id="cust_001",
            product="home_loan",
            capacity_model_version="0.1.0",
            intent_model_version="0.1.0",
            gate_result="pass",
            requesting_role="rm",
            input_feature_hash="abc123",
        )

        assert mock_cursor.execute.call_count == 1
        call_args = mock_cursor.execute.call_args
        sql = call_args[0][0]
        assert "INSERT INTO audit_log" in sql
        params = call_args[0][1]
        assert params[0] == "cust_001"
        assert params[4] == "pass"
        mock_conn.commit.assert_called_once()

    @patch("apps.api.audit.audit.psycopg2")
    def test_row_does_not_contain_raw_features(self, mock_pg: MagicMock) -> None:
        """The inserted row should contain a hash, not raw feature values."""
        from apps.api.audit.audit import log_score_request

        mock_conn = MagicMock()
        mock_pg.connect.return_value = mock_conn
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = lambda s: mock_cursor
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        feature_hash = hash_features({"salary_amount_median": 50000.0})

        log_score_request(
            customer_id="cust_001",
            product="home_loan",
            capacity_model_version="0.1.0",
            intent_model_version="0.1.0",
            gate_result="fail",
            requesting_role="admin",
            input_feature_hash=feature_hash,
        )

        call_args = mock_cursor.execute.call_args
        params = call_args[0][1]
        # The feature hash param (index 6) should be a SHA-256 hex, not raw values
        assert len(params[6]) == 64
        assert "50000" not in params[6]
