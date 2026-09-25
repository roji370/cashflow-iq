"""Tests for JWT auth and role-based access control.

Self-contained: registers a throwaway ``/_test/admin-only`` route directly
in the test app so the 403 check is testable in D1 without forward-
referencing D2/D3 endpoints that don't exist yet.
"""

import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from jose import jwt

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


def _make_token(role: str, expired: bool = False) -> str:
    """Mint a JWT for testing.

    Args:
        role: Role claim to embed.
        expired: If True, the token is already expired.

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": f"test-{role}",
        "role": role,
        "iat": now,
        "exp": now + timedelta(hours=-1 if expired else 24),
    }
    return jwt.encode(payload, TEST_JWT_SECRET, algorithm="HS256")


@pytest.fixture()
def app() -> FastAPI:
    """Build a test FastAPI app with auth-protected endpoints.

    Returns:
        FastAPI application with /health, /_test/rm-only,
        /_test/admin-only, and /auth/dev-token routes.
    """
    test_app = FastAPI()

    @test_app.get("/health")
    def health() -> dict[str, str]:
        """Unprotected health check."""
        return {"status": "ok"}

    @test_app.get("/_test/rm-only")
    def rm_only(user: dict = Depends(require_role("rm"))) -> dict:
        """Test endpoint requiring rm role.

        Args:
            user: Decoded token payload from auth dependency.
        """
        return {"role": user.get("role"), "ok": True}

    @test_app.get("/_test/admin-only")
    def admin_only(user: dict = Depends(require_role("admin"))) -> dict:
        """Test endpoint requiring admin role.

        Args:
            user: Decoded token payload from auth dependency.
        """
        return {"role": user.get("role"), "ok": True}

    # Register dev token router
    dev_router = create_dev_token_router()
    if dev_router:
        test_app.include_router(dev_router)

    return test_app


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    """Create a test client from the test app.

    Args:
        app: The test FastAPI application.

    Returns:
        TestClient instance.
    """
    return TestClient(app)


# ── Auth: missing / invalid tokens → 401 ─────────────────────────────

class TestMissingToken:
    """Tests for requests without an auth token."""

    def test_rm_endpoint_no_token_returns_401(self, client: TestClient) -> None:
        """GET /_test/rm-only without token should return 401."""
        resp = client.get("/_test/rm-only")
        assert resp.status_code == 401

    def test_admin_endpoint_no_token_returns_401(self, client: TestClient) -> None:
        """GET /_test/admin-only without token should return 401."""
        resp = client.get("/_test/admin-only")
        assert resp.status_code == 401


class TestInvalidToken:
    """Tests for requests with malformed or expired tokens."""

    def test_garbage_token_returns_401(self, client: TestClient) -> None:
        """A random string as Bearer token should return 401."""
        resp = client.get(
            "/_test/rm-only",
            headers={"Authorization": "Bearer not-a-real-jwt"},
        )
        assert resp.status_code == 401

    def test_expired_token_returns_401(self, client: TestClient) -> None:
        """An expired JWT should return 401."""
        token = _make_token("rm", expired=True)
        resp = client.get(
            "/_test/rm-only",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401

    def test_wrong_secret_returns_401(self, client: TestClient) -> None:
        """A JWT signed with the wrong secret should return 401."""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": "test-rm",
            "role": "rm",
            "iat": now,
            "exp": now + timedelta(hours=24),
        }
        bad_token = jwt.encode(payload, "wrong-secret", algorithm="HS256")
        resp = client.get(
            "/_test/rm-only",
            headers={"Authorization": f"Bearer {bad_token}"},
        )
        assert resp.status_code == 401


# ── Auth: valid tokens with correct roles → 200 ──────────────────────

class TestValidAccess:
    """Tests for requests with valid tokens and sufficient roles."""

    def test_rm_token_on_rm_route_returns_200(self, client: TestClient) -> None:
        """RM token should access rm-only endpoint."""
        token = _make_token("rm")
        resp = client.get(
            "/_test/rm-only",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "rm"

    def test_admin_token_on_rm_route_returns_200(self, client: TestClient) -> None:
        """Admin token should also access rm-only endpoint (role hierarchy)."""
        token = _make_token("admin")
        resp = client.get(
            "/_test/rm-only",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"

    def test_admin_token_on_admin_route_returns_200(self, client: TestClient) -> None:
        """Admin token should access admin-only endpoint."""
        token = _make_token("admin")
        resp = client.get(
            "/_test/admin-only",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["role"] == "admin"


# ── Auth: insufficient role → 403 ────────────────────────────────────

class TestInsufficientRole:
    """Tests for requests with valid tokens but insufficient role level."""

    def test_rm_token_on_admin_route_returns_403(self, client: TestClient) -> None:
        """RM token should be forbidden from admin-only endpoint."""
        token = _make_token("rm")
        resp = client.get(
            "/_test/admin-only",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403
        assert "rm" in resp.json()["detail"]

    def test_unknown_role_on_rm_route_returns_403(self, client: TestClient) -> None:
        """A token with an unknown role should be forbidden."""
        token = _make_token("viewer")
        resp = client.get(
            "/_test/rm-only",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403


# ── Health endpoint stays unauthenticated ─────────────────────────────

class TestHealthNoAuth:
    """Confirm /health is accessible without any token."""

    def test_health_no_token_returns_200(self, client: TestClient) -> None:
        """GET /health should return 200 without auth."""
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── Dev token endpoint ────────────────────────────────────────────────

class TestDevToken:
    """Tests for the dev-only token minting endpoint."""

    def test_mint_rm_token(self, client: TestClient) -> None:
        """POST /auth/dev-token?role=rm should return a valid JWT."""
        resp = client.post("/auth/dev-token?role=rm")
        assert resp.status_code == 200
        token = resp.json()["token"]

        # Decode it and verify claims
        payload = jwt.decode(token, TEST_JWT_SECRET, algorithms=["HS256"])
        assert payload["role"] == "rm"
        assert payload["sub"] == "dev-rm"

    def test_mint_admin_token(self, client: TestClient) -> None:
        """POST /auth/dev-token?role=admin should return a valid JWT."""
        resp = client.post("/auth/dev-token?role=admin")
        assert resp.status_code == 200
        token = resp.json()["token"]
        payload = jwt.decode(token, TEST_JWT_SECRET, algorithms=["HS256"])
        assert payload["role"] == "admin"

    def test_invalid_role_returns_400(self, client: TestClient) -> None:
        """POST /auth/dev-token?role=hacker should return 400."""
        resp = client.post("/auth/dev-token?role=hacker")
        assert resp.status_code == 400


class TestDevTokenDisabled:
    """Confirm dev-token is not registered when ENVIRONMENT != local."""

    def test_non_local_env_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """create_dev_token_router() returns None in non-local environments."""
        monkeypatch.setenv("ENVIRONMENT", "staging")
        router = create_dev_token_router()
        assert router is None

    def test_unset_env_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """create_dev_token_router() returns None when ENVIRONMENT is unset."""
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        router = create_dev_token_router()
        assert router is None
