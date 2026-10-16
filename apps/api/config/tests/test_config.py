"""Tests for admin config endpoint (gating thresholds).

Validates:
- Valid PUT updates the YAML and is reflected in subsequent GET.
- Invalid values (e.g., max_dti > 1) return 422.
- Non-admin tokens get 403.
- Threshold changes are logged to config_change_log.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jose import jwt

from apps.api.config.config_router import router as config_router

# ── Fixtures ──────────────────────────────────────────────────────────

TEST_JWT_SECRET = "test-secret-do-not-use"

SAMPLE_YAML = {
    "home_loan": {
        "max_dti": 0.50,
        "min_vintage_months": 6,
        "min_confidence": 0.30,
    }
}


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
def tmp_yaml(tmp_path: Path) -> Path:
    """Create a temporary gating_thresholds.yaml for testing.

    Args:
        tmp_path: pytest tmp_path fixture.

    Returns:
        Path to the temporary YAML file.
    """
    yaml_path = tmp_path / "gating_thresholds.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(SAMPLE_YAML, f)
    return yaml_path


@pytest.fixture()
def app(tmp_yaml: Path) -> FastAPI:
    """Build a test FastAPI app with config routes and patched YAML path.

    Args:
        tmp_yaml: Path to the temporary YAML file.

    Returns:
        FastAPI application with config router.
    """
    test_app = FastAPI()
    test_app.include_router(config_router)
    return test_app


@pytest.fixture()
def client(app: FastAPI, tmp_yaml: Path) -> TestClient:
    """Create a test client with patched YAML path.

    Args:
        app: The test FastAPI application.
        tmp_yaml: Path to temporary YAML.

    Returns:
        TestClient instance.
    """
    with patch("apps.api.config.config_router._THRESHOLDS_PATH", tmp_yaml):
        yield TestClient(app)


# ── Role-based access ────────────────────────────────────────────────

class TestConfigRoles:
    """Tests for role-based access to config endpoints."""

    def test_rm_cannot_get_thresholds(self, client: TestClient) -> None:
        """RM token should get 403 on GET /config/gating-thresholds."""
        token = _make_token("rm")
        resp = client.get(
            "/config/gating-thresholds",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_rm_cannot_put_thresholds(self, client: TestClient) -> None:
        """RM token should get 403 on PUT /config/gating-thresholds."""
        token = _make_token("rm")
        resp = client.put(
            "/config/gating-thresholds",
            json={"max_dti": 0.45},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 403

    def test_no_token_returns_401(self, client: TestClient) -> None:
        """Missing token should get 401."""
        resp = client.get("/config/gating-thresholds")
        assert resp.status_code == 401

    def test_admin_can_get_thresholds(self, client: TestClient) -> None:
        """Admin token should get 200 on GET /config/gating-thresholds."""
        token = _make_token("admin")
        resp = client.get(
            "/config/gating-thresholds",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "home_loan" in data
        assert data["home_loan"]["max_dti"] == 0.50


# ── Valid PUT updates ─────────────────────────────────────────────────

class TestValidPut:
    """Tests for valid PUT requests updating thresholds."""

    @patch("apps.api.config.config_router._log_config_change")
    def test_put_updates_yaml_and_reflects_in_get(
        self, mock_log: MagicMock, client: TestClient,
    ) -> None:
        """PUT should update the file and subsequent GET reflects the change."""
        token = _make_token("admin")

        # PUT new value
        resp = client.put(
            "/config/gating-thresholds",
            json={"max_dti": 0.45},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["home_loan"]["max_dti"] == 0.45

        # GET should reflect the new value
        resp = client.get(
            "/config/gating-thresholds",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["home_loan"]["max_dti"] == 0.45

    @patch("apps.api.config.config_router._log_config_change")
    def test_partial_update_preserves_other_fields(
        self, mock_log: MagicMock, client: TestClient,
    ) -> None:
        """Updating one field should not change others."""
        token = _make_token("admin")

        resp = client.put(
            "/config/gating-thresholds",
            json={"min_confidence": 0.50},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()["home_loan"]
        assert data["min_confidence"] == 0.50
        assert data["max_dti"] == 0.50  # unchanged
        assert data["min_vintage_months"] == 6  # unchanged

    @patch("apps.api.config.config_router._log_config_change")
    def test_change_is_logged(
        self, mock_log: MagicMock, client: TestClient,
    ) -> None:
        """PUT should log each changed field to config_change_log."""
        token = _make_token("admin")

        client.put(
            "/config/gating-thresholds",
            json={"max_dti": 0.40, "min_confidence": 0.25},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert mock_log.call_count == 2
        calls = mock_log.call_args_list

        # Check both fields were logged with correct old/new values
        logged_fields = {c.kwargs.get("field_name", c[1] if len(c[0]) > 2 else None) for c in calls}
        # Use positional args
        field_names = {c[1]["field_name"] if isinstance(c[1], dict) else c[0][2] for c in calls}
        assert "max_dti" in field_names
        assert "min_confidence" in field_names

    @patch("apps.api.config.config_router._log_config_change")
    def test_no_log_when_value_unchanged(
        self, mock_log: MagicMock, client: TestClient,
    ) -> None:
        """PUT with same values should not log any changes."""
        token = _make_token("admin")

        client.put(
            "/config/gating-thresholds",
            json={"max_dti": 0.50},  # same as initial
            headers={"Authorization": f"Bearer {token}"},
        )

        mock_log.assert_not_called()


# ── Invalid values → 422 ─────────────────────────────────────────────

class TestInvalidValues:
    """Tests for validation of PUT request bodies."""

    def test_max_dti_above_1_returns_422(self, client: TestClient) -> None:
        """max_dti > 1 should be rejected."""
        token = _make_token("admin")
        resp = client.put(
            "/config/gating-thresholds",
            json={"max_dti": 1.5},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

    def test_max_dti_below_0_returns_422(self, client: TestClient) -> None:
        """max_dti < 0 should be rejected."""
        token = _make_token("admin")
        resp = client.put(
            "/config/gating-thresholds",
            json={"max_dti": -0.1},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

    def test_min_confidence_above_1_returns_422(self, client: TestClient) -> None:
        """min_confidence > 1 should be rejected."""
        token = _make_token("admin")
        resp = client.put(
            "/config/gating-thresholds",
            json={"min_confidence": 2.0},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

    def test_min_vintage_months_zero_returns_422(self, client: TestClient) -> None:
        """min_vintage_months = 0 should be rejected (must be positive)."""
        token = _make_token("admin")
        resp = client.put(
            "/config/gating-thresholds",
            json={"min_vintage_months": 0},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

    def test_min_vintage_months_negative_returns_422(self, client: TestClient) -> None:
        """min_vintage_months < 0 should be rejected."""
        token = _make_token("admin")
        resp = client.put(
            "/config/gating-thresholds",
            json={"min_vintage_months": -3},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422
