"""Admin config endpoint for gating thresholds.

Provides GET and PUT for ``/config/gating-thresholds``, both admin-only.
Uses the exact field names from ``gating_thresholds.yaml``:
``max_dti``, ``min_vintage_months``, ``min_confidence``.

On PUT, validates values, writes back to YAML, and logs each changed
field to ``config_change_log`` (separate table from ``audit_log`` —
see DECISIONS.md).
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional

import psycopg2
import yaml
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, field_validator

from apps.api.auth.auth import require_role

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["config"])

# Path to the gating thresholds YAML — same file eligibility_gate.py reads.
_THRESHOLDS_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "apps" / "ml" / "config" / "gating_thresholds.yaml"
)


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


def create_config_change_log_table() -> None:
    """Create the config_change_log table if it does not exist.

    Called from the API lifespan handler on startup.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS config_change_log (
        id              SERIAL PRIMARY KEY,
        changed_by_role TEXT NOT NULL,
        product         TEXT NOT NULL,
        field_name      TEXT NOT NULL,
        old_value       TEXT NOT NULL,
        new_value       TEXT NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_config_change_log_created
        ON config_change_log (created_at DESC);
    """
    try:
        conn = psycopg2.connect(_get_db_url())
        try:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
            logger.info("config_change_log table ensured.")
        finally:
            conn.close()
    except Exception as exc:
        logger.error("Failed to create config_change_log table: %s", exc)
        raise


def _log_config_change(
    changed_by_role: str,
    product: str,
    field_name: str,
    old_value: str,
    new_value: str,
) -> None:
    """Log a single field change to the config_change_log table.

    Args:
        changed_by_role: Role of the admin making the change.
        product: Product whose thresholds are being changed.
        field_name: Name of the changed field.
        old_value: Previous value (as string).
        new_value: New value (as string).
    """
    sql = """
    INSERT INTO config_change_log
        (changed_by_role, product, field_name, old_value, new_value)
    VALUES (%s, %s, %s, %s, %s)
    """
    try:
        conn = psycopg2.connect(_get_db_url())
        try:
            with conn.cursor() as cur:
                cur.execute(sql, (
                    changed_by_role, product, field_name, old_value, new_value,
                ))
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.error("Failed to log config change: %s", exc)


def _read_thresholds_yaml() -> dict[str, Any]:
    """Read the gating thresholds YAML file.

    Returns:
        The full YAML contents as a dict.

    Raises:
        HTTPException: 500 if the file cannot be read.
    """
    if not _THRESHOLDS_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gating thresholds file not found: {_THRESHOLDS_PATH}",
        )
    with open(_THRESHOLDS_PATH) as f:
        return yaml.safe_load(f)


def _write_thresholds_yaml(config: dict[str, Any]) -> None:
    """Write updated thresholds back to the YAML file.

    Preserves a header comment for readability.

    Args:
        config: The full config dict to write.
    """
    header = (
        "# Gating thresholds for eligibility rules.\n"
        "#\n"
        "# Each product has its own set of thresholds. The eligibility gate module\n"
        "# loads this file and checks each customer against these thresholds before\n"
        "# a lead is surfaced to the RM.\n"
        "#\n"
        "# These are configurable — no code change required to adjust thresholds.\n"
        "# Any change should be logged in docs/DECISIONS.md with rationale.\n\n"
    )
    with open(_THRESHOLDS_PATH, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ── Pydantic model for PUT validation ────────────────────────────────

class ThresholdUpdate(BaseModel):
    """Partial update schema for gating thresholds.

    Uses the exact field names from gating_thresholds.yaml:
    max_dti, min_vintage_months, min_confidence.
    All fields are optional (partial update supported).
    """

    max_dti: Optional[float] = None
    min_vintage_months: Optional[int] = None
    min_confidence: Optional[float] = None

    @field_validator("max_dti")
    @classmethod
    def validate_max_dti(cls, v: Optional[float]) -> Optional[float]:
        """Validate max_dti is between 0 and 1.

        Args:
            v: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValueError: If out of range.
        """
        if v is not None and not (0 <= v <= 1):
            raise ValueError(
                f"max_dti must be between 0 and 1, got {v}"
            )
        return v

    @field_validator("min_confidence")
    @classmethod
    def validate_min_confidence(cls, v: Optional[float]) -> Optional[float]:
        """Validate min_confidence is between 0 and 1.

        Args:
            v: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValueError: If out of range.
        """
        if v is not None and not (0 <= v <= 1):
            raise ValueError(
                f"min_confidence must be between 0 and 1, got {v}"
            )
        return v

    @field_validator("min_vintage_months")
    @classmethod
    def validate_min_vintage_months(cls, v: Optional[int]) -> Optional[int]:
        """Validate min_vintage_months is a positive integer.

        Args:
            v: The value to validate.

        Returns:
            The validated value.

        Raises:
            ValueError: If not positive.
        """
        if v is not None and v < 1:
            raise ValueError(
                f"min_vintage_months must be a positive integer, got {v}"
            )
        return v


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/gating-thresholds")
def get_gating_thresholds(
    user: dict = Depends(require_role("admin")),
) -> dict[str, Any]:
    """Return the current gating thresholds as JSON.

    Admin-only. Reads directly from the YAML file.

    Args:
        user: Decoded JWT payload (injected by auth dependency).

    Returns:
        The full thresholds config dict.
    """
    return _read_thresholds_yaml()


@router.put("/gating-thresholds")
def update_gating_thresholds(
    update: ThresholdUpdate,
    user: dict = Depends(require_role("admin")),
    product: str = "home_loan",
) -> dict[str, Any]:
    """Update gating thresholds (partial or full).

    Admin-only. Validates values, writes back to YAML, and logs each
    changed field to config_change_log.

    Args:
        update: The threshold fields to update (partial supported).
        user: Decoded JWT payload (injected by auth dependency).
        product: Product whose thresholds to update.

    Returns:
        The updated thresholds config dict.

    Raises:
        HTTPException: 422 for invalid values, 404 if product not found.
    """
    config = _read_thresholds_yaml()

    if product not in config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No thresholds configured for product '{product}'.",
        )

    current = config[product]
    role = user.get("role", "admin")

    # Apply each non-None field from the update
    update_dict = update.model_dump(exclude_none=True)

    for field_name, new_value in update_dict.items():
        old_value = current.get(field_name)
        if old_value != new_value:
            _log_config_change(
                changed_by_role=role,
                product=product,
                field_name=field_name,
                old_value=str(old_value),
                new_value=str(new_value),
            )
            current[field_name] = new_value
            logger.info(
                "Threshold updated: %s.%s: %s → %s (by %s)",
                product, field_name, old_value, new_value, role,
            )

    config[product] = current
    _write_thresholds_yaml(config)

    return config
