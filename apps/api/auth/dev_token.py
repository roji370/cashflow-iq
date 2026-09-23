"""Dev-only JWT token minting endpoint.

Registers ``POST /auth/dev-token?role={role}`` **only** when the
``ENVIRONMENT`` env var is set to ``"local"``. If ``ENVIRONMENT`` is
anything else (or unset), this router is never created — the route
cannot be called at all in non-local environments.

This ensures a test-token endpoint can never accidentally ship active
in staging or production.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, status
from jose import jwt

logger = logging.getLogger(__name__)

_VALID_ROLES = {"rm", "admin"}


def create_dev_token_router() -> APIRouter | None:
    """Create the dev-token router if ENVIRONMENT=local.

    Returns:
        An APIRouter with the dev-token endpoint, or None if the
        environment is not 'local'.
    """
    environment = os.environ.get("ENVIRONMENT", "")
    if environment != "local":
        logger.info(
            "ENVIRONMENT=%r (not 'local') — dev-token endpoint disabled.",
            environment,
        )
        return None

    router = APIRouter(prefix="/auth", tags=["auth"])

    @router.post("/dev-token")
    def mint_dev_token(
        role: str = Query(
            ...,
            description="Role to embed in the token: 'rm' or 'admin'.",
        ),
    ) -> dict[str, str]:
        """Mint a JWT for local development/testing.

        Args:
            role: The role to assign ('rm' or 'admin').

        Returns:
            Dict with 'token' key containing the signed JWT.

        Raises:
            HTTPException: 400 if role is not valid.
        """
        if role not in _VALID_ROLES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid role '{role}'. Must be one of: {sorted(_VALID_ROLES)}",
            )

        secret = os.environ.get("JWT_SECRET")
        if not secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="JWT_SECRET is not configured.",
            )

        now = datetime.now(timezone.utc)
        payload = {
            "sub": f"dev-{role}",
            "role": role,
            "iat": now,
            "exp": now + timedelta(hours=24),
        }
        token = jwt.encode(payload, secret, algorithm="HS256")

        logger.info("Dev token minted for role=%s", role)
        return {"token": token}

    return router
