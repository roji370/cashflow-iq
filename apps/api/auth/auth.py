"""JWT-based role-based access control for Cashflow IQ API.

Provides a ``require_role`` FastAPI dependency factory that validates
Bearer JWT tokens and enforces role-based access. Two roles are defined:

- **rm** (relationship manager): can call scoring endpoints.
- **admin**: can call scoring, audit, and config endpoints.

JWT secret is read from the ``JWT_SECRET`` environment variable — never
hardcoded. Tokens are HS256-signed and include ``sub`` (subject),
``role``, ``exp`` (expiry), and ``iat`` (issued-at) claims.
"""

import logging
import os
from typing import Callable

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# Role hierarchy — higher roles inherit access from lower ones.
# "admin" can do everything "rm" can do, plus admin-only routes.
_ROLE_HIERARCHY: dict[str, int] = {
    "rm": 1,
    "admin": 2,
}

_bearer_scheme = HTTPBearer(auto_error=False)


def _get_jwt_secret() -> str:
    """Read the JWT signing secret from environment.

    Returns:
        The JWT secret string.

    Raises:
        RuntimeError: If JWT_SECRET is not set.
    """
    secret = os.environ.get("JWT_SECRET")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET environment variable is not set. "
            "Set it in .env or your deployment config."
        )
    return secret


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token.

    Args:
        token: The raw JWT string.

    Returns:
        The decoded payload dict.

    Raises:
        HTTPException: 401 if the token is invalid, expired, or malformed.
    """
    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=["HS256"])
        return payload
    except JWTError as exc:
        logger.warning("JWT decode failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(role: str) -> Callable:
    """Create a FastAPI dependency that enforces a minimum role.

    Usage::

        @app.get("/admin-thing", dependencies=[Depends(require_role("admin"))])
        def admin_thing(): ...

        @app.get("/score/{cid}")
        def score(cid: str, user=Depends(require_role("rm"))): ...

    Args:
        role: The minimum role required (``"rm"`` or ``"admin"``).

    Returns:
        A FastAPI dependency function that returns the decoded token payload.
    """
    required_level = _ROLE_HIERARCHY.get(role, 0)

    async def _dependency(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    ) -> dict:
        """Validate the Bearer token and check role level.

        Args:
            credentials: Extracted from the Authorization header by HTTPBearer.

        Returns:
            Decoded JWT payload dict.

        Raises:
            HTTPException: 401 if no/invalid token, 403 if role insufficient.
        """
        if credentials is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

        payload = decode_token(credentials.credentials)

        token_role = payload.get("role", "")
        token_level = _ROLE_HIERARCHY.get(token_role, 0)

        if token_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{token_role}' does not have access. "
                       f"Requires '{role}' or higher.",
            )

        return payload

    return _dependency
