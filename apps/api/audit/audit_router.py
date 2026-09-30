"""Audit trail endpoint for Cashflow IQ API.

Provides ``GET /audit/{customer_id}`` for admin-only access to the
scoring audit trail. Each entry shows model versions, gate results,
and a feature hash — never raw feature values.
"""

from typing import Any

from fastapi import APIRouter, Depends

from apps.api.audit.audit import get_audit_trail
from apps.api.auth.auth import require_role

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/{customer_id}")
def audit_trail(
    customer_id: str,
    user: dict = Depends(require_role("admin")),
) -> list[dict[str, Any]]:
    """Return the audit trail for a customer, most recent first.

    Admin-only endpoint. Each entry includes model versions, gate result,
    requesting role, feature hash, and timestamp.

    Args:
        customer_id: The customer whose audit trail to retrieve.
        user: Decoded JWT payload (injected by auth dependency).

    Returns:
        List of audit log entries ordered by timestamp descending.
    """
    return get_audit_trail(customer_id)
