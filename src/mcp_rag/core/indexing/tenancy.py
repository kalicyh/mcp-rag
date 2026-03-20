"""Tenant-aware collection naming utilities."""

from __future__ import annotations

import re

from .models import TenantContext

_COLLECTION_RE = re.compile(r"^u(\d+)_a(\d+)_(.+)$")
_USER_COLLECTION_RE = re.compile(r"^u(\d+)_(.+)$")
_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_collection_name(name: str) -> str:
    """Make a collection name safe for Chroma."""

    cleaned = _SANITIZE_RE.sub("_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "default"


def build_collection_name(
    base_collection: str = "default",
    user_id: int | None = None,
    agent_id: int | None = None,
) -> str:
    """Build the actual collection name for a tenant scope."""

    base = sanitize_collection_name(base_collection)
    if user_id is None:
        return base
    if agent_id is None:
        return f"u{user_id}_{base}"
    return f"u{user_id}_a{agent_id}_{base}"


def parse_collection_name(collection_name: str) -> TenantContext:
    """Parse a tenant-aware collection name."""

    match = _COLLECTION_RE.match(collection_name)
    if match:
        return TenantContext(
            base_collection=match.group(3),
            user_id=int(match.group(1)),
            agent_id=int(match.group(2)),
        )

    match = _USER_COLLECTION_RE.match(collection_name)
    if match:
        return TenantContext(
            base_collection=match.group(2),
            user_id=int(match.group(1)),
            agent_id=None,
        )

    return TenantContext(base_collection=collection_name)


def resolve_collection_name(
    collection_name: str = "default",
    tenant: TenantContext | None = None,
    user_id: int | None = None,
    agent_id: int | None = None,
) -> str:
    """Resolve the actual collection name for a request."""

    if tenant is not None:
        return build_collection_name(
            base_collection=tenant.base_collection or collection_name,
            user_id=tenant.user_id,
            agent_id=tenant.agent_id,
        )
    return build_collection_name(
        base_collection=collection_name,
        user_id=user_id,
        agent_id=agent_id,
    )

