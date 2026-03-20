"""Shared contracts for the MCP-RAG shell layer.

This module keeps HTTP and MCP adapters on the same request/response shapes
while normalizing tenant scope to the new core.indexing model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from mcp_rag.core.indexing.models import TenantContext as CoreTenantContext


@dataclass(slots=True)
class TenantSpec:
    """Tenant scope used by the shell layer."""

    base_collection: str = "default"
    user_id: int | None = None
    agent_id: int | None = None

    def to_core(self) -> CoreTenantContext:
        return CoreTenantContext(
            base_collection=self.base_collection or "default",
            user_id=self.user_id,
            agent_id=self.agent_id,
        )


def normalize_tenant(
    tenant: TenantSpec | Dict[str, Any] | None = None,
    *,
    base_collection: str = "default",
    user_id: int | None = None,
    agent_id: int | None = None,
) -> TenantSpec:
    """Normalize legacy tenant payloads into a single spec object."""

    if tenant is None:
        return TenantSpec(
            base_collection=base_collection or "default",
            user_id=user_id,
            agent_id=agent_id,
        )

    if isinstance(tenant, TenantSpec):
        return TenantSpec(
            base_collection=tenant.base_collection or base_collection or "default",
            user_id=tenant.user_id if tenant.user_id is not None else user_id,
            agent_id=tenant.agent_id if tenant.agent_id is not None else agent_id,
        )

    resolved_base = str(tenant.get("base_collection") or tenant.get("collection") or base_collection or "default")
    resolved_user = tenant.get("user_id", tenant.get("_user_id", user_id))
    resolved_agent = tenant.get("agent_id", tenant.get("_agent_id", agent_id))
    return TenantSpec(
        base_collection=resolved_base,
        user_id=resolved_user,
        agent_id=resolved_agent,
    )


@dataclass(slots=True)
class DocumentRequest:
    """Normalized document input."""

    content: str
    collection: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    tenant: TenantSpec = field(default_factory=TenantSpec)


@dataclass(slots=True)
class SearchRequest:
    """Normalized search / rag_ask input."""

    query: str
    collection: str = "default"
    limit: int = 5
    threshold: float = 0.7
    mode: str = "raw"
    tenant: TenantSpec = field(default_factory=TenantSpec)


@dataclass(slots=True)
class ChatRequest:
    """Normalized chat input."""

    query: str
    collection: str = "default"
    limit: int = 5
    tenant: TenantSpec = field(default_factory=TenantSpec)


@dataclass(slots=True)
class DeleteRequest:
    """Normalized delete input."""

    identifier: str
    collection: str = "default"
    tenant: TenantSpec = field(default_factory=TenantSpec)


@dataclass(slots=True)
class SearchResultView:
    """Compatibility view for search results."""

    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    filename: str = ""
    retrieval_method: str = "hybrid"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchResponse:
    """Compatibility response for search / rag_ask."""

    query: str
    collection: str
    results: List[SearchResultView] = field(default_factory=list)
    summary: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.summary is None:
            payload.pop("summary", None)
        return payload


@dataclass(slots=True)
class ChatResponse:
    """Compatibility response for chat."""

    query: str
    collection: str
    response: str
    sources: List[SearchResultView] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class UploadFileResult:
    """Per-file upload status."""

    filename: str
    file_type: str
    content_length: int
    processed: bool
    error: str = ""
    preview: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BatchUploadResponse:
    """Batch upload response."""

    total_files: int
    successful: int
    failed: int
    results: List[UploadFileResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
