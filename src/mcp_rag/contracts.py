"""Shared contracts for the MCP-RAG shell layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from .context import RequestContext, TenantSpec, normalize_request_context, normalize_tenant


@dataclass(slots=True)
class DocumentRequest:
    """Normalized document input."""

    content: str
    collection: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    tenant: TenantSpec = field(default_factory=TenantSpec)
    context: RequestContext | None = None

    def __post_init__(self) -> None:
        self.tenant = normalize_tenant(self.tenant, base_collection=self.collection)
        self.context = normalize_request_context(
            self.context,
            tenant=self.tenant,
            base_collection=self.collection,
            operation="add_document",
        )
        self.collection = self.context.tenant.base_collection
        self.tenant = self.context.tenant


@dataclass(slots=True)
class SearchRequest:
    """Normalized search / rag_ask input."""

    query: str
    collection: str = "default"
    limit: int = 5
    threshold: float = 0.7
    mode: str = "raw"
    tenant: TenantSpec = field(default_factory=TenantSpec)
    context: RequestContext | None = None

    def __post_init__(self) -> None:
        self.tenant = normalize_tenant(self.tenant, base_collection=self.collection)
        self.context = normalize_request_context(
            self.context,
            tenant=self.tenant,
            base_collection=self.collection,
            operation="search",
        )
        self.collection = self.context.tenant.base_collection
        self.tenant = self.context.tenant


@dataclass(slots=True)
class ChatRequest:
    """Normalized chat input."""

    query: str
    collection: str = "default"
    limit: int = 5
    tenant: TenantSpec = field(default_factory=TenantSpec)
    context: RequestContext | None = None

    def __post_init__(self) -> None:
        self.tenant = normalize_tenant(self.tenant, base_collection=self.collection)
        self.context = normalize_request_context(
            self.context,
            tenant=self.tenant,
            base_collection=self.collection,
            operation="chat",
        )
        self.collection = self.context.tenant.base_collection
        self.tenant = self.context.tenant


@dataclass(slots=True)
class DeleteRequest:
    """Normalized delete input."""

    identifier: str
    collection: str = "default"
    tenant: TenantSpec = field(default_factory=TenantSpec)
    context: RequestContext | None = None

    def __post_init__(self) -> None:
        self.tenant = normalize_tenant(self.tenant, base_collection=self.collection)
        self.context = normalize_request_context(
            self.context,
            tenant=self.tenant,
            base_collection=self.collection,
            operation="delete",
        )
        self.collection = self.context.tenant.base_collection
        self.tenant = self.context.tenant


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
