"""Canonical tenant and request context helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from uuid import uuid4

from fastapi import Request

from .core.indexing.models import TenantContext as CoreTenantContext


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _new_request_id() -> str:
    return uuid4().hex


def _trace_id_from_traceparent(traceparent: str | None) -> str | None:
    text = _coerce_optional_text(traceparent)
    if text is None:
        return None
    parts = text.split("-")
    if len(parts) >= 3 and parts[1]:
        return parts[1]
    return text


@dataclass(slots=True, frozen=True)
class TenantContext:
    """Tenant scope shared across HTTP, MCP, and service layers."""

    base_collection: str = "default"
    user_id: int | None = None
    agent_id: int | None = None
    tenant_key: str | None = None

    def to_core(self) -> CoreTenantContext:
        return CoreTenantContext(
            base_collection=self.base_collection or "default",
            user_id=self.user_id,
            agent_id=self.agent_id,
        )

    def canonical_key(self, *, fallback: str = "default") -> str:
        if self.tenant_key:
            return self.tenant_key
        base = self.base_collection or fallback
        if self.user_id is None:
            return base
        if self.agent_id is None:
            return f"u{self.user_id}_{base}"
        return f"u{self.user_id}_a{self.agent_id}_{base}"


TenantSpec = TenantContext


def normalize_tenant(
    tenant: TenantContext | Mapping[str, Any] | None = None,
    *,
    base_collection: str = "default",
    user_id: int | None = None,
    agent_id: int | None = None,
    tenant_key: str | None = None,
) -> TenantContext:
    """Normalize legacy tenant payloads into a canonical context."""

    resolved_base = _coerce_optional_text(base_collection) or "default"
    resolved_user = _coerce_optional_int(user_id)
    resolved_agent = _coerce_optional_int(agent_id)
    resolved_tenant_key = _coerce_optional_text(tenant_key)

    if tenant is None:
        return TenantContext(
            base_collection=resolved_base,
            user_id=resolved_user,
            agent_id=resolved_agent,
            tenant_key=resolved_tenant_key,
        )

    if isinstance(tenant, TenantContext):
        return TenantContext(
            base_collection=tenant.base_collection or resolved_base,
            user_id=tenant.user_id if tenant.user_id is not None else resolved_user,
            agent_id=tenant.agent_id if tenant.agent_id is not None else resolved_agent,
            tenant_key=tenant.tenant_key if tenant.tenant_key is not None else resolved_tenant_key,
        )

    raw_base = tenant.get("base_collection") or tenant.get("collection") or resolved_base
    raw_user = tenant.get("user_id", tenant.get("_user_id", resolved_user))
    raw_agent = tenant.get("agent_id", tenant.get("_agent_id", resolved_agent))
    raw_tenant_key = tenant.get("tenant_key", tenant.get("tenant_id", resolved_tenant_key))
    return TenantContext(
        base_collection=_coerce_optional_text(raw_base) or resolved_base,
        user_id=_coerce_optional_int(raw_user),
        agent_id=_coerce_optional_int(raw_agent),
        tenant_key=_coerce_optional_text(raw_tenant_key),
    )


@dataclass(slots=True, frozen=True)
class RequestContext:
    """Standard request identity shared by HTTP and MCP transports."""

    tenant: TenantContext = field(default_factory=TenantContext)
    transport: str = "internal"
    operation: str = "unknown"
    api_key: str | None = None
    request_id: str = field(default_factory=_new_request_id)
    trace_id: str | None = None
    client_host: str | None = None
    subject: str | None = None
    rate_limit_subject: str | None = None
    quota_subject: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.trace_id is None:
            object.__setattr__(self, "trace_id", self.request_id)
        normalized_tenant = normalize_tenant(self.tenant)
        object.__setattr__(self, "tenant", normalized_tenant)
        tenant_key = normalized_tenant.canonical_key(
            fallback=self.client_host or normalized_tenant.base_collection or "default"
        )
        if self.rate_limit_subject is None:
            object.__setattr__(self, "rate_limit_subject", self.subject or tenant_key)
        if self.quota_subject is None:
            object.__setattr__(self, "quota_subject", normalized_tenant.canonical_key())

    @property
    def tenant_key(self) -> str:
        return self.tenant.canonical_key()

    def subject_key(self, *, fallback: str = "default") -> str:
        if self.rate_limit_subject:
            return self.rate_limit_subject
        if self.subject:
            return self.subject
        if self.client_host:
            return self.tenant.canonical_key(fallback=self.client_host)
        return self.tenant.canonical_key(fallback=fallback)

    @classmethod
    def from_http(
        cls,
        request: Request,
        *,
        tenant: TenantContext | Mapping[str, Any] | None = None,
        base_collection: str = "default",
        user_id: int | None = None,
        agent_id: int | None = None,
        api_key: str | None = None,
        operation: str = "request",
        subject: str | None = None,
    ) -> RequestContext:
        state = getattr(request, "state", None)
        request_id = _coerce_optional_text(getattr(state, "request_id", None)) or _coerce_optional_text(
            request.headers.get("x-request-id")
        ) or _new_request_id()
        trace_id = _coerce_optional_text(getattr(state, "trace_id", None)) or _coerce_optional_text(
            request.headers.get("x-trace-id")
        ) or _trace_id_from_traceparent(request.headers.get("traceparent")) or request_id
        client_host = request.client.host if request.client and request.client.host else None
        return cls(
            tenant=normalize_tenant(
                tenant,
                base_collection=base_collection,
                user_id=user_id,
                agent_id=agent_id,
            ),
            transport="http",
            operation=operation,
            api_key=_coerce_optional_text(api_key),
            request_id=request_id,
            trace_id=trace_id,
            client_host=client_host,
            subject=_coerce_optional_text(subject),
        )

    @classmethod
    def from_mcp(
        cls,
        payload: Mapping[str, Any] | None = None,
        *,
        tenant: TenantContext | Mapping[str, Any] | None = None,
        base_collection: str = "default",
        user_id: int | None = None,
        agent_id: int | None = None,
        api_key: str | None = None,
        operation: str = "mcp",
        subject: str | None = None,
    ) -> RequestContext:
        payload = payload or {}
        request_id = _coerce_optional_text(payload.get("request_id")) or _new_request_id()
        trace_id = _coerce_optional_text(payload.get("trace_id")) or request_id
        return cls(
            tenant=normalize_tenant(
                tenant if tenant is not None else payload.get("tenant"),
                base_collection=base_collection,
                user_id=user_id if user_id is not None else payload.get("user_id", payload.get("_user_id")),
                agent_id=agent_id if agent_id is not None else payload.get("agent_id", payload.get("_agent_id")),
            ),
            transport="mcp",
            operation=operation,
            api_key=_coerce_optional_text(api_key if api_key is not None else payload.get("api_key")),
            request_id=request_id,
            trace_id=trace_id,
            client_host=_coerce_optional_text(payload.get("client_host")),
            subject=_coerce_optional_text(subject),
            metadata={"tool_name": _coerce_optional_text(payload.get("name"))} if payload.get("name") else {},
        )


def normalize_request_context(
    request_context: RequestContext | Mapping[str, Any] | None = None,
    *,
    tenant: TenantContext | Mapping[str, Any] | None = None,
    base_collection: str | None = None,
    collection: str | None = None,
    user_id: int | None = None,
    agent_id: int | None = None,
    transport: str = "internal",
    operation: str = "internal",
    api_key: str | None = None,
    subject: str | None = None,
    client_host: str | None = None,
    request_id: str | None = None,
    trace_id: str | None = None,
) -> RequestContext:
    """Normalize arbitrary request identity payloads into `RequestContext`."""

    resolved_collection = _coerce_optional_text(base_collection) or _coerce_optional_text(collection) or "default"

    if isinstance(request_context, RequestContext):
        tenant_spec = normalize_tenant(
            request_context.tenant,
            base_collection=resolved_collection,
            user_id=user_id,
            agent_id=agent_id,
        )
        return RequestContext(
            tenant=tenant_spec,
            transport=request_context.transport or transport,
            operation=request_context.operation or operation,
            api_key=_coerce_optional_text(api_key) or request_context.api_key,
            request_id=_coerce_optional_text(request_id) or request_context.request_id or _new_request_id(),
            trace_id=_coerce_optional_text(trace_id) or request_context.trace_id or request_context.request_id,
            client_host=_coerce_optional_text(client_host) or request_context.client_host,
            subject=_coerce_optional_text(subject) or request_context.subject,
            rate_limit_subject=request_context.rate_limit_subject,
            quota_subject=request_context.quota_subject,
            metadata=dict(request_context.metadata),
        )

    if isinstance(request_context, Mapping):
        tenant = tenant if tenant is not None else request_context.get("tenant")
        transport = _coerce_optional_text(request_context.get("transport")) or transport
        operation = _coerce_optional_text(request_context.get("operation")) or operation
        api_key = _coerce_optional_text(api_key) or _coerce_optional_text(request_context.get("api_key"))
        request_id = _coerce_optional_text(request_id) or _coerce_optional_text(request_context.get("request_id"))
        trace_id = _coerce_optional_text(trace_id) or _coerce_optional_text(request_context.get("trace_id"))
        client_host = _coerce_optional_text(client_host) or _coerce_optional_text(request_context.get("client_host"))
        subject = _coerce_optional_text(subject) or _coerce_optional_text(request_context.get("subject"))

    resolved_request_id = _coerce_optional_text(request_id) or _new_request_id()
    resolved_trace_id = _coerce_optional_text(trace_id) or resolved_request_id
    return RequestContext(
        tenant=normalize_tenant(
            tenant,
            base_collection=resolved_collection,
            user_id=user_id,
            agent_id=agent_id,
        ),
        transport=_coerce_optional_text(transport) or "internal",
        operation=_coerce_optional_text(operation) or "internal",
        api_key=_coerce_optional_text(api_key),
        request_id=resolved_request_id,
        trace_id=resolved_trace_id,
        client_host=_coerce_optional_text(client_host),
        subject=_coerce_optional_text(subject),
    )
