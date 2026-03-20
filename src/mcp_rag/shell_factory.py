"""Thin shell/runtime assembly for HTTP, MCP, and CLI entry points."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request

from .config import settings
from .contracts import TenantSpec
from .observability import ObservabilityCollector
from .security import (
    AuthenticationError,
    AuthorizationError,
    RateLimitExceededError,
    SecurityPolicy,
    TokenBucketRateLimiter,
)
from .service_facade import RagService
from .services import ServiceRuntime

ServiceProvider = Callable[[], Awaitable[Any] | Any]


@dataclass(slots=True)
class ShellContext:
    """Runtime and shell guardrails shared by HTTP and MCP entry points."""

    settings: Any
    runtime: ServiceRuntime
    security_policy: SecurityPolicy
    rate_limiter: TokenBucketRateLimiter
    observability: ObservabilityCollector
    service_provider: ServiceProvider | None = None
    service: Any | None = None
    service_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    bootstrapped: bool = False


_default_shell_context: ShellContext | None = None


def create_shell_context(
    *,
    settings_obj=settings,
    runtime: ServiceRuntime | None = None,
    service_provider: ServiceProvider | None = None,
    security_policy: SecurityPolicy | None = None,
    rate_limiter: TokenBucketRateLimiter | None = None,
    observability: ObservabilityCollector | None = None,
) -> ShellContext:
    """Create a shell context without eagerly warming heavy dependencies."""

    resolved_runtime = runtime or ServiceRuntime(settings_obj=settings_obj)
    context = ShellContext(
        settings=settings_obj,
        runtime=resolved_runtime,
        security_policy=security_policy or SecurityPolicy.from_settings(settings_obj),
        rate_limiter=rate_limiter or TokenBucketRateLimiter.from_settings(settings_obj),
        observability=observability or ObservabilityCollector.from_settings(settings_obj),
    )

    if service_provider is not None:
        context.service_provider = service_provider
        return context

    async def _default_service_provider() -> RagService:
        if context.service is not None:
            return context.service

        async with context.service_lock:
            if context.service is None:
                context.service = RagService(runtime=context.runtime)
            return context.service

    context.service_provider = _default_service_provider
    return context


def get_default_shell_context() -> ShellContext:
    """Return the process-wide shell context shared by default shells."""

    global _default_shell_context
    if _default_shell_context is None:
        _default_shell_context = create_shell_context()
    return _default_shell_context


def create_http_app(*, context: ShellContext | None = None) -> FastAPI:
    """Create the default HTTP shell app."""

    app = FastAPI(title="MCP-RAG HTTP API", description="API for configuring MCP-RAG and adding documents")
    app.state.shell_context = context or get_default_shell_context()
    return app


def get_shell_context(target: Request | Any) -> ShellContext:
    """Fetch or attach a shell context from a request-like object."""

    app = getattr(target, "app", None)
    if app is None:
        raise TypeError("Shell context target must expose an app attribute")

    context = getattr(app.state, "shell_context", None)
    if context is None:
        context = get_default_shell_context()
        app.state.shell_context = context
    return context


async def resolve_shell_service(target: Request | Any) -> Any:
    """Resolve the service instance via the shell context."""

    context = get_shell_context(target)
    provider = context.service_provider
    if provider is None:
        raise RuntimeError("Shell context is missing a service provider")

    service = provider()
    if inspect.isawaitable(service):
        return await service
    return service


def runtime_snapshot(context: ShellContext) -> dict[str, bool]:
    """Return runtime readiness signals without initializing heavy providers."""

    runtime = context.runtime
    return {
        "document_processor": getattr(runtime, "_document_processor", None) is not None,
        "embedding_model": getattr(runtime, "_embedding_model", None) is not None,
        "vector_store": getattr(runtime, "_vector_store", None) is not None,
        "hybrid_service": getattr(runtime, "_hybrid_service", None) is not None,
        "llm_model": getattr(runtime, "_llm_model", None) is not None,
    }


def health_payload(context: ShellContext) -> dict[str, Any]:
    """Build a lightweight health response."""

    collector_payload = context.observability.as_dict()
    health = dict(collector_payload["health"])
    health["ready"] = context.bootstrapped
    health["runtime"] = runtime_snapshot(context)
    return health


def ready_payload(context: ShellContext) -> dict[str, Any]:
    """Build a readiness response."""

    return {
        "ready": context.bootstrapped,
        "runtime": runtime_snapshot(context),
    }


def metrics_payload(context: ShellContext) -> dict[str, Any]:
    """Expose the current metrics snapshot."""

    return context.observability.as_dict()


def request_api_key(request: Request, explicit_api_key: str | None = None) -> str | None:
    """Read an API key from explicit args, headers, or query params."""

    if explicit_api_key:
        token = explicit_api_key.strip()
        if token:
            return token

    api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if api_key:
        token = api_key.strip()
        if token:
            return token

    authorization = request.headers.get("authorization") or request.headers.get("Authorization")
    if authorization:
        prefix = "bearer "
        if authorization.lower().startswith(prefix):
            token = authorization[len(prefix):].strip()
            if token:
                return token

        token = authorization.strip()
        if token:
            return token

    query_token = request.query_params.get("api_key")
    if query_token:
        token = query_token.strip()
        if token:
            return token

    return None


def tenant_subject(tenant: TenantSpec | None, *, fallback: str = "default") -> str:
    """Build a stable subject key from tenant scope."""

    if tenant is None:
        return fallback

    core = tenant.to_core()
    if core.user_id is None:
        return core.base_collection or fallback
    if core.agent_id is None:
        return f"u{core.user_id}_{core.base_collection or fallback}"
    return f"u{core.user_id}_a{core.agent_id}_{core.base_collection or fallback}"


def request_subject(request: Request, tenant: TenantSpec | None, *, fallback: str = "default") -> str:
    """Build a rate-limit subject for an HTTP request."""

    client_host = request.client.host if request.client and request.client.host else fallback
    return tenant_subject(tenant, fallback=client_host)


def enforce_guardrails(
    context: ShellContext,
    *,
    request: Request | None = None,
    tenant: TenantSpec | None = None,
    api_key: str | None = None,
    subject: str | None = None,
) -> None:
    """Apply auth and rate limiting with the shared shell context."""

    auth = context.security_policy.validate(api_key, tenant=tenant)
    if not auth.allowed:
        if auth.reason == "api key required":
            raise AuthenticationError(auth.reason)
        raise AuthorizationError(auth.reason)

    resolved_subject = subject
    if resolved_subject is None and request is not None:
        resolved_subject = request_subject(request, tenant, fallback=auth.tenant_key or "default")
    if resolved_subject is None:
        resolved_subject = auth.tenant_key or tenant_subject(tenant, fallback="default")

    if context.rate_limiter.limit <= 0:
        return

    decision = context.rate_limiter.allow(resolved_subject)
    if decision.allowed:
        return

    raise RateLimitExceededError(
        f"rate limit exceeded for {resolved_subject}: retry after {decision.retry_after_seconds:.2f}s"
    )


def enforce_http_guardrails(
    request: Request,
    *,
    tenant: TenantSpec | None = None,
    api_key: str | None = None,
    subject: str | None = None,
) -> None:
    """Apply shell guardrails and convert failures into HTTP errors."""

    context = get_shell_context(request)
    try:
        enforce_guardrails(
            context,
            request=request,
            tenant=tenant,
            api_key=request_api_key(request, api_key),
            subject=subject,
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except AuthorizationError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RateLimitExceededError as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
            headers={"Retry-After": "1"},
        ) from exc
