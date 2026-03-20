"""Runtime assembly for HTTP, MCP, and CLI entry points."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException, Request

from .config import ConfigManager, config_manager
from .context import RequestContext, TenantSpec, normalize_tenant
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
class AppContext:
    """Runtime and transport guardrails shared by HTTP and MCP entry points."""

    settings: Any
    runtime: ServiceRuntime
    security_policy: SecurityPolicy
    rate_limiter: TokenBucketRateLimiter
    observability: ObservabilityCollector
    config_manager: ConfigManager | None = None
    service_provider: ServiceProvider | None = None
    service: Any | None = None
    service_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    reload_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    bootstrapped: bool = False
    config_revision: int = 0


ShellContext = AppContext

_default_app_context: AppContext | None = None


def create_app_context(
    *,
    settings_obj: Any | None = None,
    config_manager_obj: ConfigManager | None = None,
    runtime: ServiceRuntime | None = None,
    service_provider: ServiceProvider | None = None,
    security_policy: SecurityPolicy | None = None,
    rate_limiter: TokenBucketRateLimiter | None = None,
    observability: ObservabilityCollector | None = None,
) -> AppContext:
    """Create an app context without eagerly warming heavy dependencies."""

    manager = config_manager_obj
    resolved_settings = settings_obj
    if resolved_settings is None:
        if manager is None:
            manager = config_manager
        resolved_settings = manager.ensure_config_file()

    resolved_runtime = runtime or ServiceRuntime(settings_obj=resolved_settings)
    context = AppContext(
        settings=resolved_settings,
        runtime=resolved_runtime,
        security_policy=security_policy or SecurityPolicy.from_settings(resolved_settings),
        rate_limiter=rate_limiter or TokenBucketRateLimiter.from_settings(resolved_settings),
        observability=observability or ObservabilityCollector.from_settings(resolved_settings),
        config_manager=manager,
        config_revision=manager.revision if manager is not None else 0,
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


create_shell_context = create_app_context


def get_default_app_context() -> AppContext:
    """Return the process-wide app context shared by default transports."""

    global _default_app_context
    if _default_app_context is None:
        _default_app_context = create_app_context(config_manager_obj=config_manager)
    return _default_app_context


get_default_shell_context = get_default_app_context


async def sync_app_context(context: AppContext, *, force: bool = False) -> bool:
    """Rebuild runtime-scoped dependencies when config changes."""

    manager = context.config_manager
    if manager is None:
        return False

    async with context.reload_lock:
        settings_obj = manager.reload() if force else (manager.reload_if_changed() or manager.settings)
        if not force and context.config_revision == manager.revision:
            return False

        context.settings = settings_obj
        context.security_policy = SecurityPolicy.from_settings(settings_obj)
        context.rate_limiter = TokenBucketRateLimiter.from_settings(settings_obj)

        configure = getattr(context.observability, "configure_from_settings", None)
        if callable(configure):
            configure(settings_obj)
        else:
            context.observability = ObservabilityCollector.from_settings(settings_obj)

        reload_runtime = getattr(context.runtime, "reload_settings", None)
        if callable(reload_runtime):
            await reload_runtime(settings_obj)
        else:
            old_runtime = context.runtime
            context.runtime = ServiceRuntime(settings_obj=settings_obj)
            context.service = None
            if old_runtime is not context.runtime and hasattr(old_runtime, "close"):
                await old_runtime.close()

        context.config_revision = manager.revision
        return True


async def reload_app_context(context: AppContext, *, settings_obj: Any | None = None) -> bool:
    """Force a live runtime reload, optionally with an explicit settings object."""

    if settings_obj is None:
        return await sync_app_context(context, force=True)

    async with context.reload_lock:
        context.settings = settings_obj
        context.security_policy = SecurityPolicy.from_settings(settings_obj)
        context.rate_limiter = TokenBucketRateLimiter.from_settings(settings_obj)

        configure = getattr(context.observability, "configure_from_settings", None)
        if callable(configure):
            configure(settings_obj)
        else:
            context.observability = ObservabilityCollector.from_settings(settings_obj)

        reload_runtime = getattr(context.runtime, "reload_settings", None)
        if callable(reload_runtime):
            await reload_runtime(settings_obj)
        else:
            old_runtime = context.runtime
            context.runtime = ServiceRuntime(settings_obj=settings_obj)
            context.service = None
            if old_runtime is not context.runtime and hasattr(old_runtime, "close"):
                await old_runtime.close()

        if context.config_manager is not None:
            context.config_revision = context.config_manager.revision
        return True


async def ensure_app_context_current(target: Request | Any, *, force: bool = False) -> AppContext:
    """Refresh the app context for request-driven transports."""

    context = get_app_context(target)
    await sync_app_context(context, force=force)
    return context


def create_http_app(*, context: AppContext | None = None) -> FastAPI:
    """Create the default HTTP shell app."""

    app = FastAPI(title="MCP-RAG HTTP API", description="API for configuring MCP-RAG and adding documents")
    app.state.shell_context = context or get_default_app_context()
    return app


def get_app_context(target: Request | Any) -> AppContext:
    """Fetch or attach an app context from a request-like object."""

    app = getattr(target, "app", None)
    if app is None:
        raise TypeError("App context target must expose an app attribute")

    context = getattr(app.state, "shell_context", None)
    if context is None:
        context = get_default_app_context()
        app.state.shell_context = context
    return context


get_shell_context = get_app_context


async def resolve_app_service(target: Request | Any) -> Any:
    """Resolve the service instance via the shared app context."""

    context = await ensure_app_context_current(target)
    provider = context.service_provider
    if provider is None:
        raise RuntimeError("App context is missing a service provider")

    service = provider()
    if inspect.isawaitable(service):
        return await service
    return service


resolve_shell_service = resolve_app_service


def runtime_snapshot(context: AppContext) -> dict[str, Any]:
    """Return runtime readiness signals without initializing heavy providers."""

    runtime = context.runtime
    snapshot_fn = getattr(runtime, "readiness_snapshot", None)
    if callable(snapshot_fn):
        return snapshot_fn()
    return {
        "document_processor": getattr(runtime, "_document_processor", None) is not None,
        "embedding_model": getattr(runtime, "_embedding_model", None) is not None,
        "vector_store": getattr(runtime, "_vector_store", None) is not None,
        "hybrid_service": getattr(runtime, "_hybrid_service", None) is not None,
        "llm_model": getattr(runtime, "_llm_model", None) is not None,
    }


def health_payload(context: AppContext) -> dict[str, Any]:
    """Build a lightweight health response."""

    collector_payload = context.observability.as_dict()
    health = dict(collector_payload["health"])
    health["ready"] = context.bootstrapped
    health["runtime"] = runtime_snapshot(context)
    health["config_revision"] = context.config_revision
    return health


def ready_payload(context: AppContext) -> dict[str, Any]:
    """Build a readiness response."""

    return {
        "ready": context.bootstrapped,
        "runtime": runtime_snapshot(context),
        "config_revision": context.config_revision,
    }


def metrics_payload(context: AppContext) -> dict[str, Any]:
    """Expose the current metrics snapshot."""

    payload = context.observability.as_dict()
    payload["config_revision"] = context.config_revision
    return payload


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


def build_http_request_context(
    request: Request,
    *,
    tenant: TenantSpec | dict[str, Any] | None = None,
    base_collection: str = "default",
    user_id: int | None = None,
    agent_id: int | None = None,
    api_key: str | None = None,
    operation: str = "request",
    subject: str | None = None,
) -> RequestContext:
    """Build and attach the canonical request context for an HTTP request."""

    request_context = RequestContext.from_http(
        request,
        tenant=tenant,
        base_collection=base_collection,
        user_id=user_id,
        agent_id=agent_id,
        api_key=request_api_key(request, api_key),
        operation=operation,
        subject=subject,
    )
    request.state.request_context = request_context
    request.state.request_id = request_context.request_id
    request.state.trace_id = request_context.trace_id
    return request_context


def build_mcp_request_context(
    payload: dict[str, Any],
    *,
    tenant: TenantSpec | dict[str, Any] | None = None,
    base_collection: str = "default",
    user_id: int | None = None,
    agent_id: int | None = None,
    api_key: str | None = None,
    operation: str = "mcp",
    subject: str | None = None,
) -> RequestContext:
    """Build the canonical request context for an MCP tool call."""

    return RequestContext.from_mcp(
        payload,
        tenant=tenant,
        base_collection=base_collection,
        user_id=user_id,
        agent_id=agent_id,
        api_key=api_key,
        operation=operation,
        subject=subject,
    )


def tenant_subject(tenant: TenantSpec | None, *, fallback: str = "default") -> str:
    """Build a stable subject key from tenant scope."""

    if tenant is None:
        return fallback
    return normalize_tenant(tenant).canonical_key(fallback=fallback)


def request_subject(request: Request, tenant: TenantSpec | None, *, fallback: str = "default") -> str:
    """Build a rate-limit subject for an HTTP request."""

    client_host = request.client.host if request.client and request.client.host else fallback
    return tenant_subject(tenant, fallback=client_host)


def enforce_guardrails(
    context: AppContext,
    *,
    request_context: RequestContext,
) -> None:
    """Apply auth and rate limiting with the shared app context."""

    auth = context.security_policy.validate(request_context.api_key, tenant=request_context.tenant)
    if not auth.allowed:
        if auth.reason == "api key required":
            raise AuthenticationError(auth.reason)
        raise AuthorizationError(auth.reason)

    resolved_subject = request_context.subject_key(fallback=auth.tenant_key or "default")
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
    request_context: RequestContext,
) -> None:
    """Apply app guardrails and convert failures into HTTP errors."""

    context = get_app_context(request)
    try:
        enforce_guardrails(context, request_context=request_context)
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
