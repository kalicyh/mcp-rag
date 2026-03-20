"""Thin shell/runtime assembly for HTTP, MCP, and CLI entry points."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, Request

from .config import settings
from .contracts import TenantSpec
from .observability import ObservabilityCollector
from .security import SecurityPolicy, TokenBucketRateLimiter
from .service_facade import get_rag_service
from .services import ServiceRuntime

ServiceProvider = Callable[[], Awaitable[Any] | Any]


@dataclass(slots=True)
class ShellContext:
    """Runtime and guardrail dependencies shared by shell layers."""

    settings: Any
    runtime: ServiceRuntime
    service_provider: ServiceProvider
    security_policy: SecurityPolicy
    rate_limiter: TokenBucketRateLimiter
    observability: ObservabilityCollector
    bootstrapped: bool = False


def create_shell_context(
    *,
    settings_obj=settings,
    runtime: ServiceRuntime | None = None,
    service_provider: ServiceProvider | None = None,
    security_policy: SecurityPolicy | None = None,
    rate_limiter: TokenBucketRateLimiter | None = None,
    observability: ObservabilityCollector | None = None,
) -> ShellContext:
    """Create the default shell context without warming large models."""

    resolved_runtime = runtime or ServiceRuntime(settings_obj=settings_obj)
    return ShellContext(
        settings=settings_obj,
        runtime=resolved_runtime,
        service_provider=service_provider or get_rag_service,
        security_policy=security_policy or SecurityPolicy.from_settings(settings_obj),
        rate_limiter=rate_limiter or TokenBucketRateLimiter.from_settings(settings_obj),
        observability=observability or ObservabilityCollector.from_settings(settings_obj),
    )


def create_http_app(*, context: ShellContext | None = None) -> FastAPI:
    """Create a FastAPI app with shell context attached."""

    app = FastAPI(title="MCP-RAG HTTP API", description="API for configuring MCP-RAG and adding documents")
    app.state.shell_context = context or create_shell_context()
    return app


def get_shell_context(request: Request) -> ShellContext:
    """Fetch the shell context from a request."""

    context = getattr(request.app.state, "shell_context", None)
    if context is None:
        context = create_shell_context()
        request.app.state.shell_context = context
    return context


async def resolve_shell_service(request: Request) -> Any:
    """Resolve the current service instance via shell injection."""

    provider = get_shell_context(request).service_provider
    service = provider()
    if inspect.isawaitable(service):
        return await service
    return service


def runtime_snapshot(context: ShellContext) -> dict[str, bool]:
    """Return readiness signals without forcing model initialization."""

    runtime = context.runtime
    return {
        "document_processor": getattr(runtime, "_document_processor", None) is not None,
        "embedding_model": getattr(runtime, "_embedding_model", None) is not None,
        "vector_store": getattr(runtime, "_vector_store", None) is not None,
        "hybrid_service": getattr(runtime, "_hybrid_service", None) is not None,
        "llm_model": getattr(runtime, "_llm_model", None) is not None,
    }


def health_payload(context: ShellContext) -> dict[str, Any]:
    """Build a health payload without warming the service runtime."""

    collector_payload = context.observability.as_dict()
    health = dict(collector_payload["health"])
    health["ready"] = context.bootstrapped
    health["runtime"] = runtime_snapshot(context)
    return health


def ready_payload(context: ShellContext) -> dict[str, Any]:
    """Build a lightweight readiness payload."""

    return {
        "ready": context.bootstrapped,
        "runtime": runtime_snapshot(context),
    }


def metrics_payload(context: ShellContext) -> dict[str, Any]:
    """Expose metrics without forcing any service initialization."""

    return context.observability.as_dict()


def set_bootstrapped(request: Request) -> None:
    """Mark the shell context as bootstrapped after startup."""

    get_shell_context(request).bootstrapped = True


def tenant_subject(tenant: TenantSpec | None, *, fallback: str = "default") -> str:
    """Build a stable rate-limit subject from tenant scope."""

    if tenant is None:
        return fallback
    core = tenant.to_core()
    if core.user_id is None:
        return core.base_collection or fallback
    if core.agent_id is None:
        return f"u{core.user_id}_{core.base_collection or fallback}"
    return f"u{core.user_id}_a{core.agent_id}_{core.base_collection or fallback}"


def request_api_key(request: Request) -> str | None:
    """Read an API key from HTTP headers."""

    api_key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if api_key:
        return api_key.strip() or None

    authorization = request.headers.get("authorization") or request.headers.get("Authorization")
    if not authorization:
        return None

    prefix = "bearer "
    if authorization.lower().startswith(prefix):
        token = authorization[len(prefix):].strip()
        return token or None
    return authorization.strip() or None


def request_subject(request: Request, tenant: TenantSpec | None, *, fallback: str = "default") -> str:
    """Build a rate-limit subject for HTTP requests."""

    client_host = request.client.host if request.client and request.client.host else fallback
    return tenant_subject(tenant, fallback=client_host)


def enforce_request_guards(
    context: ShellContext,
    *,
    operation: str,
    tenant: TenantSpec | None,
    subject: str,
    api_key: str | None,
) -> None:
    """Apply authentication and rate limiting to a shell request."""

    context.security_policy.require(api_key, tenant=tenant)
    if context.rate_limiter.limit > 0:
        context.rate_limiter.require(subject)
