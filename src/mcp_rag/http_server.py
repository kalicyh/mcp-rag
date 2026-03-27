"""HTTP server for MCP-RAG configuration and document management."""

import asyncio
from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from uuid import uuid4
from fastapi import HTTPException, UploadFile, File, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.responses import PlainTextResponse

try:
    import httpx
except ImportError:  # pragma: no cover - optional dependency guard
    httpx = None

from .config import canonical_provider_name, config_manager
from .contracts import ChatRequest, ChatResponse, DocumentRequest, SearchRequest, SearchResponse, SearchResultView, normalize_tenant
from .knowledge_bases import KnowledgeBaseAccessError
from .shell_factory import (
    build_http_request_context,
    create_http_app,
    ensure_app_context_current,
    enforce_http_guardrails,
    get_default_shell_context,
    health_payload,
    metrics_payload,
    ready_payload,
    reload_shell_context,
    request_subject,
    resolve_shell_service,
)
from .mcp_server import mcp_server
from .security import QuotaExceededError
from .spa_assets import render_missing_spa_html, resolve_spa_entry
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

logger = logging.getLogger(__name__)

_RETRIEVAL_CONFIG_KEYS = {
    "cache",
    "embedding_provider",
    "embedding_fallback_provider",
    "enable_cache",
    "enable_llm_summary",
    "enable_reranker",
    "llm_model",
    "llm_provider",
    "llm_fallback_provider",
    "max_retrieval_results",
    "provider_budget",
    "provider_configs",
    "similarity_threshold",
}

def _build_streamable_http_manager(*, shell_context=None) -> StreamableHTTPSessionManager:
    if shell_context is not None:
        mcp_server.shell_context = shell_context
    return StreamableHTTPSessionManager(
        app=mcp_server.server,
        json_response=True,
        stateless=True,
    )


def _get_streamable_http_manager() -> StreamableHTTPSessionManager:
    manager = getattr(app.state, "streamable_http_manager", None)
    if manager is None:
        raise RuntimeError("Streamable HTTP transport is not running")
    return manager


@asynccontextmanager
async def _app_lifespan(lifespan_app):
    await reload_shell_context(lifespan_app.state.shell_context)
    lifespan_app.state.shell_context.bootstrapped = True

    manager = _build_streamable_http_manager(shell_context=lifespan_app.state.shell_context)
    lifespan_app.state.streamable_http_manager = manager
    async with manager.run():
        try:
            yield
        finally:
            lifespan_app.state.streamable_http_manager = None


app = create_http_app(context=get_default_shell_context(), lifespan=_app_lifespan)

# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


async def _streamable_http_asgi(scope, receive, send):
    if scope.get("type") != "http":
        response = PlainTextResponse("Streamable HTTP supports only HTTP requests", status_code=405)
        await response(scope, receive, send)
        return

    try:
        await _get_streamable_http_manager().handle_request(scope, receive, send)
    except RuntimeError as err:  # pragma: no cover - defensive
        logger.error("Streamable HTTP transport unavailable: %s", err)
        response = PlainTextResponse("MCP transport unavailable", status_code=503)
        await response(scope, receive, send)


app.mount("/mcp", _streamable_http_asgi, name="streamable-mcp")
app.mount("/mcp/", _streamable_http_asgi, name="streamable-mcp-slash")
app.mount("/sse", _streamable_http_asgi, name="sse")


@app.middleware("http")
async def _request_identity_middleware(request: Request, call_next):
    await ensure_app_context_current(request)
    request_id = (request.headers.get("x-request-id") or "").strip() or uuid4().hex
    trace_id = (request.headers.get("x-trace-id") or "").strip()
    if not trace_id:
        traceparent = (request.headers.get("traceparent") or "").strip()
        trace_id = traceparent.split("-")[1] if traceparent.count("-") >= 3 else traceparent or request_id

    request.state.request_id = request_id
    request.state.trace_id = trace_id or request_id

    response = await call_next(request)
    response.headers.setdefault("X-Request-Id", request_id)
    response.headers.setdefault("X-Trace-Id", trace_id or request_id)
    return response


async def get_rag_service(request: Request):
    """Compatibility wrapper used by unit tests and the shell routes."""

    return await resolve_shell_service(request)


def _config_affects_retrieval(*keys: str) -> bool:
    normalized = {str(key).split(".", 1)[0] for key in keys if key}
    return bool(normalized & _RETRIEVAL_CONFIG_KEYS)


class ConfigUpdate(BaseModel):
    """Configuration update model."""
    key: str
    value: Any


class BulkConfigUpdate(BaseModel):
    """Bulk configuration update model."""
    updates: Dict[str, Any]


class DocumentAdd(BaseModel):
    """Document addition model."""
    content: str
    collection: str = "default"
    kb_id: Optional[int] = None
    scope: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[int] = None
    agent_id: Optional[int] = None
    api_key: Optional[str] = None


class DeleteDocumentRequest(BaseModel):
    """Delete document request model."""
    document_id: str
    collection: str = "default"
    kb_id: Optional[int] = None
    scope: Optional[str] = None
    user_id: Optional[int] = None
    agent_id: Optional[int] = None
    api_key: Optional[str] = None


class DeleteFileRequest(BaseModel):
    """Delete file request model."""
    filename: str
    collection: str = "default"
    kb_id: Optional[int] = None
    scope: Optional[str] = None
    user_id: Optional[int] = None
    agent_id: Optional[int] = None
    api_key: Optional[str] = None


class KnowledgeBaseCreate(BaseModel):
    """Create knowledge base request."""

    name: str
    scope: str = "public"
    owner_user_id: Optional[int] = None
    owner_agent_id: Optional[int] = None
    api_key: Optional[str] = None


class MCPDebugCall(BaseModel):
    """Debug MCP tool invocation request."""

    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    api_key: Optional[str] = None


def _provider_config_entry(provider: str) -> Any | None:
    provider_name = canonical_provider_name(provider)
    provider_configs = getattr(config_manager.settings, "provider_configs", {}) or {}
    return provider_configs.get(provider_name)


def _infer_openai_model_family(model_id: str) -> str:
    model_name = str(model_id or "").strip().lower()
    embedding_markers = ("embedding", "bge-", "m3e", "e5", "rerank", "text-embedding")
    if any(marker in model_name for marker in embedding_markers):
        return "embedding"
    return "chat"


async def _fetch_openai_compatible_models(base_url: str, api_key: str) -> list[dict[str, str]]:
    if httpx is None:
        raise RuntimeError("httpx is not installed")

    async with httpx.AsyncClient(
        base_url=base_url.rstrip("/"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=20.0,
    ) as client:
        response = await client.get("/models")
        if response.status_code != 200:
            raise RuntimeError(f"Model API error: {response.status_code} - {response.text}")

        payload = response.json()
        items = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []

        models = []
        for item in items:
            model_id = str((item or {}).get("id") or "").strip()
            if not model_id:
                continue
            models.append(
                {
                    "id": model_id,
                    "label": model_id,
                    "family": _infer_openai_model_family(model_id),
                    "source": "remote",
                }
            )
        return models


async def _fetch_ollama_models(base_url: str) -> list[dict[str, str]]:
    if httpx is None:
        raise RuntimeError("httpx is not installed")

    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=20.0) as client:
        response = await client.get("/api/tags")
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")

        payload = response.json()
        items = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []

        models = []
        for item in items:
            model_id = str((item or {}).get("name") or "").strip()
            if not model_id:
                continue
            models.append(
                {
                    "id": model_id,
                    "label": model_id,
                    "family": "chat",
                    "source": "remote",
                }
            )
        return models


def _legacy_collection_key(collection: str, *, scope: str, user_id: int | None, agent_id: int | None) -> str:
    if scope == "public":
        return f"legacy:public:{collection or 'default'}"
    return f"legacy:agent_private:{user_id}:{agent_id}:{collection or 'default'}"


def _resolve_request_knowledge_base(
    request: Request,
    *,
    kb_id: int | None = None,
    scope: str | None = None,
    collection: str = "default",
    user_id: int | None = None,
    agent_id: int | None = None,
    api_key: str | None = None,
    operation: str = "request",
) -> tuple:
    shell_context = request.app.state.shell_context
    resolved_scope = shell_context.knowledge_bases.normalize_scope(scope, user_id=user_id, agent_id=agent_id)
    try:
        resolution = shell_context.knowledge_bases.resolve(
            kb_id=kb_id,
            scope=resolved_scope,
            user_id=user_id,
            agent_id=agent_id,
            legacy_collection=collection,
            legacy_collection_key=_legacy_collection_key(
                collection,
                scope=resolved_scope,
                user_id=user_id,
                agent_id=agent_id,
            ),
        )
    except KnowledgeBaseAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    effective_user_id = resolution.knowledge_base.owner_user_id if resolution.scope == "agent_private" else None
    effective_agent_id = resolution.knowledge_base.owner_agent_id if resolution.scope == "agent_private" else None
    tenant = normalize_tenant(
        base_collection=resolution.name,
        user_id=effective_user_id,
        agent_id=effective_agent_id,
    )
    request_context = build_http_request_context(
        request,
        tenant=tenant,
        base_collection=resolution.name,
        user_id=effective_user_id,
        agent_id=effective_agent_id,
        api_key=api_key,
        kb_id=resolution.kb_id,
        kb_scope=resolution.scope,
        kb_name=resolution.name,
        resolved_collection=resolution.collection_name,
        operation=operation,
        subject=request_subject(request, tenant, fallback=f"{operation}:{resolution.collection_name}"),
    )
    return resolution, request_context


def _parse_kb_ids(raw_value: Any) -> list[int]:
    if raw_value is None or raw_value == "":
        return []
    if isinstance(raw_value, list):
        values = raw_value
    else:
        values = str(raw_value).split(",")
    parsed: list[int] = []
    for item in values:
        try:
            value = int(str(item).strip())
        except (TypeError, ValueError):
            continue
        if value not in parsed:
            parsed.append(value)
    return parsed


async def _search_across_knowledge_bases(
    *,
    request: Request,
    service,
    query: str,
    collection: str,
    kb_ids: list[int],
    scope: str | None,
    limit: int,
    user_id: int | None,
    agent_id: int | None,
    api_key: str | None,
    operation: str,
) -> tuple[list[Any], SearchResponse]:
    contexts = [
        _resolve_request_knowledge_base(
            request,
            kb_id=kb_id,
            scope=scope,
            collection=collection,
            user_id=user_id,
            agent_id=agent_id,
            api_key=api_key,
            operation=operation,
        )[1]
        for kb_id in kb_ids
    ]
    if contexts:
        enforce_http_guardrails(request, request_context=contexts[0])
    responses = await asyncio.gather(
        *[
            service.search(
                SearchRequest(
                    query=query,
                    collection=request_context.kb_name or collection,
                    limit=limit,
                    kb_id=request_context.kb_id,
                    scope=request_context.kb_scope,
                    tenant=request_context.tenant,
                    context=request_context,
                )
            )
            for request_context in contexts
        ]
    )

    merged_results: list[SearchResultView] = []
    for response, request_context in zip(responses, contexts):
        for item in response.results:
            metadata = dict(item.metadata or {})
            metadata.setdefault("knowledge_base_id", request_context.kb_id)
            metadata.setdefault("knowledge_base_name", request_context.kb_name)
            metadata.setdefault("knowledge_base_scope", request_context.kb_scope)
            metadata.setdefault("owner_user_id", request_context.tenant.user_id)
            metadata.setdefault("owner_agent_id", request_context.tenant.agent_id)
            merged_results.append(
                SearchResultView(
                    content=item.content,
                    score=item.score,
                    vector_score=item.vector_score,
                    keyword_score=item.keyword_score,
                    metadata=metadata,
                    source=item.source,
                    filename=item.filename,
                    retrieval_method=item.retrieval_method,
                )
            )
    merged_results.sort(key=lambda item: item.score, reverse=True)
    merged_results = merged_results[:limit]

    summary = None
    if merged_results and getattr(service.runtime.settings, "enable_llm_summary", False):
        try:
            llm_model = await service.runtime.ensure_llm_model()
            summary_context = "\n\n".join(
                f"知识库 {item.metadata.get('knowledge_base_name', '未知')} · 文档 {index + 1} (相似度: {item.score:.3f}):\n{item.content}"
                for index, item in enumerate(merged_results)
            )
            summary = await llm_model.summarize(summary_context, query)
        except Exception as exc:
            logger.warning("LLM summary failed for multi-kb search, falling back to raw results: %s", exc)

    return contexts, SearchResponse(
        query=query,
        collection="multi_kb",
        results=merged_results,
        summary=summary,
    )


def _format_chat_context(results: list[SearchResultView]) -> str:
    return "\n\n".join(
        f"知识库 {item.metadata.get('knowledge_base_name', '未知')} / 文档 {index + 1} ({item.filename or item.source}):\n{item.content}"
        for index, item in enumerate(results)
    )


def _build_chat_prompt(query: str, context: str) -> str:
    return (
        "基于以下知识库内容回答用户的问题。如果知识库内容不足以回答问题，请说明无法找到相关信息。\n\n"
        f"知识库内容:\n{context}\n\n"
        f"用户问题: {query}\n\n"
        "请提供准确、简洁的回答:"
    )


def _format_llm_fallback_response(context: str, error: Exception) -> str:
    detail = str(error).strip() or error.__class__.__name__
    return (
        "### Retrieved Context\n\n"
        f"{context}\n\n"
        "### Note\n"
        "LLM is not available. The above context was retrieved for your query.\n\n"
        f"LLM error: {detail}"
    )


@app.get("/", tags=["系统"], summary="根入口")
async def root():
    """Root endpoint - redirect to the SPA shell."""
    return RedirectResponse(url="/app")


@app.get("/doc", tags=["系统"], summary="文档入口重定向")
async def doc_redirect():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["系统"], summary="健康检查")
async def health(request: Request):
    """Lightweight health summary without warming the runtime."""
    return health_payload(request.app.state.shell_context)


@app.get("/ready", tags=["系统"], summary="就绪检查")
async def ready(request: Request):
    """Readiness signal for shell wiring."""
    payload = ready_payload(request.app.state.shell_context)
    return JSONResponse(status_code=200 if payload["ready"] else 503, content=payload)


@app.get("/metrics", tags=["系统"], summary="运行指标")
async def metrics(request: Request):
    """Observability snapshot."""
    return metrics_payload(request.app.state.shell_context)


def _spa_redirect_target(view: str | None = None) -> str:
    """Build a stable SPA redirect target for legacy page routes."""

    if not view:
        return "/app"
    return f"/app/{view}"


def _serve_spa_entry(*, request_path: str):
    """Serve the prebuilt SPA entry file or a clear fallback page."""

    entry = resolve_spa_entry(static_path)
    if entry is None:
        return HTMLResponse(
            content=render_missing_spa_html(static_dir=static_path, request_path=request_path),
            status_code=503,
        )
    return FileResponse(entry, media_type="text/html")


@app.get("/app", response_class=HTMLResponse, tags=["应用"], summary="前端应用入口")
@app.get("/app/{spa_path:path}", response_class=HTMLResponse, tags=["应用"], summary="前端应用子路径")
async def spa_entry(spa_path: str = ""):
    """Serve the SPA shell for browser clients."""

    request_path = "/app" if not spa_path else f"/app/{spa_path}"
    return _serve_spa_entry(request_path=request_path)


@app.get("/documents-page", tags=["应用"], summary="文档页兼容入口")
async def documents_page():
    """Backward-compatible redirect to the SPA documents view."""

    return RedirectResponse(url=_spa_redirect_target("documents"))


@app.get("/config-page", tags=["应用"], summary="配置页兼容入口")
async def config_page():
    """Backward-compatible redirect to the SPA config view."""

    return RedirectResponse(url=_spa_redirect_target("config"))


@app.get("/config", tags=["配置"], summary="读取配置")
async def get_config(request: Request, api_key: Optional[str] = None):
    """Get current configuration."""
    context = request.app.state.shell_context
    async with context.observability.timer("config.get"):
        request_context = build_http_request_context(request, api_key=api_key, subject="config")
        enforce_http_guardrails(request, request_context=request_context)
        return config_manager.get_all_settings()


@app.post("/config", tags=["配置"], summary="更新单个配置项")
async def update_config(config: ConfigUpdate, request: Request):
    """Update a single configuration setting."""
    context = request.app.state.shell_context
    async with context.observability.timer("config.update"):
        request_context = build_http_request_context(request, subject="config")
        enforce_http_guardrails(request, request_context=request_context)
        success = config_manager.update_setting(config.key, config.value)
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to update config {config.key}")
        await reload_shell_context(context, settings_obj=config_manager.settings)
        if _config_affects_retrieval(config.key):
            await context.runtime.invalidate_all_retrieval_cache()
        return {"message": f"Config {config.key} updated successfully", "reloaded": True}


@app.post("/config/bulk", tags=["配置"], summary="批量更新配置")
async def update_config_bulk(config: BulkConfigUpdate, request: Request):
    """Update multiple configuration settings."""
    context = request.app.state.shell_context
    async with context.observability.timer("config.bulk_update"):
        request_context = build_http_request_context(request, subject="config")
        enforce_http_guardrails(request, request_context=request_context)
        success = config_manager.update_settings(config.updates)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to update config")
        await reload_shell_context(context, settings_obj=config_manager.settings)
        if _config_affects_retrieval(*config.updates.keys()):
            await context.runtime.invalidate_all_retrieval_cache()
        return {"message": "Config updated successfully", "reloaded": True}


@app.post("/config/reset", tags=["配置"], summary="重置默认配置")
async def reset_config(request: Request):
    """Reset configuration to defaults."""
    context = request.app.state.shell_context
    async with context.observability.timer("config.reset"):
        request_context = build_http_request_context(request, subject="config")
        enforce_http_guardrails(request, request_context=request_context)
        success = config_manager.reset_to_defaults()
        if not success:
            raise HTTPException(status_code=400, detail="Failed to reset config")
        await reload_shell_context(context, settings_obj=config_manager.settings)
        await context.runtime.invalidate_all_retrieval_cache()
        return {"message": "Config reset to defaults successfully", "reloaded": True}


@app.post("/config/reload", tags=["配置"], summary="从磁盘重载配置")
async def reload_config(request: Request):
    """Reload configuration from disk and rebuild the live runtime."""
    context = request.app.state.shell_context
    async with context.observability.timer("config.reload"):
        request_context = build_http_request_context(request, subject="config")
        enforce_http_guardrails(request, request_context=request_context)
        settings_obj = config_manager.reload()
        await reload_shell_context(context, settings_obj=settings_obj)
        await context.runtime.invalidate_all_retrieval_cache()
        return {"message": "Config reloaded successfully", "reloaded": True}


@app.get("/providers/{provider}/models", tags=["服务商"], summary="获取服务商模型列表")
async def get_provider_models(
    provider: str,
    request: Request,
    family: Optional[str] = None,
):
    """Fetch provider model list from remote service when supported."""
    context = request.app.state.shell_context
    async with context.observability.timer("provider_models.get"):
        request_context = build_http_request_context(request, subject=f"provider_models:{provider}")
        enforce_http_guardrails(request, request_context=request_context)

        provider_name = canonical_provider_name(provider)
        provider_config = _provider_config_entry(provider_name)
        requested_family = str(family or "").strip().lower() or None

        try:
            if provider_name in {"m3e-small", "e5-small"}:
                local_model = provider_name
                model_family = "embedding"
                if requested_family and requested_family != model_family:
                    return {"provider": provider_name, "family": requested_family, "models": []}
                return {
                    "provider": provider_name,
                    "family": model_family,
                    "models": [{"id": local_model, "label": local_model, "family": model_family, "source": "local"}],
                }

            if provider_name == "ollama":
                base_url = str(getattr(provider_config, "base_url", "") or "http://localhost:11434")
                models = await _fetch_ollama_models(base_url)
            else:
                if provider_config is None:
                    raise HTTPException(status_code=404, detail=f"Unknown provider: {provider_name}")
                api_key = getattr(provider_config, "api_key", None)
                base_url = str(getattr(provider_config, "base_url", "") or "")
                if not base_url:
                    raise HTTPException(status_code=400, detail="Provider base_url is missing")
                if not api_key:
                    raise HTTPException(status_code=400, detail="Provider api_key is missing")
                models = await _fetch_openai_compatible_models(base_url, api_key)
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Failed to fetch models for provider %s: %s", provider_name, exc)
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        if requested_family:
            models = [item for item in models if item.get("family") == requested_family]

        return {
            "provider": provider_name,
            "family": requested_family,
            "models": models,
        }


@app.post("/add-document", tags=["文档"], summary="新增单条文档")
async def add_document(doc: DocumentAdd, request: Request):
    """Add a single document."""
    try:
        _, request_context = _resolve_request_knowledge_base(
            request,
            kb_id=doc.kb_id,
            scope=doc.scope,
            collection=doc.collection,
            user_id=doc.user_id,
            agent_id=doc.agent_id,
            api_key=doc.api_key,
            operation="add_document",
        )
        context = request.app.state.shell_context
        async with context.observability.timer("add_document"):
            enforce_http_guardrails(request, request_context=request_context)
            service = await get_rag_service(request)
            return await service.add_document(
                DocumentRequest(
                    content=doc.content,
                    collection=request_context.kb_name or doc.collection,
                    metadata=doc.metadata,
                    kb_id=request_context.kb_id,
                    scope=request_context.kb_scope,
                    tenant=request_context.tenant,
                    context=request_context,
                )
            )
    except QuotaExceededError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except KnowledgeBaseAccessError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/upload-files", tags=["文档"], summary="上传文件")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    collection: str = Form("default"),
    kb_id: Optional[int] = Form(None),
    scope: Optional[str] = Form(None),
    user_id: Optional[int] = Form(None),
    agent_id: Optional[int] = Form(None),
    api_key: Optional[str] = Form(None),
):
    """Upload and process multiple files."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    _, request_context = _resolve_request_knowledge_base(
        request,
        kb_id=kb_id,
        scope=scope,
        collection=collection,
        user_id=user_id,
        agent_id=agent_id,
        api_key=api_key,
        operation="upload_files",
    )
    context = request.app.state.shell_context
    async with context.observability.timer("upload_files"):
        enforce_http_guardrails(request, request_context=request_context)
        service = await get_rag_service(request)
        return await service.upload_files(
            files,
            collection=collection,
            request_context=request_context,
        )


@app.get("/collections", tags=["知识库"], summary="列出历史集合")
async def list_collections(
    request: Request,
    user_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    api_key: Optional[str] = None,
):
    """List all collections."""
    request_context = build_http_request_context(
        request,
        tenant=normalize_tenant(base_collection="collections", user_id=user_id, agent_id=agent_id),
        base_collection="collections",
        user_id=user_id,
        agent_id=agent_id,
        api_key=api_key,
        operation="list_collections",
        subject=request_subject(
            request,
            normalize_tenant(base_collection="collections", user_id=user_id, agent_id=agent_id),
            fallback="collections",
        ),
    )
    context = request.app.state.shell_context
    async with context.observability.timer("list_collections"):
        enforce_http_guardrails(request, request_context=request_context)
        collections = [item.collection_name for item in context.knowledge_bases.list_accessible(user_id=user_id, agent_id=agent_id)]
    return {"collections": collections}


@app.get("/knowledge-bases", tags=["知识库"], summary="列出知识库")
async def list_knowledge_bases(
    request: Request,
    user_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    api_key: Optional[str] = None,
):
    """List accessible knowledge bases."""

    tenant = normalize_tenant(base_collection="knowledge_bases", user_id=user_id, agent_id=agent_id)
    request_context = build_http_request_context(
        request,
        tenant=tenant,
        base_collection="knowledge_bases",
        user_id=user_id,
        agent_id=agent_id,
        api_key=api_key,
        operation="list_knowledge_bases",
        subject=request_subject(request, tenant, fallback="knowledge_bases"),
    )
    context = request.app.state.shell_context
    async with context.observability.timer("list_knowledge_bases"):
        enforce_http_guardrails(request, request_context=request_context)
        items = context.knowledge_bases.list_accessible(user_id=user_id, agent_id=agent_id)
    return {"knowledge_bases": [item.to_dict() for item in items]}


@app.post("/knowledge-bases", tags=["知识库"], summary="创建知识库")
async def create_knowledge_base(payload: KnowledgeBaseCreate, request: Request):
    """Create a public or agent-private knowledge base."""

    tenant = normalize_tenant(
        base_collection="knowledge_bases",
        user_id=payload.owner_user_id,
        agent_id=payload.owner_agent_id,
    )
    request_context = build_http_request_context(
        request,
        tenant=tenant,
        base_collection="knowledge_bases",
        user_id=payload.owner_user_id,
        agent_id=payload.owner_agent_id,
        api_key=payload.api_key,
        operation="create_knowledge_base",
        subject=request_subject(request, tenant, fallback="knowledge_bases:create"),
    )
    context = request.app.state.shell_context
    async with context.observability.timer("create_knowledge_base"):
        enforce_http_guardrails(request, request_context=request_context)
        knowledge_base = context.knowledge_bases.create_knowledge_base(
            name=payload.name,
            scope=payload.scope,
            owner_user_id=payload.owner_user_id,
            owner_agent_id=payload.owner_agent_id,
        )
    return knowledge_base.to_dict()


@app.get("/debug/mcp/tools", tags=["MCP 调试"], summary="列出 MCP 工具")
async def debug_mcp_tools(request: Request, api_key: Optional[str] = None):
    """List MCP tools for the debug UI."""

    tenant = normalize_tenant(base_collection="mcp_debug")
    request_context = build_http_request_context(
        request,
        tenant=tenant,
        base_collection="mcp_debug",
        api_key=api_key,
        operation="debug_mcp_tools",
        subject="mcp_debug",
    )
    context = request.app.state.shell_context
    async with context.observability.timer("debug_mcp_tools"):
        enforce_http_guardrails(request, request_context=request_context)
        return {"tools": mcp_server.debug_tools()}


@app.post("/debug/mcp/call", tags=["MCP 调试"], summary="调试调用 MCP 工具")
async def debug_mcp_call(payload: MCPDebugCall, request: Request):
    """Call one MCP tool through an HTTP debug facade."""

    tenant = normalize_tenant(base_collection="mcp_debug")
    request_context = build_http_request_context(
        request,
        tenant=tenant,
        base_collection="mcp_debug",
        api_key=payload.api_key,
        operation="debug_mcp_call",
        subject="mcp_debug",
    )
    context = request.app.state.shell_context
    async with context.observability.timer("debug_mcp_call"):
        enforce_http_guardrails(request, request_context=request_context)
        result = await mcp_server.debug_call_tool(payload.tool, payload.arguments)
    return result


@app.post("/chat", tags=["对话与检索"], summary="对话问答")
async def chat_with_knowledge_base(chat_request: dict, request: Request):
    """Chat with knowledge base using LLM."""
    query = chat_request.get("query", "")
    collection = chat_request.get("collection", "default")
    kb_id = chat_request.get("kb_id")
    kb_ids = _parse_kb_ids(chat_request.get("kb_ids"))
    scope = chat_request.get("scope")
    limit = int(chat_request.get("limit", 5) or 5)
    user_id = chat_request.get("user_id")
    agent_id = chat_request.get("agent_id")
    api_key = chat_request.get("api_key")

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    context = request.app.state.shell_context
    async with context.observability.timer("chat"):
        service = await get_rag_service(request)
        if kb_ids:
            contexts, search_response = await _search_across_knowledge_bases(
                request=request,
                service=service,
                query=query,
                collection=collection,
                kb_ids=kb_ids,
                scope=scope,
                limit=limit,
                user_id=user_id,
                agent_id=agent_id,
                api_key=api_key,
                operation="chat",
            )
            context_text = _format_chat_context(search_response.results)
            prompt = _build_chat_prompt(query, context_text)
            try:
                llm_model = await service.runtime.ensure_llm_model()
                answer = await llm_model.generate(prompt)
            except Exception as exc:
                logger.warning("LLM generation failed for multi-kb chat, using retrieval context fallback: %s", exc)
                answer = _format_llm_fallback_response(context_text, exc)
            response = ChatResponse(
                query=query,
                collection="multi_kb",
                response=answer,
                sources=search_response.results,
            )
        else:
            _, request_context = _resolve_request_knowledge_base(
                request,
                kb_id=kb_id,
                scope=scope,
                collection=collection,
                user_id=user_id,
                agent_id=agent_id,
                api_key=api_key,
                operation="chat",
            )
            enforce_http_guardrails(request, request_context=request_context)
            response = await service.chat(
                ChatRequest(
                    query=query,
                    collection=request_context.kb_name or collection,
                    limit=limit,
                    kb_id=request_context.kb_id,
                    scope=request_context.kb_scope,
                    tenant=request_context.tenant,
                    context=request_context,
                )
            )
    return response.to_dict()

@app.get("/search", tags=["对话与检索"], summary="检索知识库")
async def search_documents(
    request: Request,
    query: str,
    collection: str = "default",
    kb_id: Optional[int] = None,
    kb_ids: Optional[str] = None,
    scope: Optional[str] = None,
    limit: int = 5,
    user_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    api_key: Optional[str] = None,
):
    """Search documents."""
    logger.info("Searching for '%s' in collection '%s'", query, collection)
    parsed_kb_ids = _parse_kb_ids(kb_ids)
    context = request.app.state.shell_context
    async with context.observability.timer("search"):
        service = await get_rag_service(request)
        if parsed_kb_ids:
            contexts, response = await _search_across_knowledge_bases(
                request=request,
                service=service,
                query=query,
                collection=collection,
                kb_ids=parsed_kb_ids,
                scope=scope,
                limit=limit,
                user_id=user_id,
                agent_id=agent_id,
                api_key=api_key,
                operation="search",
            )
        else:
            _, request_context = _resolve_request_knowledge_base(
                request,
                kb_id=kb_id,
                scope=scope,
                collection=collection,
                user_id=user_id,
                agent_id=agent_id,
                api_key=api_key,
                operation="search",
            )
            enforce_http_guardrails(request, request_context=request_context)
            response = await service.search(
                SearchRequest(
                    query=query,
                    collection=request_context.kb_name or collection,
                    limit=limit,
                    kb_id=request_context.kb_id,
                    scope=request_context.kb_scope,
                    tenant=request_context.tenant,
                    context=request_context,
                )
            )
    return response.to_dict()

@app.get("/list-documents", tags=["文档"], summary="列出文档记录")
async def list_documents(
    request: Request,
    collection: str = "default",
    kb_id: Optional[int] = None,
    scope: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    filename: str = None,
    user_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    api_key: Optional[str] = None,
):
    """List documents in a collection."""
    _, request_context = _resolve_request_knowledge_base(
        request,
        kb_id=kb_id,
        scope=scope,
        collection=collection,
        user_id=user_id,
        agent_id=agent_id,
        api_key=api_key,
        operation="list_documents",
    )
    context = request.app.state.shell_context
    async with context.observability.timer("list_documents"):
        enforce_http_guardrails(request, request_context=request_context)
        service = await get_rag_service(request)
        result = await service.list_documents(
            collection=request_context.kb_name or collection,
            limit=limit,
            offset=offset,
            filename=filename,
            request_context=request_context,
        )
    return result


@app.delete("/delete-document", tags=["文档"], summary="删除文档记录")
async def delete_document(document_request: DeleteDocumentRequest, request: Request):
    """Delete a document."""
    try:
        _, request_context = _resolve_request_knowledge_base(
            request,
            kb_id=document_request.kb_id,
            scope=document_request.scope,
            collection=document_request.collection,
            user_id=document_request.user_id,
            agent_id=document_request.agent_id,
            api_key=document_request.api_key,
            operation="delete_document",
        )
        context = request.app.state.shell_context
        async with context.observability.timer("delete_document"):
            enforce_http_guardrails(request, request_context=request_context)
            service = await get_rag_service(request)
            success = await service.delete_document(
                document_id=document_request.document_id,
                collection=request_context.kb_name or document_request.collection,
                request_context=request_context,
            )
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found or failed to delete")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-files", tags=["文档"], summary="列出文件")
async def list_files(
    request: Request,
    collection: str = "default",
    kb_id: Optional[int] = None,
    scope: Optional[str] = None,
    user_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    api_key: Optional[str] = None,
):
    """List files in a collection."""
    try:
        _, request_context = _resolve_request_knowledge_base(
            request,
            kb_id=kb_id,
            scope=scope,
            collection=collection,
            user_id=user_id,
            agent_id=agent_id,
            api_key=api_key,
            operation="list_files",
        )
        context = request.app.state.shell_context
        async with context.observability.timer("list_files"):
            enforce_http_guardrails(request, request_context=request_context)
            service = await get_rag_service(request)
            result = await service.list_files(
                collection=request_context.kb_name or collection,
                request_context=request_context,
            )
        return {"files": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-file", tags=["文档"], summary="删除文件")
async def delete_file(file_request: DeleteFileRequest, request: Request):
    """Delete a file."""
    try:
        _, request_context = _resolve_request_knowledge_base(
            request,
            kb_id=file_request.kb_id,
            scope=file_request.scope,
            collection=file_request.collection,
            user_id=file_request.user_id,
            agent_id=file_request.agent_id,
            api_key=file_request.api_key,
            operation="delete_file",
        )
        context = request.app.state.shell_context
        async with context.observability.timer("delete_file"):
            enforce_http_guardrails(request, request_context=request_context)
            service = await get_rag_service(request)
            success = await service.delete_file(
                filename=file_request.filename,
                collection=request_context.kb_name or file_request.collection,
                request_context=request_context,
            )
        if success:
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found or failed to delete")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
