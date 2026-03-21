"""HTTP server for MCP-RAG configuration and document management."""

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

from .config import config_manager
from .contracts import ChatRequest, DocumentRequest, SearchRequest, normalize_tenant
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
    metadata: Dict[str, Any] = Field(default_factory=dict)
    user_id: Optional[int] = None
    agent_id: Optional[int] = None
    api_key: Optional[str] = None


class DeleteDocumentRequest(BaseModel):
    """Delete document request model."""
    document_id: str
    collection: str = "default"
    user_id: Optional[int] = None
    agent_id: Optional[int] = None
    api_key: Optional[str] = None


class DeleteFileRequest(BaseModel):
    """Delete file request model."""
    filename: str
    collection: str = "default"
    user_id: Optional[int] = None
    agent_id: Optional[int] = None
    api_key: Optional[str] = None


@app.get("/")
async def root():
    """Root endpoint - redirect to the SPA shell."""
    return RedirectResponse(url="/app")


@app.get("/doc")
async def doc_redirect():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health(request: Request):
    """Lightweight health summary without warming the runtime."""
    return health_payload(request.app.state.shell_context)


@app.get("/ready")
async def ready(request: Request):
    """Readiness signal for shell wiring."""
    payload = ready_payload(request.app.state.shell_context)
    return JSONResponse(status_code=200 if payload["ready"] else 503, content=payload)


@app.get("/metrics")
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


@app.get("/app", response_class=HTMLResponse)
@app.get("/app/{spa_path:path}", response_class=HTMLResponse)
async def spa_entry(spa_path: str = ""):
    """Serve the SPA shell for browser clients."""

    request_path = "/app" if not spa_path else f"/app/{spa_path}"
    return _serve_spa_entry(request_path=request_path)


@app.get("/documents-page")
async def documents_page():
    """Backward-compatible redirect to the SPA documents view."""

    return RedirectResponse(url=_spa_redirect_target("documents"))


@app.get("/config-page")
async def config_page():
    """Backward-compatible redirect to the SPA config view."""

    return RedirectResponse(url=_spa_redirect_target("config"))


@app.get("/config")
async def get_config(request: Request, api_key: Optional[str] = None):
    """Get current configuration."""
    context = request.app.state.shell_context
    async with context.observability.timer("config.get"):
        request_context = build_http_request_context(request, api_key=api_key, subject="config")
        enforce_http_guardrails(request, request_context=request_context)
        return config_manager.get_all_settings()


@app.post("/config")
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


@app.post("/config/bulk")
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


@app.post("/config/reset")
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


@app.post("/config/reload")
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


@app.post("/add-document")
async def add_document(doc: DocumentAdd, request: Request):
    """Add a single document."""
    try:
        request_context = build_http_request_context(
            request,
            base_collection=doc.collection,
            user_id=doc.user_id,
            agent_id=doc.agent_id,
            api_key=doc.api_key,
            subject=request_subject(
                request,
                normalize_tenant(
                    base_collection=doc.collection,
                    user_id=doc.user_id,
                    agent_id=doc.agent_id,
                ),
                fallback="documents",
            ),
        )
        context = request.app.state.shell_context
        async with context.observability.timer("add_document"):
            enforce_http_guardrails(request, request_context=request_context)
            service = await get_rag_service(request)
            return await service.add_document(
                DocumentRequest(
                    content=doc.content,
                    collection=doc.collection,
                    metadata=doc.metadata,
                    tenant=request_context.tenant,
                    context=request_context,
                )
            )
    except QuotaExceededError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


@app.post("/upload-files")
async def upload_files(
    request: Request,
    files: List[UploadFile] = File(...),
    collection: str = Form("default"),
    user_id: Optional[int] = Form(None),
    agent_id: Optional[int] = Form(None),
    api_key: Optional[str] = Form(None),
):
    """Upload and process multiple files."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    tenant = normalize_tenant(
        base_collection=collection,
        user_id=user_id,
        agent_id=agent_id,
    )
    request_context = build_http_request_context(
        request,
        tenant=tenant,
        base_collection=collection,
        user_id=user_id,
        agent_id=agent_id,
        api_key=api_key,
        subject=request_subject(request, tenant, fallback=f"upload:{collection}"),
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


@app.get("/collections")
async def list_collections(
    request: Request,
    user_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    api_key: Optional[str] = None,
):
    """List all collections."""
    tenant = normalize_tenant(
        base_collection="default",
        user_id=user_id,
        agent_id=agent_id,
    )
    request_context = build_http_request_context(
        request,
        tenant=tenant,
        base_collection="default",
        user_id=user_id,
        agent_id=agent_id,
        api_key=api_key,
        subject=request_subject(request, tenant, fallback="collections"),
    )
    context = request.app.state.shell_context
    async with context.observability.timer("list_collections"):
        enforce_http_guardrails(request, request_context=request_context)
        service = await get_rag_service(request)
        collections = await service.list_collections(request_context=request_context)
    return {"collections": collections}


@app.post("/chat")
async def chat_with_knowledge_base(chat_request: dict, request: Request):
    """Chat with knowledge base using LLM."""
    query = chat_request.get("query", "")
    collection = chat_request.get("collection", "default")
    limit = int(chat_request.get("limit", 5) or 5)
    tenant = normalize_tenant(
        chat_request.get("tenant"),
        base_collection=collection,
        user_id=chat_request.get("user_id"),
        agent_id=chat_request.get("agent_id"),
    )
    request_context = build_http_request_context(
        request,
        tenant=tenant,
        base_collection=collection,
        user_id=chat_request.get("user_id"),
        agent_id=chat_request.get("agent_id"),
        api_key=chat_request.get("api_key"),
        subject=request_subject(request, tenant, fallback=f"chat:{collection}"),
    )

    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    context = request.app.state.shell_context
    async with context.observability.timer("chat"):
        enforce_http_guardrails(request, request_context=request_context)
        service = await get_rag_service(request)
        response = await service.chat(
            ChatRequest(
                query=query,
                collection=collection,
                limit=limit,
                tenant=request_context.tenant,
                context=request_context,
            )
        )
    return response.to_dict()

@app.get("/search")
async def search_documents(
    request: Request,
    query: str,
    collection: str = "default",
    limit: int = 5,
    user_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    api_key: Optional[str] = None,
):
    """Search documents."""
    logger.info("Searching for '%s' in collection '%s'", query, collection)
    tenant = normalize_tenant(
        base_collection=collection,
        user_id=user_id,
        agent_id=agent_id,
    )
    request_context = build_http_request_context(
        request,
        tenant=tenant,
        base_collection=collection,
        user_id=user_id,
        agent_id=agent_id,
        api_key=api_key,
        subject=request_subject(request, tenant, fallback=f"search:{collection}"),
    )
    context = request.app.state.shell_context
    async with context.observability.timer("search"):
        enforce_http_guardrails(request, request_context=request_context)
        service = await get_rag_service(request)
        response = await service.search(
            SearchRequest(
                query=query,
                collection=collection,
                limit=limit,
                tenant=request_context.tenant,
                context=request_context,
            )
        )
    return response.to_dict()

@app.get("/list-documents")
async def list_documents(
    request: Request,
    collection: str = "default",
    limit: int = 100,
    offset: int = 0,
    filename: str = None,
    user_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    api_key: Optional[str] = None,
):
    """List documents in a collection."""
    tenant = normalize_tenant(
        base_collection=collection,
        user_id=user_id,
        agent_id=agent_id,
    )
    request_context = build_http_request_context(
        request,
        tenant=tenant,
        base_collection=collection,
        user_id=user_id,
        agent_id=agent_id,
        api_key=api_key,
        subject=request_subject(request, tenant, fallback=f"list_documents:{collection}"),
    )
    context = request.app.state.shell_context
    async with context.observability.timer("list_documents"):
        enforce_http_guardrails(request, request_context=request_context)
        service = await get_rag_service(request)
        result = await service.list_documents(
            collection=collection,
            limit=limit,
            offset=offset,
            filename=filename,
            request_context=request_context,
        )
    return result


@app.delete("/delete-document")
async def delete_document(document_request: DeleteDocumentRequest, request: Request):
    """Delete a document."""
    try:
        tenant = normalize_tenant(
            base_collection=document_request.collection,
            user_id=document_request.user_id,
            agent_id=document_request.agent_id,
        )
        request_context = build_http_request_context(
            request,
            tenant=tenant,
            base_collection=document_request.collection,
            user_id=document_request.user_id,
            agent_id=document_request.agent_id,
            api_key=document_request.api_key,
            subject=request_subject(request, tenant, fallback=f"delete_document:{document_request.collection}"),
        )
        context = request.app.state.shell_context
        async with context.observability.timer("delete_document"):
            enforce_http_guardrails(request, request_context=request_context)
            service = await get_rag_service(request)
            success = await service.delete_document(
                document_id=document_request.document_id,
                collection=document_request.collection,
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


@app.get("/list-files")
async def list_files(
    request: Request,
    collection: str = "default",
    user_id: Optional[int] = None,
    agent_id: Optional[int] = None,
    api_key: Optional[str] = None,
):
    """List files in a collection."""
    try:
        tenant = normalize_tenant(
            base_collection=collection,
            user_id=user_id,
            agent_id=agent_id,
        )
        request_context = build_http_request_context(
            request,
            tenant=tenant,
            base_collection=collection,
            user_id=user_id,
            agent_id=agent_id,
            api_key=api_key,
            subject=request_subject(request, tenant, fallback=f"list_files:{collection}"),
        )
        context = request.app.state.shell_context
        async with context.observability.timer("list_files"):
            enforce_http_guardrails(request, request_context=request_context)
            service = await get_rag_service(request)
            result = await service.list_files(
                collection=collection,
                request_context=request_context,
            )
        return {"files": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-file")
async def delete_file(file_request: DeleteFileRequest, request: Request):
    """Delete a file."""
    try:
        tenant = normalize_tenant(
            base_collection=file_request.collection,
            user_id=file_request.user_id,
            agent_id=file_request.agent_id,
        )
        request_context = build_http_request_context(
            request,
            tenant=tenant,
            base_collection=file_request.collection,
            user_id=file_request.user_id,
            agent_id=file_request.agent_id,
            api_key=file_request.api_key,
            subject=request_subject(request, tenant, fallback=f"delete_file:{file_request.collection}"),
        )
        context = request.app.state.shell_context
        async with context.observability.timer("delete_file"):
            enforce_http_guardrails(request, request_context=request_context)
            service = await get_rag_service(request)
            success = await service.delete_file(
                filename=file_request.filename,
                collection=file_request.collection,
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
