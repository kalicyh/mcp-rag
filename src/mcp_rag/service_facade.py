"""Shell-facing RAG service facade.

The facade keeps HTTP and MCP handlers thin while delegating all indexing and
retrieval work to the new core.indexing + retrieval primitives.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

from fastapi import UploadFile

from .config import settings
from .contracts import (
    BatchUploadResponse,
    ChatRequest,
    ChatResponse,
    DocumentRequest,
    SearchRequest,
    SearchResponse,
    SearchResultView,
    TenantSpec,
    UploadFileResult,
    normalize_tenant,
)
from .core.indexing import (
    ChromaVectorStore,
    DocumentProcessor,
    IndexingSettings,
    OpenAICompatibleEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from .core.indexing.models import TenantContext as CoreTenantContext
from .retrieval import HybridRetrievalService
from .llm import get_llm_model

logger = logging.getLogger(__name__)

_FILE_NAME_RE = re.compile(r"[/\\:*?\"<>|]+")


class RagService:
    """Tenant-aware service facade used by HTTP and MCP shells."""

    def __init__(
        self,
        *,
        settings_obj=settings,
        document_processor: DocumentProcessor | None = None,
        embedding_model=None,
        vector_store: ChromaVectorStore | None = None,
        hybrid_service: HybridRetrievalService | None = None,
        llm_model=None,
    ):
        self.settings = settings_obj
        self._document_processor = document_processor
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._hybrid_service = hybrid_service
        self._llm_model = llm_model
        self._init_lock = asyncio.Lock()

        # Injected dependencies are assumed to be ready.
        self._embedding_ready = embedding_model is not None
        self._vector_ready = vector_store is not None
        self._hybrid_ready = hybrid_service is not None
        self._llm_ready = llm_model is not None
        self._document_ready = document_processor is not None

    async def add_document(self, request: DocumentRequest) -> Dict[str, Any]:
        tenant = request.tenant.to_core()
        processor = await self._ensure_document_processor()
        vector_store = await self._ensure_vector_store()
        embedding_model = await self._ensure_embedding_model()
        self._attach_embedding_model(vector_store, embedding_model)

        filename = self._resolve_filename(request.metadata, fallback="manual_input")
        processed = processor.process_text(
            request.content,
            source=request.metadata.get("source", "manual_input"),
            filename=filename,
            file_type=request.metadata.get("file_type", "text"),
            metadata=request.metadata,
        )
        chunks = processor.chunk_document(processed)
        if not chunks:
            raise ValueError("No content extracted from document")

        await vector_store.upsert_chunks(
            chunks,
            tenant=tenant,
            collection_name=request.collection,
        )
        await self._refresh_keywords(request.collection, tenant)

        return {
            "message": "Document added successfully",
            "document_id": processed.document_id,
            "chunk_count": len(chunks),
        }

    async def upload_files(
        self,
        files: Sequence[UploadFile],
        *,
        collection: str = "default",
        tenant: TenantSpec | Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        tenant_spec = normalize_tenant(tenant, base_collection=collection)
        processor = await self._ensure_document_processor()
        vector_store = await self._ensure_vector_store()
        embedding_model = await self._ensure_embedding_model()
        self._attach_embedding_model(vector_store, embedding_model)

        results: list[UploadFileResult] = []
        for upload in files:
            temp_path: Path | None = None
            try:
                temp_path = await self._write_upload_to_tempfile(upload)
                processed_doc = processor.process_file(
                    temp_path,
                    metadata={"source": "upload", "filename": upload.filename},
                    filename=upload.filename,
                )
                if processed_doc.error or not processed_doc.content.strip():
                    results.append(
                        UploadFileResult(
                            filename=upload.filename,
                            file_type=processed_doc.file_type,
                            content_length=len(processed_doc.content),
                            processed=False,
                            error=processed_doc.error or "No content extracted",
                            preview="",
                        )
                    )
                    continue

                chunks = processor.chunk_document(processed_doc)
                await vector_store.upsert_chunks(
                    chunks,
                    tenant=tenant_spec.to_core(),
                    collection_name=tenant_spec.base_collection,
                )
                await self._refresh_keywords(tenant_spec.base_collection, tenant_spec.to_core())

                preview = processed_doc.content[:500]
                if len(processed_doc.content) > 500:
                    preview += "..."

                results.append(
                    UploadFileResult(
                        filename=upload.filename,
                        file_type=processed_doc.file_type,
                        content_length=len(processed_doc.content),
                        processed=True,
                        error="",
                        preview=preview,
                    )
                )
            except Exception as exc:
                logger.exception("Failed to process upload %s", getattr(upload, "filename", "unknown"))
                results.append(
                    UploadFileResult(
                        filename=getattr(upload, "filename", "unknown"),
                        file_type="unknown",
                        content_length=0,
                        processed=False,
                        error=str(exc),
                        preview="",
                    )
                )
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

        return BatchUploadResponse(
            total_files=len(files),
            successful=len([item for item in results if item.processed]),
            failed=len([item for item in results if not item.processed]),
            results=results,
        ).to_dict()

    async def search(
        self,
        request: SearchRequest,
    ) -> SearchResponse:
        tenant = request.tenant.to_core()
        hybrid = await self._ensure_hybrid_service()
        hits = await hybrid.retrieve(
            request.query,
            collection_name=request.collection,
            tenant=tenant,
            limit=request.limit,
            threshold=request.threshold,
        )

        results = [
            SearchResultView(
                content=hit.content,
                score=hit.score,
                metadata=dict(hit.metadata or {}),
                source=hit.source,
                filename=hit.filename,
                retrieval_method=hit.retrieval_method,
            )
            for hit in hits
        ]

        summary: str | None = None
        if hits and self.settings.enable_llm_summary:
            try:
                llm_model = await self._ensure_llm_model()
                combined_content = "\n\n".join(
                    f"文档 {index + 1} (相似度: {hit.score:.3f}):\n{hit.content}"
                    for index, hit in enumerate(hits)
                )
                summary = await llm_model.summarize(combined_content, request.query)
            except Exception as exc:
                logger.warning("LLM summary failed, falling back to raw results: %s", exc)

        return SearchResponse(
            query=request.query,
            collection=request.collection,
            results=results,
            summary=summary,
        )

    async def chat(
        self,
        request: ChatRequest,
    ) -> ChatResponse:
        search_response = await self.search(
            SearchRequest(
                query=request.query,
                collection=request.collection,
                limit=request.limit,
                threshold=0.7,
                tenant=request.tenant,
            )
        )

        context = "\n\n".join(
            f"文档 {index + 1} ({item.filename or item.source}):\n{item.content}"
            for index, item in enumerate(search_response.results)
        )
        prompt = (
            "基于以下知识库内容回答用户的问题。如果知识库内容不足以回答问题，请说明无法找到相关信息。\n\n"
            f"知识库内容:\n{context}\n\n"
            f"用户问题: {request.query}\n\n"
            "请提供准确、简洁的回答:"
        )

        try:
            llm_model = await self._ensure_llm_model()
            answer = await llm_model.generate(prompt)
        except Exception as exc:
            logger.warning("LLM generation failed, using retrieval context fallback: %s", exc)
            answer = (
                "### Retrieved Context\n\n"
                f"{context}\n\n"
                "### Note\n"
                "LLM is not available. The above context was retrieved for your query."
            )

        return ChatResponse(
            query=request.query,
            collection=request.collection,
            response=answer,
            sources=search_response.results,
        )

    async def ask(self, request: SearchRequest) -> SearchResponse:
        """Compatibility wrapper for rag_ask."""

        response = await self.search(request)
        if request.mode == "summary" and response.summary is None:
            try:
                llm_model = await self._ensure_llm_model()
                combined_content = "\n\n".join(
                    f"文档 {index + 1} (相似度: {item.score:.3f}):\n{item.content}"
                    for index, item in enumerate(response.results)
                )
                response.summary = await llm_model.summarize(combined_content, request.query)
            except Exception as exc:
                logger.warning("Summary mode fallback failed: %s", exc)
                response.summary = "摘要生成功能暂未启用。"
        return response

    async def list_documents(
        self,
        *,
        collection: str = "default",
        limit: int = 100,
        offset: int = 0,
        filename: str | None = None,
        tenant: TenantSpec | Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        vector_store = await self._ensure_vector_store()
        tenant_spec = normalize_tenant(tenant, base_collection=collection)
        return await vector_store.list_documents(
            collection_name=collection,
            limit=limit,
            offset=offset,
            filename=filename,
            tenant=tenant_spec.to_core(),
        )

    async def list_files(
        self,
        *,
        collection: str = "default",
        tenant: TenantSpec | Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
        vector_store = await self._ensure_vector_store()
        tenant_spec = normalize_tenant(tenant, base_collection=collection)
        return await vector_store.list_files(
            collection_name=collection,
            tenant=tenant_spec.to_core(),
        )

    async def list_collections(
        self,
        *,
        tenant: TenantSpec | Dict[str, Any] | None = None,
    ) -> List[str]:
        vector_store = await self._ensure_vector_store()
        collections = await vector_store.list_collections()
        names = [entry["name"] for entry in collections]

        tenant_spec = normalize_tenant(tenant) if tenant is not None else None
        if tenant_spec is None:
            return sorted(names)

        filtered: list[str] = []
        for item in collections:
            if tenant_spec.user_id is not None and item.get("user_id") != tenant_spec.user_id:
                continue
            if tenant_spec.agent_id is not None and item.get("agent_id") != tenant_spec.agent_id:
                continue
            filtered.append(str(item["name"]))
        return sorted(filtered)

    async def delete_document(
        self,
        *,
        document_id: str,
        collection: str = "default",
        tenant: TenantSpec | Dict[str, Any] | None = None,
    ) -> bool:
        vector_store = await self._ensure_vector_store()
        tenant_spec = normalize_tenant(tenant, base_collection=collection)
        result = await vector_store.delete_document(
            document_id,
            collection_name=collection,
            tenant=tenant_spec.to_core(),
        )
        await self._refresh_keywords(collection, tenant_spec.to_core())
        return result

    async def delete_file(
        self,
        *,
        filename: str,
        collection: str = "default",
        tenant: TenantSpec | Dict[str, Any] | None = None,
    ) -> bool:
        vector_store = await self._ensure_vector_store()
        tenant_spec = normalize_tenant(tenant, base_collection=collection)
        result = await vector_store.delete_file(
            filename,
            collection_name=collection,
            tenant=tenant_spec.to_core(),
        )
        await self._refresh_keywords(collection, tenant_spec.to_core())
        return result

    async def _ensure_document_processor(self) -> DocumentProcessor:
        if self._document_processor is None:
            indexing_settings = self._build_indexing_settings()
            self._document_processor = DocumentProcessor(indexing_settings)
            self._document_ready = True
        return self._document_processor

    async def _ensure_embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = self._build_embedding_model()

        if not self._embedding_ready:
            initializer = getattr(self._embedding_model, "initialize", None)
            if callable(initializer):
                await initializer()
            self._embedding_ready = True

        if self._vector_store is not None and hasattr(self._vector_store, "embedding_model"):
            self._vector_store.embedding_model = self._embedding_model
        return self._embedding_model

    async def _ensure_vector_store(self) -> ChromaVectorStore:
        if self._vector_store is None:
            self._vector_store = ChromaVectorStore(
                persist_directory=self.settings.chroma_persist_directory,
                embedding_model=self._embedding_model,
            )

        if not self._vector_ready:
            initializer = getattr(self._vector_store, "initialize", None)
            if callable(initializer):
                await initializer()
            self._vector_ready = True
        return self._vector_store

    async def _ensure_hybrid_service(self) -> HybridRetrievalService:
        if self._hybrid_service is None:
            embedding_model = await self._ensure_embedding_model()
            vector_store = await self._ensure_vector_store()
            self._hybrid_service = HybridRetrievalService(
                vector_store=vector_store,
                embedding_model=embedding_model,
                rerank_enabled=bool(self.settings.enable_reranker),
                candidate_pool_size=max(10, int(self.settings.max_retrieval_results)),
            )
        self._hybrid_ready = True
        return self._hybrid_service

    async def _ensure_llm_model(self):
        if self._llm_model is None:
            self._llm_model = await get_llm_model()
        self._llm_ready = True
        return self._llm_model

    def _build_indexing_settings(self) -> IndexingSettings:
        provider = (self.settings.embedding_provider or "zhipu").lower()
        provider_config = self.settings.provider_configs.get(provider)
        if provider_config is None:
            provider_config = next(iter(self.settings.provider_configs.values()))

        return IndexingSettings(
            persist_directory=self.settings.chroma_persist_directory,
            embedding_provider=provider,
            embedding_model=provider_config.model,
            embedding_base_url=provider_config.base_url,
            embedding_api_key=provider_config.api_key,
            embedding_device=self.settings.embedding_device,
            embedding_cache_dir=self.settings.embedding_cache_dir,
        )

    def _build_embedding_model(self):
        provider = (self.settings.embedding_provider or "").lower()
        if provider in {"doubao", "zhipu", "openai"}:
            provider_config = self.settings.provider_configs.get(provider)
            if provider_config is None:
                raise ValueError(f"Provider configuration not found for '{provider}'")
            return OpenAICompatibleEmbeddingModel(
                api_key=provider_config.api_key,
                base_url=provider_config.base_url,
                model=provider_config.model,
            )

        return SentenceTransformerEmbeddingModel(
            model_name=provider or "m3e-small",
            device=self.settings.embedding_device,
            cache_dir=self.settings.embedding_cache_dir,
        )

    def _attach_embedding_model(self, vector_store: ChromaVectorStore, embedding_model) -> None:
        if hasattr(vector_store, "embedding_model"):
            vector_store.embedding_model = embedding_model

    async def _refresh_keywords(self, collection: str, tenant: CoreTenantContext) -> None:
        if self._hybrid_service is not None:
            collection_index = getattr(self._hybrid_service, "collection_index", None)
            if collection_index is not None and hasattr(collection_index, "refresh"):
                try:
                    await collection_index.refresh(collection_name=collection, tenant=tenant)
                except Exception as exc:
                    logger.warning("Keyword index refresh failed for %s: %s", collection, exc)

    async def _write_upload_to_tempfile(self, upload: UploadFile) -> Path:
        suffix = Path(upload.filename or "").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(upload.file, temp_file)
            return Path(temp_file.name)

    def _resolve_filename(self, metadata: Dict[str, Any], fallback: str = "manual_input") -> str:
        raw = str(metadata.get("filename") or metadata.get("title") or fallback or "manual_input").strip()
        raw = _FILE_NAME_RE.sub("_", raw)
        return raw[:80] or "manual_input"


_rag_service: RagService | None = None
_rag_service_lock: asyncio.Lock | None = None


async def get_rag_service() -> RagService:
    """Get the singleton shell service."""

    global _rag_service
    global _rag_service_lock
    if _rag_service is None:
        if _rag_service_lock is None:
            _rag_service_lock = asyncio.Lock()
        async with _rag_service_lock:
            if _rag_service is None:
                _rag_service = RagService()
    return _rag_service
