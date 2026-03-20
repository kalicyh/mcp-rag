"""Shared runtime container for MCP-RAG services."""

from __future__ import annotations

import asyncio
import logging
from threading import Lock
from typing import Any

from ..config import settings
from ..core.indexing import (
    ChromaVectorStore,
    DocumentProcessor,
    IndexingSettings,
    OpenAICompatibleEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from ..core.indexing.models import TenantContext as CoreTenantContext
from ..core.indexing.tenancy import resolve_collection_name
from ..llm import get_llm_model
from ..retrieval import HybridRetrievalService
from .retrieval_cache import RetrievalCache, build_scope_token

logger = logging.getLogger(__name__)


class ServiceRuntime:
    """Lazy dependency container shared by the service layer."""

    def __init__(
        self,
        *,
        settings_obj=settings,
        document_processor: DocumentProcessor | None = None,
        embedding_model: Any | None = None,
        vector_store: ChromaVectorStore | None = None,
        hybrid_service: HybridRetrievalService | None = None,
        llm_model: Any | None = None,
    ):
        self.settings = settings_obj
        self._document_processor = document_processor
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._hybrid_service = hybrid_service
        self._llm_model = llm_model

        self._document_ready = document_processor is not None
        self._embedding_ready = embedding_model is not None
        self._vector_ready = vector_store is not None
        self._hybrid_ready = hybrid_service is not None
        self._llm_ready = llm_model is not None

        self._document_lock = asyncio.Lock()
        self._embedding_lock = asyncio.Lock()
        self._vector_lock = asyncio.Lock()
        self._hybrid_lock = asyncio.Lock()
        self._llm_lock = asyncio.Lock()
        self._reload_lock = asyncio.Lock()
        self._retrieval_cache: RetrievalCache | None = None
        self._retrieval_cache_lock = Lock()

    async def ensure_document_processor(self) -> DocumentProcessor:
        if self._document_processor is not None and self._document_ready:
            return self._document_processor

        async with self._document_lock:
            if self._document_processor is None:
                self._document_processor = DocumentProcessor(self.build_indexing_settings())
            self._document_ready = True
            return self._document_processor

    async def ensure_embedding_model(self):
        if self._embedding_model is None:
            async with self._embedding_lock:
                if self._embedding_model is None:
                    self._embedding_model = self.build_embedding_model()

        if not self._embedding_ready:
            initializer = getattr(self._embedding_model, "initialize", None)
            if callable(initializer):
                await initializer()
            self._embedding_ready = True

        if self._vector_store is not None and hasattr(self._vector_store, "embedding_model"):
            self._vector_store.embedding_model = self._embedding_model
        return self._embedding_model

    async def ensure_vector_store(self) -> ChromaVectorStore:
        if self._vector_store is None:
            async with self._vector_lock:
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

    async def ensure_hybrid_service(self) -> HybridRetrievalService:
        if self._hybrid_service is None:
            async with self._hybrid_lock:
                if self._hybrid_service is None:
                    embedding_model = await self.ensure_embedding_model()
                    vector_store = await self.ensure_vector_store()
                    self._hybrid_service = HybridRetrievalService(
                        vector_store=vector_store,
                        embedding_model=embedding_model,
                        rerank_enabled=bool(self.settings.enable_reranker),
                        candidate_pool_size=max(10, int(self.settings.max_retrieval_results)),
                    )
        self._hybrid_ready = True
        return self._hybrid_service

    async def ensure_llm_model(self):
        if self._llm_model is None:
            async with self._llm_lock:
                if self._llm_model is None:
                    self._llm_model = await get_llm_model(self.settings)
        self._llm_ready = True
        return self._llm_model

    def get_retrieval_cache(self) -> RetrievalCache | None:
        cache_settings = getattr(self.settings, "cache", None)
        enabled = bool(getattr(cache_settings, "enabled", False) or getattr(self.settings, "enable_cache", False))
        if not enabled:
            return None

        if self._retrieval_cache is None:
            with self._retrieval_cache_lock:
                if self._retrieval_cache is None:
                    self._retrieval_cache = RetrievalCache()
        return self._retrieval_cache

    async def invalidate_retrieval_cache(self, *, collection: str, tenant: CoreTenantContext) -> int:
        cache = self._retrieval_cache
        if cache is None:
            return 0

        actual_collection = resolve_collection_name(collection, tenant=tenant)
        scope_token = build_scope_token(
            actual_collection=actual_collection,
            base_collection=tenant.base_collection or collection,
            user_id=tenant.user_id,
            agent_id=tenant.agent_id,
        )
        return cache.invalidate_scope(scope_token=scope_token)

    async def invalidate_all_retrieval_cache(self) -> int:
        cache = self._retrieval_cache
        if cache is None:
            return 0
        return cache.invalidate_all()

    async def reload_settings(self, settings_obj) -> None:
        """Swap runtime settings and reset cached providers when needed."""

        async with self._reload_lock:
            previous_embedding_signature = self._embedding_signature()
            previous_runtime_signature = self._runtime_signature()
            previous_llm_signature = self._llm_signature()
            previous_retrieval_signature = self._retrieval_signature()

            self.settings = settings_obj

            if previous_embedding_signature != self._embedding_signature():
                self._embedding_model = None
                self._embedding_ready = False
                self._vector_store = None
                self._vector_ready = False
                self._hybrid_service = None
                self._hybrid_ready = False

            if previous_runtime_signature != self._runtime_signature():
                self._document_processor = None
                self._document_ready = False

            if previous_llm_signature != self._llm_signature():
                await self._close_component(self._llm_model)
                self._llm_model = None
                self._llm_ready = False

            if self._hybrid_service is not None:
                self._hybrid_service.rerank_enabled = bool(getattr(settings_obj, "enable_reranker", False))
                self._hybrid_service.candidate_pool_size = max(
                    10,
                    int(getattr(settings_obj, "max_retrieval_results", 5) or 5),
                )

            cache_settings = getattr(settings_obj, "cache", None)
            cache_enabled = bool(getattr(cache_settings, "enabled", False) or getattr(settings_obj, "enable_cache", False))
            if not cache_enabled:
                self._retrieval_cache = None
            elif previous_retrieval_signature != self._retrieval_signature():
                await self.invalidate_all_retrieval_cache()

    def build_indexing_settings(self) -> IndexingSettings:
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

    def build_embedding_model(self):
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

    async def refresh_keywords(self, collection: str, tenant: CoreTenantContext) -> None:
        if self._hybrid_service is None:
            return

        collection_index = getattr(self._hybrid_service, "collection_index", None)
        if collection_index is None or not hasattr(collection_index, "refresh"):
            return

        try:
            await collection_index.refresh(collection_name=collection, tenant=tenant)
        except Exception as exc:
            logger.warning("Keyword index refresh failed for %s: %s", collection, exc)

    def attach_embedding_model(self, vector_store: ChromaVectorStore, embedding_model) -> None:
        if hasattr(vector_store, "embedding_model"):
            vector_store.embedding_model = embedding_model

    def _runtime_signature(self) -> tuple[Any, ...]:
        indexing_settings = self.build_indexing_settings()
        return (
            indexing_settings.persist_directory,
            indexing_settings.embedding_provider,
            indexing_settings.embedding_model,
            indexing_settings.embedding_base_url,
            indexing_settings.embedding_api_key,
            indexing_settings.embedding_device,
            indexing_settings.embedding_cache_dir,
            bool(self.settings.enable_reranker),
            int(self.settings.max_retrieval_results),
        )

    def _embedding_signature(self) -> tuple[Any, ...]:
        indexing_settings = self.build_indexing_settings()
        return (
            indexing_settings.persist_directory,
            indexing_settings.embedding_provider,
            indexing_settings.embedding_model,
            indexing_settings.embedding_base_url,
            indexing_settings.embedding_api_key,
            indexing_settings.embedding_device,
            indexing_settings.embedding_cache_dir,
        )

    def _llm_signature(self) -> tuple[Any, ...]:
        return (
            str(getattr(self.settings, "llm_provider", "doubao") or "doubao").lower(),
            getattr(self.settings, "llm_model", ""),
            getattr(self.settings, "llm_base_url", ""),
            getattr(self.settings, "llm_api_key", None),
            bool(getattr(self.settings, "enable_thinking", True)),
        )

    def _retrieval_signature(self) -> tuple[Any, ...]:
        cache_settings = getattr(self.settings, "cache", None)
        return (
            bool(getattr(cache_settings, "enabled", False) or getattr(self.settings, "enable_cache", False)),
            bool(getattr(self.settings, "enable_reranker", False)),
            int(getattr(self.settings, "max_retrieval_results", 5) or 5),
            float(getattr(self.settings, "similarity_threshold", 0.7) or 0.7),
            bool(getattr(self.settings, "enable_llm_summary", False)),
            str(getattr(self.settings, "llm_provider", "doubao") or "doubao").lower(),
            getattr(self.settings, "llm_model", ""),
        )

    async def _close_component(self, component: Any) -> None:
        if component is None:
            return

        closer = getattr(component, "close", None)
        if callable(closer):
            await closer()
            return

        async_closer = getattr(component, "aclose", None)
        if callable(async_closer):
            await async_closer()

    async def close(self) -> None:
        """Close runtime-managed resources when they expose async teardown."""

        await self._close_component(self._llm_model)


RuntimeContainer = ServiceRuntime
