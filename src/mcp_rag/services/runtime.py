"""Shared runtime container for MCP-RAG services."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import monotonic
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
from ..llm import DoubaoLLMModel, LLMModel, OllamaModel, get_llm_model
from ..retrieval import HybridRetrievalService
from ..security import TokenBucketRateLimiter
from .retrieval_cache import RetrievalCache, build_scope_token

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ProviderBudgetState:
    """In-memory budget and simple circuit-breaker state for one provider family."""

    limiter: TokenBucketRateLimiter
    failure_threshold: int
    cooldown_seconds: int
    consecutive_failures: int = 0
    cooldown_until: float = 0.0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def allow(self, subject: str) -> tuple[bool, float]:
        with self._lock:
            now = monotonic()
            if self.cooldown_until > now:
                return False, max(0.0, self.cooldown_until - now)

        if self.limiter.limit <= 0:
            return True, 0.0

        decision = self.limiter.allow(subject)
        return decision.allowed, decision.retry_after_seconds

    def record(self, *, success: bool) -> None:
        with self._lock:
            if success:
                self.consecutive_failures = 0
                self.cooldown_until = 0.0
                return

            self.consecutive_failures += 1
            if self.failure_threshold > 0 and self.consecutive_failures >= self.failure_threshold:
                self.cooldown_until = monotonic() + max(0, self.cooldown_seconds)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = monotonic()
            cooldown_remaining = max(0.0, self.cooldown_until - now)
            return {
                "enabled": True,
                "status": "cooldown" if cooldown_remaining > 0 else "enabled",
                "ready": True,
                "failure_threshold": self.failure_threshold,
                "consecutive_failures": self.consecutive_failures,
                "cooldown_seconds": self.cooldown_seconds,
                "cooldown_remaining_seconds": cooldown_remaining,
                "limit": self.limiter.limit,
                "window_seconds": self.limiter.window_seconds,
                "burst": self.limiter.burst,
            }


class _BudgetedEmbeddingModel:
    """Embedding wrapper that routes calls through runtime governance hooks."""

    def __init__(self, runtime: ServiceRuntime, model: Any):
        self._runtime = runtime
        self._model = model

    async def initialize(self) -> None:
        initializer = getattr(self._model, "initialize", None)
        if callable(initializer):
            await initializer()

    async def encode(self, texts):
        return await self._runtime._invoke_embedding("encode", self._model.encode, texts)

    async def encode_single(self, text: str):
        return await self._runtime._invoke_embedding("encode_single", self._model.encode_single, text)

    async def close(self) -> None:
        await self._runtime._close_component(self._model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class _BudgetedLLMModel:
    """LLM wrapper that routes calls through runtime governance hooks."""

    def __init__(self, runtime: ServiceRuntime, model: LLMModel):
        self._runtime = runtime
        self._model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        return await self._runtime._invoke_llm("generate", self._model.generate, prompt, **kwargs)

    async def summarize(self, content: str, query: str) -> str:
        return await self._runtime._invoke_llm("summarize", self._model.summarize, content, query)

    async def close(self) -> None:
        await self._runtime._close_component(self._model)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)


class ServiceRuntime:
    """Lazy dependency container shared by the service layer."""

    def __init__(
        self,
        *,
        settings_obj=settings,
        observability=None,
        document_processor: DocumentProcessor | None = None,
        embedding_model: Any | None = None,
        vector_store: ChromaVectorStore | None = None,
        hybrid_service: HybridRetrievalService | None = None,
        llm_model: Any | None = None,
    ):
        self.settings = settings_obj
        self.observability = observability
        self._document_processor = document_processor
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._hybrid_service = hybrid_service
        self._llm_model = llm_model
        self._fallback_embedding_model: Any | None = None
        self._fallback_llm_model: Any | None = None
        self._retrieval_cache: RetrievalCache | None = None

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
        self._fallback_embedding_lock = asyncio.Lock()
        self._fallback_llm_lock = asyncio.Lock()
        self._retrieval_cache_lock = Lock()
        self._provider_budgets = self._build_provider_budgets()

    async def ensure_document_processor(self) -> DocumentProcessor:
        if self._document_processor is not None and self._document_ready:
            return self._document_processor

        async with self._document_lock:
            if self._document_processor is None:
                self._document_processor = DocumentProcessor(self.build_indexing_settings())
            self._document_ready = True
            return self._document_processor

    async def ensure_embedding_model(self):
        started = self._clock()
        provider_label = self._embedding_provider_name()
        try:
            if self._embedding_model is None:
                async with self._embedding_lock:
                    if self._embedding_model is None:
                        self._embedding_model = self._budget_embedding_model(self.build_embedding_model())
            elif not isinstance(self._embedding_model, _BudgetedEmbeddingModel):
                self._embedding_model = self._budget_embedding_model(self._embedding_model)

            if not self._embedding_ready:
                initializer = getattr(self._embedding_model, "initialize", None)
                if callable(initializer):
                    await initializer()
                self._embedding_ready = True

            if self._vector_store is not None and hasattr(self._vector_store, "embedding_model"):
                self._vector_store.embedding_model = self._embedding_model

            self._record_provider_latency(provider_label, "ensure", started, success=True)
            return self._embedding_model
        except Exception as exc:
            self._record_provider_latency(provider_label, "ensure", started, success=False, error=exc.__class__.__name__)
            raise

    async def ensure_vector_store(self) -> ChromaVectorStore:
        started = self._clock()
        try:
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

            self._record_provider_latency("vector_store", "ensure", started, success=True)
            return self._vector_store
        except Exception as exc:
            self._record_provider_latency("vector_store", "ensure", started, success=False, error=exc.__class__.__name__)
            raise

    async def ensure_hybrid_service(self) -> HybridRetrievalService:
        started = self._clock()
        try:
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
            self._record_provider_latency("retrieval", "ensure", started, success=True)
            return self._hybrid_service
        except Exception as exc:
            self._record_provider_latency("retrieval", "ensure", started, success=False, error=exc.__class__.__name__)
            raise

    async def ensure_llm_model(self):
        started = self._clock()
        provider_label = self._llm_provider_name()
        try:
            if self._llm_model is None:
                async with self._llm_lock:
                    if self._llm_model is None:
                        self._llm_model = self._budget_llm_model(await get_llm_model(self.settings))
            elif not isinstance(self._llm_model, _BudgetedLLMModel):
                self._llm_model = self._budget_llm_model(self._llm_model)

            self._llm_ready = True
            self._record_provider_latency(provider_label, "ensure", started, success=True)
            return self._llm_model
        except Exception as exc:
            self._record_provider_latency(provider_label, "ensure", started, success=False, error=exc.__class__.__name__)
            raise

    async def reload_settings(self, settings_obj) -> None:
        """Swap runtime settings and reset cached providers when needed."""

        async with self._reload_lock:
            previous_embedding_signature = self._embedding_signature()
            previous_runtime_signature = self._runtime_signature()
            previous_llm_signature = self._llm_signature()
            previous_cache_signature = self._cache_signature()
            previous_embedding_fallback_signature = self._embedding_fallback_signature()
            previous_llm_fallback_signature = self._llm_fallback_signature()

            self.settings = settings_obj
            self._provider_budgets = self._build_provider_budgets()

            if previous_embedding_signature != self._embedding_signature():
                await self._close_component(self._embedding_model)
                await self._close_component(self._fallback_embedding_model)
                self._embedding_model = None
                self._fallback_embedding_model = None
                self._embedding_ready = False
                self._vector_store = None
                self._vector_ready = False
                self._hybrid_service = None
                self._hybrid_ready = False
            elif previous_embedding_fallback_signature != self._embedding_fallback_signature():
                await self._close_component(self._fallback_embedding_model)
                self._fallback_embedding_model = None

            if previous_runtime_signature != self._runtime_signature():
                self._document_processor = None
                self._document_ready = False

            if previous_llm_signature != self._llm_signature():
                await self._close_component(self._llm_model)
                self._llm_model = None
                self._llm_ready = False
            if previous_llm_fallback_signature != self._llm_fallback_signature():
                await self._close_component(self._fallback_llm_model)
                self._fallback_llm_model = None

            if self._hybrid_service is not None:
                self._hybrid_service.rerank_enabled = bool(getattr(settings_obj, "enable_reranker", False))
                self._hybrid_service.candidate_pool_size = max(
                    10,
                    int(getattr(settings_obj, "max_retrieval_results", 5) or 5),
                )

            if previous_cache_signature != self._cache_signature():
                self.reset_retrieval_cache()

    def build_indexing_settings(self) -> IndexingSettings:
        provider = self._embedding_provider_name()
        provider_config = self._provider_config(provider)

        return IndexingSettings(
            persist_directory=self.settings.chroma_persist_directory,
            embedding_provider=provider,
            embedding_model=(
                getattr(provider_config, "embedding_model", None)
                or getattr(provider_config, "model", None)
                or provider
                or "m3e-small"
            ),
            embedding_base_url=getattr(provider_config, "base_url", None),
            embedding_api_key=getattr(provider_config, "api_key", None),
            embedding_device=self.settings.embedding_device,
            embedding_cache_dir=self.settings.embedding_cache_dir,
        )

    def build_embedding_model(self, provider: str | None = None):
        provider_name = str(provider or self._embedding_provider_name() or "").lower()
        provider_config = self._provider_config(provider_name)
        if provider_config is not None:
            return OpenAICompatibleEmbeddingModel(
                api_key=provider_config.api_key,
                base_url=provider_config.base_url,
                model=provider_config.model,
            )

        return SentenceTransformerEmbeddingModel(
            model_name=provider_name or "m3e-small",
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

    def get_retrieval_cache(self) -> RetrievalCache | None:
        if not self._cache_enabled():
            return None

        cache_settings = getattr(self.settings, "cache", None)
        max_entries = int(getattr(cache_settings, "max_entries", 256) or 256)
        ttl_seconds = int(getattr(cache_settings, "ttl_seconds", 300) or 300)

        with self._retrieval_cache_lock:
            if self._retrieval_cache is None:
                self._retrieval_cache = RetrievalCache(max_entries=max_entries, ttl_seconds=ttl_seconds)
                return self._retrieval_cache

            self._retrieval_cache.reconfigure(max_entries=max_entries, ttl_seconds=ttl_seconds)
            return self._retrieval_cache

    async def invalidate_retrieval_cache(self, *, collection: str, tenant: CoreTenantContext) -> int:
        return self.invalidate_retrieval_scope(collection, tenant)

    def invalidate_retrieval_scope(self, collection: str, tenant: CoreTenantContext) -> int:
        cache = self.get_retrieval_cache()
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
        return self.invalidate_retrieval_cache_all()

    def invalidate_retrieval_cache_all(self) -> int:
        cache = self.get_retrieval_cache()
        if cache is None:
            return 0
        return cache.invalidate_all()

    def reset_retrieval_cache(self) -> None:
        with self._retrieval_cache_lock:
            self._retrieval_cache = None

    def readiness_snapshot(self) -> dict[str, Any]:
        embedding = self._embedding_readiness()
        vector_store = self._vector_store_readiness()
        llm = self._llm_readiness()
        retrieval_cache = self._retrieval_cache_readiness()
        hybrid_ready = embedding["ready"] and vector_store["ready"]

        return {
            "document_processor": {
                "ready": True,
                "status": "ready" if self._document_ready else "lazy",
            },
            "embedding_model": embedding,
            "vector_store": vector_store,
            "hybrid_service": {
                "ready": hybrid_ready,
                "status": "ready" if self._hybrid_ready else ("configured" if hybrid_ready else "blocked"),
            },
            "llm_model": llm,
            "retrieval_cache": retrieval_cache,
            "provider_budget": {
                "embeddings": self._provider_budget_snapshot("embeddings"),
                "llm": self._provider_budget_snapshot("llm"),
            },
        }

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

    def _embedding_fallback_signature(self) -> tuple[Any, ...]:
        fallback = str(getattr(self.settings, "embedding_fallback_provider", "") or "").strip().lower()
        config = self._provider_config(fallback)
        return (
            fallback,
            getattr(config, "base_url", None),
            getattr(config, "model", None),
            getattr(config, "api_key", None),
        )

    def _llm_signature(self) -> tuple[Any, ...]:
        provider_config = self._provider_config(self._llm_provider_name())
        return (
            self._llm_provider_name(),
            getattr(provider_config, "llm_model", None) or getattr(provider_config, "model", None) or getattr(self.settings, "llm_model", ""),
            getattr(provider_config, "base_url", None) or getattr(self.settings, "llm_base_url", ""),
            getattr(provider_config, "api_key", None) or getattr(self.settings, "llm_api_key", None),
            bool(getattr(self.settings, "enable_thinking", True)),
        )

    def _llm_fallback_signature(self) -> tuple[Any, ...]:
        fallback_provider = str(getattr(self.settings, "llm_fallback_provider", "") or "").strip().lower()
        provider_config = self._provider_config(fallback_provider)
        return (
            fallback_provider,
            getattr(provider_config, "llm_model", None) or getattr(provider_config, "model", None) or getattr(self.settings, "llm_model", ""),
            getattr(provider_config, "base_url", None) or getattr(self.settings, "llm_base_url", ""),
            getattr(provider_config, "api_key", None) or getattr(self.settings, "llm_api_key", None),
        )

    def _cache_signature(self) -> tuple[Any, ...]:
        cache_settings = getattr(self.settings, "cache", None)
        provider_budget = getattr(self.settings, "provider_budget", None)
        budget_signature = ()
        if provider_budget is not None:
            budget_signature = (
                bool(getattr(provider_budget, "enabled", True)),
                int(getattr(getattr(provider_budget, "embeddings", None), "requests_per_window", 0) or 0),
                int(getattr(getattr(provider_budget, "embeddings", None), "failure_threshold", 0) or 0),
                int(getattr(getattr(provider_budget, "llm", None), "requests_per_window", 0) or 0),
                int(getattr(getattr(provider_budget, "llm", None), "failure_threshold", 0) or 0),
            )
        return (
            self._cache_enabled(),
            int(getattr(cache_settings, "max_entries", 256) or 256),
            int(getattr(cache_settings, "ttl_seconds", 300) or 300),
            bool(getattr(self.settings, "enable_llm_summary", False)),
            bool(getattr(self.settings, "enable_reranker", False)),
            int(getattr(self.settings, "max_retrieval_results", 5) or 5),
            float(getattr(self.settings, "similarity_threshold", 0.7) or 0.7),
            self._embedding_provider_name(),
            str(getattr(self.settings, "embedding_fallback_provider", "") or "").strip().lower(),
            self._llm_provider_name(),
            str(getattr(self.settings, "llm_fallback_provider", "") or "").strip().lower(),
            getattr(self.settings, "llm_model", ""),
            budget_signature,
        )

    def _cache_enabled(self) -> bool:
        cache_settings = getattr(self.settings, "cache", None)
        nested_enabled = bool(getattr(cache_settings, "enabled", False))
        return bool(getattr(self.settings, "enable_cache", False) or nested_enabled)

    async def _close_component(self, component: Any) -> None:
        if component is None:
            return

        closer = getattr(component, "close", None)
        if callable(closer):
            result = closer()
            if asyncio.iscoroutine(result):
                await result
            return

        async_closer = getattr(component, "aclose", None)
        if callable(async_closer):
            await async_closer()

    async def close(self) -> None:
        """Close runtime-managed resources when they expose async teardown."""

        await self._close_component(self._llm_model)
        await self._close_component(self._fallback_llm_model)
        await self._close_component(self._embedding_model)
        await self._close_component(self._fallback_embedding_model)

    def _provider_config(self, provider: str | None):
        provider_name = str(provider or "").strip().lower()
        provider_configs = getattr(self.settings, "provider_configs", {}) or {}
        return provider_configs.get(provider_name)

    def _embedding_provider_name(self) -> str:
        return str(getattr(self.settings, "embedding_provider", "") or "zhipu").strip().lower()

    def _llm_provider_name(self) -> str:
        return str(getattr(self.settings, "llm_provider", "") or "doubao").strip().lower()

    def _budget_embedding_model(self, model: Any):
        if isinstance(model, _BudgetedEmbeddingModel):
            return model
        return _BudgetedEmbeddingModel(self, model)

    def _budget_llm_model(self, model: Any):
        if isinstance(model, _BudgetedLLMModel):
            return model
        return _BudgetedLLMModel(self, model)

    def _build_provider_budgets(self) -> dict[str, ProviderBudgetState]:
        provider_budget = getattr(self.settings, "provider_budget", None)
        if provider_budget is None or not bool(getattr(provider_budget, "enabled", True)):
            return {}

        return {
            "embeddings": ProviderBudgetState(
                limiter=TokenBucketRateLimiter(
                    limit=int(getattr(provider_budget.embeddings, "requests_per_window", 0) or 0),
                    window_seconds=float(getattr(provider_budget.embeddings, "window_seconds", 60) or 60),
                    burst=int(getattr(provider_budget.embeddings, "burst", 0) or 0),
                ),
                failure_threshold=int(getattr(provider_budget.embeddings, "failure_threshold", 0) or 0),
                cooldown_seconds=int(getattr(provider_budget.embeddings, "cooldown_seconds", 0) or 0),
            ),
            "llm": ProviderBudgetState(
                limiter=TokenBucketRateLimiter(
                    limit=int(getattr(provider_budget.llm, "requests_per_window", 0) or 0),
                    window_seconds=float(getattr(provider_budget.llm, "window_seconds", 60) or 60),
                    burst=int(getattr(provider_budget.llm, "burst", 0) or 0),
                ),
                failure_threshold=int(getattr(provider_budget.llm, "failure_threshold", 0) or 0),
                cooldown_seconds=int(getattr(provider_budget.llm, "cooldown_seconds", 0) or 0),
            ),
        }

    def _provider_budget_snapshot(self, family: str) -> dict[str, Any]:
        state = self._provider_budgets.get(family)
        if state is None:
            return {"enabled": False, "ready": True, "status": "disabled"}
        return state.snapshot()

    def _provider_subject(self, family: str) -> str:
        return f"provider:{family}"

    async def _invoke_embedding(self, operation: str, callable_obj, *args, **kwargs):
        state = self._provider_budgets.get("embeddings")
        if state is not None:
            allowed, retry_after = state.allow(self._provider_subject("embeddings"))
            if not allowed:
                fallback, label = await self._ensure_fallback_embedding_model()
                if fallback is not None and label is not None:
                    return await self._invoke_provider_callable(
                        label,
                        operation,
                        getattr(fallback, operation),
                        *args,
                        **kwargs,
                    )
                raise RuntimeError(f"embedding provider budget exceeded; retry after {retry_after:.2f}s")

        try:
            result = await self._invoke_provider_callable(
                self._embedding_provider_name(),
                operation,
                callable_obj,
                *args,
                **kwargs,
            )
        except Exception:
            if state is not None:
                state.record(success=False)
            fallback, label = await self._ensure_fallback_embedding_model()
            if fallback is not None and label is not None:
                return await self._invoke_provider_callable(
                    label,
                    operation,
                    getattr(fallback, operation),
                    *args,
                    **kwargs,
                )
            raise

        if state is not None:
            state.record(success=True)
        return result

    async def _invoke_llm(self, operation: str, callable_obj, *args, **kwargs):
        state = self._provider_budgets.get("llm")
        if state is not None:
            allowed, retry_after = state.allow(self._provider_subject("llm"))
            if not allowed:
                fallback, label = await self._ensure_fallback_llm_model()
                if fallback is not None and label is not None:
                    return await self._invoke_provider_callable(
                        label,
                        operation,
                        getattr(fallback, operation),
                        *args,
                        **kwargs,
                    )
                raise RuntimeError(f"llm provider budget exceeded; retry after {retry_after:.2f}s")

        try:
            result = await self._invoke_provider_callable(
                self._llm_provider_name(),
                operation,
                callable_obj,
                *args,
                **kwargs,
            )
        except Exception:
            if state is not None:
                state.record(success=False)
            fallback, label = await self._ensure_fallback_llm_model()
            if fallback is not None and label is not None:
                return await self._invoke_provider_callable(
                    label,
                    operation,
                    getattr(fallback, operation),
                    *args,
                    **kwargs,
                )
            raise

        if state is not None:
            state.record(success=True)
        return result

    async def _invoke_provider_callable(self, provider: str, operation: str, callable_obj, *args, **kwargs):
        started = self._clock()
        try:
            result = await callable_obj(*args, **kwargs)
        except Exception as exc:
            self._record_provider_latency(provider, operation, started, success=False, error=exc.__class__.__name__)
            raise

        self._record_provider_latency(provider, operation, started, success=True)
        return result

    async def _ensure_fallback_embedding_model(self) -> tuple[Any | None, str | None]:
        fallback_provider = str(getattr(self.settings, "embedding_fallback_provider", "") or "").strip().lower()
        if not fallback_provider or fallback_provider == self._embedding_provider_name():
            return None, None

        if self._fallback_embedding_model is None:
            async with self._fallback_embedding_lock:
                if self._fallback_embedding_model is None:
                    self._fallback_embedding_model = await self._build_fallback_embedding_model(fallback_provider)
        return self._fallback_embedding_model, fallback_provider

    async def _ensure_fallback_llm_model(self) -> tuple[Any | None, str | None]:
        fallback_provider = str(getattr(self.settings, "llm_fallback_provider", "") or "").strip().lower()
        if not fallback_provider or fallback_provider == self._llm_provider_name():
            return None, None

        if self._fallback_llm_model is None:
            async with self._fallback_llm_lock:
                if self._fallback_llm_model is None:
                    self._fallback_llm_model = await self._build_fallback_llm_model(fallback_provider)
        return self._fallback_llm_model, fallback_provider

    async def _build_fallback_embedding_model(self, fallback_provider: str) -> Any | None:
        try:
            model = self.build_embedding_model(provider=fallback_provider)
            initializer = getattr(model, "initialize", None)
            if callable(initializer):
                await initializer()
            return model
        except Exception as exc:
            logger.warning("Failed to initialize embedding fallback provider %s: %s", fallback_provider, exc)
            return None

    async def _build_fallback_llm_model(self, fallback_provider: str) -> Any | None:
        try:
            provider_config = self._provider_config(fallback_provider)
            model_name = str(
                getattr(provider_config, "llm_model", None)
                or getattr(provider_config, "model", None)
                or getattr(self.settings, "llm_model", "")
                or ""
            )
            base_url = str(
                getattr(provider_config, "base_url", None)
                or getattr(self.settings, "llm_base_url", "https://ark.cn-beijing.volces.com/api/v3")
                or ""
            )
            api_key = getattr(provider_config, "api_key", None) or getattr(self.settings, "llm_api_key", None)

            if fallback_provider == "ollama":
                return OllamaModel(
                    base_url=base_url or "http://localhost:11434",
                    model=model_name or "qwen2:7b",
                )

            if fallback_provider == "doubao":
                if not api_key:
                    return None
                llm_model = DoubaoLLMModel(
                    api_key=api_key,
                    base_url=base_url,
                    model=model_name,
                    enable_thinking=bool(getattr(self.settings, "enable_thinking", True)),
                )
                await llm_model.initialize()
                return llm_model

            if provider_config is not None:
                from ..llm import OpenAICompatibleLLMModel

                if not api_key:
                    return None
                llm_model = OpenAICompatibleLLMModel(
                    api_key=api_key,
                    base_url=base_url,
                    model=model_name,
                )
                await llm_model.initialize()
                return llm_model
        except Exception as exc:
            logger.warning("Failed to initialize llm fallback provider %s: %s", fallback_provider, exc)
            return None

        return None

    def _embedding_readiness(self) -> dict[str, Any]:
        provider = self._embedding_provider_name()
        if self._embedding_model is not None:
            return {
                "ready": self._embedding_ready,
                "status": "ready" if self._embedding_ready else "initializing",
                "provider": provider,
            }

        provider_config = self._provider_config(provider)
        if provider_config is not None and not getattr(provider_config, "api_key", None):
            return {
                "ready": False,
                "status": "misconfigured",
                "provider": provider,
                "reason": "embedding provider api_key is missing",
            }

        return {
            "ready": True,
            "status": "configured",
            "provider": provider,
        }

    def _vector_store_readiness(self) -> dict[str, Any]:
        persist_directory = str(getattr(self.settings, "chroma_persist_directory", "") or "")
        if self._vector_store is not None:
            return {
                "ready": self._vector_ready,
                "status": "ready" if self._vector_ready else "initializing",
                "persist_directory": persist_directory,
            }
        return {
            "ready": bool(persist_directory),
            "status": "configured" if persist_directory else "misconfigured",
            "persist_directory": persist_directory,
        }

    def _llm_readiness(self) -> dict[str, Any]:
        provider = self._llm_provider_name()
        provider_config = self._provider_config(provider)
        llm_api_key = getattr(provider_config, "api_key", None) or getattr(self.settings, "llm_api_key", None)
        if not bool(getattr(self.settings, "enable_llm_summary", False)):
            return {
                "ready": True,
                "status": "disabled",
                "provider": provider,
            }

        if self._llm_model is not None:
            return {
                "ready": self._llm_ready,
                "status": "ready" if self._llm_ready else "initializing",
                "provider": provider,
            }

        if provider != "ollama" and not llm_api_key:
            return {
                "ready": False,
                "status": "misconfigured",
                "provider": provider,
                "reason": "llm api_key is missing",
            }
        if provider not in {"doubao", "ollama"} and provider_config is None:
            return {
                "ready": False,
                "status": "misconfigured",
                "provider": provider,
                "reason": "unsupported llm provider",
            }

        return {
            "ready": True,
            "status": "configured",
            "provider": provider,
        }

    def _retrieval_cache_readiness(self) -> dict[str, Any]:
        if not self._cache_enabled():
            return {"ready": True, "status": "disabled"}

        cache_settings = getattr(self.settings, "cache", None)
        max_entries = int(getattr(cache_settings, "max_entries", 256) or 256)
        ttl_seconds = int(getattr(cache_settings, "ttl_seconds", 300) or 300)
        if self._retrieval_cache is None:
            return {
                "ready": True,
                "status": "configured",
                "max_entries": max_entries,
                "ttl_seconds": ttl_seconds,
            }

        snapshot = self._retrieval_cache.snapshot()
        return {
            "ready": True,
            "status": "ready",
            **snapshot,
        }

    def _record_provider_latency(
        self,
        provider: str,
        operation: str,
        started_at: float,
        *,
        success: bool,
        error: str | None = None,
    ) -> None:
        collector = self.observability
        if collector is None:
            return
        collector.record_provider_latency(
            provider,
            operation,
            (self._clock() - started_at) * 1000.0,
            success=success,
            error=error,
        )

    def _clock(self) -> float:
        return monotonic()


RuntimeContainer = ServiceRuntime
