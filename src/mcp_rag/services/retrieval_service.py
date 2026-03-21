"""Retrieval service for search and rag_ask flows."""

from __future__ import annotations

import logging
from typing import List

from ..context import normalize_request_context
from ..contracts import SearchRequest, SearchResponse, SearchResultView
from ..core.indexing.tenancy import resolve_collection_name
from .retrieval_cache import CacheGeneration, RetrievalCache, RetrievalCacheKey
from .runtime import ServiceRuntime

logger = logging.getLogger(__name__)


class RetrievalService:
    """Hybrid retrieval and summary generation."""

    def __init__(self, runtime: ServiceRuntime):
        self.runtime = runtime

    async def search(self, request: SearchRequest) -> SearchResponse:
        request_context = normalize_request_context(
            request.context,
            tenant=request.tenant,
            base_collection=request.collection,
            kb_id=request.kb_id,
            kb_scope=request.scope,
        )
        request.context = request_context
        request.tenant = request_context.tenant
        return await self._search(request)

    async def _search(
        self,
        request: SearchRequest,
        *,
        cache_lookup: tuple[RetrievalCache | None, RetrievalCacheKey | None, CacheGeneration | None] | None = None,
    ) -> SearchResponse:
        request_context = normalize_request_context(
            request.context,
            tenant=request.tenant,
            base_collection=request.collection,
            kb_id=request.kb_id,
            kb_scope=request.scope,
        )
        request.context = request_context
        request.tenant = request_context.tenant
        tenant = request_context.tenant.to_core()
        cache, cache_key, cache_generation = cache_lookup or self._prepare_cache_lookup(request, request_context=request_context)
        if cache is not None and cache_key is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached

        hybrid = await self.runtime.ensure_hybrid_service()
        hits = await hybrid.retrieve(
            request.query,
            collection_name=request_context.resolved_collection or request.collection,
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
        if hits and self.runtime.settings.enable_llm_summary:
            try:
                llm_model = await self.runtime.ensure_llm_model()
                summary = await llm_model.summarize(self._build_summary_context(hits), request.query)
            except Exception as exc:
                logger.warning("LLM summary failed, falling back to raw results: %s", exc)

        response = SearchResponse(
            query=request.query,
            collection=request.collection,
            results=results,
            summary=summary,
        )
        if cache is not None and cache_key is not None:
            cache.set(response=response, key=cache_key, expected_generation=cache_generation)
        return response

    async def ask(self, request: SearchRequest) -> SearchResponse:
        """Compatibility wrapper for rag_ask."""

        request.context = normalize_request_context(
            request.context,
            tenant=request.tenant,
            base_collection=request.collection,
            kb_id=request.kb_id,
            kb_scope=request.scope,
        )
        cache_lookup = self._prepare_cache_lookup(request, request_context=request.context)
        response = await self._search(request, cache_lookup=cache_lookup)
        if request.mode == "summary" and response.summary is None:
            try:
                llm_model = await self.runtime.ensure_llm_model()
                response.summary = await llm_model.summarize(
                    self._build_summary_context_from_views(response.results),
                    request.query,
                )
            except Exception as exc:
                logger.warning("Summary mode fallback failed: %s", exc)
                response.summary = "摘要生成功能暂未启用。"
            cache, cache_key, cache_generation = cache_lookup
            if cache is not None and cache_key is not None:
                cache.set(response=response, key=cache_key, expected_generation=cache_generation)
        return response

    def _prepare_cache_lookup(
        self,
        request: SearchRequest,
        *,
        request_context,
    ) -> tuple[RetrievalCache | None, RetrievalCacheKey | None, CacheGeneration | None]:
        cache = self.runtime.get_retrieval_cache()
        if cache is None:
            return None, None, None

        actual_collection = request_context.resolved_collection or resolve_collection_name(
            request.collection,
            tenant=request_context.tenant.to_core(),
        )
        key = self._build_cache_key(request, request_context=request_context, actual_collection=actual_collection)
        return cache, key, cache.generation_for(key)

    def _build_cache_key(
        self,
        request: SearchRequest,
        *,
        request_context,
        actual_collection: str,
    ) -> RetrievalCacheKey:
        tenant = request_context.tenant
        return RetrievalCacheKey(
            base_collection=tenant.base_collection or request.collection,
            user_id=tenant.user_id,
            agent_id=tenant.agent_id,
            actual_collection=actual_collection,
            query=request.query,
            mode=request.mode or "raw",
            limit=int(request.limit),
            threshold=round(float(request.threshold), 6),
            summary_enabled=bool(getattr(self.runtime.settings, "enable_llm_summary", False)),
            rerank_enabled=bool(getattr(self.runtime.settings, "enable_reranker", False)),
            retrieval_window=int(getattr(self.runtime.settings, "max_retrieval_results", request.limit) or request.limit),
        )

    def _build_summary_context(self, hits) -> str:
        return "\n\n".join(
            f"文档 {index + 1} (相似度: {hit.score:.3f}):\n{hit.content}"
            for index, hit in enumerate(hits)
        )

    def _build_summary_context_from_views(self, results: List[SearchResultView]) -> str:
        return "\n\n".join(
            f"文档 {index + 1} (相似度: {item.score:.3f}):\n{item.content}"
            for index, item in enumerate(results)
        )
