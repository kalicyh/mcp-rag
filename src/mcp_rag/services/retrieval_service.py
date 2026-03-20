"""Retrieval service for search and rag_ask flows."""

from __future__ import annotations

import logging
from typing import List

from ..contracts import SearchRequest, SearchResponse, SearchResultView
from .runtime import ServiceRuntime

logger = logging.getLogger(__name__)


class RetrievalService:
    """Hybrid retrieval and summary generation."""

    def __init__(self, runtime: ServiceRuntime):
        self.runtime = runtime

    async def search(self, request: SearchRequest) -> SearchResponse:
        tenant = request.tenant.to_core()
        hybrid = await self.runtime.ensure_hybrid_service()
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
        if hits and self.runtime.settings.enable_llm_summary:
            try:
                llm_model = await self.runtime.ensure_llm_model()
                summary = await llm_model.summarize(self._build_summary_context(hits), request.query)
            except Exception as exc:
                logger.warning("LLM summary failed, falling back to raw results: %s", exc)

        return SearchResponse(
            query=request.query,
            collection=request.collection,
            results=results,
            summary=summary,
        )

    async def ask(self, request: SearchRequest) -> SearchResponse:
        """Compatibility wrapper for rag_ask."""

        response = await self.search(request)
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
        return response

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
