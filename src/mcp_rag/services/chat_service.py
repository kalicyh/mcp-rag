"""Chat service for question answering over retrieved context."""

from __future__ import annotations

import logging

from ..context import normalize_request_context
from ..contracts import ChatRequest, ChatResponse, SearchRequest
from .retrieval_service import RetrievalService
from .runtime import ServiceRuntime

logger = logging.getLogger(__name__)


class ChatService:
    """Generate chat responses from retrieved knowledge base context."""

    def __init__(self, runtime: ServiceRuntime, retrieval_service: RetrievalService):
        self.runtime = runtime
        self.retrieval_service = retrieval_service

    async def chat(self, request: ChatRequest) -> ChatResponse:
        request_context = normalize_request_context(
            request.context,
            tenant=request.tenant,
            base_collection=request.collection,
        )
        search_response = await self.retrieval_service.search(
            SearchRequest(
                query=request.query,
                collection=request.collection,
                limit=request.limit,
                threshold=0.7,
                tenant=request.tenant,
                context=request_context,
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
            llm_model = await self.runtime.ensure_llm_model()
            started = self.runtime.observability._clock() if self.runtime.observability else None
            answer = await llm_model.generate(prompt)
            if started is not None:
                self.runtime.observability.record_provider_latency(
                    "llm",
                    "generate",
                    (self.runtime.observability._clock() - started) * 1000.0,
                )
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
