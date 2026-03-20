"""Thin compatibility facade over the service runtime."""

from __future__ import annotations

import asyncio

from .config import settings
from .contracts import ChatRequest, ChatResponse, DocumentRequest, SearchRequest, SearchResponse, TenantSpec
from .services import ChatService, IndexingService, RetrievalService, ServiceRuntime


class RagService:
    """Tenant-aware facade used by HTTP and MCP shells."""

    def __init__(
        self,
        *,
        settings_obj=settings,
        document_processor=None,
        embedding_model=None,
        vector_store=None,
        hybrid_service=None,
        llm_model=None,
        runtime: ServiceRuntime | None = None,
    ):
        self.runtime = runtime or ServiceRuntime(
            settings_obj=settings_obj,
            document_processor=document_processor,
            embedding_model=embedding_model,
            vector_store=vector_store,
            hybrid_service=hybrid_service,
            llm_model=llm_model,
        )
        self.settings = self.runtime.settings
        self.indexing_service = IndexingService(self.runtime)
        self.retrieval_service = RetrievalService(self.runtime)
        self.chat_service = ChatService(self.runtime, self.retrieval_service)

    async def add_document(self, request: DocumentRequest):
        return await self.indexing_service.add_document(request)

    async def upload_files(self, files, *, collection: str = "default", tenant: TenantSpec | dict | None = None):
        return await self.indexing_service.upload_files(files, collection=collection, tenant=tenant)

    async def search(self, request: SearchRequest) -> SearchResponse:
        return await self.retrieval_service.search(request)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        return await self.chat_service.chat(request)

    async def ask(self, request: SearchRequest) -> SearchResponse:
        return await self.retrieval_service.ask(request)

    async def list_documents(
        self,
        *,
        collection: str = "default",
        limit: int = 100,
        offset: int = 0,
        filename: str | None = None,
        tenant: TenantSpec | dict | None = None,
    ):
        return await self.indexing_service.list_documents(
            collection=collection,
            limit=limit,
            offset=offset,
            filename=filename,
            tenant=tenant,
        )

    async def list_files(self, *, collection: str = "default", tenant: TenantSpec | dict | None = None):
        return await self.indexing_service.list_files(collection=collection, tenant=tenant)

    async def list_collections(self, *, tenant: TenantSpec | dict | None = None):
        return await self.indexing_service.list_collections(tenant=tenant)

    async def delete_document(
        self,
        *,
        document_id: str,
        collection: str = "default",
        tenant: TenantSpec | dict | None = None,
    ) -> bool:
        return await self.indexing_service.delete_document(
            document_id=document_id,
            collection=collection,
            tenant=tenant,
        )

    async def delete_file(
        self,
        *,
        filename: str,
        collection: str = "default",
        tenant: TenantSpec | dict | None = None,
    ) -> bool:
        return await self.indexing_service.delete_file(
            filename=filename,
            collection=collection,
            tenant=tenant,
        )


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

