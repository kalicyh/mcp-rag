"""Indexing service for document ingestion and management."""

from __future__ import annotations

import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

from fastapi import UploadFile

from ..context import RequestContext, normalize_request_context
from ..contracts import BatchUploadResponse, DocumentRequest, TenantSpec, UploadFileResult
from ..core.indexing.models import TenantContext as CoreTenantContext
from ..security import IndexQuotaPolicy, QuotaExceededError, UploadQuotaPolicy
from .runtime import ServiceRuntime

logger = logging.getLogger(__name__)

_FILE_NAME_RE = re.compile(r"[/\\:*?\"<>|]+")


class IndexingService:
    """Document ingestion, listing, and deletion operations."""

    def __init__(self, runtime: ServiceRuntime):
        self.runtime = runtime

    async def add_document(self, request: DocumentRequest) -> Dict[str, Any]:
        request_context = normalize_request_context(
            request.context,
            tenant=request.tenant,
            base_collection=request.collection,
        )
        tenant = request_context.tenant.to_core()
        processor = await self.runtime.ensure_document_processor()
        vector_store = await self.runtime.ensure_vector_store()
        embedding_model = await self.runtime.ensure_embedding_model()
        self.runtime.attach_embedding_model(vector_store, embedding_model)

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
        self._enforce_index_quota(document_count=1, chunks=chunks)

        await vector_store.upsert_chunks(
            chunks,
            tenant=tenant,
            collection_name=request.collection,
        )
        await self.runtime.refresh_keywords(request.collection, tenant)
        self.runtime.invalidate_retrieval_scope(request.collection, tenant)

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
        request_context: RequestContext | None = None,
    ) -> Dict[str, Any]:
        resolved_context = normalize_request_context(
            request_context,
            tenant=tenant,
            base_collection=collection,
        )
        tenant_spec = resolved_context.tenant
        processor = await self.runtime.ensure_document_processor()
        vector_store = await self.runtime.ensure_vector_store()
        embedding_model = await self.runtime.ensure_embedding_model()
        self.runtime.attach_embedding_model(vector_store, embedding_model)

        upload_policy = UploadQuotaPolicy.from_settings(self.runtime.settings)
        file_sizes = [self._measure_upload_size(upload) for upload in files]
        try:
            upload_policy.require(file_sizes)
        except QuotaExceededError as exc:
            return BatchUploadResponse(
                total_files=len(files),
                successful=0,
                failed=len(files),
                results=[
                    UploadFileResult(
                        filename=getattr(upload, "filename", "unknown"),
                        file_type="unknown",
                        content_length=file_sizes[index],
                        processed=False,
                        error=str(exc),
                        preview="",
                    )
                    for index, upload in enumerate(files)
                ],
            ).to_dict()

        results: list[UploadFileResult] = []
        indexed_documents = 0
        indexed_chunks = 0
        indexed_chars = 0
        cache_dirty = False
        for upload in files:
            temp_path: Path | None = None
            try:
                visible_source = Path(upload.filename).name if upload.filename else "upload"
                temp_path = await self._write_upload_to_tempfile(upload)
                processed_doc = processor.process_file(
                    temp_path,
                    metadata={"source": visible_source, "filename": visible_source},
                    filename=visible_source,
                )
                processed_doc.source = visible_source
                processed_doc.filename = visible_source
                processed_doc.metadata["source"] = visible_source
                processed_doc.metadata["filename"] = visible_source
                if processed_doc.error or not processed_doc.content.strip():
                    results.append(
                        UploadFileResult(
                            filename=visible_source,
                            file_type=processed_doc.file_type,
                            content_length=len(processed_doc.content),
                            processed=False,
                            error=processed_doc.error or "No content extracted",
                            preview="",
                        )
                    )
                    continue

                chunks = processor.chunk_document(processed_doc)
                next_document_count = indexed_documents + 1
                next_chunk_count = indexed_chunks + len(chunks)
                next_total_chars = indexed_chars + sum(len(chunk.content) for chunk in chunks)
                self._enforce_index_quota(
                    document_count=next_document_count,
                    chunk_count=next_chunk_count,
                    total_chars=next_total_chars,
                )
                await vector_store.upsert_chunks(
                    chunks,
                    tenant=tenant_spec.to_core(),
                    collection_name=tenant_spec.base_collection,
                )
                await self.runtime.refresh_keywords(tenant_spec.base_collection, tenant_spec.to_core())
                cache_dirty = True
                indexed_documents = next_document_count
                indexed_chunks = next_chunk_count
                indexed_chars = next_total_chars

                preview = processed_doc.content[:500]
                if len(processed_doc.content) > 500:
                    preview += "..."

                results.append(
                    UploadFileResult(
                        filename=visible_source,
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

        if indexed_documents > 0:
            self.runtime.invalidate_retrieval_scope(tenant_spec.base_collection, tenant_spec.to_core())

        return BatchUploadResponse(
            total_files=len(files),
            successful=len([item for item in results if item.processed]),
            failed=len([item for item in results if not item.processed]),
            results=results,
        ).to_dict()

    async def list_documents(
        self,
        *,
        collection: str = "default",
        limit: int = 100,
        offset: int = 0,
        filename: str | None = None,
        tenant: TenantSpec | Dict[str, Any] | None = None,
        request_context: RequestContext | None = None,
    ) -> Dict[str, Any]:
        vector_store = await self.runtime.ensure_vector_store()
        tenant_spec = normalize_request_context(
            request_context,
            tenant=tenant,
            base_collection=collection,
        ).tenant
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
        request_context: RequestContext | None = None,
    ) -> List[Dict[str, Any]]:
        vector_store = await self.runtime.ensure_vector_store()
        tenant_spec = normalize_request_context(
            request_context,
            tenant=tenant,
            base_collection=collection,
        ).tenant
        return await vector_store.list_files(
            collection_name=collection,
            tenant=tenant_spec.to_core(),
        )

    async def list_collections(
        self,
        *,
        tenant: TenantSpec | Dict[str, Any] | None = None,
        request_context: RequestContext | None = None,
    ) -> List[str]:
        vector_store = await self.runtime.ensure_vector_store()
        collections = await vector_store.list_collections()
        names = [entry["name"] for entry in collections]

        tenant_spec = None
        if tenant is not None or request_context is not None:
            tenant_spec = normalize_request_context(request_context, tenant=tenant).tenant
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
        request_context: RequestContext | None = None,
    ) -> bool:
        vector_store = await self.runtime.ensure_vector_store()
        tenant_spec = normalize_request_context(
            request_context,
            tenant=tenant,
            base_collection=collection,
        ).tenant
        deleted = await self._delete_document_identifier(
            vector_store,
            identifier=document_id,
            collection=collection,
            tenant=tenant_spec.to_core(),
        )
        if deleted:
            await self.runtime.refresh_keywords(collection, tenant_spec.to_core())
            self.runtime.invalidate_retrieval_scope(collection, tenant_spec.to_core())
        return deleted

    async def delete_file(
        self,
        *,
        filename: str,
        collection: str = "default",
        tenant: TenantSpec | Dict[str, Any] | None = None,
        request_context: RequestContext | None = None,
    ) -> bool:
        vector_store = await self.runtime.ensure_vector_store()
        tenant_spec = normalize_request_context(
            request_context,
            tenant=tenant,
            base_collection=collection,
        ).tenant
        result = await vector_store.delete_file(
            filename,
            collection_name=collection,
            tenant=tenant_spec.to_core(),
        )
        await self.runtime.refresh_keywords(collection, tenant_spec.to_core())
        if result:
            self.runtime.invalidate_retrieval_scope(collection, tenant_spec.to_core())
        return result

    async def _delete_document_identifier(
        self,
        vector_store,
        *,
        identifier: str,
        collection: str,
        tenant: CoreTenantContext,
    ) -> bool:
        get_collection = getattr(vector_store, "_get_collection", None)
        if not callable(get_collection):
            return await vector_store.delete_document(
                identifier,
                collection_name=collection,
                tenant=tenant,
            )

        collection_handle = await get_collection(collection, tenant)
        deleted = False

        try:
            exact_chunk = collection_handle.get(ids=[identifier], include=["metadatas"])
            if exact_chunk.get("ids"):
                collection_handle.delete(ids=[identifier])
                deleted = True
        except Exception as exc:
            logger.debug("Exact chunk delete probe failed for %s: %s", identifier, exc)

        try:
            document_matches = collection_handle.get(where={"document_id": identifier}, include=["metadatas"])
            if document_matches.get("ids"):
                collection_handle.delete(where={"document_id": identifier})
                deleted = True
        except Exception as exc:
            logger.debug("Document-id delete probe failed for %s: %s", identifier, exc)

        return deleted

    async def _write_upload_to_tempfile(self, upload: UploadFile) -> Path:
        suffix = Path(upload.filename or "").suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(upload.file, temp_file)
            return Path(temp_file.name)

    def _enforce_index_quota(
        self,
        *,
        document_count: int,
        chunks: Sequence[Any] | None = None,
        chunk_count: int | None = None,
        total_chars: int | None = None,
    ) -> None:
        policy = IndexQuotaPolicy.from_settings(self.runtime.settings)
        effective_chunk_count = chunk_count if chunk_count is not None else len(list(chunks or ()))
        effective_total_chars = total_chars if total_chars is not None else sum(
            len(getattr(chunk, "content", "")) for chunk in (chunks or ())
        )
        policy.require(
            document_count=document_count,
            chunk_count=effective_chunk_count,
            total_chars=effective_total_chars,
        )

    def _measure_upload_size(self, upload: UploadFile) -> int:
        current_position = upload.file.tell()
        upload.file.seek(0, 2)
        size = int(upload.file.tell())
        upload.file.seek(current_position)
        return size

    def _resolve_filename(self, metadata: Dict[str, Any], fallback: str = "manual_input") -> str:
        raw = str(metadata.get("filename") or metadata.get("title") or fallback or "manual_input").strip()
        raw = _FILE_NAME_RE.sub("_", raw)
        return raw[:80] or "manual_input"
