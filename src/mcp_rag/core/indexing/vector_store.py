"""Chroma vector store for the indexing foundation."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb

from .embeddings import EmbeddingModel
from .models import ChunkRecord, FileSummary, SearchHit, TenantContext
from .tenancy import parse_collection_name, resolve_collection_name

logger = logging.getLogger(__name__)

_RESERVED_KEYS = {
    "id",
    "ids",
    "embedding",
    "embeddings",
    "document",
    "documents",
    "metadata",
    "metadatas",
    "distance",
    "distances",
}


class ChromaVectorStore:
    """Minimal Chroma-backed vector store with cosine similarity."""

    def __init__(
        self,
        *,
        persist_directory: str,
        collection_name: str = "default",
        embedding_model: EmbeddingModel | None = None,
        tenant: TenantContext | None = None,
    ):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.default_tenant = tenant
        self.client: chromadb.PersistentClient | None = None

    async def initialize(self) -> None:
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        await self._ensure_collection(self.collection_name, self.default_tenant)

    async def add_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
        tenant: TenantContext | None = None,
        collection_name: str | None = None,
    ) -> List[str]:
        collection = await self._get_collection(collection_name, tenant)

        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        vectors = list(embeddings) if embeddings is not None else await self._encode(texts)
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [self._encode_metadata(chunk.metadata) for chunk in chunks]

        collection.add(
            ids=ids,
            embeddings=[list(vector) for vector in vectors],
            documents=texts,
            metadatas=metadatas,
        )
        return ids

    async def upsert_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
        tenant: TenantContext | None = None,
        collection_name: str | None = None,
    ) -> List[str]:
        collection = await self._get_collection(collection_name, tenant)

        if not chunks:
            return []

        texts = [chunk.content for chunk in chunks]
        vectors = list(embeddings) if embeddings is not None else await self._encode(texts)
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas = [self._encode_metadata(chunk.metadata) for chunk in chunks]

        collection.upsert(
            ids=ids,
            embeddings=[list(vector) for vector in vectors],
            documents=texts,
            metadatas=metadatas,
        )
        return ids

    async def search(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 5,
        threshold: float = 0.7,
        tenant: TenantContext | None = None,
        collection_name: str | None = None,
    ) -> List[SearchHit]:
        collection = await self._get_collection(collection_name, tenant)
        results = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[SearchHit] = []
        if not results.get("ids"):
            return hits

        ids = results["ids"][0] or []
        distances = results["distances"][0] or []
        documents = results["documents"][0] or []
        metadatas = results["metadatas"][0] or []

        for index, chunk_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else 1.0
            score = 1.0 - distance
            if score < threshold:
                continue

            metadata = self._decode_metadata(metadatas[index] if index < len(metadatas) else {})
            hits.append(
                SearchHit(
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    score=score,
                    source=str(metadata.get("source", "")),
                    filename=str(metadata.get("filename", "")),
                    content=documents[index] if index < len(documents) else "",
                    metadata=metadata,
                )
            )

        return hits

    async def list_documents(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        filename: str | None = None,
        tenant: TenantContext | None = None,
        collection_name: str | None = None,
    ) -> Dict[str, Any]:
        collection = await self._get_collection(collection_name, tenant)
        where = {"filename": filename} if filename else None
        results = collection.get(
            limit=limit,
            offset=offset,
            where=where,
            include=["documents", "metadatas"],
        )
        documents: list[dict[str, Any]] = []
        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []
        contents = results.get("documents") or []

        for index, chunk_id in enumerate(ids):
            metadata = self._decode_metadata(metadatas[index] if index < len(metadatas) else {})
            documents.append(
                {
                    "id": chunk_id,
                    "content": contents[index] if index < len(contents) else "",
                    "metadata": metadata,
                }
            )

        total = collection.count()
        return {"total": total, "documents": documents, "limit": limit, "offset": offset}

    async def list_files(
        self,
        *,
        tenant: TenantContext | None = None,
        collection_name: str | None = None,
    ) -> List[Dict[str, Any]]:
        collection = await self._get_collection(collection_name, tenant)
        results = collection.get(include=["documents", "metadatas"])
        groups: dict[str, FileSummary] = {}
        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []

        for index, chunk_id in enumerate(ids):
            metadata = self._decode_metadata(metadatas[index] if index < len(metadatas) else {})
            filename = str(metadata.get("filename") or metadata.get("source") or chunk_id)
            document_id = str(metadata.get("document_id") or metadata.get("original_id") or "")
            file_type = str(metadata.get("file_type") or "unknown")
            source = str(metadata.get("source") or "")
            total_chars = int(metadata.get("chunk_char_count") or metadata.get("char_count") or 0)
            processed_at = metadata.get("processed_at")
            first_seen_at = self._parse_datetime(processed_at)

            if filename not in groups:
                groups[filename] = FileSummary(
                    filename=filename,
                    source=source,
                    file_type=file_type,
                    chunk_count=0,
                    total_chars=0,
                    document_id=document_id,
                    metadata=metadata,
                    first_seen_at=first_seen_at,
                )

            summary = groups[filename]
            summary.chunk_count += 1
            summary.total_chars += total_chars
            if not summary.document_id:
                summary.document_id = document_id
            if summary.first_seen_at is None:
                summary.first_seen_at = first_seen_at

        return [
            {
                "filename": summary.filename,
                "source": summary.source,
                "file_type": summary.file_type,
                "chunk_count": summary.chunk_count,
                "total_chars": summary.total_chars,
                "document_id": summary.document_id,
                "metadata": summary.metadata,
                "first_seen_at": summary.first_seen_at.isoformat() if summary.first_seen_at else None,
            }
            for summary in groups.values()
        ]

    async def delete_document(
        self,
        document_id: str,
        *,
        tenant: TenantContext | None = None,
        collection_name: str | None = None,
    ) -> bool:
        collection = await self._get_collection(collection_name, tenant)
        collection.delete(where={"document_id": document_id})
        return True

    async def delete_file(
        self,
        filename: str,
        *,
        tenant: TenantContext | None = None,
        collection_name: str | None = None,
    ) -> bool:
        collection = await self._get_collection(collection_name, tenant)
        results = collection.get(include=["metadatas"])
        ids_to_delete: list[str] = []
        ids = results.get("ids") or []
        metadatas = results.get("metadatas") or []

        for index, chunk_id in enumerate(ids):
            metadata = self._decode_metadata(metadatas[index] if index < len(metadatas) else {})
            current_filename = str(metadata.get("filename") or "")
            document_id = str(metadata.get("document_id") or "")
            if current_filename == filename:
                ids_to_delete.append(chunk_id)
                continue
            if document_id and document_id == filename:
                ids_to_delete.append(chunk_id)
                continue
            if "_chunk_" in chunk_id and chunk_id.rsplit("_chunk_", 1)[0] == filename:
                ids_to_delete.append(chunk_id)

        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
        return True

    async def delete_collection(
        self,
        *,
        tenant: TenantContext | None = None,
        collection_name: str | None = None,
    ) -> None:
        if self.client is None:
            raise RuntimeError("Vector store not initialized")
        actual = resolve_collection_name(
            collection_name or self.collection_name,
            tenant=tenant or self.default_tenant,
        )
        self.client.delete_collection(name=actual)

    async def list_collections(self) -> List[Dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("Vector store not initialized")

        collections = self.client.list_collections()
        output: list[dict[str, Any]] = []
        for collection in collections:
            parsed = parse_collection_name(collection.name)
            output.append(
                {
                    "name": collection.name,
                    "base_collection": parsed.base_collection,
                    "user_id": parsed.user_id,
                    "agent_id": parsed.agent_id,
                }
            )
        return output

    async def _get_collection(
        self,
        collection_name: str | None,
        tenant: TenantContext | None,
    ):
        if self.client is None:
            await self.initialize()
        actual_name = resolve_collection_name(
            collection_name or self.collection_name,
            tenant=tenant or self.default_tenant,
        )
        return await self._ensure_collection(actual_name, None)

    async def _ensure_collection(
        self,
        collection_name: str,
        tenant: TenantContext | None,
    ):
        if self.client is None:
            raise RuntimeError("Vector store not initialized")

        actual_name = resolve_collection_name(collection_name, tenant=tenant)
        try:
            collection = self.client.get_collection(name=actual_name)
            space = collection.metadata.get("hnsw:space") if collection.metadata else None
            if space != "cosine":
                self.client.delete_collection(name=actual_name)
                collection = self.client.create_collection(
                    name=actual_name,
                    metadata={"hnsw:space": "cosine"},
                )
            return collection
        except Exception:
            return self.client.create_collection(
                name=actual_name,
                metadata={"hnsw:space": "cosine"},
            )

    async def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        if self.embedding_model is None:
            raise RuntimeError("No embedding model configured for vector store")
        return await self.embedding_model.encode(texts)

    def _encode_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        flattened: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if key in _RESERVED_KEYS:
                raise ValueError(f"Metadata contains reserved key: {key}")

            if isinstance(value, (dict, list, tuple, set)):
                flattened[f"custom_{key}"] = json.dumps(list(value) if isinstance(value, set) else value, ensure_ascii=False)
            elif isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            else:
                flattened[f"custom_{key}"] = json.dumps(str(value), ensure_ascii=False)
        return flattened

    def _decode_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        decoded: dict[str, Any] = {}
        for key, value in metadata.items():
            if key.startswith("custom_"):
                original_key = key.removeprefix("custom_")
                if isinstance(value, str):
                    try:
                        decoded[original_key] = json.loads(value)
                        continue
                    except json.JSONDecodeError:
                        pass
                decoded[original_key] = value
            else:
                decoded[key] = value
        return decoded

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        return None

