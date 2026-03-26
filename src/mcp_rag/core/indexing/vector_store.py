"""Chroma vector store for the indexing foundation."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import chromadb

from .embeddings import EmbeddingModel
from .models import ChunkRecord, FileSummary, SearchHit, TenantContext
from .tenancy import parse_collection_name, resolve_collection_name, sanitize_collection_name

logger = logging.getLogger(__name__)

_EMBEDDING_COLLECTION_SUFFIX_RE = re.compile(r"^(?P<base>.+)__emb_(?P<version>[A-Za-z0-9_.-]+)$")

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
        actual_collection_name: str | None = None,
    ) -> List[SearchHit]:
        collection = await self._get_collection(collection_name, tenant, actual_collection_name=actual_collection_name)
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
        records = await self._list_document_records(
            collection_name=collection_name,
            tenant=tenant,
            filename=filename,
        )
        total = len(records)
        page = records[offset: offset + limit]
        return {"total": total, "documents": page, "limit": limit, "offset": offset}

    async def list_files(
        self,
        *,
        tenant: TenantContext | None = None,
        collection_name: str | None = None,
    ) -> List[Dict[str, Any]]:
        groups: dict[str, FileSummary] = {}
        for variant in await self.list_collection_variants(collection_name=collection_name, tenant=tenant):
            collection = await self._get_collection(
                collection_name,
                tenant,
                actual_collection_name=variant["name"],
            )
            results = collection.get(include=["documents", "metadatas"])
            self._collect_file_summaries(groups, results)
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

    async def list_collection_variants(
        self,
        *,
        collection_name: str | None = None,
        tenant: TenantContext | None = None,
    ) -> List[Dict[str, Any]]:
        if self.client is None:
            await self.initialize()
        if self.client is None:
            raise RuntimeError("Vector store not initialized")

        logical_name = resolve_collection_name(
            collection_name or self.collection_name,
            tenant=tenant or self.default_tenant,
        )
        current_name = self._resolve_runtime_collection_name(
            collection_name or self.collection_name,
            tenant=tenant or self.default_tenant,
        )
        variants: list[dict[str, Any]] = []
        for listed in self.client.list_collections():
            base_name, embedding_version = self._split_embedding_collection_name(listed.name)
            if base_name != logical_name:
                continue
            collection = self.client.get_collection(name=listed.name)
            metadata = dict(collection.metadata or {})
            variants.append(
                {
                    "name": listed.name,
                    "logical_name": base_name,
                    "embedding_version": embedding_version,
                    "embedding_provider": metadata.get("embedding_provider"),
                    "embedding_model": metadata.get("embedding_model"),
                    "embedding_base_url": metadata.get("embedding_base_url"),
                    "embedding_dimensions": metadata.get("embedding_dimensions"),
                    "current": listed.name == current_name,
                }
            )
        if not variants:
            variants.append(
                {
                    "name": current_name,
                    "logical_name": logical_name,
                    "embedding_version": self._embedding_version(),
                    "embedding_provider": getattr(self.embedding_model, "provider_name", None),
                    "embedding_model": getattr(self.embedding_model, "model_name", None) or getattr(self.embedding_model, "model", None),
                    "embedding_base_url": getattr(self.embedding_model, "base_url", None),
                    "embedding_dimensions": getattr(self.embedding_model, "dimensions", None),
                    "current": True,
                }
            )
        variants.sort(key=lambda item: (not item["current"], item["name"]))
        return variants

    def _collect_file_summaries(self, groups: dict[str, FileSummary], results: Dict[str, Any]) -> None:
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

    async def delete_document(
        self,
        document_id: str,
        *,
        tenant: TenantContext | None = None,
        collection_name: str | None = None,
    ) -> bool:
        for variant in await self.list_collection_variants(collection_name=collection_name, tenant=tenant):
            collection = await self._get_collection(
                collection_name,
                tenant,
                actual_collection_name=variant["name"],
            )
            collection.delete(where={"document_id": document_id})
        return True

    async def delete_file(
        self,
        filename: str,
        *,
        tenant: TenantContext | None = None,
        collection_name: str | None = None,
    ) -> bool:
        for variant in await self.list_collection_variants(collection_name=collection_name, tenant=tenant):
            collection = await self._get_collection(
                collection_name,
                tenant,
                actual_collection_name=variant["name"],
            )
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
        for variant in await self.list_collection_variants(collection_name=collection_name, tenant=tenant):
            self.client.delete_collection(name=variant["name"])

    async def list_collections(self) -> List[Dict[str, Any]]:
        if self.client is None:
            raise RuntimeError("Vector store not initialized")

        collections = self.client.list_collections()
        output: list[dict[str, Any]] = []
        for collection in collections:
            base_name, embedding_version = self._split_embedding_collection_name(collection.name)
            parsed = parse_collection_name(base_name)
            output.append(
                {
                    "name": collection.name,
                    "logical_name": base_name,
                    "base_collection": parsed.base_collection,
                    "user_id": parsed.user_id,
                    "agent_id": parsed.agent_id,
                    "embedding_version": embedding_version,
                }
            )
        return output

    async def _get_collection(
        self,
        collection_name: str | None,
        tenant: TenantContext | None,
        actual_collection_name: str | None = None,
    ):
        if self.client is None:
            await self.initialize()
        if actual_collection_name:
            return await self._ensure_collection_by_actual_name(actual_collection_name)
        actual_name = self._resolve_runtime_collection_name(
            collection_name or self.collection_name,
            tenant=tenant or self.default_tenant,
        )
        return await self._ensure_collection_by_actual_name(actual_name)

    async def _ensure_collection(
        self,
        collection_name: str,
        tenant: TenantContext | None,
    ):
        if self.client is None:
            raise RuntimeError("Vector store not initialized")

        actual_name = self._resolve_runtime_collection_name(collection_name, tenant=tenant)
        return await self._ensure_collection_by_actual_name(actual_name)

    async def _ensure_collection_by_actual_name(
        self,
        actual_name: str,
    ):
        if self.client is None:
            raise RuntimeError("Vector store not initialized")
        try:
            collection = self.client.get_collection(name=actual_name)
            space = collection.metadata.get("hnsw:space") if collection.metadata else None
            if space != "cosine":
                self.client.delete_collection(name=actual_name)
                collection = self.client.create_collection(
                    name=actual_name,
                    metadata=self._collection_metadata(),
                )
            return collection
        except Exception:
            return self.client.create_collection(
                name=actual_name,
                metadata=self._collection_metadata(),
            )

    def _resolve_runtime_collection_name(
        self,
        collection_name: str,
        tenant: TenantContext | None = None,
    ) -> str:
        logical_name = resolve_collection_name(collection_name, tenant=tenant)
        embedding_version = self._embedding_version()
        if not embedding_version:
            return logical_name
        return f"{logical_name}__emb_{embedding_version}"

    def _collection_metadata(self) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {"hnsw:space": "cosine"}
        if self.embedding_model is None:
            return metadata
        provider_name = getattr(self.embedding_model, "provider_name", None)
        model_name = getattr(self.embedding_model, "model_name", None) or getattr(self.embedding_model, "model", None)
        base_url = getattr(self.embedding_model, "base_url", None)
        dimensions = getattr(self.embedding_model, "dimensions", None)
        if provider_name:
            metadata["embedding_provider"] = str(provider_name)
        if model_name:
            metadata["embedding_model"] = str(model_name)
        if base_url:
            metadata["embedding_base_url"] = str(base_url)
        if dimensions not in (None, "", 0):
            metadata["embedding_dimensions"] = int(dimensions)
        return metadata

    def _embedding_version(self) -> str | None:
        model = self.embedding_model
        if model is None:
            return None

        provider_name = getattr(model, "provider_name", None)
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
        dimensions = getattr(model, "dimensions", None)
        parts = [str(provider_name or "").strip(), str(model_name or "").strip()]
        if dimensions not in (None, "", 0):
            parts.append(f"d{dimensions}")
        version = sanitize_collection_name("_".join(part for part in parts if part))
        return version or None

    def _split_embedding_collection_name(self, collection_name: str) -> tuple[str, str | None]:
        match = _EMBEDDING_COLLECTION_SUFFIX_RE.match(collection_name)
        if not match:
            return collection_name, None
        return match.group("base"), match.group("version")

    async def _list_document_records(
        self,
        *,
        collection_name: str | None,
        tenant: TenantContext | None,
        filename: str | None = None,
    ) -> List[Dict[str, Any]]:
        records: list[dict[str, Any]] = []
        where = {"filename": filename} if filename else None
        for variant in await self.list_collection_variants(collection_name=collection_name, tenant=tenant):
            collection = await self._get_collection(
                collection_name,
                tenant,
                actual_collection_name=variant["name"],
            )
            results = collection.get(where=where, include=["documents", "metadatas"])
            ids = results.get("ids") or []
            metadatas = results.get("metadatas") or []
            contents = results.get("documents") or []
            for index, chunk_id in enumerate(ids):
                metadata = self._decode_metadata(metadatas[index] if index < len(metadatas) else {})
                metadata.setdefault("collection_variant", variant["name"])
                records.append(
                    {
                        "id": chunk_id,
                        "content": contents[index] if index < len(contents) else "",
                        "metadata": metadata,
                    }
                )
        records.sort(key=lambda item: str(item["id"]))
        return records

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
