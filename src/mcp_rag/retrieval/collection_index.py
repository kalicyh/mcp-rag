"""Per-collection keyword index built with TF-IDF."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mcp_rag.core.indexing.models import TenantContext
from mcp_rag.core.indexing.tenancy import resolve_collection_name

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class KeywordSearchHit:
    """Keyword retrieval result."""

    chunk_id: str
    document_id: str
    score: float
    source: str
    filename: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _CollectionKeywordIndex:
    """Cached keyword index for one collection."""

    actual_collection_name: str
    vectorizer: TfidfVectorizer
    matrix: Any
    chunk_ids: List[str]
    documents: List[str]
    metadata: List[Dict[str, Any]]
    signature: str


class CollectionKeywordIndex:
    """Collection-scoped TF-IDF index manager."""

    def __init__(self, vector_store, *, page_size: int = 200):
        self._vector_store = vector_store
        self._page_size = max(1, page_size)
        self._cache: dict[str, _CollectionKeywordIndex] = {}

    async def search(
        self,
        query: str,
        *,
        collection_name: str = "default",
        tenant: TenantContext | None = None,
        limit: int = 5,
        threshold: float = 0.0,
    ) -> List[KeywordSearchHit]:
        index = await self._get_or_build(collection_name=collection_name, tenant=tenant)
        if index is None:
            return []

        query_vector = index.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, index.matrix).ravel()

        ranked: list[KeywordSearchHit] = []
        for position in scores.argsort()[::-1]:
            score = float(scores[position])
            if score <= 0.0 or score < threshold:
                continue

            metadata = index.metadata[position]
            ranked.append(
                KeywordSearchHit(
                    chunk_id=index.chunk_ids[position],
                    document_id=str(metadata.get("document_id") or metadata.get("original_id") or ""),
                    score=score,
                    source=str(metadata.get("source") or ""),
                    filename=str(metadata.get("filename") or ""),
                    content=index.documents[position],
                    metadata=metadata,
                )
            )
            if len(ranked) >= limit:
                break

        return ranked

    async def refresh(
        self,
        *,
        collection_name: str = "default",
        tenant: TenantContext | None = None,
    ) -> None:
        actual = self._actual_collection_name(collection_name, tenant)
        self._cache.pop(actual, None)
        await self._get_or_build(collection_name=collection_name, tenant=tenant)

    async def clear(self) -> None:
        self._cache.clear()

    async def _get_or_build(
        self,
        *,
        collection_name: str,
        tenant: TenantContext | None,
    ) -> _CollectionKeywordIndex | None:
        actual = self._actual_collection_name(collection_name, tenant)
        docs, signature = await self._load_documents(collection_name=collection_name, tenant=tenant)
        if not docs:
            self._cache.pop(actual, None)
            return None

        cached = self._cache.get(actual)
        if cached is not None and cached.signature == signature:
            return cached

        texts = [record["content"] for record in docs]
        try:
            vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2))
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            logger.debug("Collection %s has no usable vocabulary for TF-IDF", actual)
            self._cache.pop(actual, None)
            return None

        index = _CollectionKeywordIndex(
            actual_collection_name=actual,
            vectorizer=vectorizer,
            matrix=matrix,
            chunk_ids=[str(record["id"]) for record in docs],
            documents=texts,
            metadata=[dict(record.get("metadata") or {}) for record in docs],
            signature=signature,
        )
        self._cache[actual] = index
        return index

    async def _load_documents(
        self,
        *,
        collection_name: str,
        tenant: TenantContext | None,
    ) -> tuple[list[dict[str, Any]], str]:
        items: list[dict[str, Any]] = []
        offset = 0
        signature_parts: list[str] = []

        while True:
            result = await self._vector_store.list_documents(
                collection_name=collection_name,
                tenant=tenant,
                limit=self._page_size,
                offset=offset,
            )
            batch = result.get("documents") or []
            if not batch:
                break

            for record in batch:
                items.append(record)
                signature_parts.append(
                    json.dumps(
                        {
                            "id": record.get("id", ""),
                            "content": record.get("content", ""),
                            "metadata": record.get("metadata") or {},
                        },
                        sort_keys=True,
                        ensure_ascii=False,
                        default=str,
                    )
                )

            total = int(result.get("total") or 0)
            offset += len(batch)
            if offset >= total or len(batch) < self._page_size:
                break

        signature = hashlib.sha256("\n".join(signature_parts).encode("utf-8")).hexdigest()
        return items, signature

    def _actual_collection_name(self, collection_name: str, tenant: TenantContext | None) -> str:
        return resolve_collection_name(collection_name, tenant=tenant)
