"""Minimal hybrid retrieval service for MCP-RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

from mcp_rag.core.indexing.models import SearchHit, TenantContext

from .collection_index import CollectionKeywordIndex, KeywordSearchHit
from .query_classifier import QueryClassification, QueryClassifier, QueryIntent

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HybridSearchResult:
    """Unified retrieval result from vector and keyword channels."""

    chunk_id: str
    document_id: str
    score: float
    source: str
    filename: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_method: str = "vector"
    rank_position: int = 0


@dataclass(slots=True)
class _Candidate:
    chunk_id: str
    document_id: str
    source: str
    filename: str
    content: str
    metadata: Dict[str, Any]
    vector_score: float = 0.0
    keyword_score: float = 0.0
    vector_rank: int | None = None
    keyword_rank: int | None = None


class HybridRetrievalService:
    """Tenant-aware hybrid retrieval over core.indexing primitives."""

    def __init__(
        self,
        *,
        vector_store,
        embedding_model,
        embedding_model_factory=None,
        collection_index: CollectionKeywordIndex | None = None,
        classifier: QueryClassifier | None = None,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        candidate_pool_size: int = 10,
        rerank_enabled: bool = False,
        fusion_k: int = 60,
    ):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.embedding_model_factory = embedding_model_factory
        self.collection_index = collection_index or CollectionKeywordIndex(vector_store)
        self.classifier = classifier or QueryClassifier()
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.candidate_pool_size = max(1, candidate_pool_size)
        self.rerank_enabled = rerank_enabled
        self.fusion_k = max(1, fusion_k)

    async def retrieve(
        self,
        query: str,
        *,
        collection_name: str = "default",
        tenant: TenantContext | None = None,
        limit: int = 5,
        threshold: float = 0.7,
    ) -> List[HybridSearchResult]:
        if limit <= 0:
            return []

        classification = self.classifier.classify(query)
        vector_weight, keyword_weight = self._adapt_weights(classification)

        vector_hits = await self._search_vector_versions(
            query,
            collection_name=collection_name,
            tenant=tenant,
            limit=max(limit, self.candidate_pool_size),
            threshold=threshold,
        )
        keyword_hits = await self.collection_index.search(
            query,
            collection_name=collection_name,
            tenant=tenant,
            limit=max(limit, self.candidate_pool_size),
            threshold=0.0,
        )

        fused = self._fuse(vector_hits, keyword_hits, vector_weight=vector_weight, keyword_weight=keyword_weight)
        if not fused:
            return []

        final = fused[:limit]
        for position, item in enumerate(final, start=1):
            item.rank_position = position
        return final

    async def _search_vector_versions(
        self,
        query: str,
        *,
        collection_name: str,
        tenant: TenantContext | None,
        limit: int,
        threshold: float,
    ) -> List[SearchHit]:
        if hasattr(self.vector_store, "list_collection_variants"):
            variants = await self.vector_store.list_collection_variants(collection_name=collection_name, tenant=tenant)
        else:
            variants = [{"name": None, "current": True}]
        hits: list[SearchHit] = []
        for variant in variants:
            model = self.embedding_model if variant.get("current") else None
            if model is None:
                if self.embedding_model_factory is None:
                    continue
                model = await self.embedding_model_factory(variant)
            query_embedding = await model.encode_single(query)
            try:
                if variant.get("name") is None:
                    variant_hits = await self.vector_store.search(
                        query_embedding=query_embedding,
                        collection_name=collection_name,
                        limit=limit,
                        threshold=threshold,
                        tenant=tenant,
                    )
                else:
                    variant_hits = await self.vector_store.search(
                        query_embedding=query_embedding,
                        collection_name=collection_name,
                        actual_collection_name=variant["name"],
                        limit=limit,
                        threshold=threshold,
                        tenant=tenant,
                    )
            except Exception as exc:
                if not self._is_dimension_mismatch_error(exc):
                    raise
                logger.warning(
                    "Skipping collection variant %s for %s due to embedding dimension mismatch.",
                    variant.get("name") or collection_name,
                    collection_name,
                )
                continue
            hits.extend(variant_hits)
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def _is_dimension_mismatch_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return "expecting embedding with dimension" in message and "got" in message

    def _fuse(
        self,
        vector_hits: Sequence[SearchHit],
        keyword_hits: Sequence[KeywordSearchHit],
        *,
        vector_weight: float,
        keyword_weight: float,
    ) -> List[HybridSearchResult]:
        candidates: dict[str, _Candidate] = {}

        for rank, hit in enumerate(vector_hits, start=1):
            candidate = candidates.setdefault(
                hit.chunk_id,
                _Candidate(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    source=hit.source,
                    filename=hit.filename,
                    content=hit.content,
                    metadata=dict(hit.metadata or {}),
                ),
            )
            candidate.vector_score = max(candidate.vector_score, float(hit.score))
            candidate.vector_rank = rank
            if not candidate.document_id:
                candidate.document_id = hit.document_id
            if not candidate.source:
                candidate.source = hit.source
            if not candidate.filename:
                candidate.filename = hit.filename
            if not candidate.content:
                candidate.content = hit.content
            if hit.metadata:
                candidate.metadata = self._merge_metadata(candidate.metadata, hit.metadata)

        for rank, hit in enumerate(keyword_hits, start=1):
            candidate = candidates.setdefault(
                hit.chunk_id,
                _Candidate(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    source=hit.source,
                    filename=hit.filename,
                    content=hit.content,
                    metadata=dict(hit.metadata or {}),
                ),
            )
            candidate.keyword_score = max(candidate.keyword_score, float(hit.score))
            candidate.keyword_rank = rank
            if not candidate.document_id:
                candidate.document_id = hit.document_id
            if not candidate.source:
                candidate.source = hit.source
            if not candidate.filename:
                candidate.filename = hit.filename
            if not candidate.content:
                candidate.content = hit.content
            if hit.metadata:
                candidate.metadata = self._merge_metadata(candidate.metadata, hit.metadata)

        ranked: list[tuple[float, HybridSearchResult]] = []
        for candidate in candidates.values():
            score = self._combined_score(
                candidate.vector_rank,
                candidate.keyword_rank,
                candidate.vector_score,
                candidate.keyword_score,
                vector_weight=vector_weight,
                keyword_weight=keyword_weight,
            )
            retrieval_method = "hybrid" if candidate.vector_rank and candidate.keyword_rank else (
                "vector" if candidate.vector_rank else "keyword"
            )
            ranked.append(
                (
                    score,
                    HybridSearchResult(
                        chunk_id=candidate.chunk_id,
                        document_id=candidate.document_id,
                        score=score,
                        source=candidate.source,
                        filename=candidate.filename,
                        content=candidate.content,
                        metadata=candidate.metadata,
                        retrieval_method=retrieval_method,
                    ),
                )
            )

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked]

    def _combined_score(
        self,
        vector_rank: int | None,
        keyword_rank: int | None,
        vector_score: float,
        keyword_score: float,
        *,
        vector_weight: float,
        keyword_weight: float,
    ) -> float:
        score = 0.0
        if vector_rank is not None:
            score += vector_weight * (1.0 / (self.fusion_k + vector_rank))
            score += vector_weight * 0.05 * vector_score
        if keyword_rank is not None:
            score += keyword_weight * (1.0 / (self.fusion_k + keyword_rank))
            score += keyword_weight * 0.05 * keyword_score
        return score

    def _adapt_weights(self, classification: QueryClassification) -> tuple[float, float]:
        intent = classification.primary_intent
        if intent == QueryIntent.TROUBLESHOOTING:
            return 0.4, 0.6
        if intent in {QueryIntent.CONCEPTUAL, QueryIntent.BEST_PRACTICES}:
            return 0.8, 0.2
        if intent == QueryIntent.TECHNICAL_DOCS:
            return 0.6, 0.4
        if intent == QueryIntent.CODE_EXPLANATION:
            return 0.75, 0.25
        if intent == QueryIntent.HOW_TO:
            return 0.65, 0.35
        return self.vector_weight, self.keyword_weight

    def _merge_metadata(self, base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(base)
        for key, value in incoming.items():
            if key not in merged or merged[key] in (None, ""):
                merged[key] = value
        return merged
