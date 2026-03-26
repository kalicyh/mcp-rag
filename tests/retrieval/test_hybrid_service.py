from __future__ import annotations

import math
import unittest
from dataclasses import dataclass

from mcp_rag.core.indexing.models import SearchHit, TenantContext
from mcp_rag.retrieval.hybrid_service import HybridRetrievalService
from mcp_rag.retrieval.query_classifier import QueryIntent


@dataclass
class _Doc:
    id: str
    content: str
    metadata: dict
    embedding: list[float]


class _FakeEmbeddingModel:
    async def encode_single(self, text: str):
        text = text.lower()
        if "fastapi" in text or "python" in text:
            return [1.0, 0.0]
        return [0.0, 1.0]


class _LegacyEmbeddingModel:
    async def encode_single(self, text: str):
        text = text.lower()
        if "legacy" in text or "history" in text:
            return [0.0, 1.0]
        return [1.0, 0.0]


class _FakeVectorStore:
    def __init__(self, documents_by_collection: dict[str, list[_Doc]]):
        self.documents_by_collection = documents_by_collection
        self.search_calls: list[tuple[str, str | None, int, float]] = []
        self.list_calls: list[tuple[str, str | None, int, int]] = []

    async def list_documents(self, *, collection_name: str = "default", tenant=None, limit: int = 100, offset: int = 0):
        actual = self._actual(collection_name, tenant)
        self.list_calls.append((collection_name, actual, limit, offset))
        docs = self.documents_by_collection.get(actual, [])
        slice_ = docs[offset : offset + limit]
        return {
            "total": len(docs),
            "documents": [
                {"id": doc.id, "content": doc.content, "metadata": dict(doc.metadata)}
                for doc in slice_
            ],
        }

    async def search(self, *, query_embedding, collection_name: str = "default", limit: int = 5, threshold: float = 0.7, tenant=None):
        actual = self._actual(collection_name, tenant)
        self.search_calls.append((collection_name, actual, limit, threshold))
        docs = self.documents_by_collection.get(actual, [])
        hits = []
        for doc in docs:
            score = self._cosine(query_embedding, doc.embedding)
            if score >= threshold:
                hits.append(
                    SearchHit(
                        chunk_id=doc.id,
                        document_id=str(doc.metadata.get("document_id", "")),
                        score=score,
                        source=str(doc.metadata.get("source", "")),
                        filename=str(doc.metadata.get("filename", "")),
                        content=doc.content,
                        metadata=dict(doc.metadata),
                    )
                )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    def _actual(self, collection_name: str, tenant) -> str:
        if tenant is None:
            return collection_name
        if tenant.agent_id is not None:
            return f"u{tenant.user_id}_a{tenant.agent_id}_{tenant.base_collection}"
        return f"u{tenant.user_id}_{tenant.base_collection}"

    def _cosine(self, left, right) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)


class _FakeVersionedVectorStore(_FakeVectorStore):
    def __init__(self, documents_by_collection: dict[str, list[_Doc]], variants_by_collection: dict[str, list[dict]]):
        super().__init__(documents_by_collection)
        self.variants_by_collection = variants_by_collection

    async def list_collection_variants(self, *, collection_name: str = "default", tenant=None):
        return list(self.variants_by_collection.get(self._actual(collection_name, tenant), []))

    async def search(
        self,
        *,
        query_embedding,
        collection_name: str = "default",
        limit: int = 5,
        threshold: float = 0.7,
        tenant=None,
        actual_collection_name: str | None = None,
    ):
        actual = actual_collection_name or self._actual(collection_name, tenant)
        self.search_calls.append((collection_name, actual, limit, threshold))
        docs = self.documents_by_collection.get(actual, [])
        hits = []
        for doc in docs:
            score = self._cosine(query_embedding, doc.embedding)
            if score >= threshold:
                hits.append(
                    SearchHit(
                        chunk_id=doc.id,
                        document_id=str(doc.metadata.get("document_id", "")),
                        score=score,
                        source=str(doc.metadata.get("source", "")),
                        filename=str(doc.metadata.get("filename", "")),
                        content=doc.content,
                        metadata=dict(doc.metadata),
                    )
                )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]


class _DimensionMismatchVersionedVectorStore(_FakeVersionedVectorStore):
    async def search(
        self,
        *,
        query_embedding,
        collection_name: str = "default",
        limit: int = 5,
        threshold: float = 0.7,
        tenant=None,
        actual_collection_name: str | None = None,
    ):
        actual = actual_collection_name or self._actual(collection_name, tenant)
        if actual == "u1_docs":
            raise RuntimeError("Collection expecting embedding with dimension of 2048, got 1024")
        return await super().search(
            query_embedding=query_embedding,
            collection_name=collection_name,
            limit=limit,
            threshold=threshold,
            tenant=tenant,
            actual_collection_name=actual_collection_name,
        )


class _RecordingClassifier:
    def classify(self, query: str):
        from mcp_rag.retrieval.query_classifier import QueryClassification

        return QueryClassification(
            primary_intent=QueryIntent.HOW_TO if "how to" in query.lower() else QueryIntent.GENERAL_QA,
            confidence=0.9,
            keywords=["fastapi", "python"],
            is_technical=True,
            all_intents={},
        )


class HybridRetrievalServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_hybrid_retrieval_merges_vector_and_keyword_hits(self) -> None:
        store = _FakeVectorStore(
            {
                "u1_docs": [
                    _Doc("c1", "FastAPI routing and dependencies", {"source": "a.md", "filename": "a.md", "document_id": "doc-a"}, [1.0, 0.0]),
                    _Doc("c2", "Python packaging basics", {"source": "b.md", "filename": "b.md", "document_id": "doc-b"}, [1.0, 0.0]),
                    _Doc("c3", "Build an API server", {"source": "c.md", "filename": "c.md", "document_id": "doc-c"}, [0.0, 1.0]),
                ]
            }
        )
        service = HybridRetrievalService(
            vector_store=store,
            embedding_model=_FakeEmbeddingModel(),
            classifier=_RecordingClassifier(),
            vector_weight=0.7,
            keyword_weight=0.3,
            candidate_pool_size=3,
        )
        results = await service.retrieve(
            "How to build FastAPI API",
            collection_name="docs",
            tenant=TenantContext(base_collection="docs", user_id=1),
            limit=3,
            threshold=0.0,
        )

        self.assertEqual([result.chunk_id for result in results][0], "c1")
        self.assertEqual(results[0].retrieval_method, "hybrid")
        self.assertEqual(results[0].rank_position, 1)
        self.assertEqual(store.search_calls[0][1], "u1_docs")
        self.assertEqual(store.list_calls[0][1], "u1_docs")

    async def test_empty_collection_returns_empty_list(self) -> None:
        store = _FakeVectorStore({})
        service = HybridRetrievalService(
            vector_store=store,
            embedding_model=_FakeEmbeddingModel(),
        )
        results = await service.retrieve(
            "FastAPI",
            collection_name="docs",
            tenant=TenantContext(base_collection="docs", user_id=7),
            limit=5,
            threshold=0.0,
        )
        self.assertEqual(results, [])

    async def test_tenant_isolation_uses_actual_collection_name(self) -> None:
        store = _FakeVectorStore(
            {
                "u1_docs": [
                    _Doc("c1", "FastAPI routing", {"source": "a.md", "filename": "a.md", "document_id": "doc-a"}, [1.0, 0.0]),
                ],
                "u2_docs": [
                    _Doc("c2", "Different tenant content", {"source": "b.md", "filename": "b.md", "document_id": "doc-b"}, [0.0, 1.0]),
                ],
            }
        )
        service = HybridRetrievalService(
            vector_store=store,
            embedding_model=_FakeEmbeddingModel(),
            classifier=_RecordingClassifier(),
            candidate_pool_size=2,
        )

        left = await service.retrieve("FastAPI", collection_name="docs", tenant=TenantContext(base_collection="docs", user_id=1), limit=1, threshold=0.0)
        right = await service.retrieve("FastAPI", collection_name="docs", tenant=TenantContext(base_collection="docs", user_id=2), limit=1, threshold=0.1)

        self.assertEqual(left[0].chunk_id, "c1")
        self.assertEqual(right, [])

    async def test_versioned_retrieval_queries_all_collection_variants(self) -> None:
        async def legacy_factory(_variant):
            return _LegacyEmbeddingModel()

        store = _FakeVersionedVectorStore(
            {
                "u1_docs__emb_new": [
                    _Doc("c-new", "FastAPI current guide", {"source": "new.md", "filename": "new.md", "document_id": "doc-new"}, [1.0, 0.0]),
                ],
                "u1_docs__emb_legacy": [
                    _Doc("c-old", "Legacy history notes", {"source": "old.md", "filename": "old.md", "document_id": "doc-old"}, [0.0, 1.0]),
                ],
            },
            {
                "u1_docs": [
                    {"name": "u1_docs__emb_new", "current": True},
                    {"name": "u1_docs__emb_legacy", "current": False, "embedding_provider": "aliyun", "embedding_model": "legacy-model"},
                ]
            },
        )
        service = HybridRetrievalService(
            vector_store=store,
            embedding_model=_FakeEmbeddingModel(),
            embedding_model_factory=legacy_factory,
            classifier=_RecordingClassifier(),
            candidate_pool_size=4,
        )

        results = await service.retrieve(
            "legacy history",
            collection_name="docs",
            tenant=TenantContext(base_collection="docs", user_id=1),
            limit=3,
            threshold=0.0,
        )

        self.assertEqual(results[0].chunk_id, "c-old")
        self.assertEqual({call[1] for call in store.search_calls}, {"u1_docs__emb_new", "u1_docs__emb_legacy"})

    async def test_dimension_mismatch_variant_is_skipped(self) -> None:
        async def legacy_factory(_variant):
            return _LegacyEmbeddingModel()

        store = _DimensionMismatchVersionedVectorStore(
            {
                "u1_docs__emb_new": [
                    _Doc("c-new", "FastAPI current guide", {"source": "new.md", "filename": "new.md", "document_id": "doc-new"}, [1.0, 0.0]),
                ]
            },
            {
                "u1_docs": [
                    {"name": "u1_docs", "current": False, "embedding_provider": "aliyun", "embedding_model": "legacy-broken"},
                    {"name": "u1_docs__emb_new", "current": True},
                ]
            },
        )
        service = HybridRetrievalService(
            vector_store=store,
            embedding_model=_FakeEmbeddingModel(),
            embedding_model_factory=legacy_factory,
            classifier=_RecordingClassifier(),
            candidate_pool_size=3,
        )

        results = await service.retrieve(
            "FastAPI",
            collection_name="docs",
            tenant=TenantContext(base_collection="docs", user_id=1),
            limit=3,
            threshold=0.0,
        )

        self.assertEqual([item.chunk_id for item in results], ["c-new"])


if __name__ == "__main__":
    unittest.main()
