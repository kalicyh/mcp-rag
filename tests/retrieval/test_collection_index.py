from __future__ import annotations

import unittest
from dataclasses import dataclass

from mcp_rag.core.indexing.models import TenantContext
from mcp_rag.retrieval.collection_index import CollectionKeywordIndex


@dataclass
class _Doc:
    id: str
    content: str
    metadata: dict


class _FakeVectorStore:
    def __init__(self, documents_by_collection: dict[str, list[_Doc]]):
        self.documents_by_collection = documents_by_collection
        self.calls: list[tuple[str, str | None, int, int]] = []

    async def list_documents(self, *, collection_name: str = "default", tenant=None, limit: int = 100, offset: int = 0):
        actual = self._actual(collection_name, tenant)
        self.calls.append((collection_name, actual, limit, offset))
        docs = self.documents_by_collection.get(actual, [])
        slice_ = docs[offset : offset + limit]
        return {
            "total": len(docs),
            "documents": [
                {"id": doc.id, "content": doc.content, "metadata": dict(doc.metadata)}
                for doc in slice_
            ],
        }

    def _actual(self, collection_name: str, tenant) -> str:
        if tenant is None:
            return collection_name
        return f"u{tenant.user_id}_a{tenant.agent_id}_{tenant.base_collection}" if tenant.agent_id is not None else f"u{tenant.user_id}_{tenant.base_collection}"


class CollectionKeywordIndexTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_builds_index_and_returns_best_match(self) -> None:
        store = _FakeVectorStore(
            {
                "u1_docs": [
                    _Doc("c1", "FastAPI routing and dependencies", {"source": "a.md", "filename": "a.md"}),
                    _Doc("c2", "Database migrations with Alembic", {"source": "b.md", "filename": "b.md"}),
                ]
            }
        )
        index = CollectionKeywordIndex(store)
        hits = await index.search("FastAPI dependencies", collection_name="docs", tenant=TenantContext(base_collection="docs", user_id=1), limit=2)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk_id, "c1")
        self.assertEqual(store.calls[0][1], "u1_docs")

    async def test_empty_collection_returns_empty_list(self) -> None:
        store = _FakeVectorStore({})
        index = CollectionKeywordIndex(store)
        hits = await index.search("anything", collection_name="docs", tenant=TenantContext(base_collection="docs", user_id=9), limit=3)
        self.assertEqual(hits, [])

    async def test_cache_is_scoped_by_actual_collection(self) -> None:
        store = _FakeVectorStore(
            {
                "u1_docs": [
                    _Doc("c1", "FastAPI routing", {"source": "a.md", "filename": "a.md"}),
                ],
                "u2_docs": [
                    _Doc("c2", "Python packaging", {"source": "b.md", "filename": "b.md"}),
                ],
            }
        )
        index = CollectionKeywordIndex(store)
        first = await index.search("FastAPI", collection_name="docs", tenant=TenantContext(base_collection="docs", user_id=1), limit=1)
        second = await index.search("Python", collection_name="docs", tenant=TenantContext(base_collection="docs", user_id=2), limit=1)

        self.assertEqual(first[0].chunk_id, "c1")
        self.assertEqual(second[0].chunk_id, "c2")


if __name__ == "__main__":
    unittest.main()
