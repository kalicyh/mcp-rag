from __future__ import annotations

import unittest

from mcp_rag.contracts import SearchResponse, SearchResultView
from mcp_rag.services.retrieval_cache import RetrievalCache, RetrievalCacheKey


def _build_key(**overrides) -> RetrievalCacheKey:
    payload = {
        "base_collection": "docs",
        "user_id": 7,
        "agent_id": 2,
        "actual_collection": "u7_a2_docs",
        "query": "fastapi",
        "mode": "summary",
        "limit": 3,
        "threshold": 0.7,
        "summary_enabled": True,
        "rerank_enabled": False,
        "retrieval_window": 5,
    }
    payload.update(overrides)
    return RetrievalCacheKey(**payload)


def _build_response(summary: str = "summary text") -> SearchResponse:
    return SearchResponse(
        query="fastapi",
        collection="docs",
        results=[
            SearchResultView(
                content="FastAPI routing and dependencies",
                score=0.91,
                metadata={"source": "guide.md"},
                source="guide.md",
                filename="guide.md",
                retrieval_method="hybrid",
            )
        ],
        summary=summary,
    )


class RetrievalCacheTests(unittest.TestCase):
    def test_get_returns_cloned_response(self) -> None:
        cache = RetrievalCache()
        key = _build_key()

        self.assertTrue(cache.set(key, _build_response()))
        first = cache.get(key)
        self.assertIsNotNone(first)
        first.results[0].metadata["source"] = "mutated.md"

        second = cache.get(key)
        self.assertEqual(second.results[0].metadata["source"], "guide.md")

        snapshot = cache.snapshot()
        self.assertEqual(snapshot["entries"], 1)
        self.assertEqual(snapshot["hits"], 2)
        self.assertEqual(snapshot["writes"], 1)

    def test_scope_invalidation_rejects_stale_write(self) -> None:
        cache = RetrievalCache()
        key = _build_key()
        generation = cache.generation_for(key)

        self.assertTrue(cache.set(key, _build_response(), expected_generation=generation))
        self.assertEqual(cache.invalidate_scope(scope_token=key.scope_token), 1)
        self.assertIsNone(cache.get(key))
        self.assertFalse(cache.set(key, _build_response("stale"), expected_generation=generation))

    def test_global_invalidation_rejects_stale_write(self) -> None:
        cache = RetrievalCache()
        key = _build_key()
        generation = cache.generation_for(key)

        self.assertTrue(cache.set(key, _build_response(), expected_generation=generation))
        self.assertEqual(cache.invalidate_all(), 1)
        self.assertFalse(cache.set(key, _build_response("stale"), expected_generation=generation))
        self.assertIsNone(cache.get(key))

    def test_reconfigure_updates_limits(self) -> None:
        cache = RetrievalCache(max_entries=10, ttl_seconds=30)

        cache.reconfigure(max_entries=2, ttl_seconds=5)

        snapshot = cache.snapshot()
        self.assertEqual(snapshot["max_entries"], 2)
        self.assertEqual(snapshot["ttl_seconds"], 5)

    def test_expired_entries_are_evicted_on_read(self) -> None:
        ticks = iter([0.0, 5.0])
        cache = RetrievalCache(ttl_seconds=3, clock=lambda: next(ticks))
        key = _build_key()

        self.assertTrue(cache.set(key, _build_response()))
        self.assertIsNone(cache.get(key))
        self.assertEqual(cache.snapshot()["entries"], 0)

if __name__ == "__main__":
    unittest.main()
