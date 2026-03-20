"""In-memory request-level cache for retrieval responses."""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Callable

from ..contracts import SearchResponse, SearchResultView

CacheGeneration = tuple[int, int]


def build_scope_token(
    *,
    actual_collection: str,
    base_collection: str,
    user_id: int | None,
    agent_id: int | None,
) -> str:
    """Build a stable tenant-aware invalidation scope."""

    return "|".join(
        (
            actual_collection or "default",
            base_collection or "default",
            "none" if user_id is None else str(user_id),
            "none" if agent_id is None else str(agent_id),
        )
    )


def clone_search_response(response: SearchResponse) -> SearchResponse:
    """Copy a response so callers cannot mutate cache state."""

    return SearchResponse(
        query=response.query,
        collection=response.collection,
        results=[
            SearchResultView(
                content=item.content,
                score=item.score,
                metadata=dict(item.metadata or {}),
                source=item.source,
                filename=item.filename,
                retrieval_method=item.retrieval_method,
            )
            for item in response.results
        ],
        summary=response.summary,
    )


@dataclass(frozen=True, slots=True)
class RetrievalCacheKey:
    """Lookup key for request-level retrieval caching."""

    base_collection: str
    user_id: int | None
    agent_id: int | None
    actual_collection: str
    query: str
    mode: str
    limit: int
    threshold: float
    summary_enabled: bool
    rerank_enabled: bool
    retrieval_window: int
    schema_version: str = "v1"

    @property
    def scope_token(self) -> str:
        return build_scope_token(
            actual_collection=self.actual_collection,
            base_collection=self.base_collection,
            user_id=self.user_id,
            agent_id=self.agent_id,
        )


@dataclass(slots=True)
class RetrievalCacheEntry:
    """Stored retrieval response and the generation it belongs to."""

    response: SearchResponse
    created_at: float
    generation: CacheGeneration


class RetrievalCache:
    """Thread-safe in-memory cache with scope invalidation and TTL."""

    def __init__(
        self,
        *,
        max_entries: int = 256,
        ttl_seconds: int = 300,
        clock: Callable[[], float] = monotonic,
    ):
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(1, int(ttl_seconds))
        self._clock = clock
        self._entries: dict[RetrievalCacheKey, RetrievalCacheEntry] = {}
        self._scope_index: dict[str, set[RetrievalCacheKey]] = {}
        self._scope_generations: dict[str, int] = {}
        self._global_generation = 0
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._evictions = 0
        self._lock = RLock()

    def reconfigure(self, *, max_entries: int, ttl_seconds: int) -> None:
        with self._lock:
            self.max_entries = max(1, int(max_entries))
            self.ttl_seconds = max(1, int(ttl_seconds))
            self._purge_expired_unlocked()
            self._enforce_capacity_unlocked()

    def generation_for(self, key: RetrievalCacheKey) -> CacheGeneration:
        with self._lock:
            return self._current_generation_unlocked(key.scope_token)

    def get(self, key: RetrievalCacheKey) -> SearchResponse | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None

            current_generation = self._current_generation_unlocked(key.scope_token)
            if entry.generation != current_generation or self._is_expired(entry):
                self._discard_key_unlocked(key)
                self._misses += 1
                return None

            self._hits += 1
            return clone_search_response(entry.response)

    def set(
        self,
        key: RetrievalCacheKey,
        response: SearchResponse,
        *,
        expected_generation: CacheGeneration | None = None,
    ) -> bool:
        with self._lock:
            current_generation = self._current_generation_unlocked(key.scope_token)
            if expected_generation is not None and current_generation != expected_generation:
                return False

            self._entries[key] = RetrievalCacheEntry(
                response=clone_search_response(response),
                created_at=self._clock(),
                generation=current_generation,
            )
            self._scope_index.setdefault(key.scope_token, set()).add(key)
            self._writes += 1
            self._enforce_capacity_unlocked()
            return True

    def invalidate_scope(self, *, scope_token: str) -> int:
        with self._lock:
            removed = self._remove_scope_entries_unlocked(scope_token)
            self._scope_generations[scope_token] = self._scope_generations.get(scope_token, 0) + 1
            self._evictions += removed
            return removed

    def invalidate_all(self) -> int:
        with self._lock:
            removed = len(self._entries)
            self._entries.clear()
            self._scope_index.clear()
            self._global_generation += 1
            self._evictions += removed
            return removed

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "scopes": len(self._scope_index),
                "hits": self._hits,
                "misses": self._misses,
                "writes": self._writes,
                "evictions": self._evictions,
                "global_generation": self._global_generation,
                "max_entries": self.max_entries,
                "ttl_seconds": self.ttl_seconds,
            }

    def _current_generation_unlocked(self, scope_token: str) -> CacheGeneration:
        return self._global_generation, self._scope_generations.get(scope_token, 0)

    def _discard_key_unlocked(self, key: RetrievalCacheKey) -> None:
        self._entries.pop(key, None)
        keys = self._scope_index.get(key.scope_token)
        if not keys:
            return
        keys.discard(key)
        if not keys:
            self._scope_index.pop(key.scope_token, None)

    def _remove_scope_entries_unlocked(self, scope_token: str) -> int:
        keys = self._scope_index.pop(scope_token, set())
        for key in keys:
            self._entries.pop(key, None)
        return len(keys)

    def _enforce_capacity_unlocked(self) -> None:
        overflow = len(self._entries) - self.max_entries
        if overflow <= 0:
            return

        stale_keys = sorted(self._entries.items(), key=lambda item: item[1].created_at)[:overflow]
        for key, _entry in stale_keys:
            self._discard_key_unlocked(key)
        self._evictions += len(stale_keys)

    def _is_expired(self, entry: RetrievalCacheEntry) -> bool:
        return (self._clock() - entry.created_at) >= float(self.ttl_seconds)

    def _purge_expired_unlocked(self) -> None:
        expired_keys = [key for key, entry in self._entries.items() if self._is_expired(entry)]
        for key in expired_keys:
            self._discard_key_unlocked(key)
        self._evictions += len(expired_keys)
