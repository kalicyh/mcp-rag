"""Persistent knowledge base registry and request resolution helpers."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

_ACTIVE_STATUS = "active"
_PUBLIC_SCOPE = "public"
_AGENT_PRIVATE_SCOPE = "agent_private"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_text(value: Any, default: str | None = None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text or default


def _normalize_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


@dataclass(slots=True, frozen=True)
class KnowledgeBase:
    """Business-facing knowledge base entity."""

    id: int
    name: str
    scope: str
    owner_user_id: int | None
    owner_agent_id: int | None
    collection_name: str
    legacy_collection_key: str | None
    status: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_public(self) -> bool:
        return self.scope == _PUBLIC_SCOPE


@dataclass(slots=True, frozen=True)
class KnowledgeBaseResolution:
    """Resolved knowledge base information for one request."""

    knowledge_base: KnowledgeBase
    selected_via: str

    @property
    def kb_id(self) -> int:
        return self.knowledge_base.id

    @property
    def collection_name(self) -> str:
        return self.knowledge_base.collection_name

    @property
    def scope(self) -> str:
        return self.knowledge_base.scope

    @property
    def name(self) -> str:
        return self.knowledge_base.name


class KnowledgeBaseAccessError(RuntimeError):
    """Raised when a caller cannot access a knowledge base."""


class KnowledgeBaseStore:
    """SQLite-backed knowledge base registry."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    owner_user_id INTEGER,
                    owner_agent_id INTEGER,
                    collection_name TEXT NOT NULL UNIQUE,
                    legacy_collection_key TEXT,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_knowledge_bases_legacy
                    ON knowledge_bases(legacy_collection_key)
                    WHERE legacy_collection_key IS NOT NULL;
                CREATE INDEX IF NOT EXISTS idx_knowledge_bases_access
                    ON knowledge_bases(scope, owner_user_id, owner_agent_id, status);
                """
            )

    def _row_to_model(self, row: sqlite3.Row | None) -> KnowledgeBase | None:
        if row is None:
            return None
        return KnowledgeBase(
            id=int(row["id"]),
            name=str(row["name"]),
            scope=str(row["scope"]),
            owner_user_id=_normalize_int(row["owner_user_id"]),
            owner_agent_id=_normalize_int(row["owner_agent_id"]),
            collection_name=str(row["collection_name"]),
            legacy_collection_key=_normalize_text(row["legacy_collection_key"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def get(self, kb_id: int) -> KnowledgeBase | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_bases WHERE id = ? LIMIT 1",
                (int(kb_id),),
            ).fetchone()
        return self._row_to_model(row)

    def get_by_legacy_key(self, legacy_collection_key: str | None) -> KnowledgeBase | None:
        key = _normalize_text(legacy_collection_key)
        if key is None:
            return None
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_bases WHERE legacy_collection_key = ? LIMIT 1",
                (key,),
            ).fetchone()
        return self._row_to_model(row)

    def get_public_default(self) -> KnowledgeBase | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM knowledge_bases
                WHERE scope = ? AND owner_user_id IS NULL AND owner_agent_id IS NULL AND status = ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (_PUBLIC_SCOPE, _ACTIVE_STATUS),
            ).fetchone()
        return self._row_to_model(row)

    def get_default_agent_private(self, *, user_id: int, agent_id: int) -> KnowledgeBase | None:
        with self._lock, closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM knowledge_bases
                WHERE scope = ? AND owner_user_id = ? AND owner_agent_id = ? AND status = ?
                ORDER BY id ASC
                LIMIT 1
                """,
                (_AGENT_PRIVATE_SCOPE, int(user_id), int(agent_id), _ACTIVE_STATUS),
            ).fetchone()
        return self._row_to_model(row)

    def list_accessible(self, *, user_id: int | None = None) -> list[KnowledgeBase]:
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM knowledge_bases
                WHERE status = ?
                  AND (
                    scope = ?
                    OR (? IS NOT NULL AND owner_user_id = ?)
                  )
                ORDER BY CASE WHEN scope = ? THEN 0 ELSE 1 END, name COLLATE NOCASE ASC, id ASC
                """,
                (_ACTIVE_STATUS, _PUBLIC_SCOPE, user_id, user_id, _PUBLIC_SCOPE),
            ).fetchall()
        return [item for row in rows if (item := self._row_to_model(row)) is not None]

    def create(
        self,
        *,
        name: str,
        scope: str,
        owner_user_id: int | None = None,
        owner_agent_id: int | None = None,
        legacy_collection_key: str | None = None,
    ) -> KnowledgeBase:
        resolved_name = _normalize_text(name, default="Knowledge Base") or "Knowledge Base"
        resolved_scope = _normalize_text(scope, default=_PUBLIC_SCOPE) or _PUBLIC_SCOPE
        now = _utc_now()
        temporary_collection = f"pending_{uuid4().hex}"
        with self._lock, closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO knowledge_bases(
                    name,
                    scope,
                    owner_user_id,
                    owner_agent_id,
                    collection_name,
                    legacy_collection_key,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    resolved_name,
                    resolved_scope,
                    _normalize_int(owner_user_id),
                    _normalize_int(owner_agent_id),
                    temporary_collection,
                    _normalize_text(legacy_collection_key),
                    _ACTIVE_STATUS,
                    now,
                    now,
                ),
            )
            kb_id = int(cursor.lastrowid)
            collection_name = f"kb_{kb_id}"
            connection.execute(
                "UPDATE knowledge_bases SET collection_name = ?, updated_at = ? WHERE id = ?",
                (collection_name, now, kb_id),
            )
            connection.commit()
        created = self.get(kb_id)
        if created is None:
            raise RuntimeError(f"Knowledge base {kb_id} was not persisted")
        return created


class KnowledgeBaseService:
    """Higher-level knowledge base resolution with default creation."""

    def __init__(self, db_path: str):
        self.store = KnowledgeBaseStore(db_path)

    def ensure_public_default(self) -> KnowledgeBase:
        existing = self.store.get_public_default()
        if existing is not None:
            return existing
        return self.store.create(
            name="公共知识库",
            scope=_PUBLIC_SCOPE,
            legacy_collection_key="legacy:public:default",
        )

    def ensure_agent_private_default(self, *, user_id: int, agent_id: int) -> KnowledgeBase:
        existing = self.store.get_default_agent_private(user_id=int(user_id), agent_id=int(agent_id))
        if existing is not None:
            return existing
        return self.store.create(
            name=f"Agent {int(agent_id)} 知识库",
            scope=_AGENT_PRIVATE_SCOPE,
            owner_user_id=int(user_id),
            owner_agent_id=int(agent_id),
            legacy_collection_key=f"legacy:agent_private:{int(user_id)}:{int(agent_id)}:default",
        )

    def create_knowledge_base(
        self,
        *,
        name: str,
        scope: str,
        owner_user_id: int | None = None,
        owner_agent_id: int | None = None,
        legacy_collection_key: str | None = None,
    ) -> KnowledgeBase:
        resolved_scope = self.normalize_scope(scope, user_id=owner_user_id, agent_id=owner_agent_id)
        if resolved_scope == _PUBLIC_SCOPE:
            owner_user_id = None
            owner_agent_id = None
        if resolved_scope == _AGENT_PRIVATE_SCOPE:
            if owner_user_id is None:
                raise ValueError("agent_private knowledge bases require owner_user_id")
        return self.store.create(
            name=name,
            scope=resolved_scope,
            owner_user_id=owner_user_id,
            owner_agent_id=owner_agent_id,
            legacy_collection_key=legacy_collection_key,
        )

    def list_accessible(self, *, user_id: int | None = None, agent_id: int | None = None) -> list[KnowledgeBase]:
        items = self.store.list_accessible(user_id=user_id)
        self.ensure_public_default()
        if user_id is not None and agent_id is not None:
            default_private = self.ensure_agent_private_default(user_id=user_id, agent_id=agent_id)
            if all(item.id != default_private.id for item in items):
                items.append(default_private)
        public_default = self.ensure_public_default()
        if all(item.id != public_default.id for item in items):
            items.append(public_default)
        return sorted(
            items,
            key=lambda item: (0 if item.scope == _PUBLIC_SCOPE else 1, item.name.lower(), item.id),
        )

    def get(self, kb_id: int) -> KnowledgeBase | None:
        return self.store.get(int(kb_id))

    def resolve(
        self,
        *,
        kb_id: int | None = None,
        scope: str | None = None,
        user_id: int | None = None,
        agent_id: int | None = None,
        legacy_collection: str | None = None,
        legacy_collection_key: str | None = None,
    ) -> KnowledgeBaseResolution:
        resolved_user_id = _normalize_int(user_id)
        resolved_agent_id = _normalize_int(agent_id)
        resolved_kb_id = _normalize_int(kb_id)

        if resolved_kb_id is not None:
            knowledge_base = self.store.get(resolved_kb_id)
            if knowledge_base is None:
                raise KnowledgeBaseAccessError(f"knowledge base {resolved_kb_id} not found")
            self._ensure_access(knowledge_base, user_id=resolved_user_id)
            return KnowledgeBaseResolution(knowledge_base=knowledge_base, selected_via="kb_id")

        legacy_key = _normalize_text(legacy_collection_key)
        if legacy_key is not None:
            knowledge_base = self.store.get_by_legacy_key(legacy_key)
            if knowledge_base is not None:
                self._ensure_access(knowledge_base, user_id=resolved_user_id)
                return KnowledgeBaseResolution(knowledge_base=knowledge_base, selected_via="legacy_key")

        resolved_scope = self.normalize_scope(scope, user_id=resolved_user_id, agent_id=resolved_agent_id)
        if resolved_scope == _PUBLIC_SCOPE:
            collection_name = _normalize_text(legacy_collection, default="公共知识库")
            if collection_name and collection_name not in {"default", "knowledge", "公共知识库"}:
                legacy_key = legacy_key or f"legacy:public:{collection_name}"
                existing = self.store.get_by_legacy_key(legacy_key)
                if existing is not None:
                    return KnowledgeBaseResolution(knowledge_base=existing, selected_via="legacy_public")
                created = self.create_knowledge_base(
                    name=collection_name,
                    scope=_PUBLIC_SCOPE,
                    legacy_collection_key=legacy_key,
                )
                return KnowledgeBaseResolution(knowledge_base=created, selected_via="legacy_public")
            return KnowledgeBaseResolution(knowledge_base=self.ensure_public_default(), selected_via="scope")

        if resolved_user_id is None or resolved_agent_id is None:
            raise KnowledgeBaseAccessError("agent_private knowledge base requires user_id and agent_id")

        if legacy_collection and legacy_collection not in {"default", "knowledge"}:
            legacy_key = legacy_key or f"legacy:agent_private:{resolved_user_id}:{resolved_agent_id}:{legacy_collection}"
            existing = self.store.get_by_legacy_key(legacy_key)
            if existing is not None:
                self._ensure_access(existing, user_id=resolved_user_id)
                return KnowledgeBaseResolution(knowledge_base=existing, selected_via="legacy_private")
            created = self.create_knowledge_base(
                name=legacy_collection,
                scope=_AGENT_PRIVATE_SCOPE,
                owner_user_id=resolved_user_id,
                owner_agent_id=resolved_agent_id,
                legacy_collection_key=legacy_key,
            )
            return KnowledgeBaseResolution(knowledge_base=created, selected_via="legacy_private")

        return KnowledgeBaseResolution(
            knowledge_base=self.ensure_agent_private_default(user_id=resolved_user_id, agent_id=resolved_agent_id),
            selected_via="scope",
        )

    def normalize_scope(self, scope: str | None, *, user_id: int | None, agent_id: int | None) -> str:
        resolved_scope = (_normalize_text(scope) or "").lower()
        if resolved_scope in {"public", _PUBLIC_SCOPE}:
            return _PUBLIC_SCOPE
        if resolved_scope in {"agent_private", "private", _AGENT_PRIVATE_SCOPE}:
            return _AGENT_PRIVATE_SCOPE
        if user_id is not None and agent_id is not None:
            return _AGENT_PRIVATE_SCOPE
        return _PUBLIC_SCOPE

    def _ensure_access(self, knowledge_base: KnowledgeBase, *, user_id: int | None) -> None:
        if knowledge_base.scope == _PUBLIC_SCOPE:
            return
        if user_id is None or knowledge_base.owner_user_id != int(user_id):
            raise KnowledgeBaseAccessError(f"access denied for knowledge base {knowledge_base.id}")
