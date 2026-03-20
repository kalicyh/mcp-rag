"""Core indexing models for MCP-RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


@dataclass(slots=True)
class TenantContext:
    """Tenant scope used to resolve collection names."""

    base_collection: str = "default"
    user_id: Optional[int] = None
    agent_id: Optional[int] = None


@dataclass(slots=True)
class IndexingSettings:
    """Lightweight settings object for the indexing foundation."""

    default_collection: str = "default"
    persist_directory: str = "./data/chroma"
    chunk_size: int = 4000
    chunk_overlap: int = 200
    separators: Sequence[str] = field(default_factory=lambda: ("\n\n", "\n", " ", ""))
    schema_version: int = 1
    embedding_provider: str = "openai-compatible"
    embedding_model: str = "doubao-embedding-text-240715"
    embedding_base_url: str = "https://ark.cn-beijing.volces.com/api/v3"
    embedding_api_key: Optional[str] = None
    embedding_device: str = "cpu"
    embedding_cache_dir: Optional[str] = None
    embedding_dimensions: Optional[int] = None


@dataclass(slots=True)
class ProcessedDocument:
    """Document ready for chunking and indexing."""

    document_id: str
    source: str
    filename: str
    file_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass(slots=True)
class ChunkRecord:
    """Chunk emitted by the document processor."""

    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    source: str
    filename: str
    file_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchHit:
    """Search result returned by the vector store."""

    chunk_id: str
    document_id: str
    score: float
    source: str
    filename: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FileSummary:
    """Aggregated file view from stored chunks."""

    filename: str
    source: str
    file_type: str
    chunk_count: int
    total_chars: int
    document_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    first_seen_at: Optional[datetime] = None

