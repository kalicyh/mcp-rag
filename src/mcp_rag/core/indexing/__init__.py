"""Core indexing foundation for MCP-RAG."""

from .document_processor import DocumentProcessor
from .embeddings import (
    DoubaoEmbeddingModel,
    EmbeddingModel,
    OpenAICompatibleEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from .models import (
    ChunkRecord,
    FileSummary,
    IndexingSettings,
    ProcessedDocument,
    SearchHit,
    TenantContext,
)
from .tenancy import build_collection_name, parse_collection_name, resolve_collection_name
from .text_splitter import RecursiveCharacterTextSplitter, split_text
from .vector_store import ChromaVectorStore

