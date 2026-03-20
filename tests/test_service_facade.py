from __future__ import annotations

import io
import unittest
from dataclasses import dataclass
from pathlib import Path

from mcp_rag.contracts import (
    ChatRequest,
    DocumentRequest,
    SearchRequest,
    SearchResultView,
    TenantSpec,
)
from mcp_rag.service_facade import RagService


@dataclass
class _FakeProviderConfig:
    base_url: str = "https://example.com/v1"
    model: str = "fake-model"
    api_key: str | None = "fake-key"


@dataclass
class _FakeSettings:
    chroma_persist_directory: str = "./data/chroma"
    embedding_provider: str = "zhipu"
    embedding_device: str = "cpu"
    embedding_cache_dir: str | None = None
    provider_configs: dict[str, _FakeProviderConfig] = None  # type: ignore[assignment]
    enable_llm_summary: bool = True
    max_retrieval_results: int = 5
    enable_reranker: bool = False

    def __post_init__(self):
        if self.provider_configs is None:
            self.provider_configs = {"zhipu": _FakeProviderConfig()}


@dataclass
class _FakeChunk:
    chunk_id: str
    content: str
    metadata: dict


@dataclass
class _FakeProcessedDocument:
    document_id: str
    source: str
    filename: str
    file_type: str
    content: str
    metadata: dict
    error: str | None = None


class _FakeDocumentProcessor:
    def __init__(self):
        self.process_text_calls = []
        self.process_file_calls = []

    def process_text(self, text, *, source="inline_text", filename=None, file_type="text", metadata=None):
        self.process_text_calls.append((text, source, filename, file_type, dict(metadata or {})))
        return _FakeProcessedDocument(
            document_id="doc-1",
            source=source,
            filename=filename or source,
            file_type=file_type,
            content=text,
            metadata=dict(metadata or {}),
        )

    def process_file(self, file_path, *, metadata=None, filename=None):
        self.process_file_calls.append((str(file_path), filename, dict(metadata or {})))
        content = Path(file_path).read_text(encoding="utf-8")
        return _FakeProcessedDocument(
            document_id="doc-upload",
            source=str(file_path),
            filename=filename or Path(file_path).name,
            file_type="txt",
            content=content,
            metadata=dict(metadata or {}),
        )

    def chunk_document(self, document, *, chunk_size=None, chunk_overlap=None):
        return [
            _FakeChunk(
                chunk_id=f"{document.document_id}_chunk_0000",
                content=document.content,
                metadata={
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "source": document.source,
                    "file_type": document.file_type,
                    **document.metadata,
                },
            )
        ]


class _FakeVectorStore:
    def __init__(self):
        self.embedding_model = None
        self.upserts = []
        self.deletes = []
        self.collections = [
            {"name": "default", "base_collection": "default", "user_id": None, "agent_id": None},
            {"name": "u7_docs", "base_collection": "docs", "user_id": 7, "agent_id": None},
        ]

    async def initialize(self):
        return None

    async def upsert_chunks(self, chunks, *, tenant=None, collection_name=None, embeddings=None):
        self.upserts.append((collection_name, tenant, list(chunks)))
        return [chunk.chunk_id for chunk in chunks]

    async def list_documents(self, *, collection_name="default", limit=100, offset=0, filename=None, tenant=None):
        return {
            "total": 1,
            "documents": [
                {
                    "id": "doc-1_chunk_0000",
                    "content": "stored content",
                    "metadata": {
                        "document_id": "doc-1",
                        "filename": filename or "sample.txt",
                        "source": "sample.txt",
                        "file_type": "txt",
                    },
                }
            ],
            "limit": limit,
            "offset": offset,
        }

    async def list_files(self, *, collection_name="default", tenant=None):
        return [
            {
                "filename": "sample.txt",
                "source": "sample.txt",
                "file_type": "txt",
                "chunk_count": 1,
                "total_chars": 12,
                "document_id": "doc-1",
                "metadata": {"filename": "sample.txt"},
                "first_seen_at": None,
            }
        ]

    async def list_collections(self):
        return list(self.collections)

    async def delete_document(self, document_id, *, tenant=None, collection_name=None):
        self.deletes.append(("document", document_id, collection_name, tenant))
        return True

    async def delete_file(self, filename, *, tenant=None, collection_name=None):
        self.deletes.append(("file", filename, collection_name, tenant))
        return True


class _FakeHybridService:
    def __init__(self):
        self.calls = []

    async def retrieve(self, query, *, collection_name="default", tenant=None, limit=5, threshold=0.7):
        self.calls.append((query, collection_name, tenant, limit, threshold))
        return [
            SearchResultView(
                content="FastAPI routing and dependencies",
                score=0.91,
                metadata={"source": "guide.md"},
                source="guide.md",
                filename="guide.md",
                retrieval_method="hybrid",
            )
        ]


class _FakeLLM:
    def __init__(self):
        self.generate_calls = []
        self.summarize_calls = []

    async def generate(self, prompt: str, **kwargs) -> str:
        self.generate_calls.append(prompt)
        return "mock answer"

    async def summarize(self, content: str, query: str) -> str:
        self.summarize_calls.append((content, query))
        return f"summary for {query}"


class RagServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = _FakeSettings()
        self.processor = _FakeDocumentProcessor()
        self.vector_store = _FakeVectorStore()
        self.hybrid = _FakeHybridService()
        self.llm = _FakeLLM()
        self.service = RagService(
            settings_obj=self.settings,
            document_processor=self.processor,
            vector_store=self.vector_store,
            hybrid_service=self.hybrid,
            llm_model=self.llm,
        )

    async def test_add_document_uses_tenant_and_upsert(self):
        result = await self.service.add_document(
            DocumentRequest(
                content="alpha beta gamma",
                collection="docs",
                metadata={"title": "Intro"},
                tenant=TenantSpec(base_collection="docs", user_id=7, agent_id=2),
            )
        )

        self.assertEqual(result["document_id"], "doc-1")
        self.assertEqual(result["chunk_count"], 1)
        self.assertEqual(len(self.vector_store.upserts), 1)
        collection_name, tenant, chunks = self.vector_store.upserts[0]
        self.assertEqual(collection_name, "docs")
        self.assertEqual(tenant.base_collection, "docs")
        self.assertEqual(tenant.user_id, 7)
        self.assertEqual(tenant.agent_id, 2)
        self.assertEqual(chunks[0].metadata["filename"], "Intro")

    async def test_upload_files_returns_batch_result(self):
        class _Upload:
            def __init__(self, filename: str, content: bytes):
                self.filename = filename
                self.file = io.BytesIO(content)

        payload = await self.service.upload_files(
            [_Upload("sample.txt", b"file content")],
            collection="docs",
            tenant=TenantSpec(base_collection="docs", user_id=7),
        )

        self.assertEqual(payload["total_files"], 1)
        self.assertEqual(payload["successful"], 1)
        self.assertEqual(payload["failed"], 0)
        self.assertEqual(len(self.vector_store.upserts), 1)

    async def test_search_and_chat_use_hybrid_and_llm(self):
        search = await self.service.search(
            SearchRequest(
                query="fastapi",
                collection="docs",
                limit=3,
                tenant=TenantSpec(base_collection="docs", user_id=7),
            )
        )
        self.assertEqual(search.query, "fastapi")
        self.assertEqual(search.collection, "docs")
        self.assertEqual(search.summary, "summary for fastapi")
        self.assertEqual(search.results[0].source, "guide.md")
        self.assertEqual(self.hybrid.calls[0][1], "docs")

        chat = await self.service.chat(
            ChatRequest(
                query="What is FastAPI?",
                collection="docs",
                limit=3,
                tenant=TenantSpec(base_collection="docs", user_id=7),
            )
        )
        self.assertEqual(chat.response, "mock answer")
        self.assertEqual(chat.sources[0].filename, "guide.md")
        self.assertTrue(self.llm.generate_calls)

    async def test_management_calls_are_tenant_aware(self):
        docs = await self.service.list_documents(
            collection="docs",
            tenant=TenantSpec(base_collection="docs", user_id=7),
        )
        files = await self.service.list_files(
            collection="docs",
            tenant=TenantSpec(base_collection="docs", user_id=7),
        )
        collections = await self.service.list_collections(tenant=TenantSpec(base_collection="docs", user_id=7))
        deleted_document = await self.service.delete_document(
            document_id="doc-1",
            collection="docs",
            tenant=TenantSpec(base_collection="docs", user_id=7),
        )
        deleted_file = await self.service.delete_file(
            filename="sample.txt",
            collection="docs",
            tenant=TenantSpec(base_collection="docs", user_id=7),
        )

        self.assertEqual(docs["total"], 1)
        self.assertEqual(files[0]["filename"], "sample.txt")
        self.assertIn("u7_docs", collections)
        self.assertTrue(deleted_document)
        self.assertTrue(deleted_file)
        self.assertEqual(len(self.vector_store.deletes), 2)


if __name__ == "__main__":
    unittest.main()
