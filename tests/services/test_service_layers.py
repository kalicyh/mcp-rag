from __future__ import annotations

import io
import unittest
from dataclasses import dataclass
from pathlib import Path

from mcp_rag.contracts import ChatRequest, DocumentRequest, SearchRequest, SearchResultView, TenantSpec
from mcp_rag.security import QuotaExceededError
from mcp_rag.services import ChatService, IndexingService, RetrievalService, ServiceRuntime


@dataclass
class _FakeProviderConfig:
    base_url: str = "https://example.com/v1"
    model: str = "fake-model"
    api_key: str | None = "fake-key"


@dataclass
class _FakeQuotas:
    max_upload_files: int = 100
    max_upload_bytes: int = 10_000_000
    max_upload_file_bytes: int = 10_000_000
    max_index_documents: int = 100
    max_index_chunks: int = 1000
    max_index_chars: int = 10_000_000


@dataclass
class _FakeSettings:
    chroma_persist_directory: str = "./data/chroma"
    embedding_provider: str = "zhipu"
    embedding_device: str = "cpu"
    embedding_cache_dir: str | None = None
    provider_configs: dict[str, _FakeProviderConfig] = None  # type: ignore[assignment]
    enable_llm_summary: bool = True
    enable_cache: bool = False
    max_retrieval_results: int = 5
    enable_reranker: bool = False
    quotas: _FakeQuotas | None = None

    def __post_init__(self):
        if self.provider_configs is None:
            self.provider_configs = {"zhipu": _FakeProviderConfig()}
        if self.quotas is None:
            self.quotas = _FakeQuotas()


@dataclass
class _FakeChunk:
    chunk_id: str
    document_id: str
    chunk_index: int
    total_chunks: int
    source: str
    filename: str
    file_type: str
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


class _FakeCollectionIndex:
    def __init__(self):
        self.refresh_calls = []

    async def refresh(self, *, collection_name=None, tenant=None):
        self.refresh_calls.append((collection_name, tenant))


class _FakeCollection:
    def __init__(self):
        self.records = [
            {
                "id": "doc-1_chunk_0000",
                "document_id": "doc-1",
                "content": "stored content",
                "metadata": {
                    "document_id": "doc-1",
                    "filename": "sample.txt",
                    "source": "sample.txt",
                    "file_type": "txt",
                },
            }
        ]
        self.delete_calls = []

    def get(self, ids=None, where=None, include=None, limit=None, offset=None):
        matches = self._match_records(ids=ids, where=where)
        return {
            "ids": [record["id"] for record in matches],
            "documents": [record["content"] for record in matches],
            "metadatas": [dict(record["metadata"]) for record in matches],
        }

    def delete(self, ids=None, where=None):
        self.delete_calls.append(
            {
                "ids": list(ids) if ids is not None else None,
                "where": dict(where) if where is not None else None,
            }
        )
        self.records = [
            record
            for record in self.records
            if not self._matches_record(record, ids=ids, where=where)
        ]

    def _match_records(self, *, ids=None, where=None):
        return [
            record
            for record in self.records
            if self._matches_record(record, ids=ids, where=where)
        ]

    def _matches_record(self, record, *, ids=None, where=None):
        if ids is not None and record["id"] not in set(ids):
            return False
        if where and "document_id" in where and record["document_id"] != where["document_id"]:
            return False
        return True


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
                document_id=document.document_id,
                chunk_index=0,
                total_chunks=1,
                source=document.source,
                filename=document.filename,
                file_type=document.file_type,
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


class _FakeEmbeddingModel:
    def __init__(self):
        self.initialized = False

    async def initialize(self):
        self.initialized = True


class _FakeVectorStore:
    def __init__(self):
        self.embedding_model = None
        self.upserts = []
        self.deletes = []
        self.collections = [
            {"name": "default", "base_collection": "default", "user_id": None, "agent_id": None},
            {"name": "u7_docs", "base_collection": "docs", "user_id": 7, "agent_id": None},
        ]
        self.collection = _FakeCollection()

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

    async def _get_collection(self, collection_name, tenant):
        return self.collection

    async def delete_file(self, filename, *, tenant=None, collection_name=None):
        self.deletes.append(("file", filename, collection_name, tenant))
        return True


class _FakeHybridService:
    def __init__(self):
        self.calls = []
        self.collection_index = _FakeCollectionIndex()

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
    def __init__(self, *, fail_summarize: bool = False, fail_generate: bool = False):
        self.fail_summarize = fail_summarize
        self.fail_generate = fail_generate
        self.generate_calls = []
        self.summarize_calls = []

    async def generate(self, prompt: str, **kwargs) -> str:
        self.generate_calls.append(prompt)
        if self.fail_generate:
            raise RuntimeError("generate failed")
        return "mock answer"

    async def summarize(self, content: str, query: str) -> str:
        self.summarize_calls.append((content, query))
        if self.fail_summarize:
            raise RuntimeError("summarize failed")
        return f"summary for {query}"


class ServiceRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_indexing_settings_uses_provider_config(self):
        runtime = ServiceRuntime(settings_obj=_FakeSettings())

        indexing_settings = runtime.build_indexing_settings()

        self.assertEqual(indexing_settings.embedding_model, "fake-model")
        self.assertEqual(indexing_settings.embedding_base_url, "https://example.com/v1")
        self.assertEqual(indexing_settings.persist_directory, "./data/chroma")

    async def test_attach_embedding_model_sets_vector_store(self):
        runtime = ServiceRuntime(settings_obj=_FakeSettings())
        vector_store = _FakeVectorStore()
        embedding_model = _FakeEmbeddingModel()

        runtime.attach_embedding_model(vector_store, embedding_model)

        self.assertIs(vector_store.embedding_model, embedding_model)


class IndexingServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = _FakeSettings()
        self.processor = _FakeDocumentProcessor()
        self.embedding = _FakeEmbeddingModel()
        self.vector_store = _FakeVectorStore()
        self.hybrid = _FakeHybridService()
        self.runtime = ServiceRuntime(
            settings_obj=self.settings,
            document_processor=self.processor,
            embedding_model=self.embedding,
            vector_store=self.vector_store,
            hybrid_service=self.hybrid,
        )
        self.service = IndexingService(self.runtime)

    async def test_add_document_uses_tenant_and_refreshes_keywords(self):
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
        self.assertEqual(self.hybrid.collection_index.refresh_calls[0][0], "docs")

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
        collection_name, tenant, chunks = self.vector_store.upserts[0]
        self.assertEqual(collection_name, "docs")
        self.assertEqual(chunks[0].source, "sample.txt")
        self.assertEqual(chunks[0].metadata["source"], "sample.txt")
        self.assertEqual(chunks[0].metadata["filename"], "sample.txt")

    async def test_add_document_enforces_index_quota(self):
        limited_runtime = ServiceRuntime(
            settings_obj=_FakeSettings(quotas=_FakeQuotas(max_index_documents=1, max_index_chunks=0, max_index_chars=10)),
            document_processor=self.processor,
            embedding_model=self.embedding,
            vector_store=self.vector_store,
            hybrid_service=self.hybrid,
        )
        limited_service = IndexingService(limited_runtime)

        with self.assertRaises(QuotaExceededError):
            await limited_service.add_document(
                DocumentRequest(
                    content="alpha beta gamma",
                    collection="docs",
                    metadata={"title": "Intro"},
                    tenant=TenantSpec(base_collection="docs", user_id=7, agent_id=2),
                )
            )

    async def test_upload_files_marks_batch_failed_when_upload_quota_is_exceeded(self):
        limited_runtime = ServiceRuntime(
            settings_obj=_FakeSettings(quotas=_FakeQuotas(max_upload_files=1, max_upload_bytes=4, max_upload_file_bytes=4)),
            document_processor=self.processor,
            embedding_model=self.embedding,
            vector_store=self.vector_store,
            hybrid_service=self.hybrid,
        )
        limited_service = IndexingService(limited_runtime)

        class _Upload:
            def __init__(self, filename: str, content: bytes):
                self.filename = filename
                self.file = io.BytesIO(content)

        payload = await limited_service.upload_files(
            [_Upload("sample.txt", b"file content")],
            collection="docs",
            tenant=TenantSpec(base_collection="docs", user_id=7),
        )

        self.assertEqual(payload["successful"], 0)
        self.assertEqual(payload["failed"], 1)
        self.assertIn("file too large", payload["results"][0]["error"])

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
        self.assertEqual(self.vector_store.collection.delete_calls[0]["where"], {"document_id": "doc-1"})
        self.assertEqual(len(self.vector_store.deletes), 1)
        self.assertEqual(self.vector_store.deletes[0][0], "file")

    async def test_delete_document_accepts_chunk_id(self):
        deleted = await self.service.delete_document(
            document_id="doc-1_chunk_0000",
            collection="docs",
            tenant=TenantSpec(base_collection="docs", user_id=7),
        )

        self.assertTrue(deleted)
        self.assertEqual(self.vector_store.collection.delete_calls[0]["ids"], ["doc-1_chunk_0000"])


class RetrievalAndChatServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.settings = _FakeSettings(enable_llm_summary=True, enable_cache=True)
        self.processor = _FakeDocumentProcessor()
        self.embedding = _FakeEmbeddingModel()
        self.vector_store = _FakeVectorStore()
        self.hybrid = _FakeHybridService()
        self.llm = _FakeLLM()
        self.runtime = ServiceRuntime(
            settings_obj=self.settings,
            document_processor=self.processor,
            embedding_model=self.embedding,
            vector_store=self.vector_store,
            hybrid_service=self.hybrid,
            llm_model=self.llm,
        )
        self.retrieval = RetrievalService(self.runtime)
        self.chat = ChatService(self.runtime, self.retrieval)
        self.indexing = IndexingService(self.runtime)

    async def test_search_and_chat_use_runtime_dependencies(self):
        search = await self.retrieval.search(
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

        cached = await self.retrieval.search(
            SearchRequest(
                query="fastapi",
                collection="docs",
                limit=3,
                tenant=TenantSpec(base_collection="docs", user_id=7),
            )
        )
        self.assertEqual(cached.summary, "summary for fastapi")
        self.assertEqual(len(self.hybrid.calls), 1)
        self.assertEqual(len(self.llm.summarize_calls), 1)

        chat = await self.chat.chat(
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

    async def test_cache_key_isolated_by_scope_and_request_shape(self):
        request = SearchRequest(
            query="fastapi",
            collection="docs",
            limit=3,
            threshold=0.7,
            tenant=TenantSpec(base_collection="docs", user_id=7),
        )
        await self.retrieval.search(request)
        await self.retrieval.search(request)
        self.assertEqual(len(self.hybrid.calls), 1)

        await self.retrieval.search(
            SearchRequest(
                query="fastapi",
                collection="docs",
                limit=4,
                threshold=0.7,
                tenant=TenantSpec(base_collection="docs", user_id=7),
            )
        )
        await self.retrieval.search(
            SearchRequest(
                query="fastapi",
                collection="docs",
                limit=3,
                threshold=0.5,
                tenant=TenantSpec(base_collection="docs", user_id=7),
            )
        )
        await self.retrieval.search(
            SearchRequest(
                query="fastapi",
                collection="docs",
                limit=3,
                threshold=0.7,
                tenant=TenantSpec(base_collection="docs", user_id=8),
            )
        )
        self.assertEqual(len(self.hybrid.calls), 4)

    async def test_ask_falls_back_when_summary_generation_fails(self):
        fallback_runtime = ServiceRuntime(
            settings_obj=_FakeSettings(enable_llm_summary=False, enable_cache=True),
            document_processor=self.processor,
            embedding_model=self.embedding,
            vector_store=self.vector_store,
            hybrid_service=self.hybrid,
            llm_model=_FakeLLM(fail_summarize=True),
        )
        fallback_service = RetrievalService(fallback_runtime)

        response = await fallback_service.ask(
            SearchRequest(
                query="fastapi",
                collection="docs",
                mode="summary",
                tenant=TenantSpec(base_collection="docs", user_id=7),
            )
        )

        self.assertEqual(response.summary, "摘要生成功能暂未启用。")

    async def test_summary_mode_fallback_is_cached(self):
        llm = _FakeLLM()
        fallback_runtime = ServiceRuntime(
            settings_obj=_FakeSettings(enable_llm_summary=False, enable_cache=True),
            document_processor=self.processor,
            embedding_model=self.embedding,
            vector_store=self.vector_store,
            hybrid_service=self.hybrid,
            llm_model=llm,
        )
        fallback_service = RetrievalService(fallback_runtime)
        request = SearchRequest(
            query="fastapi",
            collection="docs",
            mode="summary",
            tenant=TenantSpec(base_collection="docs", user_id=7),
        )

        first = await fallback_service.ask(request)
        second = await fallback_service.ask(request)

        self.assertEqual(first.summary, "summary for fastapi")
        self.assertEqual(second.summary, "summary for fastapi")
        self.assertEqual(len(self.hybrid.calls), 1)
        self.assertEqual(len(llm.summarize_calls), 1)

    async def test_indexing_writes_invalidate_cached_search_scope(self):
        request = SearchRequest(
            query="fastapi",
            collection="docs",
            tenant=TenantSpec(base_collection="docs", user_id=7),
        )
        await self.retrieval.search(request)
        await self.retrieval.search(request)
        self.assertEqual(len(self.hybrid.calls), 1)

        await self.indexing.add_document(
            DocumentRequest(
                content="fresh content",
                collection="docs",
                metadata={"title": "New"},
                tenant=TenantSpec(base_collection="docs", user_id=7),
            )
        )
        await self.retrieval.search(request)
        self.assertEqual(len(self.hybrid.calls), 2)

        await self.indexing.delete_file(
            filename="sample.txt",
            collection="docs",
            tenant=TenantSpec(base_collection="docs", user_id=7),
        )
        await self.retrieval.search(request)
        self.assertEqual(len(self.hybrid.calls), 3)


if __name__ == "__main__":
    unittest.main()
