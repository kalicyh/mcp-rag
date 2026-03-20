import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_rag.core.indexing import (
    ChromaVectorStore,
    DocumentProcessor,
    IndexingSettings,
    OpenAICompatibleEmbeddingModel,
    RecursiveCharacterTextSplitter,
    TenantContext,
    build_collection_name,
    parse_collection_name,
)
from mcp_rag.core.indexing.models import ChunkRecord


class TenancyTests(unittest.TestCase):
    def test_build_and_parse_collection_name(self):
        self.assertEqual(build_collection_name("default"), "default")
        self.assertEqual(build_collection_name("docs", user_id=7), "u7_docs")
        self.assertEqual(build_collection_name("docs", user_id=7, agent_id=3), "u7_a3_docs")

        parsed = parse_collection_name("u7_a3_docs")
        self.assertEqual(parsed.base_collection, "docs")
        self.assertEqual(parsed.user_id, 7)
        self.assertEqual(parsed.agent_id, 3)

    def test_parse_legacy_collection_name(self):
        parsed = parse_collection_name("plain_collection")
        self.assertEqual(parsed.base_collection, "plain_collection")
        self.assertIsNone(parsed.user_id)
        self.assertIsNone(parsed.agent_id)


class TextSplitterTests(unittest.TestCase):
    def test_split_text_with_overlap_window(self):
        splitter = RecursiveCharacterTextSplitter(chunk_size=10, chunk_overlap=2, separators=("",))
        chunks = splitter.split_text("abcdefghijklmno")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 10 for chunk in chunks))
        self.assertEqual(chunks[0], "abcdefghij")


class DocumentProcessorTests(unittest.TestCase):
    def setUp(self):
        self.processor = DocumentProcessor(IndexingSettings(chunk_size=20, chunk_overlap=5))

    def test_process_text_and_chunk(self):
        doc = self.processor.process_text(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            source="inline",
            filename="inline.txt",
            metadata={"topic": "demo"},
        )
        self.assertEqual(doc.filename, "inline.txt")
        self.assertEqual(doc.metadata["topic"], "demo")

        chunks = self.processor.chunk_document(doc)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(isinstance(chunk, ChunkRecord) for chunk in chunks))
        self.assertEqual(chunks[0].metadata["document_id"], doc.document_id)

    def test_process_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.txt"
            path.write_text("hello\nworld", encoding="utf-8")

            doc = self.processor.process_file(path)
            self.assertEqual(doc.filename, "sample.txt")
            self.assertEqual(doc.file_type, "txt")
            self.assertIn("char_count", doc.metadata)
            self.assertIn("content_hash", doc.metadata)


class EmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_openai_compatible_batch_and_single_encode(self):
        fake_response = MagicMock()
        fake_response.status_code = 200
        fake_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }

        fake_client = AsyncMock()
        fake_client.post = AsyncMock(return_value=fake_response)
        fake_client.aclose = AsyncMock()

        with patch("mcp_rag.core.indexing.embeddings.httpx.AsyncClient", return_value=fake_client):
            model = OpenAICompatibleEmbeddingModel(
                api_key="test-key",
                base_url="https://example.com/v1",
                model="test-model",
            )
            await model.initialize()

            batch = await model.encode(["first", "second"])
            self.assertEqual(batch[0], [0.1, 0.2, 0.3])
            single = await model.encode_single("first")
            self.assertEqual(single, [0.1, 0.2, 0.3])

            await model.close()
            fake_client.aclose.assert_awaited()


class VectorStoreTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = ChromaVectorStore(persist_directory=self.tmpdir.name)
        await self.store.initialize()

    async def asyncTearDown(self):
        self.tmpdir.cleanup()

    async def test_add_search_list_delete(self):
        chunks = [
            ChunkRecord(
                chunk_id="doc1_chunk_0000",
                document_id="doc1",
                chunk_index=0,
                total_chunks=2,
                source="sample.txt",
                filename="sample.txt",
                file_type="txt",
                content="alpha beta gamma",
                metadata={
                    "document_id": "doc1",
                    "filename": "sample.txt",
                    "file_type": "txt",
                    "source": "sample.txt",
                    "chunk_char_count": 17,
                    "processed_at": "2026-01-01T00:00:00+00:00",
                },
            ),
            ChunkRecord(
                chunk_id="doc1_chunk_0001",
                document_id="doc1",
                chunk_index=1,
                total_chunks=2,
                source="sample.txt",
                filename="sample.txt",
                file_type="txt",
                content="delta epsilon zeta",
                metadata={
                    "document_id": "doc1",
                    "filename": "sample.txt",
                    "file_type": "txt",
                    "source": "sample.txt",
                    "chunk_char_count": 19,
                    "processed_at": "2026-01-01T00:00:00+00:00",
                },
            ),
        ]
        embeddings = [[1.0, 0.0, 0.0], [0.8, 0.1, 0.1]]

        ids = await self.store.add_chunks(chunks, embeddings=embeddings)
        self.assertEqual(ids, [chunk.chunk_id for chunk in chunks])

        results = await self.store.search([1.0, 0.0, 0.0], limit=2, threshold=0.0)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0].filename, "sample.txt")

        files = await self.store.list_files()
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0]["chunk_count"], 2)

        docs = await self.store.list_documents(filename="sample.txt")
        self.assertEqual(docs["total"], 2)
        self.assertEqual(len(docs["documents"]), 2)

        self.assertTrue(await self.store.delete_file("sample.txt"))
        after_delete = await self.store.list_documents(filename="sample.txt")
        self.assertEqual(len(after_delete["documents"]), 0)

    async def test_tenant_collections(self):
        tenant = TenantContext(base_collection="docs", user_id=9, agent_id=2)
        await self.store.upsert_chunks(
            [
                ChunkRecord(
                    chunk_id="doc2_chunk_0000",
                    document_id="doc2",
                    chunk_index=0,
                    total_chunks=1,
                    source="tenant.txt",
                    filename="tenant.txt",
                    file_type="txt",
                    content="tenant content",
                    metadata={
                        "document_id": "doc2",
                        "filename": "tenant.txt",
                        "file_type": "txt",
                        "source": "tenant.txt",
                        "chunk_char_count": 14,
                    },
                )
            ],
            embeddings=[[0.2, 0.2, 0.6]],
            tenant=tenant,
        )

        collections = await self.store.list_collections()
        self.assertTrue(any(item["name"] == "u9_a2_docs" for item in collections))

        hits = await self.store.search([0.2, 0.2, 0.6], tenant=tenant, threshold=0.0)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].document_id, "doc2")


if __name__ == "__main__":
    unittest.main()
