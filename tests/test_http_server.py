from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from mcp_rag.contracts import ChatResponse, SearchResponse, SearchResultView
from mcp_rag.http_server import app


class HttpServerFacadeTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()

    def _fake_service(self):
        service = type("FakeService", (), {})()
        service.add_document = AsyncMock(return_value={"message": "Document added successfully", "document_id": "doc-1", "chunk_count": 1})
        service.upload_files = AsyncMock(return_value={"total_files": 1, "successful": 1, "failed": 0, "results": []})
        service.list_collections = AsyncMock(return_value=["default", "u7_docs"])
        service.chat = AsyncMock(
            return_value=ChatResponse(
                query="What is FastAPI?",
                collection="docs",
                response="mock answer",
                sources=[
                    SearchResultView(
                        content="FastAPI routing and dependencies",
                        score=0.91,
                        metadata={"source": "guide.md"},
                        source="guide.md",
                        filename="guide.md",
                    )
                ],
            )
        )
        service.search = AsyncMock(
            return_value=SearchResponse(
                query="fastapi",
                collection="docs",
                results=[
                    SearchResultView(
                        content="FastAPI routing and dependencies",
                        score=0.91,
                        metadata={"source": "guide.md"},
                        source="guide.md",
                        filename="guide.md",
                    )
                ],
                summary="summary text",
            )
        )
        service.list_documents = AsyncMock(return_value={"total": 1, "documents": [{"id": "doc-1", "content": "content", "metadata": {"filename": "sample.txt"}}], "limit": 100, "offset": 0})
        service.delete_document = AsyncMock(return_value=True)
        service.list_files = AsyncMock(return_value=[{"filename": "sample.txt", "chunk_count": 1}])
        service.delete_file = AsyncMock(return_value=True)
        return service

    def test_search_chat_upload_and_collections_use_facade(self):
        service = self._fake_service()
        with patch("mcp_rag.http_server.get_rag_service", new=AsyncMock(return_value=service)):
            search = self.client.get("/search", params={"query": "fastapi", "collection": "docs", "limit": 3, "user_id": 7, "agent_id": 2})
            self.assertEqual(search.status_code, 200)
            self.assertEqual(search.json()["results"][0]["filename"], "guide.md")

            chat = self.client.post(
                "/chat",
                json={"query": "What is FastAPI?", "collection": "docs", "user_id": 7, "agent_id": 2},
            )
            self.assertEqual(chat.status_code, 200)
            self.assertEqual(chat.json()["response"], "mock answer")

            collections = self.client.get("/collections", params={"user_id": 7, "agent_id": 2})
            self.assertEqual(collections.status_code, 200)
            self.assertEqual(collections.json()["collections"], ["default", "u7_docs"])

            upload = self.client.post(
                "/upload-files",
                data={"collection": "docs", "user_id": 7, "agent_id": 2},
                files=[("files", ("sample.txt", b"alpha beta gamma", "text/plain"))],
            )
            self.assertEqual(upload.status_code, 200)
            self.assertEqual(upload.json()["successful"], 1)

            self.assertTrue(service.search.await_count)
            self.assertTrue(service.chat.await_count)
            self.assertTrue(service.upload_files.await_count)
            self.assertTrue(service.list_collections.await_count)

    def test_document_management_routes_use_facade(self):
        service = self._fake_service()
        with patch("mcp_rag.http_server.get_rag_service", new=AsyncMock(return_value=service)):
            add = self.client.post(
                "/add-document",
                json={
                    "content": "alpha beta gamma",
                    "collection": "docs",
                    "metadata": {"title": "Intro"},
                    "user_id": 7,
                    "agent_id": 2,
                },
            )
            self.assertEqual(add.status_code, 200)
            self.assertEqual(add.json()["document_id"], "doc-1")

            docs = self.client.get("/list-documents", params={"collection": "docs", "limit": 10, "offset": 0, "user_id": 7, "agent_id": 2})
            self.assertEqual(docs.status_code, 200)
            self.assertEqual(docs.json()["total"], 1)

            files = self.client.get("/list-files", params={"collection": "docs", "user_id": 7, "agent_id": 2})
            self.assertEqual(files.status_code, 200)
            self.assertEqual(files.json()["files"][0]["filename"], "sample.txt")

            delete_doc = self.client.request(
                "DELETE",
                "/delete-document",
                json={"document_id": "doc-1", "collection": "docs", "user_id": 7, "agent_id": 2},
            )
            self.assertEqual(delete_doc.status_code, 200)
            self.assertEqual(delete_doc.json()["message"], "Document deleted successfully")

            delete_file = self.client.request(
                "DELETE",
                "/delete-file",
                json={"filename": "sample.txt", "collection": "docs", "user_id": 7, "agent_id": 2},
            )
            self.assertEqual(delete_file.status_code, 200)
            self.assertEqual(delete_file.json()["message"], "File deleted successfully")

            self.assertTrue(service.add_document.await_count)
            self.assertTrue(service.list_documents.await_count)
            self.assertTrue(service.list_files.await_count)
            self.assertTrue(service.delete_document.await_count)
            self.assertTrue(service.delete_file.await_count)


if __name__ == "__main__":
    unittest.main()
