from __future__ import annotations

import tempfile
import unittest
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

import mcp_rag.http_server as http_server_module
from mcp_rag.config import ConfigManager
from mcp_rag.contracts import ChatResponse, SearchResponse, SearchResultView
from mcp_rag.http_server import ConfigUpdate, app
from mcp_rag.observability import ObservabilityCollector
from mcp_rag.security import SecurityPolicy, TokenBucketRateLimiter
from mcp_rag.services import ServiceRuntime
from mcp_rag.services.retrieval_cache import RetrievalCacheKey
from mcp_rag.shell_factory import create_shell_context


class HttpServerFacadeTests(unittest.TestCase):
    def setUp(self):
        self.original_context = app.state.shell_context
        self.original_config_manager = http_server_module.config_manager
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        app.state.shell_context = self.original_context
        http_server_module.config_manager = self.original_config_manager

    def _set_context(self, context):
        self.client.close()
        app.state.shell_context = context
        self.client = TestClient(app)

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
        original_provider = app.state.shell_context.service_provider
        app.state.shell_context.service_provider = AsyncMock(return_value=service)
        try:
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
        finally:
            app.state.shell_context.service_provider = original_provider

    def test_document_management_routes_use_facade(self):
        service = self._fake_service()
        original_provider = app.state.shell_context.service_provider
        app.state.shell_context.service_provider = AsyncMock(return_value=service)
        try:
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
        finally:
            app.state.shell_context.service_provider = original_provider

    def test_health_metrics_and_guardrails_use_shell_context(self):
        service = self._fake_service()
        context = create_shell_context(
            service_provider=AsyncMock(return_value=service),
            security_policy=SecurityPolicy(enabled=True, allow_anonymous=False, api_keys=["secret"]),
            rate_limiter=TokenBucketRateLimiter(limit=1, window_seconds=60),
            observability=ObservabilityCollector(),
        )
        context.bootstrapped = True
        self._set_context(context)

        ready = self.client.get("/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertTrue(ready.json()["ready"])

        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ready"])

        denied = self.client.get("/search", params={"query": "fastapi", "collection": "docs"})
        self.assertEqual(denied.status_code, 401)

        allowed = self.client.get(
            "/search",
            params={"query": "fastapi", "collection": "docs"},
            headers={"x-api-key": "secret"},
        )
        self.assertEqual(allowed.status_code, 200)

        throttled = self.client.get(
            "/search",
            params={"query": "fastapi", "collection": "docs"},
            headers={"x-api-key": "secret"},
        )
        self.assertEqual(throttled.status_code, 429)

        metrics = self.client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        payload = metrics.json()["metrics"]
        self.assertGreaterEqual(payload["total_requests"], 3)
        self.assertEqual(payload["operations"]["search"]["count"], 3)
        self.assertEqual(service.search.await_count, 1)

    def test_http_routes_expose_request_and_trace_headers(self):
        service = self._fake_service()
        original_provider = app.state.shell_context.service_provider
        app.state.shell_context.service_provider = AsyncMock(return_value=service)
        try:
            response = self.client.get(
                "/search",
                params={"query": "fastapi", "collection": "docs"},
                headers={"x-request-id": "req-1", "x-trace-id": "trace-1"},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-request-id"], "req-1")
            self.assertEqual(response.headers["x-trace-id"], "trace-1")
            called_request = service.search.await_args.args[0]
            self.assertEqual(called_request.context.request_id, "req-1")
            self.assertEqual(called_request.context.trace_id, "trace-1")
        finally:
            app.state.shell_context.service_provider = original_provider

    def test_config_update_hot_reloads_shell_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_manager = ConfigManager(config_file=f"{tmpdir}/config.json")
            http_server_module.config_manager = config_manager
            context = create_shell_context(
                settings_obj=config_manager.settings,
                config_manager_obj=config_manager,
                service_provider=AsyncMock(return_value=self._fake_service()),
            )
            context.bootstrapped = True
            self._set_context(context)

            response = self.client.post(
                "/config",
                json={
                    "key": "rate_limit",
                    "value": {"requests_per_window": 9, "window_seconds": 30, "burst": 2},
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["reloaded"])
            self.assertEqual(app.state.shell_context.settings.rate_limit.requests_per_window, 9)
            self.assertEqual(app.state.shell_context.rate_limiter.limit, 9)
            self.assertEqual(app.state.shell_context.rate_limiter.window_seconds, 30.0)

    def test_config_update_invalidates_runtime_retrieval_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ConfigManager(config_file=f"{tmpdir}/config.json")
            manager.update_setting("enable_cache", True)
            runtime = ServiceRuntime(settings_obj=manager.settings)
            context = create_shell_context(
                runtime=runtime,
                settings_obj=runtime.settings,
                config_manager_obj=manager,
            )
            context.bootstrapped = True
            self._set_context(context)

            original_manager = http_server_module.config_manager
            http_server_module.config_manager = manager
            try:
                cache = runtime.get_retrieval_cache()
                key = RetrievalCacheKey(
                    base_collection="docs",
                    user_id=7,
                    agent_id=None,
                    actual_collection="u7_docs",
                    query="fastapi",
                    mode="raw",
                    limit=3,
                    threshold=0.7,
                    summary_enabled=False,
                    rerank_enabled=False,
                    retrieval_window=5,
                )
                cache.set(
                    key,
                    SearchResponse(
                        query="fastapi",
                        collection="docs",
                        results=[
                            SearchResultView(
                                content="cached",
                                score=0.9,
                                metadata={},
                                source="guide.md",
                                filename="guide.md",
                            )
                        ],
                    ),
                )

                response = self.client.post("/config", json=ConfigUpdate(key="enable_reranker", value=True).model_dump())

                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["reloaded"])
                self.assertIsNotNone(runtime.get_retrieval_cache())
                self.assertEqual(runtime.get_retrieval_cache().snapshot()["entries"], 0)
                self.assertTrue(runtime.settings.enable_reranker)
            finally:
                http_server_module.config_manager = original_manager


if __name__ == "__main__":
    unittest.main()
