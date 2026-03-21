from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import mcp_rag.http_server as http_server_module
from mcp_rag.config import ConfigManager, Settings
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
        service.ask = AsyncMock(
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
            self.assertTrue(collections.json()["collections"])
            self.assertTrue(all(name.startswith("kb_") for name in collections.json()["collections"]))

            knowledge_bases = self.client.get("/knowledge-bases", params={"user_id": 7, "agent_id": 2})
            self.assertEqual(knowledge_bases.status_code, 200)
            self.assertTrue(knowledge_bases.json()["knowledge_bases"])

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

    def test_mcp_debug_routes_expose_tools_and_calls(self):
        tools = self.client.get("/debug/mcp/tools")
        self.assertEqual(tools.status_code, 200)
        payload = tools.json()["tools"]
        self.assertTrue(payload)
        self.assertEqual(payload[0]["name"], "rag_ask")

        with patch.object(http_server_module.mcp_server, "debug_call_tool", AsyncMock(return_value={"tool": "rag_ask", "contents": [{"type": "text", "text": "ok"}]})):
            result = self.client.post(
                "/debug/mcp/call",
                json={"tool": "rag_ask", "arguments": {"query": "fastapi", "scope": "public"}},
            )
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["tool"], "rag_ask")
            self.assertTrue(result.json()["contents"])

    def test_health_metrics_and_guardrails_use_shell_context(self):
        service = self._fake_service()
        ready_settings = Settings().model_copy(update={"embedding_provider": "m3e-small"})
        context = create_shell_context(
            settings_obj=ready_settings,
            runtime=ServiceRuntime(settings_obj=ready_settings),
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
        self.assertIn("runtime", ready.json())
        self.assertIn("embedding_model", ready.json()["runtime"])
        self.assertEqual(ready.json()["runtime"]["embedding_model"]["status"], "configured")

        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json()["ready"])
        self.assertIn("config_revision", health.json())

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
        metrics_json = metrics.json()
        payload = metrics_json["metrics"]
        self.assertGreaterEqual(payload["total_requests"], 3)
        self.assertEqual(payload["operations"]["search"]["count"], 3)
        self.assertEqual(service.search.await_count, 1)
        self.assertIn("providers", payload)
        self.assertIn("config_revision", metrics_json)

    def test_ready_returns_503_when_runtime_is_misconfigured(self):
        context = create_shell_context(
            settings_obj=Settings(),
            runtime=ServiceRuntime(settings_obj=Settings()),
            service_provider=AsyncMock(return_value=self._fake_service()),
        )
        context.bootstrapped = True
        self._set_context(context)

        ready = self.client.get("/ready")

        self.assertEqual(ready.status_code, 503)
        self.assertFalse(ready.json()["ready"])
        self.assertEqual(ready.json()["runtime"]["embedding_model"]["status"], "misconfigured")

    def test_root_and_legacy_pages_redirect_to_spa_shell(self):
        root = self.client.get("/", follow_redirects=False)
        self.assertEqual(root.status_code, 307)
        self.assertEqual(root.headers["location"], "/app")

        documents = self.client.get("/documents-page", follow_redirects=False)
        self.assertEqual(documents.status_code, 307)
        self.assertEqual(documents.headers["location"], "/app/documents")

        config = self.client.get("/config-page", follow_redirects=False)
        self.assertEqual(config.status_code, 307)
        self.assertEqual(config.headers["location"], "/app/config")

    def test_app_returns_clear_503_when_spa_bundle_is_missing(self):
        with patch.object(http_server_module, "resolve_spa_entry", return_value=None):
            response = self.client.get("/app")

        self.assertEqual(response.status_code, 503)
        self.assertIn("SPA assets are unavailable", response.text)
        self.assertIn("The existing JSON APIs remain available.", response.text)

    def test_app_serves_prebuilt_spa_entry_when_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = Path(tmpdir) / "index.html"
            entry.write_text("<!DOCTYPE html><html><body><div id='app'></div></body></html>", encoding="utf-8")

            with patch.object(http_server_module, "resolve_spa_entry", return_value=entry):
                response = self.client.get("/app/documents")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("<div id='app'></div>", response.text)

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
            self.assertEqual(app.state.shell_context.config_revision, config_manager.revision)

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

    def test_streamable_http_mcp_smoke(self):
        service = self._fake_service()
        ready_settings = Settings().model_copy(update={"embedding_provider": "m3e-small"})
        context = create_shell_context(
            settings_obj=ready_settings,
            runtime=ServiceRuntime(settings_obj=ready_settings),
            service_provider=AsyncMock(return_value=service),
            security_policy=SecurityPolicy(enabled=True, allow_anonymous=False, api_keys=["secret"]),
            observability=ObservabilityCollector(),
        )
        context.bootstrapped = True
        self._set_context(context)

        with TestClient(app) as client:
            init_response = client.post(
                "/mcp",
                headers={"accept": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0"},
                    },
                },
            )
            self.assertEqual(init_response.status_code, 200)

            tools_response = client.post(
                "/mcp",
                headers={"accept": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {},
                },
            )
            self.assertEqual(tools_response.status_code, 200)
            tools_payload = tools_response.json()["result"]["tools"]
            self.assertTrue(any(tool["name"] == "rag_ask" for tool in tools_payload))

            call_response = client.post(
                "/mcp",
                headers={"accept": "application/json"},
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "rag_ask",
                        "arguments": {
                            "query": "fastapi",
                            "collection": "docs",
                            "api_key": "secret",
                        },
                    },
                },
            )
            self.assertEqual(call_response.status_code, 200)
            self.assertIn("为查询 'fastapi' 找到 1 个相关文档", call_response.json()["result"]["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
