from __future__ import annotations

import asyncio
import hashlib
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import httpx
from fastapi.testclient import TestClient
from httpx import ASGITransport
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from mcp_rag.core.indexing import ChromaVectorStore, DocumentProcessor, IndexingSettings
from mcp_rag.http_server import app
from mcp_rag.mcp_server import mcp_server
from mcp_rag.observability import ObservabilityCollector
from mcp_rag.security import SecurityPolicy, TokenBucketRateLimiter
from mcp_rag.service_facade import RagService
from mcp_rag.services import ServiceRuntime
from mcp_rag.shell_factory import ShellContext, create_shell_context


@dataclass
class _IntegrationSettings:
    chroma_persist_directory: str
    enable_llm_summary: bool = True
    max_retrieval_results: int = 5
    enable_reranker: bool = False
    quotas: object | None = None


class _FakeEmbeddingModel:
    def __init__(self):
        self.initialized = False

    async def initialize(self):
        self.initialized = True

    async def encode(self, texts):
        return [self._vector(text) for text in texts]

    async def encode_single(self, text: str):
        vectors = await self.encode([text])
        return vectors[0]

    def _vector(self, text: str):
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()
        vector = [float(byte) for byte in digest[:8]]
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]


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


async def _build_shell_context(
    *,
    persist_directory: Path,
    enable_llm_summary: bool = True,
    rate_limit: int = 3,
) -> tuple[ShellContext, RagService, _FakeLLM]:
    settings = _IntegrationSettings(
        chroma_persist_directory=str(persist_directory),
        enable_llm_summary=enable_llm_summary,
        max_retrieval_results=5,
        enable_reranker=False,
        quotas=type(
            "_IntegrationQuotas",
            (),
            {
                "max_upload_files": 100,
                "max_upload_bytes": 10_000_000,
                "max_upload_file_bytes": 10_000_000,
                "max_index_documents": 100,
                "max_index_chunks": 1000,
                "max_index_chars": 10_000_000,
            },
        )(),
    )
    embedding_model = _FakeEmbeddingModel()
    vector_store = ChromaVectorStore(
        persist_directory=str(persist_directory),
        embedding_model=embedding_model,
    )
    await vector_store.initialize()
    document_processor = DocumentProcessor(IndexingSettings(persist_directory=str(persist_directory)))
    llm = _FakeLLM()
    runtime = ServiceRuntime(
        settings_obj=settings,
        document_processor=document_processor,
        embedding_model=embedding_model,
        vector_store=vector_store,
        llm_model=llm,
    )
    service = RagService(runtime=runtime)
    context = ShellContext(
        settings=settings,
        runtime=runtime,
        service_provider=lambda: service,
        security_policy=SecurityPolicy(enabled=True, allow_anonymous=False, api_keys=["test-key"]),
        rate_limiter=TokenBucketRateLimiter(limit=rate_limit, window_seconds=60.0),
        observability=ObservabilityCollector(),
    )
    return context, service, llm


def _httpx_client_factory_for_app(*, headers=None, timeout=None, auth=None):
    return httpx.AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
        headers=headers,
        timeout=timeout,
        auth=auth,
        follow_redirects=True,
    )


class HttpShellIntegrationTests(unittest.TestCase):
    def setUp(self):
        self._original_http_context = app.state.shell_context
        self._original_mcp_context = mcp_server.shell_context

    def tearDown(self):
        app.state.shell_context = self._original_http_context
        mcp_server.shell_context = self._original_mcp_context

    def test_health_ready_and_metrics_do_not_warm_runtime(self):
        context = create_shell_context()
        app.state.shell_context = context
        mcp_server.shell_context = context

        with TestClient(app) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["ready"], True)
            self.assertFalse(health.json()["runtime"]["embedding_model"])
            self.assertFalse(health.json()["runtime"]["llm_model"])

            ready = client.get("/ready")
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(ready.json()["ready"], True)
            self.assertFalse(ready.json()["runtime"]["embedding_model"])
            self.assertFalse(ready.json()["runtime"]["llm_model"])

            metrics = client.get("/metrics")
            self.assertEqual(metrics.status_code, 200)
            self.assertEqual(metrics.json()["metrics"]["total_requests"], 0)

        self.assertIsNone(context.runtime._embedding_model)
        self.assertIsNone(context.runtime._llm_model)

    def test_add_search_chat_and_rate_limit_flow(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            context, _service, _llm = asyncio.run(_build_shell_context(persist_directory=Path(tmpdir)))
            app.state.shell_context = context
            mcp_server.shell_context = context

            with TestClient(app) as client:
                unauthorized = client.post(
                    "/add-document",
                    json={
                        "content": "fastapi",
                        "collection": "docs",
                        "metadata": {"title": "Intro"},
                        "user_id": 7,
                        "agent_id": 2,
                    },
                )
                self.assertEqual(unauthorized.status_code, 401)

                headers = {"x-api-key": "test-key"}
                add = client.post(
                    "/add-document",
                    json={
                        "content": "fastapi",
                        "collection": "docs",
                        "metadata": {"title": "Intro"},
                        "user_id": 7,
                        "agent_id": 2,
                    },
                    headers=headers,
                )
                self.assertEqual(add.status_code, 200)
                self.assertEqual(add.json()["message"], "Document added successfully")

                search = client.get(
                    "/search",
                    params={
                        "query": "fastapi",
                        "collection": "docs",
                        "limit": 3,
                        "user_id": 7,
                        "agent_id": 2,
                    },
                    headers=headers,
                )
                self.assertEqual(search.status_code, 200)
                self.assertEqual(search.json()["summary"], "summary for fastapi")
                self.assertTrue(search.json()["results"])

                chat = client.post(
                    "/chat",
                    json={
                        "query": "fastapi",
                        "collection": "docs",
                        "limit": 3,
                        "user_id": 7,
                        "agent_id": 2,
                    },
                    headers=headers,
                )
                self.assertEqual(chat.status_code, 200)
                self.assertEqual(chat.json()["response"], "mock answer")

                rate_limited = client.get(
                    "/search",
                    params={
                        "query": "fastapi",
                        "collection": "docs",
                        "limit": 3,
                        "user_id": 7,
                        "agent_id": 2,
                    },
                    headers=headers,
                )
                self.assertEqual(rate_limited.status_code, 429)

                metrics = client.get("/metrics")
                self.assertGreaterEqual(metrics.json()["metrics"]["total_requests"], 4)


class McpShellIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self._original_http_context = app.state.shell_context
        self._original_mcp_context = mcp_server.shell_context

    async def asyncTearDown(self):
        await app.router.shutdown()
        app.state.shell_context = self._original_http_context
        mcp_server.shell_context = self._original_mcp_context

    async def test_rag_ask_uses_shared_shell_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            context, _service, _llm = await _build_shell_context(persist_directory=Path(tmpdir))
            app.state.shell_context = context
            mcp_server.shell_context = context

            await app.router.startup()

            async with httpx.AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
                follow_redirects=True,
            ) as http_client:
                add = await http_client.post(
                    "/add-document",
                    json={
                        "content": "fastapi",
                        "collection": "docs",
                        "metadata": {"title": "Intro"},
                        "user_id": 7,
                        "agent_id": 2,
                    },
                    headers={"x-api-key": "test-key"},
                )
                self.assertEqual(add.status_code, 200)

            async with streamablehttp_client(
                "http://testserver/mcp",
                httpx_client_factory=_httpx_client_factory_for_app,
            ) as (read_stream, write_stream, _get_session_id):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    unauthorized = await session.call_tool(
                        "rag_ask",
                        {
                            "query": "fastapi",
                            "collection": "docs",
                        },
                    )
                    self.assertIn("api key required", unauthorized.content[0].text)

                    result = await session.call_tool(
                        "rag_ask",
                        {
                            "query": "fastapi",
                            "collection": "docs",
                            "limit": 3,
                            "threshold": 0.1,
                            "tenant": {"base_collection": "docs", "user_id": 7, "agent_id": 2},
                            "api_key": "test-key",
                        },
                    )
                    text = result.content[0].text
                    self.assertIn("为查询 'fastapi' 找到", text)
                    self.assertIn("fastapi", text)


if __name__ == "__main__":
    unittest.main()
