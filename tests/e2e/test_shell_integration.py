from __future__ import annotations

import tempfile
import unittest

from fastapi.testclient import TestClient

from mcp_rag.config import Settings
from mcp_rag.core.indexing import ChromaVectorStore, DocumentProcessor
from mcp_rag.http_server import app
from mcp_rag.mcp_server import MCPServer
from mcp_rag.observability import ObservabilityCollector
from mcp_rag.security import SecurityPolicy, TokenBucketRateLimiter
from mcp_rag.services import ServiceRuntime
from mcp_rag.shell_factory import create_shell_context


class _FakeEmbeddingModel:
    def __init__(self) -> None:
        self.single_calls: list[str] = []

    async def initialize(self) -> None:
        return None

    async def encode(self, texts):
        return [self._vectorize(text) for text in texts]

    async def encode_single(self, text: str):
        self.single_calls.append(text)
        return self._vectorize(text)

    def _vectorize(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(lowered.count("fastapi")) + float(lowered.count("api")) * 0.1,
            float(lowered.count("python")) + 1.0,
            float(lowered.count("starlette")) + float(lowered.count("pydantic")),
            float(len(lowered.split()) or 1),
        ]


class _FakeLLM:
    async def generate(self, prompt: str, **kwargs) -> str:
        if "FastAPI" in prompt:
            return "FastAPI is an ASGI framework built on Starlette and Pydantic."
        return "mock answer"

    async def summarize(self, content: str, query: str) -> str:
        return f"summary for {query}"


class ShellIntegrationE2ETests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_context = app.state.shell_context

        settings = Settings().model_copy(
            update={
                "chroma_persist_directory": self.temp_dir.name,
                "enable_cache": True,
                "enable_llm_summary": True,
                "max_retrieval_results": 5,
            }
        )
        embedding_model = _FakeEmbeddingModel()
        self.embedding_model = embedding_model
        vector_store = ChromaVectorStore(
            persist_directory=self.temp_dir.name,
            embedding_model=embedding_model,
        )
        runtime = ServiceRuntime(
            settings_obj=settings,
            document_processor=DocumentProcessor(),
            embedding_model=embedding_model,
            vector_store=vector_store,
            llm_model=_FakeLLM(),
        )
        self.context = create_shell_context(
            settings_obj=settings,
            runtime=runtime,
            observability=ObservabilityCollector(),
            security_policy=SecurityPolicy(enabled=True, allow_anonymous=False, api_keys=["secret"]),
            rate_limiter=TokenBucketRateLimiter(limit=50, window_seconds=60),
        )
        self.context.bootstrapped = True
        app.state.shell_context = self.context
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        app.state.shell_context = self.original_context
        self.temp_dir.cleanup()

    async def test_http_and_mcp_share_real_chroma_backing_store(self) -> None:
        upload = self.client.post(
            "/upload-files",
            headers={"x-api-key": "secret"},
            data={"collection": "docs"},
            files=[
                (
                    "files",
                    (
                        "fastapi.txt",
                        b"FastAPI is a Python web framework built on Starlette and Pydantic.",
                        "text/plain",
                    ),
                )
            ],
        )
        self.assertEqual(upload.status_code, 200)
        self.assertEqual(upload.json()["successful"], 1)

        search = self.client.get(
            "/search",
            headers={"x-api-key": "secret"},
            params={"query": "FastAPI", "collection": "docs", "limit": 3},
        )
        self.assertEqual(search.status_code, 200)
        search_payload = search.json()
        self.assertEqual(len(search_payload["results"]), 1)
        self.assertEqual(search_payload["results"][0]["filename"], "fastapi.txt")
        self.assertEqual(search_payload["summary"], "summary for FastAPI")
        self.assertEqual(self.context.runtime.get_retrieval_cache().snapshot()["entries"], 1)

        second_search = self.client.get(
            "/search",
            headers={"x-api-key": "secret"},
            params={"query": "FastAPI", "collection": "docs", "limit": 3},
        )
        self.assertEqual(second_search.status_code, 200)
        self.assertEqual(len(self.embedding_model.single_calls), 1)

        chat = self.client.post(
            "/chat",
            headers={"x-api-key": "secret"},
            json={"query": "What is FastAPI?", "collection": "docs", "limit": 3},
        )
        self.assertEqual(chat.status_code, 200)
        self.assertIn("Starlette", chat.json()["response"])

        ready = self.client.get("/ready")
        self.assertEqual(ready.status_code, 200)
        self.assertTrue(ready.json()["ready"])

        server = MCPServer(shell_context=self.context)
        mcp_result = await server.handle_rag_ask(
            {"query": "FastAPI", "collection": "docs", "mode": "summary", "api_key": "secret"}
        )
        self.assertEqual(len(mcp_result), 1)
        self.assertIn("找到 1 个相关文档", mcp_result[0].text)
        self.assertIn("summary for FastAPI", mcp_result[0].text)

        delete_file = self.client.request(
            "DELETE",
            "/delete-file",
            headers={"x-api-key": "secret"},
            json={"filename": "fastapi.txt", "collection": "docs"},
        )
        self.assertEqual(delete_file.status_code, 200)
        self.assertEqual(self.context.runtime.get_retrieval_cache().snapshot()["entries"], 0)

        after_delete = self.client.get(
            "/search",
            headers={"x-api-key": "secret"},
            params={"query": "FastAPI", "collection": "docs", "limit": 3},
        )
        self.assertEqual(after_delete.status_code, 200)
        self.assertEqual(after_delete.json()["results"], [])
        self.assertEqual(len(self.embedding_model.single_calls), 4)

        metrics = self.client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        metrics_payload = metrics.json()["metrics"]
        self.assertGreaterEqual(metrics_payload["total_requests"], 6)
        self.assertGreaterEqual(metrics_payload["operations"]["upload_files"]["count"], 1)
        self.assertGreaterEqual(metrics_payload["operations"]["search"]["count"], 3)
        self.assertGreaterEqual(metrics_payload["operations"]["chat"]["count"], 1)
        self.assertGreaterEqual(self.context.observability.snapshot().operations["mcp.rag_ask"].count, 1)
