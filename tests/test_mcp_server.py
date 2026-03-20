from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from mcp_rag.contracts import SearchResponse, SearchResultView
from mcp_rag.mcp_server import MCPServer


class McpServerFacadeTests(unittest.IsolatedAsyncioTestCase):
    async def test_rag_ask_formats_results_and_summary(self):
        service = type("FakeService", (), {})()
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

        server = MCPServer()
        with patch("mcp_rag.mcp_server.get_rag_service", new=AsyncMock(return_value=service)):
            content = await server.handle_rag_ask(
                {
                    "query": "fastapi",
                    "collection": "docs",
                    "mode": "summary",
                    "limit": 3,
                    "threshold": 0.5,
                    "tenant": {"base_collection": "docs", "user_id": 7, "agent_id": 2},
                }
            )

        self.assertEqual(len(content), 1)
        text = content[0].text
        self.assertIn("为查询 'fastapi' 找到 1 个相关文档", text)
        self.assertIn("相似度: 0.910", text)
        self.assertIn("summary text", text)
        self.assertTrue(service.ask.await_count)

    async def test_rag_ask_accepts_legacy_tenant_fields(self):
        service = type("FakeService", (), {})()
        service.ask = AsyncMock(
            return_value=SearchResponse(
                query="fastapi",
                collection="docs",
                results=[],
            )
        )

        server = MCPServer()
        with patch("mcp_rag.mcp_server.get_rag_service", new=AsyncMock(return_value=service)):
            content = await server.handle_rag_ask(
                {
                    "query": "fastapi",
                    "collection": "docs",
                    "_user_id": 7,
                    "_agent_id": 2,
                }
            )

        self.assertEqual(len(content), 1)
        self.assertIn("未找到相关文档", content[0].text)
        called_request = service.ask.await_args.args[0]
        self.assertEqual(called_request.tenant.user_id, 7)
        self.assertEqual(called_request.tenant.agent_id, 2)


if __name__ == "__main__":
    unittest.main()

