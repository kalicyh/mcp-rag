"""MCP Server implementation for RAG service."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

from mcp import Tool, types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .contracts import SearchRequest, normalize_tenant
from .shell_factory import (
    ShellContext,
    enforce_guardrails,
    get_default_shell_context,
    resolve_shell_service,
    tenant_subject,
)
from .security import AuthenticationError, AuthorizationError, RateLimitExceededError

logger = logging.getLogger(__name__)


async def get_rag_service(request_adapter):
    """Compatibility wrapper used by tests and shell code."""

    return await resolve_shell_service(request_adapter)


class MCPServer:
    """MCP Server for RAG operations."""

    def __init__(self, shell_context: ShellContext | None = None):
        self.server = Server("mcp-rag")
        self.shell_context = shell_context or get_default_shell_context()
        self._setup_mcp_tools()

    def _setup_mcp_tools(self):
        """Setup MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """列出可用的MCP工具。"""
            return [
                Tool(
                    name="rag_ask",
                    description="向RAG知识库提问查询信息",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "搜索查询",
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["raw", "summary"],
                                "description": "检索模式",
                                "default": "raw",
                            },
                            "collection": {
                                "type": "string",
                                "description": "要搜索的集合名称",
                                "default": "default",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "最大结果数量",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20,
                            },
                            "threshold": {
                                "type": "number",
                                "description": "相似度阈值",
                                "default": 0.7,
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                            "tenant": {
                                "type": "object",
                                "description": "Tenant scope (base_collection, user_id, agent_id)",
                                "properties": {
                                    "base_collection": {"type": "string", "default": "default"},
                                    "user_id": {"type": "integer"},
                                    "agent_id": {"type": "integer"},
                                },
                            },
                            "user_id": {
                                "type": "integer",
                                "description": "Legacy user id parameter",
                            },
                            "agent_id": {
                                "type": "integer",
                                "description": "Legacy agent id parameter",
                            },
                            "_user_id": {
                                "type": "integer",
                                "description": "Legacy hidden user id parameter",
                            },
                            "_agent_id": {
                                "type": "integer",
                                "description": "Legacy hidden agent id parameter",
                            },
                            "api_key": {
                                "type": "string",
                                "description": "Optional API key for stdio clients",
                            },
                        },
                        "required": ["query"],
                    },
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[types.TextContent]:
            """调用MCP工具。"""
            if name == "rag_ask":
                return await self.handle_rag_ask(arguments)
            raise ValueError(f"未知工具: {name}")

    async def handle_rag_ask(self, arguments: Dict[str, Any]) -> Sequence[types.TextContent]:
        """Handle rag_ask tool call with shell-level formatting."""

        try:
            query = str(arguments.get("query", "")).strip()
            if not query:
                return [types.TextContent(type="text", text="检索过程中出错: Missing required parameter: query")]

            mode = str(arguments.get("mode", "raw") or "raw")
            collection = str(arguments.get("collection", "default") or "default")
            limit = int(arguments.get("limit", 5) or 5)
            threshold = float(arguments.get("threshold", 0.7) or 0.7)
            api_key = str(arguments.get("api_key", "") or "").strip() or None
            tenant = normalize_tenant(
                arguments.get("tenant"),
                base_collection=collection,
                user_id=arguments.get("user_id", arguments.get("_user_id")),
                agent_id=arguments.get("agent_id", arguments.get("_agent_id")),
            )

            logger.info("开始处理RAG检索请求: %s", query)
            request = self._request_adapter()
            with self.shell_context.observability.timer("mcp.rag_ask"):
                enforce_guardrails(
                    self.shell_context,
                    tenant=tenant,
                    api_key=api_key,
                    subject=tenant_subject(tenant, fallback=api_key or "mcp"),
                )
                service = await get_rag_service(request)
                response = await service.ask(
                    SearchRequest(
                        query=query,
                        collection=collection,
                        limit=limit,
                        threshold=threshold,
                        mode=mode,
                        tenant=tenant,
                    )
                )

            if not response.results:
                text = f"为查询 '{query}' 未找到相关文档"
                if mode == "summary":
                    text += "\n\n--- 摘要模式 ---\n摘要生成功能暂未启用。"
                return [types.TextContent(type="text", text=text)]

            lines = [f"为查询 '{query}' 找到 {len(response.results)} 个相关文档", ""]
            for index, result in enumerate(response.results, 1):
                lines.append(f"[{index}] 相似度: {result.score:.3f}")
                lines.append(f"内容: {result.content}")
                if result.source:
                    lines.append(f"来源: {result.source}")
                lines.append("")

            if mode == "summary":
                lines.append("--- 摘要模式 ---")
                lines.append(response.summary or "摘要生成功能暂未启用。")

            logger.info("RAG检索完成")
            return [types.TextContent(type="text", text="\n".join(lines).rstrip())]
        except (AuthenticationError, AuthorizationError, RateLimitExceededError) as exc:
            return [types.TextContent(type="text", text=f"检索过程中出错: {str(exc)}")]
        except Exception as e:
            logger.error("工具调用失败: %s", e, exc_info=True)
            return [types.TextContent(type="text", text=f"检索过程中出错: {str(e)}")]

    def _request_adapter(self):
        """Adapt the shell context to the HTTP helper interface."""

        class _Adapter:
            def __init__(self, context: ShellContext):
                self.app = type("_App", (), {"state": type("_State", (), {"shell_context": context})()})()

        return _Adapter(self.shell_context)

    async def start_stdio_server(self):
        """启动MCP stdio服务器。"""
        logger.info("启动MCP-RAG stdio服务器...")
        try:
            logger.info("初始化组件...")
            await get_rag_service(self._request_adapter())
            self.shell_context.bootstrapped = True
            logger.info("组件初始化完成")

            async with stdio_server() as (read_stream, write_stream):
                initialization_options = self.server.create_initialization_options()
                await self.server.run(read_stream, write_stream, initialization_options)
        except Exception as e:
            logger.error("MCP服务器启动失败: %s", e)
            raise


# 全局服务器实例
mcp_server = MCPServer(shell_context=get_default_shell_context())
