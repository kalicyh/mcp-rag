"""MCP Server implementation for RAG service."""

import logging
from typing import Any, Dict, List, Optional, Sequence

from mcp import Tool, types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import settings
from .database import get_vector_database, Document, SearchResult
from .embedding import get_embedding_model

logger = logging.getLogger(__name__)


class MCPServer:
    """MCP Server for RAG operations."""

    def __init__(self):
        self.server = Server("mcp-rag")
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
                                "description": "搜索查询"
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["raw", "summary"],
                                "description": "检索模式",
                                "default": "raw"
                            },
                            "collection": {
                                "type": "string",
                                "description": "要搜索的集合名称",
                                "default": "default"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "最大结果数量",
                                "default": 5,
                                "minimum": 1,
                                "maximum": 20
                            },
                            "threshold": {
                                "type": "number",
                                "description": "相似度阈值",
                                "default": 0.7,
                                "minimum": 0.0,
                                "maximum": 1.0
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> Sequence[types.TextContent]:
            """调用MCP工具。"""
            if name == "rag_ask":
                try:
                    logger.info(f"开始处理RAG检索请求: {arguments.get('query', 'unknown')}")
                    
                    # 获取组件
                    vector_db = await get_vector_database()
                    embedding_model = await get_embedding_model()

                    # 提取参数
                    query = arguments["query"]
                    mode = arguments.get("mode", "raw")
                    collection = arguments.get("collection", "default")
                    limit = arguments.get("limit", 5)
                    threshold = arguments.get("threshold", 0.7)

                    _user_id = arguments.get("_user_id")
                    _agent_id = arguments.get("_agent_id")
                    if _user_id:
                        logger.info(f"RAG Request from User: {_user_id}, Agent: {_agent_id}")

                    logger.info(f"编码查询: {query}")
                    # 编码查询
                    query_embedding = await embedding_model.encode_single(query)

                    logger.info(f"搜索数据库，集合: {collection}, 限制: {limit}")
                    # 搜索数据库
                    search_results = await vector_db.search(
                        query_embedding=query_embedding,
                        collection_name=collection,
                        limit=limit,
                        threshold=threshold,
                        user_id=_user_id,
                        agent_id=_agent_id
                    )

                    # 格式化结果
                    if not search_results:
                        logger.info("未找到相关文档")
                        return [types.TextContent(
                            type="text",
                            text=f"为查询 '{query}' 未找到相关文档"
                        )]

                    logger.info(f"找到 {len(search_results)} 个相关文档")
                    result_text = f"为查询 '{query}' 找到 {len(search_results)} 个相关文档\n\n"

                    for i, result in enumerate(search_results, 1):
                        result_text += f"[{i}] 相似度: {result.score:.3f}\n"
                        result_text += f"内容: {result.document.content}\n"
                        if result.document.metadata.get("source"):
                            result_text += f"来源: {result.document.metadata['source']}\n"
                        result_text += "\n"

                    # 对于摘要模式，添加摘要生成
                    if mode == "summary":
                        result_text += "\n--- 摘要模式 ---\n"
                        result_text += "摘要生成功能尚未实现。\n"

                    logger.info("RAG检索完成")
                    return [types.TextContent(type="text", text=result_text)]

                except Exception as e:
                    logger.error(f"工具调用失败: {e}")
                    return [types.TextContent(
                        type="text",
                        text=f"检索过程中出错: {str(e)}"
                    )]
            else:
                raise ValueError(f"未知工具: {name}")

    async def start_stdio_server(self):
        """启动MCP stdio服务器。"""
        logger.info("启动MCP-RAG stdio服务器...")
        try:
            # 初始化组件
            logger.info("初始化组件...")
            await get_vector_database()
            await get_embedding_model()
            logger.info("组件初始化完成")

            # 启动stdio服务器
            async with stdio_server() as (read_stream, write_stream):
                initialization_options = self.server.create_initialization_options()
                await self.server.run(read_stream, write_stream, initialization_options)
        except Exception as e:
            logger.error(f"MCP服务器启动失败: {e}")
            raise


# 全局服务器实例
mcp_server = MCPServer()