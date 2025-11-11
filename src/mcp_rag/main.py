"""Main entry point for MCP-RAG service."""

import logging
import asyncio
from pathlib import Path
import uvicorn

from .config import settings
from .mcp_server import mcp_server
from .http_server import app as http_app

# Setup logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def run_stdio_server():
    """Run the MCP stdio server only."""
    logger.info("启动MCP stdio服务器...")
    await mcp_server.start_stdio_server()


async def run_http_server():
    """Run the HTTP server only."""
    logger.info("启动HTTP服务器...")

    # 确保数据目录存在
    data_dir = Path(settings.chroma_persist_directory)
    data_dir.mkdir(parents=True, exist_ok=True)

    config = uvicorn.Config(
        http_app,
        host="0.0.0.0",
        port=settings.http_port if hasattr(settings, 'http_port') else 8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()


def run_stdio_server_sync():
    """同步包装器 for stdio server."""
    asyncio.run(run_stdio_server())


def run_http_server_sync():
    """同步包装器 for HTTP server."""
    asyncio.run(run_http_server())


async def main():
    """主应用入口点。"""
    logger.info("启动MCP-RAG服务...")

    try:
        # 确保数据目录存在
        data_dir = Path(settings.chroma_persist_directory)
        data_dir.mkdir(parents=True, exist_ok=True)

        # 同时运行stdio和HTTP服务器
        stdio_task = asyncio.create_task(run_stdio_server())
        http_task = asyncio.create_task(run_http_server())

        # 等待任一任务完成（通常是stdio，因为它是阻塞的）
        done, pending = await asyncio.wait(
            [stdio_task, http_task],
            return_when=asyncio.FIRST_COMPLETED
        )

        # 取消待处理的任务
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except KeyboardInterrupt:
        logger.info("正在关闭MCP-RAG服务...")
    except Exception as e:
        logger.error(f"启动MCP-RAG服务失败: {e}")
        raise


def run_server():
    """运行MCP-RAG服务器（同步包装器）。"""
    asyncio.run(main())


if __name__ == "__main__":
    run_server()