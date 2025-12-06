"""Main entry point for MCP-RAG service."""

import logging
import asyncio
from pathlib import Path
import uvicorn

from .config import settings
from .http_server import app as http_app

# Setup logging
logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def run_http_server():
    """Run the HTTP server only."""
    logger.info("启动MCP-RAG Streamable HTTP服务器...")

    # 确保数据目录存在
    data_dir = Path(settings.chroma_persist_directory)
    data_dir.mkdir(parents=True, exist_ok=True)

    config = uvicorn.Config(
        http_app,
        host="0.0.0.0",
        port=settings.http_port if hasattr(settings, 'http_port') else 8060,
        log_level="info"
    )
    
    port = settings.http_port if hasattr(settings, 'http_port') else 8060
    print(f"\n访问地址: http://127.0.0.1:{port} (Streamable MCP endpoint: http://127.0.0.1:{port}/mcp)\n")
    
    server = uvicorn.Server(config)
    await server.serve()


def run_http_server_sync():
    """同步包装器 for HTTP server."""
    asyncio.run(run_http_server())


async def main():
    """主应用入口点。"""
    logger.info("启动MCP-RAG服务...")

    try:
        await run_http_server()

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