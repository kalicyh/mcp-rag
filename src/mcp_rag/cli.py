"""CLI interface for MCP-RAG service."""

import typer
from pathlib import Path

from .main import run_http_server_sync
from .config import settings

app = typer.Typer()


@app.command()
def serve():
    """启动每个客户端都可通过 Streamable HTTP 访问的 MCP-RAG 服务。"""
    typer.echo("启动MCP-RAG Streamable HTTP服务器...")
    run_http_server_sync()


@app.command()
def web():
    """启动 HTTP 管理界面和 Streamable HTTP 端点。"""
    typer.echo("启动MCP-RAG HTTP服务器...")
    run_http_server_sync()


@app.command()
def init(
    data_dir: str = typer.Option("./data", "--data-dir", help="数据目录")
):
    """初始化MCP-RAG服务。"""
    data_path = Path(data_dir)
    chroma_dir = data_path / "chroma"

    # 创建目录
    data_path.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"在 {data_path} 初始化了MCP-RAG数据目录")


@app.callback()
def callback():
    """MCP-RAG: 基于MCP协议的低延迟RAG服务。"""
    pass


def main():
    """CLI入口点。"""
    app()


if __name__ == "__main__":
    main()
