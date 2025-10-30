"""
MCP 服务器 - 主服务器
=====================================

这是主要的 MCP 服务器，采用模块化架构。
保留了所有现有功能，并进行了更好的组织。
现在支持结构化模型（DocumentModel 和 MetadataModel）。
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from urllib.parse import urlparse

# 添加 src 目录到路径以支持导入
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

# 导入工具
from utils.logger import log, log_mcp_server
from utils.config import Config

# 导入 RAG 核心功能（云端实现）
from rag_core_openai import (
    add_text_to_knowledge_base,
    add_text_to_knowledge_base_enhanced,
    load_document_with_fallbacks,
    get_qa_chain,
    get_vector_store,
    search_with_metadata_filters,
    create_metadata_filter,
    get_document_statistics,
    get_cache_stats,
    print_cache_stats,
    clear_embedding_cache,
    optimize_vector_store,
    get_vector_store_stats,
    reindex_vector_store,
    get_optimal_vector_store_profile,
    load_document_with_elements
)

# 导入结构化模型
try:
    from models import DocumentModel, MetadataModel
    MODELS_AVAILABLE = True
    log_mcp_server("✅ 结构化模型 (DocumentModel, MetadataModel) 可用")
except ImportError as e:
    MODELS_AVAILABLE = False
    log_mcp_server(f"⚠️ 结构化模型不可用: {e}")

# --- 初始化服务器和配置 ---
load_dotenv()
mcp = FastMCP(Config.SERVER_NAME)

# 状态现在包括有关结构化模型的信息
rag_state = {
    "models_available": MODELS_AVAILABLE,
    "structured_processing": MODELS_AVAILABLE,
    "document_models": [],  # 已处理的 DocumentModel 列表
    "metadata_cache": {}    # 每个文档的 MetadataModel 缓存
}

md_converter = None

def warm_up_rag_system():
    """
    预加载 RAG 系统的重型组件，以避免首次调用工具时的延迟和冲突。
    """
    if "warmed_up" in rag_state:
        return
    
    log_mcp_server("正在预热 RAG 系统...")
    log_mcp_server("初始化云端向量存储（OpenAI-only）...")
    
    rag_state["warmed_up"] = True
    log_mcp_server("RAG 系统已预热并准备就绪。")

def ensure_converted_docs_directory():
    """确保存在用于存储转换文档的文件夹。"""
    Config.ensure_directories()
    if not os.path.exists(Config.CONVERTED_DOCS_DIR):
        os.makedirs(Config.CONVERTED_DOCS_DIR)
        log_mcp_server(f"已创建转换文档文件夹: {Config.CONVERTED_DOCS_DIR}")

def save_processed_copy(file_path: str, processed_content: str, processing_method: str = "unstructured") -> str:
    """
    保存处理后的文档副本为 Markdown 格式。

    参数：
        file_path: 原始文件路径
        processed_content: 处理后的内容
        processing_method: 使用的处理方法

    返回：
        保存的 Markdown 文件路径
    """
    ensure_converted_docs_directory()
    
    # 获取原始文件名（无扩展名）
    original_filename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(original_filename)[0]
    
    # 创建包含方法信息的 Markdown 文件名
    md_filename = f"{name_without_ext}_{processing_method}.md"
    md_filepath = os.path.join(Config.CONVERTED_DOCS_DIR, md_filename)
    
    # 保存内容到 Markdown 文件
    try:
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        log_mcp_server(f"已保存处理后的副本: {md_filepath}")
        return md_filepath
    except Exception as e:
        log_mcp_server(f"警告: 无法保存处理后的副本: {e}")
        return ""

def initialize_rag():
    """
    使用核心初始化 RAG 系统的所有组件。
    """
    if "initialized" in rag_state:
        return

    log_mcp_server("通过核心初始化 RAG 系统...")
    
    # 从云端核心获取向量存储和 QA 链
    vector_store = get_vector_store()
    qa_chain = get_qa_chain(vector_store)
    
    rag_state["vector_store"] = vector_store
    rag_state["qa_chain"] = qa_chain
    rag_state["initialized"] = True
    
    # 关于模型状态的信息
    if MODELS_AVAILABLE:
        log_mcp_server("✅ RAG 系统已初始化，支持结构化模型")
        log_mcp_server("🧠 DocumentModel 和 MetadataModel 可用于高级处理")
    else:
        log_mcp_server("⚠️ RAG 系统已初始化，但未启用结构化模型 (使用字典)")
    
    log_mcp_server("RAG 系统初始化成功。")

# --- 初始化自动化 RAG 系统 ---
log_mcp_server("自动初始化 RAG 系统...")
backend = "JSON"
log_mcp_server("RAG 后端: JSON")
initialize_rag()
warm_up_rag_system()
log_mcp_server("RAG 系统已初始化并准备就绪。")

# --- 在初始化 RAG 后配置模块化工具 ---
from tools import configure_rag_state, ALL_TOOLS

# 配置工具模块中的 RAG 状态
configure_rag_state(
    rag_state=rag_state,
    initialize_rag_func=initialize_rag,
    save_processed_copy_func=save_processed_copy
)

# --- Definir las herramientas MCP directamente en el servidor ---
@mcp.tool()
def ask_rag(query: str) -> str:
    """用户想查询已有资料或者需要知识库时调用"""
    from tools.search_tools import ask_rag as ask_rag_logic
    return ask_rag_logic(query)

mcp.ask_rag = ask_rag


# @mcp.tool()
# def get_context(**kwargs) -> str:
#     """用户想查询已有资料或者需要知识库时调用"""
#     try:
#         query = kwargs.get("query", "")
#         from tools.search_tools import get_context_tool
#         return get_context_tool(query, k=5)
#     except Exception as e:
#         log_mcp_server(f"注册工具 get_context 时出错: {e}")
#         return ""

# mcp.get_context = get_context

# --- 启动 MCP RAG 服务器 ---
if __name__ == "__main__":
    log_mcp_server("启动 MCP RAG 服务器...")
    warm_up_rag_system()  # 启动时预热系统
    log_mcp_server("🚀 服务器已启动，运行模式: stdio")
    mcp.ask_rag = ask_rag
    # mcp.get_context = get_context
    mcp.run(transport='stdio')