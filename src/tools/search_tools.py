"""
MCP 搜索工具
===========

此模块包含与知识库搜索和查询相关的工具。
从 rag_server.py 迁移而来，用于模块化架构。

注意：这些函数被设计为在主服务器中使用 @mcp.tool() 装饰器。
"""

from rag_core_openai import (
    get_qa_chain,
)
from utils.logger import log

# 导入结构化模型
try:
    from models import MetadataModel
except ImportError as e:
    log(f"警告：无法导入结构化模型：{e}")
    MetadataModel = None

# 必须在服务器中可用的全局变量
rag_state = {}
initialize_rag_func = None

def set_rag_state(state):
    """设置全局 RAG 状态。"""
    global rag_state
    rag_state = state

def set_initialize_rag_func(func):
    """设置 RAG 初始化函数。"""
    global initialize_rag_func
    initialize_rag_func = func

def initialize_rag():
    """初始化 RAG 系统。"""
    if initialize_rag_func:
        initialize_rag_func()
    elif "initialized" in rag_state:
        return
    # 此函数必须在主服务器中实现
    pass

def process_document_metadata(metadata: dict) -> dict:
    """
    使用 MetadataModel（如果可用）处理文档元数据。
    
    参数：
        metadata: 文档元数据字典
        
    返回：
        包含已处理文档信息的字典
    """
    if not metadata:
        return {"source": "未知来源"}
    
    # 如果 MetadataModel 可用，尝试创建结构化模型
    if MetadataModel is not None:
        try:
            metadata_model = MetadataModel.from_dict(metadata)
            return {
                "source": metadata_model.source,
                "file_path": metadata_model.file_path,
                "file_type": metadata_model.file_type,
                "processing_method": metadata_model.processing_method,
                "structural_info": metadata_model.structural_info,
                "titles_count": metadata_model.titles_count,
                "tables_count": metadata_model.tables_count,
                "lists_count": metadata_model.lists_count,
                "total_elements": metadata_model.total_elements,
                "is_rich_content": metadata_model.is_rich_content(),
                "chunking_method": metadata_model.chunking_method,
                "avg_chunk_size": metadata_model.avg_chunk_size
            }
        except Exception as e:
            log(f"MCP服务器警告：使用 MetadataModel 处理元数据时出错：{e}")
    
    # 回退到直接字典处理
    return {
        "source": metadata.get("source", "未知来源"),
        "file_path": metadata.get("file_path"),
        "file_type": metadata.get("file_type"),
        "processing_method": metadata.get("processing_method"),
        "structural_info": metadata.get("structural_info", {}),
        "titles_count": metadata.get("structural_titles_count", 0),
        "tables_count": metadata.get("structural_tables_count", 0),
        "lists_count": metadata.get("structural_lists_count", 0),
        "total_elements": metadata.get("structural_total_elements", 0),
        "is_rich_content": False,  # 没有模型无法确定
        "chunking_method": metadata.get("chunking_method", "未知"),
        "avg_chunk_size": metadata.get("avg_chunk_size", 0)
    }


def extract_brief_answer(full_text: str) -> str:
    """
    从增强回答文本中提取简洁回答（去掉前缀、来源和建议部分）。
    返回去掉杂项后的纯文本（如果无法提取则返回原文的简短形式或空字符串）。
    """
    if not full_text:
        return ""

    text = full_text.strip()

    # 常见前缀
    prefixes = ["🤖 回答：", "🔍 回答（已应用过滤器）：", "🔍 回答：", "回答："]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].lstrip('\n ').lstrip()
            break

    # 截断到第一个来源或建议标记
    for marker in ["📚 使用的信息来源：", "📋 应用的过滤器：", "💡 建议：", "⚠️ 注意："]:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx].rstrip()
            break

    return text.strip()

def ask_rag(query: str) -> str:
    """
    向 RAG 知识库提问并基于存储的信息返回答案。
    当您想从之前学习的知识库中获取信息时使用此功能。
    
    使用场景示例：
    - 询问特定主题或概念
    - 请求解释或定义
    - 从处理过的文档中寻求信息
    - 基于学习的文本或文档获取答案
    
    系统将搜索所有存储的信息并提供最相关的答案。

    参数：
        query: 向知识库提出的问题或查询。
    """
    log(f"MCP服务器：正在处理问题：{query}")
    initialize_rag()
    
    try:
        # 使用标准 QA 链（无过滤器）
        qa_chain = get_qa_chain(rag_state["vector_store"])
        response = qa_chain.invoke({"query": query})
        
        answer = response.get("result", "")
        source_documents = response.get("source_documents", [])

        # 优先返回简洁的回答文本（去掉来源与建议），否则退回到完整回答
        concise = extract_brief_answer(response.get("result", ""))
        if concise:
            log(f"MCP服务器：成功生成简洁回答，使用了 {len(source_documents)} 个来源")
            return concise
        # concise 为空时，返回原始 answer（可能包含更多上下文或模型信息）
        log(f"MCP服务器：未提取到简洁回答，返回完整回答（长度 {len(answer)}）")
        return answer
        
    except Exception as e:
        log(f"MCP服务器：处理问题时出错：{e}")
        return f"❌ 处理问题时出错： {e}" 

def get_context_tool(query: str, k: int = 5) -> str:
    """
    MCP tool: 返回与 QAChain.invoke 中相同的 context 内容（仅 context 字符串），便于在调试或外部流程中复用。

    使用全局 rag_state 中的 vector_store。
    """
    log(f"MCP服务器：获取 context，query={query}, k={k}")
    initialize_rag()
    try:
        vs = rag_state.get("vector_store")
        if not vs:
            raise RuntimeError("vector_store 未初始化")
        from rag_core_openai import get_context_for_query
    
        ctx = get_context_for_query(vs, query, metadata_filter=None, k=k)
        return ctx
    except Exception as e:
        log(f"MCP服务器：获取 context 时出错：{e}")
        return ""