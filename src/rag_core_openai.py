"""
Cloud-only RAG core using OpenAI APIs and an in-memory vector store.

This file replaces langchain/chroma/unstructured with minimal logic:
- Vector store: services.cloud_openai.OpenAIVectorStore
- QA chain: OpenAI chat completion over top-k retrieved chunks
- Document loading: simple text reader fallback
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from utils.logger import log
except Exception:
    def log(msg: str):
        print(msg)

from services.cloud_openai import (
    OpenAIVectorStore,
    ensure_client,
    embed_query,
)
from utils.config import Config


_VECTOR_STORE: Optional[Any] = None
_STORE_PATH = os.path.join(os.path.abspath(Config.VECTOR_STORE_DIR), "cloud_store.json")


def get_vector_store(profile: str = 'auto') -> Any:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        # Ensure configured rag directories exist
        try:
            Config.ensure_directories()
        except Exception:
            # best-effort fallback: ensure parent dir of store path exists
            try:
                os.makedirs(os.path.dirname(_STORE_PATH), exist_ok=True)
            except Exception:
                pass

        _VECTOR_STORE = OpenAIVectorStore()
        # 尝试加载持久化(JSON)
        try:
            loaded = _VECTOR_STORE.load_from_file(os.path.abspath(_STORE_PATH))
            if loaded > 0:
                log(f"核心: 已初始化 OpenAI 向量库并加载 {loaded} 条记录 (cloud-only)")
            else:
                log("核心: 已初始化 OpenAI 向量库")
        except Exception:
            log("核心: 已初始化 OpenAI 向量库")
    return _VECTOR_STORE


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(end - overlap, 0)
    return chunks


def flatten_metadata(metadata: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in (metadata or {}).items():
        key = f"{prefix}{k}" if prefix else k
        # 跳过 None
        if v is None:
            continue
        if isinstance(v, dict):
            flat.update(flatten_metadata(v, f"{key}_"))
        elif isinstance(v, (list, tuple)):
            flat[key] = str(v)
        elif isinstance(v, (str, int, float, bool)):
            flat[key] = v
        else:
            # 其他类型统一转字符串，确保类型受支持
            flat[key] = str(v)
    return flat


def add_text_to_knowledge_base_enhanced(
    text: str,
    vector_store: Any,
    source_metadata: Optional[dict] = None,
    use_semantic_chunking: bool = False,
    structural_elements: Optional[List[Any]] = None,
) -> None:
    if not text or text.isspace():
        log("核心警告: 空文本，忽略")
        return
    # simple chunking
    texts = _chunk_text(text, 1000, 200)
    metadatas: Optional[List[Dict[str, Any]]] = None
    if source_metadata:
        metadatas = []
        for i in range(len(texts)):
            m = flatten_metadata(source_metadata)
            m["chunk_index"] = i
            m["total_chunks"] = len(texts)
            m["chunking_method"] = "standard" if not use_semantic_chunking else "semantic"
            metadatas.append(m)
    vector_store.add_texts(texts, metadatas=metadatas)
    log(f"核心: 已写入 {len(texts)} 个片段到内存向量库")
    # 写入后持久化
    # JSON 后端支持文件持久化
    try:
        if hasattr(vector_store, "save_to_file"):
            vector_store.save_to_file(os.path.abspath(_STORE_PATH))
    except Exception:
        pass


def add_text_to_knowledge_base(text: str, vector_store: Any, source_metadata: dict | None = None) -> None:
    add_text_to_knowledge_base_enhanced(text, vector_store, source_metadata, use_semantic_chunking=False)


def create_metadata_filter(
    file_type: str | None = None,
    processing_method: str | None = None,
    min_tables: int | None = None,
    min_titles: int | None = None,
    source_contains: str | None = None,
) -> dict:
    conditions: List[dict] = []
    if file_type:
        conditions.append({"file_type": file_type})
    if processing_method:
        conditions.append({"processing_method": processing_method})
    if min_tables is not None:
        conditions.append({"structural_info_tables_count": {"$gte": min_tables}})
    if min_titles is not None:
        conditions.append({"structural_info_titles_count": {"$gte": min_titles}})
    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


@dataclass
class SimpleDocument:
    page_content: str
    metadata: Dict[str, Any]


class QAChain:
    def __init__(self, vector_store: OpenAIVectorStore, metadata_filter: Optional[dict] = None):
        self.vs = vector_store
        self.filter = metadata_filter

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        query = inputs.get("query", "")
        top = self.vs.search(query, k=5, filter=self.filter)
        context = "\n\n".join([x["text"][:1200] for x in top])
        client = ensure_client()
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        msgs = [
            {"role": "system", "content": "As an intelligent knowledge assistant, prioritize answering questions accurately and concisely based on the knowledge base. If the knowledge base lacks relevant information, simply state No relevant content in the knowledge base instead of making assumptions. Understand the user's intent (including typos and homophones) and answer naturally, briefly, and quickly in Chinese."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]
        try:
            base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
            if base_url == "https://ark.cn-beijing.volces.com/api/v3":
                # For Ark client, disable thinking
                resp = client.chat.completions.create(
                    model=model, 
                    messages=msgs, 
                    temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0")),
                    thinking={"type": "disabled"}
                )
            else:
                resp = client.chat.completions.create(model=model, messages=msgs, temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0")))
            answer = resp.choices[0].message.content or ""
        except Exception as e:
            answer = f"[OpenAI 调用失败] {e}"
        sources = [SimpleDocument(page_content=r["text"], metadata=r["metadata"]) for r in top]
        return {"result": answer, "source_documents": sources}


def get_qa_chain(vector_store: Any, metadata_filter: dict | None = None) -> QAChain:
    return QAChain(vector_store, metadata_filter)


def get_context_for_query(vector_store: Any, query: str, metadata_filter: dict | None = None, k: int = 5) -> str:
    """
    返回与 QAChain.invoke 中相同格式的 context 字符串，方便作为独立工具被调用。

    Args:
        vector_store: 向量库对象，需支持 search(query, k, filter) 接口。
        query: 查询字符串。
        metadata_filter: 可选的元数据过滤器。
        k: 返回的 top-k 数量。

    Returns:
        拼接后的 context 字符串。
    """
    try:
        top = vector_store.search(query, k=k, filter=metadata_filter)
        context = "\n\n".join([x.get("text", "")[:1200] for x in top])
    except Exception:
        # 在任何异常情况下返回空字符串，调用方可识别为空并记录日志
        context = ""
    return context


def search_with_metadata_filters(vector_store: Any, query: str, metadata_filter: dict | None = None, k: int = 5) -> List[SimpleDocument]:
    results = vector_store.search(query, k=k, filter=metadata_filter)
    return [SimpleDocument(page_content=r["text"], metadata=r["metadata"]) for r in results]


def get_document_statistics(vector_store: Any | None = None) -> dict:
    vs = vector_store or get_vector_store()
    # 统一通过 get() 获取全部数据（JSON 内存）
    data = vs.get() if hasattr(vs, "get") else {"documents": [], "metadatas": []}
    docs = data.get("documents", []) or []
    metas = data.get("metadatas", []) or []
    total = len(metas) if metas else len(docs)
    file_types: Dict[str, int] = {}
    methods: Dict[str, int] = {}
    structural = {
        "documents_with_tables": 0,
        "documents_with_lists": 0,
        "documents_with_titles": 0,
        "avg_tables_per_doc": 0,
        "avg_titles_per_doc": 0,
        "avg_lists_per_doc": 0,
    }
    sum_tables = sum_titles = sum_lists = 0
    for m in metas:
        ft = m.get("file_type") or "unknown"
        file_types[ft] = file_types.get(ft, 0) + 1
        pm = m.get("processing_method") or "unknown"
        methods[pm] = methods.get(pm, 0) + 1
        t = int(m.get("structural_info_tables_count", 0) or 0)
        ti = int(m.get("structural_info_titles_count", 0) or 0)
        l = int(m.get("structural_info_lists_count", 0) or 0)
        if t > 0:
            structural["documents_with_tables"] += 1
        if ti > 0:
            structural["documents_with_titles"] += 1
        if l > 0:
            structural["documents_with_lists"] += 1
        sum_tables += t
        sum_titles += ti
        sum_lists += l
    if total > 0:
        structural["avg_tables_per_doc"] = sum_tables / total
        structural["avg_titles_per_doc"] = sum_titles / total
        structural["avg_lists_per_doc"] = sum_lists / total
    return {
        "total_documents": total,
        "file_types": file_types,
        "processing_methods": methods,
        "structural_stats": structural,
    }


def get_vector_store_stats(vector_store: Any | None = None) -> dict:
    vs = vector_store or get_vector_store()
    try:
        dim = len(embed_query("__probe__"))
    except Exception:
        dim = "unknown"
    data = vs.get() if hasattr(vs, "get") else {"documents": [], "metadatas": []}
    metas = data.get("metadatas", []) or []
    return {
        "total_documents": len(metas),
        "file_types": {},
        "processing_methods": {},
        "collection_name": "openai_in_memory",
        "embedding_dimension": str(dim),
    }


def optimize_vector_store(vector_store: Any | None = None) -> dict:
    return {"status": "success", "message": "no-op for in-memory store"}


def reindex_vector_store(vector_store: Any | None = None, profile: str = 'auto') -> dict:
    return {"status": "success", "message": "no-op for in-memory store", "profile": profile}


def load_document_with_fallbacks(file_path: str) -> tuple[str, dict]:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        log(f"核心警告: 读取文件失败: {e}")
        content = ""
    metadata = {
        "source": os.path.basename(file_path),
        "file_path": file_path,
        "file_type": os.path.splitext(file_path)[1].lower(),
        "processed_date": datetime.now().isoformat(),
        "processing_method": "simple_loader",
        "structural_info": {
            "total_elements": 1,
            "titles_count": 0,
            "tables_count": 0,
            "lists_count": 0,
            "narrative_blocks": 1,
            "other_elements": 0,
            "total_text_length": len(content),
            "avg_element_length": len(content),
        },
    }
    return content, metadata


def load_document_with_elements(file_path: str) -> tuple[str, dict, List[Any]]:
    content, metadata = load_document_with_fallbacks(file_path)
    return content, metadata, []


# Cache stubs

def get_cache_stats() -> Dict[str, Any]:
    return {
        "total_requests": 0,
        "memory_hits": 0,
        "disk_hits": 0,
        "misses": 0,
        "memory_hit_rate": "0%",
        "disk_hit_rate": "0%",
        "overall_hit_rate": "0%",
        "memory_cache_size": 0,
        "max_memory_size": 0,
        "cache_directory": "",
    }


def print_cache_stats() -> None:
    log("缓存统计: 不适用（cloud-only 最简实现）")


def clear_embedding_cache() -> None:
    log("已清空缓存（无实际缓存，cloud-only 最简实现）")


def get_optimal_vector_store_profile() -> str:
    return "small"
