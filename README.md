# MCP-RAG: Low-Latency RAG Service

基于 MCP (Model Context Protocol) 协议的低延迟 RAG (Retrieval-Augmented Generation) 服务架构。

## 特性

- **极低延迟** (<100ms) 本地知识检索
- **双模式支持**: Raw 模式 (直接检索) 和 Summary 模式 (检索+摘要)
- **LLM 总结功能**: 支持 Doubao、Ollama 等 LLM 提供商进行智能摘要
- **模块化架构**: MCP Server 作为统一知识接口层
- **异步优化**: 异步调用与模型预热机制
- **可扩展设计**: 预留 reranker 与缓存模块接口

## 技术栈

- **后端框架**: FastAPI
- **向量数据库**: ChromaDB (本地部署)
- **嵌入模型**: Doubao 嵌入 API (默认), 本地模型可选 (m3e-small / e5-small via sentence-transformers)
- **LLM 模型**: Doubao API, Ollama (本地部署)
- **协议**: MCP (Model Context Protocol)
- **包管理**: uv (现代化 Python 包管理器)

## 快速开始

### 1. 环境要求

- Python >= 3.13
- uv 包管理器

### 2. 安装依赖

```bash
# 基础安装 (仅云端API)
uv sync

# 如果需要使用本地embedding模型 (m3e-small, e5-small)
uv sync --extra local-embeddings
```

### 3. 启动服务

```bash
uv run mcp-rag serve
```
> 首次启动会报错（懒得改）  
> 该命令同时启动 Streamable HTTP MCP 端点和管理界面，后续可以直接访问 HTTP 页面完成配置、上传与查询。  

- **访问配置页面**：`http://localhost:8060/config-page`  
- **访问资料管理页面**：`http://localhost:8060/documents-page`  
- **访问 Swagger API 文档**：`http://localhost:8060/docs`

### 4. 配置管理

MCP-RAG 现在使用 JSON 文件进行持久化配置管理

`data\config.json` 文件存储配置信息，支持通过 Web 界面进行修改和保存。

默认配置示例：
```JSON
{
  "host": "0.0.0.0",
  "port": 8060,
  "http_port": 8060,
  "debug": false,
  "vector_db_type": "chroma",
  "chroma_persist_directory": "./data/chroma",
  "qdrant_url": "http://localhost:6333",
  "embedding_provider": "zhipu",
  "embedding_device": "cpu",
  "embedding_cache_dir": null,
  "provider_configs": {
    "doubao": {
      "base_url": "https://ark.cn-beijing.volces.com/api/v3",
      "model": "doubao-embedding-text-240715",
      "api_key": null
    },
    "zhipu": {
      "base_url": "https://open.bigmodel.cn/api/paas/v4",
      "model": "embedding-3",
      "api_key": null
    }
  },
  "llm_provider": "doubao",
  "llm_model": "doubao-seed-1.6-250615",
  "llm_base_url": "https://ark.cn-beijing.volces.com/api/v3",
  "llm_api_key": null,
  "enable_llm_summary": false,
  "enable_thinking": true,
  "max_retrieval_results": 5,
  "similarity_threshold": 0.7,
  "enable_reranker": false,
  "enable_cache": false
}
```

> 注意：
> 
> - 仅测试豆包与智谱的向量模型，其他模型未测试
> - 豆包的向量模型好像要下线了，不推荐使用豆包的向量模型

### MCP 服务器配置

[小智go服务端](https://github.com/AnimeAIChat/xiaozhi-server-go)能通过 MCP 协议与 MCP-RAG 进行交互。以下是一个示例配置：
```JSON
{
  "mcpServers": {
    "RAG": {
      "url": "http://127.0.0.1:8060/mcp"
    }
  }
}
```

### 5. 使用 MCP 工具

```json
{
  "name": "rag_ask",
  "arguments": {
    "query": "查询内容",
    "mode": "raw",
    "limit": 5
  }
}
```

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！