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
- **嵌入模型**: m3e-small / e5-small (Sentence Transformers), Doubao 嵌入 API
- **LLM 模型**: Doubao API, Ollama (本地部署)
- **协议**: MCP (Model Context Protocol)
- **包管理**: uv (现代化 Python 包管理器)

## 快速开始

### 1. 环境要求

- Python >= 3.13
- uv 包管理器

### 2. 安装依赖

```bash
uv sync
```

### 3. 初始化服务

```bash
# 初始化数据目录
uv run mcp-rag init

# 或者手动创建
mkdir -p data/chroma
```

### 4. 配置环境变量

**注意：MCP-RAG 现在使用 JSON 文件进行配置，不再需要 `.env` 文件。**

配置将自动保存在 `./data/config.json` 中，可以通过 Web 界面或 API 修改。

### 5. 启动服务

```bash
# 使用 CLI 启动 (推荐)
uv run mcp-rag serve
```

服务启动后，可以：

- **访问配置页面**：`http://localhost:8000/config-page`
- **访问资料管理页面**：`http://localhost:8000/documents-page`
- **使用 HTTP API**：`http://localhost:8000/docs` (Swagger UI)
- **MCP stdio**：自动启动，等待客户端连接

### 6. 配置管理

打开 `http://localhost:8000/config-page` 可以：

- 查看当前所有配置
- 修改服务器、数据库、模型等设置
- 保存配置到 `./data/config.json`
- 重置为默认配置

### 7. 添加知识库资料

通过 HTTP API 或资料管理页面添加文档：

```bash
curl -X POST http://localhost:8000/add-document \\
  -H "Content-Type: application/json" \\
  -d '{
    "content": "您的文档内容...",
    "collection": "default",
    "metadata": {"source": "manual"}
  }'
```

### 7. 资料管理功能

打开 `http://localhost:8000/documents-page` 可以访问完整的资料管理界面：

- **资料上传**：支持文件上传和文本直接输入
- **资料查询**：搜索知识库内容，支持LLM智能总结
- **知识库对话**：与知识库进行自然语言对话测试

#### 资料上传功能

- **文件上传**：支持 TXT、MD、PDF、DOCX 格式
- **批量处理**：一次上传多个文件
- **文本输入**：直接输入文本内容添加到知识库
- **集合管理**：选择不同的文档集合进行组织

#### 资料查询功能

- **关键词搜索**：基于向量相似度的智能搜索
- **LLM总结**：启用时提供基于查询的智能摘要
- **结果展示**：显示相似度分数和文档内容预览

#### 知识库对话功能

- **自然对话**：与知识库进行问答对话
- **实时响应**：基于检索结果生成回答
- **参考来源**：显示回答所基于的文档片段
- **对话历史**：保持对话上下文

- **上传文件**：支持 TXT、MD、PDF、DOCX 格式
- **批量处理**：一次上传多个文件
- **实时预览**：查看处理结果和内容预览
- **资料查询**：在知识库中搜索相关内容
- **集合管理**：选择不同的文档集合

#### 支持的文件格式

- **文本文件** (.txt, .md): 直接读取文本内容
- **PDF 文件** (.pdf): 提取文本内容 (需要 PyPDF2)
- **Word 文档** (.docx): 提取段落内容 (需要 python-docx)

#### 文件上传 API

```bash
curl -X POST http://localhost:8000/upload-files \\
  -F "files=@document.pdf" \\
  -F "files=@notes.txt" \\
  -F "collection=default"
```

## API 使用

### HTTP API

服务提供 HTTP REST API 用于配置和文档管理：

#### 获取配置
```http
GET /config
```

#### 更新配置
```http
POST /config
Content-Type: application/json

{
  "key": "embedding_model",
  "value": "e5-small"
}
```

#### 添加文档
```http
POST /add-document
Content-Type: application/json

{
  "content": "文档内容...",
  "collection": "default",
  "metadata": {"source": "example"}
}
```

#### 列出集合
```http
GET /collections
```

#### 搜索文档
```http
GET /search?query=查询内容&collection=default&limit=5
```

**LLM 总结模式**: 当启用LLM总结功能时，搜索结果将包含 `summary` 字段，提供基于查询的智能摘要。

**响应示例**:
```json
{
  "query": "什么是机器学习",
  "collection": "default",
  "summary": "机器学习是人工智能的一个重要分支，通过算法让计算机从数据中学习，而不需要显式编程。它包括监督学习、无监督学习和强化学习等多种类型。",
  "results": [
    {
      "content": "机器学习是人工智能的一个重要分支...",
      "score": 0.878,
      "metadata": {"source": "test"}
    }
  ]
}
```

#### 文件上传
```http
POST /upload-files
Content-Type: multipart/form-data

files: [file1, file2, ...]
collection: default
```

#### 文本输入添加文档
```http
POST /add-document
Content-Type: application/json

{
  "content": "您的文档内容...",
  "collection": "default",
  "metadata": {"title": "文档标题", "source": "manual"}
}
```

#### 知识库对话
```http
POST /chat
Content-Type: application/json

{
  "query": "您的问题",
  "collection": "default"
}
```

### MCP 工具调用

服务同时支持 MCP 协议工具调用：

```json
{
  "name": "rag_retrieve",
  "arguments": {
    "query": "查询内容",
    "mode": "raw",
    "limit": 5
  }
}
```

## 测试

运行内置测试：

```bash
# HTTP API 测试
uv run python test_http.py

# 文件上传功能测试
uv run python test_upload.py

# LLM 功能测试
uv run python test_llm.py

# 搜索功能测试
uv run python test_search.py

# 新功能测试 (文本输入和对话)
uv run python test_new_features.py

# 配置页面演示 (自动打开浏览器)
uv run python demo_config.py

# 文件上传功能演示 (自动打开浏览器)
uv run python demo_upload.py
```

## 项目结构

```
mcp-rag/
├── src/mcp_rag/
│   ├── __init__.py
│   ├── config.py          # 配置管理 (JSON持久化)
│   ├── database.py        # 向量数据库接口
│   ├── embedding.py       # 嵌入模型管理
│   ├── llm.py            # LLM 模型管理 (新增)
│   ├── document_processor.py  # 文档处理和文件上传
│   ├── mcp_server.py     # MCP 服务器实现
│   ├── http_server.py    # HTTP API 服务器
│   ├── static/            # 静态文件目录
│   ├── main.py           # 主入口
│   └── cli.py            # CLI 接口
├── data/                 # 数据目录
│   ├── chroma/          # ChromaDB 数据
│   └── config.json      # 配置文件
├── pyproject.toml       # 项目配置
├── uv.toml             # uv 配置
├── test_http.py        # HTTP API 测试脚本
├── test_upload.py      # 文件上传测试脚本
├── test_llm.py         # LLM 功能测试脚本 (新增)
├── test_search.py      # 搜索功能测试脚本 (新增)
├── test_new_features.py # 新功能测试脚本 (新增)
├── .env.example        # 环境变量示例 (已弃用)
└── README.md           # 本文档
```

## 配置管理

MCP-RAG 现在使用 JSON 文件进行持久化配置管理，不再依赖 `.env` 文件。

### Web 配置界面

访问 `http://localhost:8000/config-page` 可以打开配置管理界面，包括：

- **服务器设置**：主机地址、HTTP端口、调试模式
- **向量数据库设置**：数据库类型、存储目录
- **嵌入模型设置**：模型选择、设备配置、API密钥
- **LLM 设置**：提供商选择、模型配置、API设置、总结功能开关
- **RAG 设置**：检索参数、功能开关

#### LLM 配置选项

- **LLM 提供商**: 选择 `doubao` (云端API) 或 `ollama` (本地部署)
- **模型名称**: 指定使用的LLM模型
- **API 基础地址**: LLM服务的API地址
- **API 密钥**: Doubao等服务的认证密钥
- **启用LLM总结**: 开启/关闭LLM智能摘要功能
- **启用深度思考**: 控制LLM的推理深度 (Doubao专用)

### 配置 API

#### 获取配置
```http
GET /config
```

#### 批量更新配置
```http
POST /config/bulk
Content-Type: application/json

{
  "updates": {
    "embedding_model": "e5-small",
    "http_port": 8080,
    "debug": true
  }
}
```

#### 重置配置
```http
POST /config/reset
```

### 配置存储

配置保存在 `./data/config.json` 文件中，支持热更新和持久化。

## 开发计划

详见 [plan.md](./plan.md)

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！