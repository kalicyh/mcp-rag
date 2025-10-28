# MCP-RAG 项目架构文档

## 项目概述

MCP-RAG 是一个基于模型上下文协议（MCP）的智能知识库系统，提供文档处理、知识问答和向量库管理功能。该项目采用模块化架构，支持多种AI模型（OpenAI、豆包等），并提供Web界面和命令行工具。

## 核心架构

### 整体架构图

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI (cli.py)  │────│ MCP Server      │────│   Web UI        │
│                 │    │ (server.py)     │    │  (web.py)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    工具层 (tools/)                              │
│  ┌─────────────────┬─────────────────┬─────────────────┐       │
│  │ document_tools │ search_tools    │ utility_tools   │       │
│  │ - learn_text   │ - ask_rag       │ - get_stats     │       │
│  │ - learn_doc    │ - ask_filtered  │ - clear_cache   │       │
│  └─────────────────┴─────────────────┴─────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   核心层 (Core Services)                        │
│  ┌─────────────────┬─────────────────┬─────────────────┐       │
│  │ RAG Core       │ Vector Store    │ Models          │       │
│  │ (rag_core_     │ (cloud_openai)  │ (document/      │       │
│  │  openai.py)    │                 │  metadata)      │       │
│  └─────────────────┴─────────────────┴─────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   基础设施层 (Infrastructure)                  │
│  ┌─────────────────┬─────────────────┬─────────────────┐       │
│  │ Config         │ Logger         │ Environment     │       │
│  │ (utils/config) │ (utils/logger) │ (.env)          │       │
│  └─────────────────┴─────────────────┴─────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## 目录结构详解

### 根目录文件

- **`main.py`** - 项目入口点（可能已废弃）
- **`pyproject.toml`** - Python项目配置，定义依赖和脚本入口
- **`README.md`** - 项目说明文档
- **`uv.lock`** - uv包管理器锁文件
- **`uv.toml`** - uv配置
- **`test_mcp_tools.py`** - 工具测试脚本

### 源代码目录 (`src/`)

#### 主模块 (`mcp_rag/`)

- **`__init__.py`** - 包初始化
- **`cli.py`** - 命令行接口
  - 提供 `serve` 和 `web` 两个子命令
  - `serve` 启动MCP服务器（stdio或HTTP模式）
  - `web` 启动Web测试界面
- **`server.py`** - MCP服务器核心
  - 初始化FastMCP服务器
  - 注册MCP工具
  - 管理RAG系统状态
  - **当前状态**: 只注册 `ask_rag` 工具
- **`web.py`** - Web界面 (1232行)
  - Flask应用提供Web界面
  - 工具分类展示（Bento风格布局）
  - 交互式工具测试

#### 工具模块 (`tools/`)

- **`__init__.py`** - 工具模块初始化
  - 导入所有工具函数
  - 提供 `configure_rag_state()` 配置函数
  - 定义 `ALL_TOOLS` 和 `TOOLS_BY_NAME` 列表

- **`document_tools.py`** - 文档处理工具 (255行)
  - `learn_text(text, source_name)` - 添加文本到知识库
  - `learn_document(file_path)` - 处理文档并添加到知识库
  - 支持25+种文档格式（PDF、DOCX、图片等）

- **`search_tools.py`** - 搜索查询工具 (252行)
  - `ask_rag(query)` - 基于知识库回答问题

#### 数据模型 (`models/`)

- **`__init__.py`** - 模型包初始化
- **`document_model.py`** - 文档数据模型 (117行)
  - `DocumentModel` 类：表示已处理文档
  - 包含文件信息、内容、分块信息、结构元素等
- **`metadata_model.py`** - 元数据模型 (212行)
  - `MetadataModel` 类：文档元数据结构
  - 处理方法、结构信息、嵌入信息等

#### 服务层 (`services/`)

- **`__init__.py`** - 服务包初始化
- **`cloud_openai.py`** - OpenAI云服务适配器 (190行)
  - `OpenAIVectorStore` 类：内存向量存储
  - 嵌入API封装
  - 支持持久化到JSON文件

#### 工具库 (`utils/`)

- **`__init__.py`** - 工具库初始化
- **`config.py`** - 配置管理 (150+行)
  - `Config` 类：集中配置管理
  - 路径配置、Unstructured参数、环境变量
  - 支持25+种文件类型的处理配置
- **`logger.py`** - 日志系统
  - Rich增强的控制台输出
  - 不同类型的日志函数

#### 核心文件

- **`rag_core_openai.py`** - RAG核心逻辑 (313行)
  - 云端RAG实现（仅使用OpenAI API）
  - 文本分块、向量存储、QA链
  - 文档加载和处理

## 核心工作流程

### 1. 系统初始化

```
CLI (serve) → server.py → rag_core_openai.py 初始化
    ↓
加载配置 (utils/config.py)
    ↓
初始化向量存储 (services/cloud_openai.py)
    ↓
预热系统 (warm_up_rag_system)
```

### 2. 工具调用流程

```
MCP客户端 → server.py (FastMCP) → 具体工具函数
    ↓
工具函数 → rag_core_openai.py 核心逻辑
    ↓
调用 OpenAI API (嵌入/聊天)
    ↓
返回结果给客户端
```

### 3. 文档处理流程

```
learn_document(file_path)
    ↓
rag_core_openai.load_document_with_fallbacks()
    ↓
Unstructured 处理文档
    ↓
分块文本 (chunk_text)
    ↓
生成嵌入 (embed_texts)
    ↓
存储到向量库 (add_texts)
```

### 4. 问答流程

```
ask_rag(query)
    ↓
嵌入查询 (embed_query)
    ↓
向量搜索 (search)
    ↓
检索相关文档片段
    ↓
OpenAI 聊天完成 (chat.completions.create)
    ↓
返回答案
```

## 配置系统

### 环境变量配置

```bash
# OpenAI 配置
OPENAI_API_KEY=your_key
OPENAI_API_BASE=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_TEMPERATURE=0

# 豆包配置
OPENAI_API_KEY=doubao_key
OPENAI_API_BASE=https://ark.cn-beijing.volces.com/api/v3
OPENAI_MODEL=doubao-1-5-pro-32k-250115
OPENAI_EMBEDDING_MODEL=doubao-embedding-text-240715
```

### 文件路径配置

- **转换文档**: `./data/documents/`
- **向量存储**: `./data/vector_store/`
- **嵌入缓存**: `./embedding_cache/`

## 重构建议

### 1. 架构问题

- **紧耦合**: server.py直接导入所有工具，难以扩展
- **全局状态**: 工具模块使用全局变量传递状态
- **混合职责**: server.py既是服务器又是工具注册器

### 2. 重构方向

#### 插件化工具系统
```
tools/
├── base.py          # 工具基类
├── registry.py      # 工具注册器
├── document/
│   ├── __init__.py
│   └── processors.py
├── search/
│   ├── __init__.py
│   └── engines.py
└── utility/
    ├── __init__.py
    └── managers.py
```

#### 依赖注入
```python
# 替换全局状态
@dataclass
class RAGContext:
    vector_store: VectorStore
    config: Config
    logger: Logger

class Tool:
    def __init__(self, context: RAGContext):
        self.context = context
```

#### 服务层重构
```python
# 抽象服务接口
class VectorStoreService(ABC):
    @abstractmethod
    def add_texts(self, texts: List[str]) -> List[str]:
        pass
    
    @abstractmethod
    def search(self, query: str, k: int) -> List[Dict]:
        pass

# 具体实现
class OpenAIVectorStoreService(VectorStoreService):
    pass

class ChromaVectorStoreService(VectorStoreService):
    pass
```

### 3. 代码质量改进

- **类型注解**: 完善类型提示
- **错误处理**: 统一异常处理机制
- **测试覆盖**: 增加单元测试和集成测试
- **文档**: 完善API文档和代码注释
- **配置管理**: 使用Pydantic进行配置验证

### 4. 性能优化

- **异步处理**: 支持异步工具调用
- **缓存策略**: 改进嵌入缓存机制
- **批处理**: 支持批量文档处理
- **流式响应**: 支持大文档流式处理

### 5. 可扩展性

- **插件系统**: 支持第三方工具插件
- **多后端支持**: 支持多种向量数据库
- **多模型支持**: 更灵活的模型配置
- **API扩展**: 提供REST API接口

## 总结

当前项目实现了基本的MCP RAG功能，但存在架构设计上的问题。建议通过插件化、服务层抽象、依赖注入等重构手段提升代码质量和可维护性。</content>
<parameter name="filePath">c:\Users\kalicyh\Documents\GitHub\mcp-rag\PROJECT_ARCHITECTURE.md