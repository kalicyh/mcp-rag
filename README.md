# MCP-RAG

> ✨100% 由 AI 编写

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/3103638f-38cc-47fe-8314-97a08c7c73e9" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/5f36a470-fd9d-42cd-bdd0-24f3bb0efa1e" width="100%"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/e8cccdaa-e6c3-4c5c-a4f9-31bf11f98b6e" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/d387d5f6-9ca2-455a-a63c-091736850da6" width="100%"></td>
  </tr>
</table>

面向 AI 客户端的服务优先 RAG 服务，当前以 FastAPI HTTP 服务和 Streamable HTTP MCP 端点为主。

代码当前提供的是一个统一后端壳层：
- FastAPI HTTP 服务
- Streamable HTTP MCP
- 共享运行时、配置热更新、鉴权、限流、配额、观测
- 基于知识库注册表的检索与文档管理

## 当前能力

- 文档导入：支持直接添加文本，以及上传 `txt`、`md`、`pdf`、`docx`
- 检索：向量检索 + 关键词检索融合
- 问答：`/search`、`/chat`、MCP `rag_ask`
- 多知识库：支持单知识库和 `kb_ids` 多知识库聚合检索/对话
- 知识库作用域：`public` 和 `agent_private`
- 租户上下文：`base_collection + user_id + agent_id`
- 运行时治理：API key、内存限流、上传/索引配额、request-level retrieval cache
- provider 治理：provider budget、熔断、fallback
- 可观测性：`/health`、`/ready`、`/metrics`
- 前端：内置单页管理面板 `/app`

## 架构

主链路：

```text
HTTP / MCP
  -> app_factory.py
  -> http_server.py / mcp_server.py
  -> context.py
  -> service_facade.py
  -> services/
       - runtime.py
       - indexing_service.py
       - retrieval_service.py
       - chat_service.py
  -> knowledge_bases.py
  -> core/indexing/
  -> retrieval/
```

关键文件：
- `src/mcp_rag/cli.py`: CLI 入口，提供 `serve` 和 `init`
- `src/mcp_rag/main.py`: HTTP 服务启动入口
- `src/mcp_rag/http_server.py`: HTTP API、SPA 入口、Streamable HTTP MCP 挂载
- `src/mcp_rag/mcp_server.py`: MCP 工具定义与 `rag_ask`
- `src/mcp_rag/app_factory.py`: 统一装配 app context、runtime、guardrails
- `src/mcp_rag/knowledge_bases.py`: 知识库注册表与默认知识库解析
- `src/mcp_rag/config.py`: 配置模型、JSON/SQLite 持久化、热更新

## 环境要求

- Python `>= 3.13`
- `uv`

## 安装

安装 CLI：

```bash
uv tool install mcp-rag
```

安装后直接运行：

```bash
mcp-rag serve
```

在仓库里开发：

```bash
uv sync
```

如果需要本地 embedding：

```bash
uv sync --extra local-embeddings
```

边界说明：
- 使用 `uv tool install mcp-rag` 的安装用户不需要 Node.js，也不需要 `pnpm`
- `pnpm` 只用于维护前端构建，不是服务运行时依赖

## 启动与初始化

启动服务：

```bash
uv run mcp-rag serve
```

初始化数据目录：

```bash
uv run mcp-rag init --data-dir ./data
```

默认端口是 `8060`，服务默认监听 `0.0.0.0:8060`。

常用入口：
- 管理面板：`http://127.0.0.1:8060/app`
- API 文档：`http://127.0.0.1:8060/docs`
- MCP 端点：`http://127.0.0.1:8060/mcp`

兼容入口：
- `/` 会重定向到 `/app`
- `/doc` 会重定向到 `/docs`
- `/documents-page` 会重定向到 `/app/documents`
- `/config-page` 会重定向到 `/app/config`

首次启动行为：
- 如果 `./data/config.json` 不存在，读取配置时会先使用默认值
- 服务启动时会调用 `ensure_config_file()`，把默认配置写入磁盘
- 数据目录中的 `./data/chroma` 和相关 SQLite 文件会按需创建

## 前端与静态资源

发布包会把 `src/mcp_rag/static/` 一并打进 wheel / sdist。

这意味着：
- 安装用户运行 `uv tool install mcp-rag` 后可以直接访问 `/app`
- 不需要单独构建前端，也不需要 Node.js
- 前端维护者需要在发版前生成最新静态资源

前端源码在 `frontend/`，构建输出到 `src/mcp_rag/static/app`。

典型流程：

```bash
cd frontend
pnpm install
pnpm build
```

## 知识库模型

当前项目不再只靠裸 `collection` 组织数据，而是以知识库注册表为主。

知识库特性：
- 持久化注册表在 `knowledge_base_db_path` 指向的 SQLite 文件中
- 默认会确保存在一个公共知识库
- 当传入 `user_id + agent_id` 时，会确保存在对应的默认 `agent_private` 知识库
- 新建知识库后会分配稳定的内部集合名，例如 `kb_<id>`

接口层仍然保留 `collection` 参数，原因是需要兼容旧调用方式。当前实际行为是：
- 可以显式传 `kb_id`
- 也可以继续传旧 `collection`
- 服务会把请求解析到具体知识库和实际集合名

## HTTP 接口

系统接口：
- `GET /health`
- `GET /ready`
- `GET /metrics`

配置接口：
- `GET /config`
- `POST /config`
- `POST /config/bulk`
- `POST /config/reset`
- `POST /config/reload`

服务商接口：
- `GET /providers/{provider}/models`

知识库接口：
- `GET /collections`
- `GET /knowledge-bases`
- `POST /knowledge-bases`

文档接口：
- `POST /add-document`
- `POST /upload-files`
- `GET /list-documents`
- `DELETE /delete-document`
- `GET /list-files`
- `DELETE /delete-file`

检索与问答：
- `GET /search`
- `POST /chat`

MCP 调试接口：
- `GET /debug/mcp/tools`
- `POST /debug/mcp/call`

几点需要明确：
- `/search` 和 `/chat` 支持 `kb_id`
- `/search` 和 `/chat` 也支持 `kb_ids` 做多知识库聚合
- `/upload-files` 使用 `multipart/form-data`
- `/delete-document` 和 `/delete-file` 通过请求体传删除参数

如果启用了安全策略，API key 可以通过以下方式传入：
- HTTP Header: `x-api-key`
- Header: `Authorization: Bearer <token>`
- 查询参数、JSON body 或 form 中的 `api_key`

## MCP

当前主形态是 Streamable HTTP MCP：

```json
{
  "mcpServers": {
    "rag": {
      "url": "http://127.0.0.1:8060/mcp"
    }
  }
}
```

已实现的 MCP 工具：
- `rag_ask`

`rag_ask` 主要参数：
- `query`
- `mode`: `raw` 或 `summary`
- `collection`
- `kb_id`
- `scope`
- `limit`
- `threshold`
- `tenant`
- `user_id` / `agent_id`
- `_user_id` / `_agent_id`
- `api_key`
- `request_id`
- `trace_id`

示例：

```json
{
  "name": "rag_ask",
  "arguments": {
    "query": "FastAPI 是什么",
    "kb_id": 1,
    "mode": "summary",
    "limit": 5
  }
}
```

## 配置

默认配置文件：

```text
./data/config.json
```

默认知识库数据库：

```text
./data/knowledge_bases.sqlite3
```

当前配置有一个重要变化：
- 普通运行配置保存在 `config.json`
- provider 相关配置会持久化到 SQLite，而不是继续完整写回 `config.json`

也就是说，这些字段会存到 SQLite 中的 `service_provider_settings`：
- `embedding_provider`
- `embedding_fallback_provider`
- `provider_configs`
- `llm_provider`
- `llm_fallback_provider`
- `llm_model`
- `llm_base_url`
- `llm_api_key`

其余配置仍然保存在 `config.json`，例如：

```json
{
  "http_port": 8060,
  "chroma_persist_directory": "./data/chroma",
  "knowledge_base_db_path": "./data/knowledge_bases.sqlite3",
  "enable_llm_summary": false,
  "security": {
    "enabled": false,
    "allow_anonymous": true,
    "api_keys": [],
    "tenant_api_keys": {}
  },
  "rate_limit": {
    "requests_per_window": 120,
    "window_seconds": 60,
    "burst": 30
  },
  "quotas": {
    "max_upload_files": 20,
    "max_upload_bytes": 52428800,
    "max_upload_file_bytes": 10485760,
    "max_index_documents": 500,
    "max_index_chunks": 2000,
    "max_index_chars": 500000
  },
  "cache": {
    "enabled": false,
    "max_entries": 256,
    "ttl_seconds": 300
  },
  "provider_budget": {
    "enabled": true
  }
}
```

当前内置 provider 相关能力：
- embedding provider 默认值是 `zhipu`
- LLM provider 默认值是 `doubao`
- 内置 provider 配置包含 `doubao`、`zhipu`、`aliyun`
- `qwen` / `dashscope` 会规范化为 `aliyun`
- `/providers/{provider}/models` 支持从兼容 OpenAI 的模型服务拉取模型列表
- 本地 embedding 支持 `m3e-small` 和 `e5-small`
- LLM 额外支持 `ollama`

## 热更新与运行时刷新

热更新行为：
- 通过 `/config`、`/config/bulk`、`/config/reset`、`/config/reload` 修改后，运行时会立即刷新
- 请求进入时会通过 `reload_if_changed()` 检测磁盘配置是否变化
- provider 设置或检索配置变化后，会重建相关运行时依赖并清理检索缓存

## Readiness 与 Metrics

- `/health` 返回健康摘要、运行时快照和 `config_revision`
- `/ready` 在未完成 bootstrap 或关键依赖未就绪时返回 `503`
- `/metrics` 返回按 operation / provider 聚合的观测指标

当前 readiness 快照会包含：
- `document_processor`
- `embedding_model`
- `vector_store`
- `hybrid_service`
- `llm_model`
- `retrieval_cache`
- `provider_budget`

## 测试

运行全量测试：

```bash
uv run python -m unittest discover -s tests
```

编译检查：

```bash
uv run python -m compileall src
```

当前测试覆盖：
- 配置默认值、磁盘重载与 provider 配置迁移
- HTTP 壳层与 MCP 壳层行为
- request context / tenant 解析
- request-level retrieval cache
- provider budget / fallback
- readiness / health / metrics
- 打包元数据与静态资源

## 许可证

MIT
