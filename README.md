# MCP-RAG

面向 AI 客户端的服务优先 RAG 服务。

当前主形态是：
- FastAPI HTTP 服务
- Streamable HTTP MCP 端点
- 基于 Chroma 的多集合检索
- HTTP / MCP 共用运行时、鉴权、限流、配额、观测与租户上下文

## 当前状态

这版已经完成：
- 正式 `RequestContext` / `TenantContext`
- HTTP / MCP 统一生成标准化上下文
- service 层统一消费标准化 context
- 配置文件热更新
- 首次运行没有配置文件也不会报错，并会在启动时写入默认配置
- request-level retrieval cache
- provider 级预算、熔断、fallback
- `/health`、`/ready`、`/metrics`
- 更细的 runtime readiness / dependency health
- Streamable HTTP MCP smoke test
- Chroma-backed 端到端测试

当前仍然不是“生产平台级完成态”，还缺：
- 分布式限流、缓存、provider budget 状态共享
- 更正式的身份体系，例如 OIDC / workspace 级鉴权
- 更完整的 tracing / log correlation / metrics export
- collection 生命周期治理、后台任务和资源回收策略

## 架构

当前主链路：

```text
HTTP / MCP
  -> app_factory.py
  -> context.py
  -> service_facade.py
  -> services/
       - runtime.py
       - indexing_service.py
       - retrieval_service.py
       - chat_service.py
  -> core/indexing/
  -> retrieval/
```

关键模块：
- `src/mcp_rag/main.py`: 服务启动入口
- `src/mcp_rag/cli.py`: CLI 启服与初始化
- `src/mcp_rag/http_server.py`: HTTP API 与 Streamable HTTP MCP 挂载
- `src/mcp_rag/mcp_server.py`: MCP 工具定义与 `rag_ask`
- `src/mcp_rag/app_factory.py`: HTTP / MCP 共享运行时装配
- `src/mcp_rag/shell_factory.py`: 向后兼容导出
- `src/mcp_rag/context.py`: `RequestContext` / `TenantContext`
- `src/mcp_rag/services/runtime.py`: provider、cache、readiness、热更新
- `src/mcp_rag/services/retrieval_cache.py`: request-level retrieval cache

## 主要能力

- 文档上传、切分、入库
- 向量检索 + 轻量关键词检索融合
- `search` / `chat` / MCP `rag_ask`
- `collection + user_id + agent_id` 的 tenant 隔离
- API key 鉴权
- 进程内 rate limit
- 上传 / 索引配额
- request-level retrieval cache
- provider-side budget、熔断、fallback
- readiness / health / metrics

## 环境要求

- Python `>= 3.13`
- `uv`

## 安装

发布版用户可以直接安装 CLI：

```bash
uv tool install mcp-rag
```

安装完成后可以直接运行：

```bash
mcp-rag serve
```

如果你是在本仓库里开发：

```bash
uv sync
```

如果需要本地 embedding：

```bash
uv sync --extra local-embeddings
```

这里有一个边界要明确：
- 使用 `uv tool install mcp-rag` 的安装用户不需要 Node.js，也不需要 `pnpm`
- `pnpm` 只用于维护前端时的开发构建，不是服务运行时依赖

## 启动

```bash
uv run mcp-rag serve
```

默认端口是 `8060`。

常用入口：
- 管理面板：`http://127.0.0.1:8060/app`
- Swagger：`http://127.0.0.1:8060/docs`
- MCP 端点：`http://127.0.0.1:8060/mcp`

兼容入口：
- `/documents-page` 会重定向到 `/app/documents`
- `/config-page` 会重定向到 `/app/config`

初始化数据目录：

```bash
uv run mcp-rag init --data-dir ./data
```

首次启动行为：
- 如果 `./data/config.json` 不存在，服务不会报错
- 启动阶段会自动写入默认配置
- `ConfigManager.reload_if_changed()` 会在运行中拾取外部配置变更

## 前端与静态资源

当前 Python 包会把 `src/mcp_rag/static/` 下的静态文件一并打进 wheel / sdist。

这意味着：
- 如果你是安装用户，`uv tool install mcp-rag` 后直接运行即可，不需要单独处理前端构建
- 如果你在维护前端，需要在发版前把构建产物放到 `src/mcp_rag/static/`
- 前端开发构建应使用 `pnpm`，但不要把 Node 作为安装用户的前置条件

典型流程：

```bash
cd frontend
pnpm install
pnpm build
```

发版前需要确认前端构建产物的输出目录就是 `src/mcp_rag/static/`，否则安装包里不会带上对应页面资源。

## HTTP 接口

基础接口：
- `GET /health`
- `GET /ready`
- `GET /metrics`

配置接口：
- `GET /config`
- `POST /config`
- `POST /config/bulk`
- `POST /config/reset`

RAG 接口：
- `POST /add-document`
- `POST /upload-files`
- `GET /collections`
- `GET /search`
- `POST /chat`
- `GET /list-documents`
- `DELETE /delete-document`
- `GET /list-files`
- `DELETE /delete-file`

如果启用了安全策略，可以通过以下方式传 API key：
- HTTP Header: `x-api-key`
- Header: `Authorization: Bearer <token>`
- 查询参数或 body/form 中的 `api_key`

## MCP

当前主要面向 Streamable HTTP MCP：

```json
{
  "mcpServers": {
    "rag": {
      "url": "http://127.0.0.1:8060/mcp"
    }
  }
}
```

支持的 MCP 工具：
- `rag_ask`

示例：

```json
{
  "name": "rag_ask",
  "arguments": {
    "query": "FastAPI 是什么",
    "collection": "default",
    "mode": "summary",
    "limit": 5
  }
}
```

如果启用了安全策略，MCP 工具调用可以传：

```json
{
  "api_key": "your-token"
}
```

## Request Context 与 Tenant

标准请求上下文字段在 `src/mcp_rag/context.py`：
- `tenant.base_collection`
- `tenant.user_id`
- `tenant.agent_id`
- `tenant.tenant_key`
- `transport`
- `operation`
- `api_key`
- `request_id`
- `trace_id`
- `subject`
- `rate_limit_subject`
- `quota_subject`

兼容输入：
- `collection`
- `tenant`
- `user_id` / `agent_id`
- `_user_id` / `_agent_id`

service 层不再自己拼 tenant / request 身份，而是统一消费标准化后的 context。

## 配置

配置文件默认在：

```text
./data/config.json
```

当前治理相关配置示例：

```json
{
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
  "observability": {
    "warning_error_rate": 0.05,
    "critical_error_rate": 0.2,
    "slow_request_ms": 1000.0,
    "latency_window_size": 512
  },
  "provider_budget": {
    "enabled": true,
    "embeddings": {
      "requests_per_window": 300,
      "window_seconds": 60,
      "burst": 60,
      "failure_threshold": 3,
      "cooldown_seconds": 30
    },
    "llm": {
      "requests_per_window": 120,
      "window_seconds": 60,
      "burst": 20,
      "failure_threshold": 3,
      "cooldown_seconds": 30
    }
  }
}
```

热更新行为：
- 通过 `/config`、`/config/bulk`、`/config/reset` 修改后，运行时会同步刷新
- 外部直接改写配置文件后，会在请求路径上通过 `reload_if_changed()` 自动拾取
- provider、retrieval cache、guardrails 会按配置签名重新装配或失效

## Readiness 与 Metrics

- `/health` 返回整体健康摘要、错误率、慢操作和 runtime 快照
- `/ready` 在 runtime 依赖未就绪或配置缺失时返回 `503`
- `/metrics` 返回按 operation / provider 聚合的指标
- 观测输出包含 `p50 / p95 / p99`

当前 readiness 会显式暴露：
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
- `RequestContext` / `TenantContext`
- 配置热更新与首次启动默认配置
- request-level retrieval cache
- provider budget / fallback
- readiness / health / metrics
- HTTP / MCP 壳层行为
- Streamable HTTP MCP smoke
- 基于临时 Chroma 的端到端集成测试
- 打包元数据与安装说明检查

## 后续建议

如果继续往生产平台推进，优先建议做：
- 外部缓存与分布式限流
- 更正式的租户与身份接入层
- tracing 导出与 metrics backend
- collection 生命周期与异步索引任务

## 许可证

MIT
