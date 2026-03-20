# MCP-RAG

面向 AI 客户端的服务优先 RAG 服务。

当前主形态：
- FastAPI HTTP 服务
- Streamable HTTP MCP 端点
- 基于 Chroma 的多集合检索
- 统一的 AppContext / RequestContext / TenantContext
- 进程内限流、配额、检索缓存、provider budget、观测与 readiness

这一版已经补齐此前 README 里列出的核心治理缺口：
- 正式的 request context / tenant context
- request-level retrieval cache
- provider 级预算、熔断和 fallback
- p50 / p95 / p99 观测
- 更强的 readiness / dependency health
- Streamable HTTP MCP transport smoke 覆盖
- 配置热更新
- 首次运行无配置文件时不会报错

## 当前状态

当前已经完成：
- HTTP / MCP 共用一套运行时装配
- 检索、索引、聊天拆到独立 service 层
- `search` / `chat` / MCP `rag_ask`
- `/health`、`/ready`、`/metrics`
- 基础 API key 鉴权
- 进程内限流
- 上传 / 索引配额
- tenant-aware request cache
- provider budget / circuit breaker / fallback
- provider latency 和 percentile metrics
- Chroma-backed 端到端测试
- Streamable HTTP `/mcp` 冒烟测试

当前还**不是**生产级完成态，主要还差：
- 限流、缓存、provider budget 仍是单进程内存实现，不是分布式
- 还没有正式的 tenant quota / billing / 审计链路
- readiness 主要是配置和运行时状态探测，不是完整 deep probe
- 还没有 Prometheus / OpenTelemetry 这类外部观测出口
- 还没有异步重建索引、后台清理、任务队列这类运维基础设施

## 架构

当前主链路：

```text
HTTP / MCP
  -> app_factory.py
  -> service_facade.py
  -> services/
       - runtime.py
       - indexing_service.py
       - retrieval_service.py
       - chat_service.py
  -> core/indexing/
  -> retrieval/
```

关键文件：
- 服务入口：[src/mcp_rag/main.py](/Users/kalicyh/Documents/GitHub/mcp-rag/src/mcp_rag/main.py)
- CLI 启服：[src/mcp_rag/cli.py](/Users/kalicyh/Documents/GitHub/mcp-rag/src/mcp_rag/cli.py)
- HTTP API：[src/mcp_rag/http_server.py](/Users/kalicyh/Documents/GitHub/mcp-rag/src/mcp_rag/http_server.py)
- MCP 服务：[src/mcp_rag/mcp_server.py](/Users/kalicyh/Documents/GitHub/mcp-rag/src/mcp_rag/mcp_server.py)
- 运行时装配根：[src/mcp_rag/app_factory.py](/Users/kalicyh/Documents/GitHub/mcp-rag/src/mcp_rag/app_factory.py)
- 兼容导出层：[src/mcp_rag/shell_factory.py](/Users/kalicyh/Documents/GitHub/mcp-rag/src/mcp_rag/shell_factory.py)
- facade：[src/mcp_rag/service_facade.py](/Users/kalicyh/Documents/GitHub/mcp-rag/src/mcp_rag/service_facade.py)
- service 层：[src/mcp_rag/services](/Users/kalicyh/Documents/GitHub/mcp-rag/src/mcp_rag/services)
- 检索层：[src/mcp_rag/retrieval](/Users/kalicyh/Documents/GitHub/mcp-rag/src/mcp_rag/retrieval)

说明：
- `app_factory.py` 是实际的 composition root。
- `shell_factory.py` 现在只是兼容导出，避免旧引用断掉。

## 主要能力

- 文档上传、切分、入库
- 向量检索 + 轻量关键词检索融合
- LLM 摘要与问答
- `collection + user_id + agent_id` 的 tenant 隔离
- request-level retrieval cache
- provider budget / circuit breaker / fallback
- readiness / health / metrics
- 配置热更新和运行时重装配

## 环境要求

- Python `>= 3.13`
- `uv`

## 安装

```bash
uv sync
```

如果需要本地 embedding：

```bash
uv sync --extra local-embeddings
```

## 启动

```bash
uv run mcp-rag serve
```

默认端口是 `8060`。

常用入口：
- 文档管理页：`http://127.0.0.1:8060/documents-page`
- 配置页面：`http://127.0.0.1:8060/config-page`
- Swagger：`http://127.0.0.1:8060/docs`
- MCP 端点：`http://127.0.0.1:8060/mcp`

初始化数据目录：

```bash
uv run mcp-rag init --data-dir ./data
```

首次运行如果没有 `./data/config.json`，服务会自动以默认配置启动并生成配置文件，不会因为缺少配置文件直接报错。

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
- `POST /config/reload`

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

`/ready` 语义：
- `200`：transport 已 bootstrapped，且 runtime readiness 为 true
- `503`：服务进程已起来，但依赖配置或 runtime 状态还没 ready

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

如果启用了安全策略，stdio / MCP 工具调用可以传：

```json
{
  "api_key": "your-token"
}
```

## Tenant 与 Request Context

当前 tenant / request 相关字段：
- `collection`
- `tenant.base_collection`
- `tenant.user_id`
- `tenant.agent_id`
- 兼容字段：`user_id` / `agent_id`
- `request_id`
- `trace_id`

HTTP 和 MCP 都会先归一化成统一的 `RequestContext`，service 层只消费标准化后的 tenant / request 信息。

## 配置

配置文件默认在：

```text
./data/config.json
```

当前除了模型和检索参数，还包含以下治理相关段落：

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
  },
  "embedding_fallback_provider": "m3e-small",
  "llm_fallback_provider": "ollama"
}
```

配置更新方式：
- `POST /config`
- `POST /config/bulk`
- `POST /config/reset`
- `POST /config/reload`

这些接口会触发运行时热更新；涉及检索行为的配置更新会同时清空 request cache。

## 测试

运行全量测试：

```bash
uv run python -m unittest discover -s tests
```

编译检查：

```bash
uv run python -m compileall src tests
```

当前测试覆盖：
- service 层单测
- request cache 单测
- provider budget / fallback 单测
- HTTP / MCP 壳层测试
- `/mcp` Streamable HTTP smoke
- 配额 / 安全 / 观测测试
- 基于临时 Chroma 的端到端集成测试

## 下一步建议

如果继续往生产化推进，优先级建议是：
- 把限流、缓存、provider budget 下沉到 Redis 之类的共享状态
- 做正式的 tenant quota / billing / 审计模型
- 给 embedding / vector store / llm 加深度可控的 probe 策略
- 补 Prometheus / OpenTelemetry 导出
- 增加后台重建索引、清理和任务调度能力

## 许可证

MIT
