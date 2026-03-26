<script>
  import { onMount } from 'svelte';
  import { api } from './lib/api.js';
  import PageShell from './lib/components/PageShell.svelte';
  import PanelCard from './lib/components/PanelCard.svelte';
  import SelectField from './lib/components/SelectField.svelte';

  const STORAGE_KEY = 'mcp-rag-dashboard-state';
  const sections = [
    { id: 'overview', title: '总览', subtitle: '状态和指标' },
    { id: 'documents', title: '文档管理', subtitle: '上传、检索、删除' },
    { id: 'mcp', title: 'MCP 调试', subtitle: '工具与调用' },
    { id: 'config', title: '配置中心', subtitle: '配置和策略' },
  ];
  const documentModes = [
    { id: 'ingest', title: '导入' },
    { id: 'search', title: '检索' },
    { id: 'manage', title: '管理' },
  ];
  const configModes = [
    { id: 'provider', title: '服务商配置' },
    { id: 'system', title: '系统配置' },
    { id: 'advanced', title: '高级配置' },
  ];
  const providerCatalog = [
    {
      id: 'doubao',
      title: '豆包',
      vendor: 'Volcengine Ark',
      description: '火山方舟兼容 OpenAI 风格接口，可同时承载对话和向量能力。',
      website: 'https://www.volcengine.com/product/ark',
      defaults: {
        base_url: 'https://ark.cn-beijing.volces.com/api/v3',
      },
      families: ['chat', 'embedding'],
      models: {
        chat: [],
        embedding: [],
      },
    },
    {
      id: 'zhipu',
      title: '智谱',
      vendor: 'BigModel',
      description: '智谱开放平台，支持 GLM 系列对话模型与 embedding 接口。',
      website: 'https://open.bigmodel.cn',
      defaults: {
        base_url: 'https://open.bigmodel.cn/api/paas/v4',
      },
      families: ['chat', 'embedding'],
      models: {
        chat: [],
        embedding: [],
      },
    },
    {
      id: 'openai',
      title: 'OpenAI',
      vendor: 'OpenAI',
      description: '标准 OpenAI 接口，可配置通用对话与向量模型。',
      website: 'https://platform.openai.com',
      defaults: {
        base_url: 'https://api.openai.com/v1',
      },
      families: ['chat', 'embedding'],
      models: {
        chat: [],
        embedding: [],
      },
    },
    {
      id: 'deepseek',
      title: 'DeepSeek',
      vendor: 'DeepSeek',
      description: '深度求索对话模型，可通过兼容接口直接接入。',
      website: 'https://platform.deepseek.com',
      defaults: {
        base_url: 'https://api.deepseek.com/v1',
      },
      families: ['chat'],
      models: {
        chat: [],
      },
    },
    {
      id: 'aliyun',
      title: '阿里云百炼',
      vendor: 'Alibaba Cloud Bailian',
      description: '阿里云百炼提供 OpenAI 兼容接口，可统一接入千问对话与向量模型。',
      website: 'https://bailian.console.aliyun.com',
      defaults: {
        base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
      },
      families: ['chat', 'embedding'],
      models: {
        chat: [],
        embedding: [],
      },
    },
    {
      id: 'siliconflow',
      title: 'SiliconCloud',
      vendor: 'SiliconFlow',
      description: '硅基流动聚合平台，支持多家模型的兼容接入。',
      website: 'https://siliconflow.cn',
      defaults: {
        base_url: 'https://api.siliconflow.cn/v1',
      },
      families: ['chat', 'embedding'],
      models: {
        chat: [],
        embedding: [],
      },
    },
    {
      id: 'ollama',
      title: 'Ollama',
      vendor: 'Ollama',
      description: '本地大模型服务，当前主要用于对话模型。',
      website: 'https://ollama.com',
      defaults: {
        base_url: 'http://localhost:11434',
      },
      families: ['chat'],
      models: {
        chat: [],
      },
    },
    {
      id: 'm3e-small',
      title: '本地 M3E',
      vendor: 'Local Embedding',
      description: '本地句向量模型，不依赖远端服务商接口。',
      defaults: {},
      families: ['embedding'],
      local: true,
      models: {
        embedding: [{ value: 'm3e-small', label: 'm3e-small' }],
      },
    },
    {
      id: 'e5-small',
      title: '本地 E5',
      vendor: 'Local Embedding',
      description: '本地轻量向量模型，适合纯离线场景。',
      defaults: {},
      families: ['embedding'],
      local: true,
      models: {
        embedding: [{ value: 'e5-small', label: 'e5-small' }],
      },
    },
  ];
  const modelFamilyMeta = {
    chat: { title: 'LLM', label: '对话模型' },
    embedding: { title: 'Embedding', label: '向量模型' },
  };
  const providerAliases = {
    qwen: 'aliyun',
    dashscope: 'aliyun',
  };
  const routeSectionMap = {
    overview: 'overview',
    documents: 'documents',
    mcp: 'mcp',
    config: 'config',
    status: 'overview',
    insights: 'overview',
  };

  function emptyDraft() {
    return {
      host: '0.0.0.0',
      port: 8060,
      http_port: 8060,
      debug: false,
      vector_db_type: 'chroma',
      chroma_persist_directory: './data/chroma',
      knowledge_base_db_path: './data/knowledge_bases.sqlite3',
      qdrant_url: 'http://localhost:6333',
      embedding_provider: '',
      embedding_fallback_provider: '',
      embedding_device: 'cpu',
      embedding_cache_dir: '',
      embedding_base_url: '',
      embedding_model: '',
      embedding_api_key: '',
      llm_provider: '',
      llm_fallback_provider: '',
      llm_model: '',
      llm_base_url: '',
      llm_api_key: '',
      enable_thinking: true,
      enable_llm_summary: false,
      enable_reranker: false,
      enable_cache: false,
      max_retrieval_results: 5,
      similarity_threshold: 0.7,
      cache: {
        enabled: false,
        max_entries: 256,
        ttl_seconds: 300,
      },
      security: {
        enabled: false,
        allow_anonymous: true,
        api_keys_text: '',
        tenant_api_keys_text: '{}',
      },
      rate_limit: {
        requests_per_window: 120,
        window_seconds: 60,
        burst: 30,
      },
      quotas: {
        max_upload_files: 20,
        max_upload_bytes: 52428800,
        max_upload_file_bytes: 10485760,
        max_index_documents: 500,
        max_index_chunks: 2000,
        max_index_chars: 500000,
      },
      observability: {
        warning_error_rate: 0.05,
        critical_error_rate: 0.2,
        slow_request_ms: 1000,
        latency_window_size: 512,
      },
      provider_budget: {
        enabled: true,
        embeddings: {
          requests_per_window: 300,
          window_seconds: 60,
          burst: 60,
          failure_threshold: 3,
          cooldown_seconds: 30,
        },
        llm: {
          requests_per_window: 120,
          window_seconds: 60,
          burst: 20,
          failure_threshold: 3,
          cooldown_seconds: 30,
        },
      },
      provider_configs_text: '{}',
      full_config_text: '{}',
    };
  }

  function emptyContext() {
    return {
      userId: '',
      agentId: '',
      apiKey: '',
    };
  }

  function normalizeIdentity(value) {
    return {
      userId: value?.userId ? String(value.userId) : '',
      agentId: value?.agentId ? String(value.agentId) : '',
      apiKey: value?.apiKey ? String(value.apiKey) : '',
    };
  }

  function parseJsonOr(defaultValue, text) {
    if (!text) return defaultValue;
    try {
      return JSON.parse(text);
    } catch {
      return defaultValue;
    }
  }

  function prettyJson(value) {
    try {
      return JSON.stringify(value, null, 2);
    } catch {
      return String(value ?? '');
    }
  }

  function splitKeys(text) {
    return text
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
  }

  function toBytes(value) {
    const size = Number(value || 0);
    if (!Number.isFinite(size) || size <= 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    let current = size;
    let index = 0;
    while (current >= 1024 && index < units.length - 1) {
      current /= 1024;
      index += 1;
    }
    return `${current.toFixed(current >= 10 ? 0 : 1)} ${units[index]}`;
  }

  function formatTime(value) {
    if (!value) return '未采样';
    const date = new Date(value * 1000 || value);
    return Number.isNaN(date.getTime()) ? '未采样' : date.toLocaleString();
  }

  function pct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '0%';
    return `${(Number(value) * 100).toFixed(1)}%`;
  }

  function clampText(value, limit = 180) {
    const text = String(value ?? '');
    return text.length > limit ? `${text.slice(0, limit)}...` : text;
  }

  function statusTone(status) {
    const normalized = String(status || '').toLowerCase();
    if (['ready', 'healthy', 'configured', 'enabled', 'ok', 'success'].includes(normalized)) return 'good';
    if (['warn', 'warning', 'degraded', 'partial', 'limited'].includes(normalized)) return 'warn';
    if (['error', 'failed', 'misconfigured', 'unhealthy', 'cooldown'].includes(normalized)) return 'bad';
    return 'info';
  }

  function safeText(value, fallback = '未知') {
    return value === null || value === undefined || value === '' ? fallback : String(value);
  }

  function canonicalProviderId(providerId) {
    const normalized = String(providerId || '').trim().toLowerCase();
    return providerAliases[normalized] || normalized;
  }

  function normalizeProviderConfigs(providerConfigs = {}) {
    const normalizedConfigs = {};
    for (const [providerId, config] of Object.entries(providerConfigs || {})) {
      normalizedConfigs[canonicalProviderId(providerId)] = config;
    }
    return normalizedConfigs;
  }

  function normalizeConfigPayload(payload) {
    if (!payload) return payload;
    const normalizedProviderConfigs = normalizeProviderConfigs(payload.provider_configs || {});
    return {
      ...payload,
      embedding_provider: canonicalProviderId(payload.embedding_provider),
      embedding_fallback_provider: canonicalProviderId(payload.embedding_fallback_provider),
      llm_provider: canonicalProviderId(payload.llm_provider),
      llm_fallback_provider: canonicalProviderId(payload.llm_fallback_provider),
      provider_configs: normalizedProviderConfigs,
    };
  }

  function summarizeUploadFailures(payload) {
    const failedItems = Array.isArray(payload?.results)
      ? payload.results.filter((item) => !item?.processed)
      : [];
    if (!failedItems.length) return '';
    return failedItems
      .slice(0, 3)
      .map((item) => `${safeText(item.filename, 'unknown')}: ${safeText(item.error, '上传失败')}`)
      .join('；');
  }

  function providerDefinition(providerId) {
    const canonicalId = canonicalProviderId(providerId);
    return providerCatalog.find((item) => item.id === canonicalId) || null;
  }

  function providersForFamily(family) {
    return providerCatalog.filter((item) => item.families.includes(family));
  }

  function providerSelectOptions(family) {
    return providersForFamily(family).map((item) => ({
      value: item.id,
      label: item.title,
    }));
  }

  function providerModelField(family) {
    return family === 'chat' ? 'llm_model' : 'embedding_model';
  }

  function providerConfigValue(providerId, field, fallback = '') {
    const canonicalId = canonicalProviderId(providerId);
    const providerConfigs = normalizeProviderConfigs(parseJsonOr({}, configDraft.provider_configs_text));
    let value = providerConfigs?.[canonicalId]?.[field];
    if ((value === undefined || value === null || value === '') && field === 'embedding_model') {
      value = providerConfigs?.[canonicalId]?.model;
    }
    if ((value === undefined || value === null || value === '') && field === 'llm_model') {
      value = providerConfigs?.[canonicalId]?.llm_model ?? providerConfigs?.[canonicalId]?.model;
    }
    if (value !== undefined && value !== null && value !== '') return value;

    if (canonicalId === canonicalProviderId(configDraft.llm_provider)) {
      if (field === 'base_url' && configDraft.llm_base_url) return configDraft.llm_base_url;
      if (field === 'api_key' && configDraft.llm_api_key) return configDraft.llm_api_key;
      if (field === 'llm_model' && configDraft.llm_model) return configDraft.llm_model;
    }

    const provider = providerDefinition(providerId);
    return provider?.defaults?.[field] ?? fallback;
  }

  function providerModelOptions(providerId, family) {
    const canonicalId = canonicalProviderId(providerId);
    const provider = providerDefinition(canonicalId);
    const catalogOptions = provider?.models?.[family] ?? [];
    const providerConfigs = normalizeProviderConfigs(parseJsonOr({}, configDraft.provider_configs_text));
    const configModels = Array.isArray(providerConfigs?.[canonicalId]?.[family === 'chat' ? 'chat_models' : 'embedding_models'])
      ? providerConfigs[canonicalId][family === 'chat' ? 'chat_models' : 'embedding_models']
      : [];
    const remoteModels = Array.isArray(fetchedProviderModels?.[canonicalId]?.[family])
      ? fetchedProviderModels[canonicalId][family]
      : [];
    const mergedCatalog = [...catalogOptions];
    for (const model of configModels) {
      if (model && !mergedCatalog.some((item) => item.value === model)) {
        mergedCatalog.push({ value: model, label: `${model} (手动)` });
      }
    }
    for (const model of remoteModels) {
      if (model?.id && !mergedCatalog.some((item) => item.value === model.id)) {
        mergedCatalog.push({ value: model.id, label: `${model.label || model.id} (远程)` });
      }
    }
    const field = providerModelField(family);
    const currentValue = providerConfigValue(providerId, field, '');
    if (currentValue && !mergedCatalog.some((item) => item.value === currentValue)) {
      return [...mergedCatalog, { value: currentValue, label: `${currentValue} (当前自定义)` }];
    }
    return mergedCatalog;
  }

  function providerModelValue(providerId, family) {
    const field = providerModelField(family);
    if (family === 'chat' && canonicalProviderId(providerId) === canonicalProviderId(configDraft.llm_provider) && configDraft.llm_model) {
      return configDraft.llm_model;
    }
    return providerConfigValue(providerId, field, '');
  }

  function providerModelSourceLabel(source) {
    if (source === 'synced') return '已同步';
    if (source === 'remote') return '远程';
    if (source === 'manual') return '手动';
    if (source === 'current') return '当前';
    return '预设';
  }

  function inferModelFamily(modelId, allowedFamilies = ['chat']) {
    if (allowedFamilies.length === 1) return allowedFamilies[0];
    const normalized = String(modelId || '').trim().toLowerCase();
    const embeddingMarkers = ['embedding', 'text-embedding', 'bge-', 'm3e', 'e5', 'rerank'];
    return embeddingMarkers.some((marker) => normalized.includes(marker)) ? 'embedding' : 'chat';
  }

  function providerModelTableRows(providerId) {
    const provider = providerDefinition(providerId);
    if (!provider) return [];

    return provider.families.flatMap((family) => {
      const currentValue = providerModelValue(providerId, family);
      const rows = [];
      const seen = new Set();

      for (const item of provider.models?.[family] ?? []) {
        const modelId = String(item?.value || '').trim();
        if (!modelId || seen.has(modelId)) continue;
        seen.add(modelId);
        rows.push({
          family,
          id: modelId,
          label: item?.label || modelId,
          source: 'preset',
          selected: currentValue === modelId,
          removable: false,
        });
      }

      for (const modelId of providerSyncedModels(providerId, family)) {
        if (!modelId || seen.has(modelId)) continue;
        seen.add(modelId);
        rows.push({
          family,
          id: modelId,
          label: modelId,
          source: 'synced',
          selected: currentValue === modelId,
          removable: false,
        });
      }

      for (const modelId of providerCustomModels(providerId, family)) {
        if (!modelId || seen.has(modelId)) continue;
        seen.add(modelId);
        rows.push({
          family,
          id: modelId,
          label: modelId,
          source: 'manual',
          selected: currentValue === modelId,
          removable: true,
        });
      }

      return rows;
    });
  }

  function filteredProviderModelRows(providerId, searchQuery) {
    const normalizedQuery = String(searchQuery || '').trim().toLowerCase();
    return providerModelTableRows(providerId).filter((row) => {
      if (!normalizedQuery) return true;
      return [row.id, row.label, modelFamilyMeta[row.family]?.label]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(normalizedQuery));
    });
  }

  function filteredProviderCatalog() {
    const query = String(providerSearch || '').trim().toLowerCase();
    if (!query) return providerCatalog;
    return providerCatalog.filter((provider) =>
      [provider.title, provider.vendor, provider.id, provider.description]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(query))
    );
  }

  let activeSection = 'overview';
  let documentMode = 'ingest';
  let configMode = 'provider';
  let providerEditor = 'doubao';
  let providerSearch = '';
  let providerModelSearch = '';
  let fetchedProviderModels = {};
  let providerFetchBusy = {};
  let customModelDrafts = {};
  let managedPage = 0;
  let managedPageSize = 8;
  let queuedFiles = [];
  let identity = emptyContext();
  let knowledgeBases = [];
  let selectedKnowledgeBase = '';
  let documentKnowledgeBase = '';
  let searchKnowledgeBase = '';
  let chatKnowledgeBase = '';
  let manageKnowledgeBase = '';
  let fileFilter = '';
  let uploadBusy = false;
  let addBusy = false;
  let searchBusy = false;
  let chatBusy = false;
  let manageBusy = false;
  let configBusy = false;
  let overviewBusy = false;
  let mcpBusy = false;
  let health = null;
  let ready = null;
  let metrics = null;
  let config = null;
  let configDraft = emptyDraft();
  let mcpTools = [];
  let mcpTool = '';
  let mcpArguments = JSON.stringify({ query: 'FastAPI 是什么？', scope: 'public' }, null, 2);
  let mcpResult = '';
  let documents = [];
  let files = [];
  let documentsTotal = 0;
  let filesTotal = 0;
  let previewDocument = null;
  let searchQuery = '';
  let searchLimit = 5;
  let searchResults = [];
  let searchSummary = '';
  let documentTitle = '';
  let documentContent = '';
  let chatInput = '';
  let chatMessages = [
    {
      role: 'assistant',
      content: '先选知识库，再上传文档、修改配置或查看状态。',
    },
  ];
  let toasts = [];
  let lastRefreshAt = null;
  let loading = true;

  function pushToast(title, message = '', tone = 'info') {
    const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    toasts = [...toasts, { id, title, message, tone }];
    setTimeout(() => {
      toasts = toasts.filter((toast) => toast.id !== id);
    }, 4200);
  }

  function knowledgeBaseById(value) {
    const numeric = Number(value || 0);
    return knowledgeBases.find((item) => Number(item.id) === numeric) || null;
  }

  function currentMcpTool() {
    return mcpTools.find((item) => item.name === mcpTool) || null;
  }

  function ensureMcpToolSelected() {
    if (!mcpTool && mcpTools[0]?.name) {
      mcpTool = mcpTools[0].name;
    }
  }

  function knowledgeBaseLabel(item) {
    if (!item) return '未选择知识库';
    if (item.scope === 'public') return `公共 · ${item.name}`;
    return `Agent ${item.owner_agent_id ?? '-'} · ${item.name}`;
  }

  function knowledgeBaseOptions() {
    return knowledgeBases.map((item) => ({
      value: String(item.id),
      label: knowledgeBaseLabel(item),
    }));
  }

  function knowledgeBaseRequest(selection) {
    const knowledgeBase = knowledgeBaseById(selection);
    return {
      kbId: knowledgeBase?.id ? Number(knowledgeBase.id) : undefined,
      scope: knowledgeBase?.scope,
      collection: knowledgeBase?.name || 'default',
    };
  }

  function readState() {
    if (typeof window === 'undefined') return;
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      const parsed = JSON.parse(raw);
      if (parsed.activeSection) activeSection = routeSectionMap[parsed.activeSection] || parsed.activeSection;
      if (parsed.documentMode) documentMode = parsed.documentMode;
      if (parsed.configMode) configMode = parsed.configMode;
      if (parsed.identity) identity = normalizeIdentity(parsed.identity);
      if (parsed.selectedKnowledgeBase) selectedKnowledgeBase = parsed.selectedKnowledgeBase;
      if (parsed.searchKnowledgeBase) searchKnowledgeBase = parsed.searchKnowledgeBase;
      if (parsed.chatKnowledgeBase) chatKnowledgeBase = parsed.chatKnowledgeBase;
      if (parsed.manageKnowledgeBase) manageKnowledgeBase = parsed.manageKnowledgeBase;
      if (parsed.documentKnowledgeBase) documentKnowledgeBase = parsed.documentKnowledgeBase;
    } catch {
      // Ignore malformed local state.
    }
    const routedSection = sectionFromLocation();
    if (routedSection) {
      activeSection = routedSection;
    }
  }

  function writeState() {
    if (typeof window === 'undefined') return;
    const payload = {
      activeSection,
      documentMode,
      configMode,
      identity,
      selectedKnowledgeBase,
      documentKnowledgeBase,
      searchKnowledgeBase,
      chatKnowledgeBase,
      manageKnowledgeBase,
    };
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  }

  function sectionFromLocation() {
    if (typeof window === 'undefined') return null;
    const params = new URLSearchParams(window.location.search);
    const queryView = params.get('view');
    if (queryView && routeSectionMap[queryView]) {
      return routeSectionMap[queryView];
    }
    const pathname = window.location.pathname.replace(/\/+$/, '');
    if (!pathname.startsWith('/app')) {
      return null;
    }
    const slug = pathname.slice('/app'.length).replace(/^\/+/, '') || 'overview';
    return routeSectionMap[slug] || 'overview';
  }

  function sectionHref(section) {
    return section === 'overview' ? '/app' : `/app/${section}`;
  }

  function syncLocation(section, { replace = false } = {}) {
    if (typeof window === 'undefined') return;
    const target = sectionHref(section);
    const current = `${window.location.pathname}${window.location.search}`;
    if (current === target) return;
    const method = replace ? 'replaceState' : 'pushState';
    window.history[method]({}, '', target);
  }

  function switchSection(section) {
    activeSection = section;
    syncLocation(section);
    writeState();
    void refreshActiveView();
  }

  function switchDocumentMode(mode) {
    documentMode = mode;
    writeState();
  }

  function switchConfigMode(mode) {
    if (mode === 'advanced') {
      configDraft = {
        ...configDraft,
        full_config_text: JSON.stringify(buildConfigPayloadFromDraft(), null, 2),
      };
    }
    configMode = mode;
    writeState();
  }

  function syncProviderEditor() {
    const preferred =
      providerDefinition(configDraft.llm_provider)?.id ||
      providerDefinition(configDraft.embedding_provider)?.id ||
      providerCatalog[0]?.id ||
      'doubao';
    if (!providerDefinition(providerEditor)) {
      providerEditor = preferred;
    }
    const families = providerDefinition(providerEditor)?.families ?? [];
  }

  function setKnowledgeBase(value) {
    selectedKnowledgeBase = value;
    documentKnowledgeBase = value;
    searchKnowledgeBase = value;
    chatKnowledgeBase = value;
    manageKnowledgeBase = value;
    writeState();
  }

  function syncConfigDraft(payload) {
    payload = normalizeConfigPayload(payload);
    const draft = emptyDraft();
    if (!payload) {
      configDraft = draft;
      return;
    }

    draft.host = payload.host ?? draft.host;
    draft.port = Number(payload.port ?? draft.port);
    draft.http_port = Number(payload.http_port ?? draft.http_port);
    draft.debug = Boolean(payload.debug);
    draft.vector_db_type = payload.vector_db_type ?? draft.vector_db_type;
    draft.chroma_persist_directory = payload.chroma_persist_directory ?? draft.chroma_persist_directory;
    draft.knowledge_base_db_path = payload.knowledge_base_db_path ?? draft.knowledge_base_db_path;
    draft.qdrant_url = payload.qdrant_url ?? draft.qdrant_url;
    draft.embedding_provider = payload.embedding_provider ?? draft.embedding_provider;
    draft.embedding_fallback_provider = payload.embedding_fallback_provider ?? '';
    draft.embedding_device = payload.embedding_device ?? draft.embedding_device;
    draft.embedding_cache_dir = payload.embedding_cache_dir ?? '';
    draft.llm_provider = payload.llm_provider ?? draft.llm_provider;
    draft.llm_fallback_provider = payload.llm_fallback_provider ?? '';
    draft.llm_model = payload.llm_model ?? draft.llm_model;
    draft.llm_base_url = payload.llm_base_url ?? '';
    draft.llm_api_key = payload.llm_api_key ?? '';
    draft.enable_thinking = payload.enable_thinking ?? draft.enable_thinking;
    draft.enable_llm_summary = Boolean(payload.enable_llm_summary);
    draft.enable_reranker = Boolean(payload.enable_reranker);
    draft.enable_cache = Boolean(payload.enable_cache);
    draft.max_retrieval_results = Number(payload.max_retrieval_results ?? draft.max_retrieval_results);
    draft.similarity_threshold = Number(payload.similarity_threshold ?? draft.similarity_threshold);
    draft.cache = {
      ...draft.cache,
      ...(payload.cache || {}),
    };
    draft.security = {
      ...draft.security,
      ...(payload.security || {}),
      api_keys_text: Array.isArray(payload.security?.api_keys) ? payload.security.api_keys.join('\n') : '',
      tenant_api_keys_text: JSON.stringify(payload.security?.tenant_api_keys ?? {}, null, 2),
    };
    draft.rate_limit = {
      ...draft.rate_limit,
      ...(payload.rate_limit || {}),
    };
    draft.quotas = {
      ...draft.quotas,
      ...(payload.quotas || {}),
    };
    draft.observability = {
      ...draft.observability,
      ...(payload.observability || {}),
    };
    draft.provider_budget = {
      ...draft.provider_budget,
      ...(payload.provider_budget || {}),
      embeddings: {
        ...draft.provider_budget.embeddings,
        ...(payload.provider_budget?.embeddings || {}),
      },
      llm: {
        ...draft.provider_budget.llm,
        ...(payload.provider_budget?.llm || {}),
      },
    };
    draft.provider_configs_text = JSON.stringify(payload.provider_configs ?? {}, null, 2);
    const activeEmbeddingProvider = payload.provider_configs?.[draft.embedding_provider] ?? {};
    draft.embedding_base_url = activeEmbeddingProvider.base_url ?? '';
    draft.embedding_model = activeEmbeddingProvider.embedding_model ?? activeEmbeddingProvider.model ?? '';
    draft.embedding_api_key = activeEmbeddingProvider.api_key ?? '';
    const activeLlmProvider = payload.provider_configs?.[draft.llm_provider] ?? {};
    draft.llm_model = payload.llm_model ?? activeLlmProvider.llm_model ?? activeLlmProvider.model ?? draft.llm_model;
    draft.llm_base_url = payload.llm_base_url ?? activeLlmProvider.base_url ?? draft.llm_base_url;
    draft.llm_api_key = payload.llm_api_key ?? activeLlmProvider.api_key ?? draft.llm_api_key;
    draft.full_config_text = JSON.stringify(payload, null, 2);
    configDraft = draft;
    syncProviderEditor();
  }

  function buildConfigPayloadFromDraft() {
    const providerConfigs = normalizeProviderConfigs(parseJsonOr({}, configDraft.provider_configs_text));
    const llmProviderId = canonicalProviderId(configDraft.llm_provider);
    const embeddingProviderId = canonicalProviderId(configDraft.embedding_provider);
    const llmProviderConfig = providerConfigs[llmProviderId] || {};
    const embeddingProviderConfig = providerConfigs[embeddingProviderId] || {};
    const nextProviderConfigs = {
      ...providerConfigs,
    };

    if (embeddingProviderId) {
      nextProviderConfigs[embeddingProviderId] = {
        ...(nextProviderConfigs[embeddingProviderId] || {}),
        ...embeddingProviderConfig,
        embedding_model:
          embeddingProviderConfig.embedding_model
          || embeddingProviderConfig.model
          || configDraft.embedding_model
          || '',
        model:
          embeddingProviderConfig.embedding_model
          || embeddingProviderConfig.model
          || configDraft.embedding_model
          || '',
      };
    }

    if (llmProviderId) {
      nextProviderConfigs[llmProviderId] = {
        ...(nextProviderConfigs[llmProviderId] || {}),
        ...llmProviderConfig,
        llm_model:
          configDraft.llm_model
          || llmProviderConfig.llm_model
          || llmProviderConfig.model
          || '',
        base_url: llmProviderConfig.base_url ?? configDraft.llm_base_url,
        api_key: llmProviderConfig.api_key ?? configDraft.llm_api_key ?? null,
      };
    }

    return {
      host: configDraft.host,
      port: Number(configDraft.port),
      http_port: Number(configDraft.http_port),
      debug: Boolean(configDraft.debug),
      vector_db_type: configDraft.vector_db_type,
      chroma_persist_directory: configDraft.chroma_persist_directory,
      knowledge_base_db_path: configDraft.knowledge_base_db_path,
      qdrant_url: configDraft.qdrant_url,
      embedding_provider: embeddingProviderId,
      embedding_fallback_provider: canonicalProviderId(configDraft.embedding_fallback_provider) || null,
      embedding_device: configDraft.embedding_device,
      embedding_cache_dir: configDraft.embedding_cache_dir || null,
      llm_provider: llmProviderId,
      llm_fallback_provider: canonicalProviderId(configDraft.llm_fallback_provider) || null,
      llm_model: configDraft.llm_model || llmProviderConfig.llm_model || llmProviderConfig.model || '',
      llm_base_url: llmProviderConfig.base_url ?? configDraft.llm_base_url,
      llm_api_key: llmProviderConfig.api_key ?? configDraft.llm_api_key ?? null,
      enable_thinking: Boolean(configDraft.enable_thinking),
      enable_llm_summary: Boolean(configDraft.enable_llm_summary),
      enable_reranker: Boolean(configDraft.enable_reranker),
      enable_cache: Boolean(configDraft.enable_cache),
      max_retrieval_results: Number(configDraft.max_retrieval_results),
      similarity_threshold: Number(configDraft.similarity_threshold),
      cache: {
        enabled: Boolean(configDraft.cache.enabled),
        max_entries: Number(configDraft.cache.max_entries),
        ttl_seconds: Number(configDraft.cache.ttl_seconds),
      },
      security: {
        enabled: Boolean(configDraft.security.enabled),
        allow_anonymous: Boolean(configDraft.security.allow_anonymous),
        api_keys: splitKeys(configDraft.security.api_keys_text),
        tenant_api_keys: parseJsonOr({}, configDraft.security.tenant_api_keys_text),
      },
      rate_limit: {
        requests_per_window: Number(configDraft.rate_limit.requests_per_window),
        window_seconds: Number(configDraft.rate_limit.window_seconds),
        burst: Number(configDraft.rate_limit.burst),
      },
      quotas: {
        max_upload_files: Number(configDraft.quotas.max_upload_files),
        max_upload_bytes: Number(configDraft.quotas.max_upload_bytes),
        max_upload_file_bytes: Number(configDraft.quotas.max_upload_file_bytes),
        max_index_documents: Number(configDraft.quotas.max_index_documents),
        max_index_chunks: Number(configDraft.quotas.max_index_chunks),
        max_index_chars: Number(configDraft.quotas.max_index_chars),
      },
      observability: {
        warning_error_rate: Number(configDraft.observability.warning_error_rate),
        critical_error_rate: Number(configDraft.observability.critical_error_rate),
        slow_request_ms: Number(configDraft.observability.slow_request_ms),
        latency_window_size: Number(configDraft.observability.latency_window_size),
      },
      provider_budget: {
        enabled: Boolean(configDraft.provider_budget.enabled),
        embeddings: {
          requests_per_window: Number(configDraft.provider_budget.embeddings.requests_per_window),
          window_seconds: Number(configDraft.provider_budget.embeddings.window_seconds),
          burst: Number(configDraft.provider_budget.embeddings.burst),
          failure_threshold: Number(configDraft.provider_budget.embeddings.failure_threshold),
          cooldown_seconds: Number(configDraft.provider_budget.embeddings.cooldown_seconds),
        },
        llm: {
          requests_per_window: Number(configDraft.provider_budget.llm.requests_per_window),
          window_seconds: Number(configDraft.provider_budget.llm.window_seconds),
          burst: Number(configDraft.provider_budget.llm.burst),
          failure_threshold: Number(configDraft.provider_budget.llm.failure_threshold),
          cooldown_seconds: Number(configDraft.provider_budget.llm.cooldown_seconds),
        },
      },
      provider_configs: nextProviderConfigs,
    };
  }

  function updateEmbeddingProviderConfig(field, value) {
    const providerName = canonicalProviderId(configDraft.embedding_provider);
    if (!providerName) return;
    const providerConfigs = normalizeProviderConfigs(parseJsonOr({}, configDraft.provider_configs_text));
    providerConfigs[providerName] = {
      ...(providerConfigs[providerName] || {}),
      [field]: value,
    };
    configDraft = {
      ...configDraft,
      provider_configs_text: JSON.stringify(providerConfigs, null, 2),
      [field === 'base_url' ? 'embedding_base_url' : field === 'model' ? 'embedding_model' : 'embedding_api_key']: value,
    };
  }

  function syncEmbeddingProviderDraft(providerName) {
    const canonicalId = canonicalProviderId(providerName);
    const providerConfigs = normalizeProviderConfigs(parseJsonOr({}, configDraft.provider_configs_text));
    const activeProvider = providerConfigs?.[canonicalId] ?? {};
    configDraft = {
      ...configDraft,
      embedding_base_url: activeProvider.base_url ?? providerDefinition(canonicalId)?.defaults?.base_url ?? '',
      embedding_model: activeProvider.embedding_model ?? activeProvider.model ?? '',
      embedding_api_key: activeProvider.api_key ?? '',
    };
  }

  function syncLlmProviderDraft(providerName) {
    const canonicalId = canonicalProviderId(providerName);
    const providerConfigs = normalizeProviderConfigs(parseJsonOr({}, configDraft.provider_configs_text));
    const activeProvider = providerConfigs?.[canonicalId] ?? {};
    const provider = providerDefinition(canonicalId);
    const defaultModel = provider?.models?.chat?.[0]?.value ?? '';
    configDraft = {
      ...configDraft,
      llm_base_url: activeProvider.base_url ?? provider?.defaults?.base_url ?? '',
      llm_api_key: activeProvider.api_key ?? '',
      llm_model: activeProvider.llm_model ?? activeProvider.model ?? configDraft.llm_model ?? defaultModel,
    };
  }

  function providerConfigField(providerName, field, fallback = '') {
    const canonicalId = canonicalProviderId(providerName);
    const providerConfigs = normalizeProviderConfigs(parseJsonOr({}, configDraft.provider_configs_text));
    return providerConfigs?.[canonicalId]?.[field] ?? fallback;
  }

  function updateNamedProviderConfig(providerName, field, value) {
    const canonicalId = canonicalProviderId(providerName);
    const providerConfigs = normalizeProviderConfigs(parseJsonOr({}, configDraft.provider_configs_text));
    providerConfigs[canonicalId] = {
      ...(providerConfigs[canonicalId] || {}),
      [field]: value,
    };
    configDraft = {
      ...configDraft,
      provider_configs_text: JSON.stringify(providerConfigs, null, 2),
    };

    if (canonicalId === canonicalProviderId(configDraft.embedding_provider)) {
      syncEmbeddingProviderDraft(canonicalId);
    }
    if (canonicalId === canonicalProviderId(configDraft.llm_provider)) {
      syncLlmProviderDraft(canonicalId);
    }
  }

  function updateProviderModel(providerName, family, value) {
    const field = providerModelField(family);
    updateNamedProviderConfig(providerName, field, value);
    if (family === 'embedding') {
      updateNamedProviderConfig(providerName, 'model', value);
      if (providerName === configDraft.embedding_provider) {
        configDraft = {
          ...configDraft,
          embedding_model: value,
        };
      }
      return;
    }

    if (providerName === configDraft.llm_provider) {
      configDraft = {
        ...configDraft,
        llm_model: value,
      };
    }
  }

  function providerCustomModels(providerName, family) {
    const providerConfigs = normalizeProviderConfigs(parseJsonOr({}, configDraft.provider_configs_text));
    const field = family === 'chat' ? 'chat_models' : 'embedding_models';
    const canonicalId = canonicalProviderId(providerName);
    return Array.isArray(providerConfigs?.[canonicalId]?.[field]) ? providerConfigs[canonicalId][field] : [];
  }

  function providerSyncedModels(providerName, family) {
    const providerConfigs = normalizeProviderConfigs(parseJsonOr({}, configDraft.provider_configs_text));
    const field = family === 'chat' ? 'chat_models_synced' : 'embedding_models_synced';
    const canonicalId = canonicalProviderId(providerName);
    return Array.isArray(providerConfigs?.[canonicalId]?.[field]) ? providerConfigs[canonicalId][field] : [];
  }

  function updateProviderCustomModels(providerName, family, models) {
    const field = family === 'chat' ? 'chat_models' : 'embedding_models';
    updateNamedProviderConfig(
      providerName,
      field,
      Array.from(new Set((models || []).map((item) => String(item || '').trim()).filter(Boolean)))
    );
  }

  function replaceProviderSyncedModels(providerName, family, models) {
    const field = family === 'chat' ? 'chat_models_synced' : 'embedding_models_synced';
    const nextModels = Array.from(new Set((models || []).map((item) => String(item || '').trim()).filter(Boolean)));
    updateNamedProviderConfig(providerName, field, nextModels);
  }

  function customModelDraftKey(providerName) {
    return canonicalProviderId(providerName);
  }

  function addCustomProviderModel(providerName) {
    const canonicalId = canonicalProviderId(providerName);
    const key = customModelDraftKey(canonicalId);
    const draft = String(customModelDrafts[key] || '').trim();
    if (!draft) {
      pushToast('模型 ID 为空', '请输入要手动添加的模型 ID。', 'warning');
      return;
    }
    const provider = providerDefinition(canonicalId);
    const family = inferModelFamily(draft, provider?.families || ['chat']);
    const nextModels = [...providerCustomModels(providerName, family), draft];
    updateProviderCustomModels(providerName, family, nextModels);
    customModelDrafts = {
      ...customModelDrafts,
      [key]: '',
    };
    pushToast('模型已添加', `${draft} 已加入 ${canonicalId} 的${modelFamilyMeta[family].label}列表。`, 'success');
  }

  function removeCustomProviderModel(providerName, family, modelId) {
    const nextModels = providerCustomModels(providerName, family).filter((item) => item !== modelId);
    updateProviderCustomModels(providerName, family, nextModels);
  }

  async function fetchProviderModels(providerName, family = null) {
    providerName = canonicalProviderId(providerName);
    const provider = providerDefinition(providerName);
    const families = family ? [family] : (provider?.families || []);
    providerFetchBusy = {
      ...providerFetchBusy,
      [customModelDraftKey(providerName)]: true,
    };
    try {
      const payload = await api.providerModels({
        provider: providerName,
        family,
        ...identity,
      });
      const models = Array.isArray(payload?.models) ? payload.models : [];
      const nextFamilies = family
        ? {
            [family]: models.map((model) => ({
              ...model,
              family: family,
            })),
          }
        : families.reduce((acc, item) => {
            acc[item] = models.filter((model) => {
              const resolvedFamily = model?.family || inferModelFamily(model?.id, families);
              return resolvedFamily === item;
            }).map((model) => ({
              ...model,
              family: model?.family || inferModelFamily(model?.id, families),
            }));
            return acc;
          }, {});
      for (const modelFamily of Object.keys(nextFamilies)) {
        replaceProviderSyncedModels(
          providerName,
          modelFamily,
          (nextFamilies[modelFamily] || []).map((model) => model.id)
        );
      }
      if (family) {
        pushToast('模型列表已获取', `${providerName} 返回 ${models.length} 个${modelFamilyMeta[family].label}模型。`, 'success');
      } else {
        pushToast(
          '模型列表已获取',
          `${providerName} 共返回 ${models.length} 个模型，已按类型写入列表。`,
          'success'
        );
      }
    } catch (error) {
      pushToast('获取模型列表失败', error.message, 'warning');
    } finally {
      providerFetchBusy = {
        ...providerFetchBusy,
        [customModelDraftKey(providerName)]: false,
      };
    }
  }

  function selectEmbeddingProvider(providerName) {
    providerName = canonicalProviderId(providerName);
    const provider = providerDefinition(providerName);
    const defaultModel = provider?.local ? providerName : (provider?.models?.embedding?.[0]?.value ?? '');
    configDraft = {
      ...configDraft,
      embedding_provider: providerName,
      embedding_model: providerModelValue(providerName, 'embedding') || defaultModel,
    };
    syncEmbeddingProviderDraft(providerName);
  }

  function selectLlmProvider(providerName) {
    providerName = canonicalProviderId(providerName);
    const provider = providerDefinition(providerName);
    const defaultModel = provider?.models?.chat?.[0]?.value ?? '';
    configDraft = {
      ...configDraft,
      llm_provider: providerName,
      llm_model: providerModelValue(providerName, 'chat') || defaultModel,
    };
    syncLlmProviderDraft(providerName);
  }

  function buildRequestContext() {
    return {
      userId: identity.userId,
      agentId: identity.agentId,
      apiKey: identity.apiKey,
    };
  }

  async function refreshKnowledgeBases({ silent = false } = {}) {
    try {
      const response = await api.knowledgeBases(buildRequestContext());
      const nextKnowledgeBases = Array.isArray(response?.knowledge_bases) ? response.knowledge_bases : [];
      knowledgeBases = nextKnowledgeBases;
      const defaultId = knowledgeBases[0] ? String(knowledgeBases[0].id) : '';
      if (!knowledgeBaseById(selectedKnowledgeBase)) {
        setKnowledgeBase(defaultId);
      }
      if (!knowledgeBaseById(documentKnowledgeBase)) documentKnowledgeBase = defaultId;
      if (!knowledgeBaseById(searchKnowledgeBase)) searchKnowledgeBase = defaultId;
      if (!knowledgeBaseById(chatKnowledgeBase)) chatKnowledgeBase = defaultId;
      if (!knowledgeBaseById(manageKnowledgeBase)) manageKnowledgeBase = defaultId;
      if (!silent) pushToast('知识库已刷新', `当前可用知识库 ${knowledgeBases.length} 个`, 'success');
    } catch (error) {
      if (!silent) pushToast('知识库加载失败', error.message, 'warning');
    }
  }

  async function refreshOverview() {
    overviewBusy = true;
    try {
      const [healthResult, readyResult, metricsResult, configResult, collectionsResult] = await Promise.allSettled([
        api.health(identity),
        api.ready(identity),
        api.metrics(identity),
        api.config(identity),
        api.knowledgeBases(buildRequestContext()),
      ]);

      if (healthResult.status === 'fulfilled') health = healthResult.value;
      if (readyResult.status === 'fulfilled') ready = readyResult.value;
      if (metricsResult.status === 'fulfilled') metrics = metricsResult.value?.metrics ?? metricsResult.value;
      if (configResult.status === 'fulfilled') {
        config = configResult.value;
        syncConfigDraft(configResult.value);
      }
      if (collectionsResult.status === 'fulfilled') {
        knowledgeBases = collectionsResult.value?.knowledge_bases ?? [];
      }
      lastRefreshAt = new Date();
    } finally {
      overviewBusy = false;
    }
  }

  async function refreshConfig() {
    configBusy = true;
    try {
      config = await api.config(identity);
      syncConfigDraft(config);
      lastRefreshAt = new Date();
    } catch (error) {
      pushToast('配置加载失败', error.message, 'warning');
    } finally {
      configBusy = false;
    }
  }

  async function refreshMcpTools({ silent = false } = {}) {
    try {
      const response = await api.mcpTools(identity);
      mcpTools = Array.isArray(response?.tools) ? response.tools : [];
      ensureMcpToolSelected();
      if (!silent) pushToast('MCP 工具已刷新', `当前可用工具 ${mcpTools.length} 个`, 'success');
    } catch (error) {
      if (!silent) pushToast('MCP 工具加载失败', error.message, 'warning');
    }
  }

  async function runMcpTool() {
    const selected = currentMcpTool();
    if (!selected) {
      pushToast('没有可用工具', '请先刷新 MCP 工具列表。', 'warning');
      return;
    }

    let argumentsPayload = null;
    try {
      argumentsPayload = parseJsonOr(null, mcpArguments);
      if (!argumentsPayload || typeof argumentsPayload !== 'object' || Array.isArray(argumentsPayload)) {
        throw new Error('参数必须是 JSON 对象');
      }
    } catch (error) {
      pushToast('参数格式错误', error.message || '请输入合法 JSON。', 'error');
      return;
    }

    mcpBusy = true;
    try {
      const result = await api.mcpCall({
        tool: selected.name,
        arguments: argumentsPayload,
        api_key: identity.apiKey || null,
      });
      mcpResult = prettyJson(result);
      pushToast('MCP 调用完成', selected.name, 'success');
    } catch (error) {
      mcpResult = prettyJson(error.payload || { message: error.message });
      pushToast('MCP 调用失败', error.message, 'error');
    } finally {
      mcpBusy = false;
    }
  }

  async function refreshDocuments() {
    manageBusy = true;
    try {
      const [documentsResult, filesResult] = await Promise.allSettled([
        api.listDocuments({
          ...knowledgeBaseRequest(manageKnowledgeBase),
          limit: managedPageSize,
          offset: managedPage * managedPageSize,
          filename: fileFilter || undefined,
          ...identity,
        }),
        api.listFiles({
          ...knowledgeBaseRequest(manageKnowledgeBase),
          ...identity,
        }),
      ]);

      if (documentsResult.status === 'fulfilled') {
        documents = documentsResult.value?.documents ?? [];
        documentsTotal = Number(documentsResult.value?.total ?? documents.length);
      }
      if (filesResult.status === 'fulfilled') {
        files = filesResult.value?.files ?? [];
        filesTotal = files.length;
      }
      lastRefreshAt = new Date();
    } catch (error) {
      pushToast('文档加载失败', error.message, 'warning');
    } finally {
      manageBusy = false;
    }
  }

  async function refreshDocumentsSection() {
    if (documentMode === 'manage') {
      await refreshDocuments();
    } else {
      await refreshKnowledgeBases({ silent: true });
    }
  }

  async function refreshAll() {
    loading = true;
    try {
      await Promise.all([
        refreshKnowledgeBases({ silent: true }),
        refreshOverview(),
      ]);
      if (activeSection !== 'overview') {
        await refreshActiveView();
      }
    } finally {
      loading = false;
    }
  }

  async function uploadQueuedFiles() {
    if (queuedFiles.length === 0) {
      pushToast('没有文件', '请先拖拽或选择要上传的文件。', 'warning');
      return;
    }
    uploadBusy = true;
    try {
      const payload = await api.uploadFiles({
        files: queuedFiles,
        ...knowledgeBaseRequest(documentKnowledgeBase),
        ...identity,
      });
      const successful = Number(payload?.successful ?? 0);
      const failed = Number(payload?.failed ?? 0);
      const failureSummary = summarizeUploadFailures(payload);
      if (successful > 0) {
        queuedFiles = [];
      }
      if (failed > 0 && successful === 0) {
        pushToast('上传失败', failureSummary || `共 ${failed} 个文件处理失败。`, 'error');
      } else if (failed > 0) {
        pushToast('部分上传成功', `成功 ${successful} 个，失败 ${failed} 个。${failureSummary}`, 'warning');
      } else {
        pushToast('上传完成', `成功处理 ${successful} 个文件。`, 'success');
      }
      await refreshKnowledgeBases({ silent: true });
      if (documentMode === 'manage') await refreshDocuments();
    } catch (error) {
      pushToast('上传失败', error.message, 'error');
    } finally {
      uploadBusy = false;
    }
  }

  function addFiles(fileList) {
    const incoming = Array.from(fileList || []);
    const seen = new Set(queuedFiles.map((file) => `${file.name}:${file.size}:${file.lastModified}`));
    const merged = [...queuedFiles];
    for (const file of incoming) {
      const key = `${file.name}:${file.size}:${file.lastModified}`;
      if (!seen.has(key)) {
        merged.push(file);
        seen.add(key);
      }
    }
    queuedFiles = merged;
  }

  function handleDrop(event) {
    addFiles(event.dataTransfer?.files);
  }

  function openFilePicker() {
    document.getElementById('file-input')?.click();
  }

  function handleDropzoneKeydown(event) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      openFilePicker();
    }
  }

  function removeQueuedFile(index) {
    queuedFiles = queuedFiles.filter((_, current) => current !== index);
  }

  async function addTextDocument() {
    if (!documentContent.trim()) {
      pushToast('请输入内容', '文本输入不能为空。', 'warning');
      return;
    }
    addBusy = true;
    try {
      await api.addDocument({
        content: documentContent.trim(),
        ...knowledgeBaseRequest(documentKnowledgeBase),
        metadata: {
          title: documentTitle.trim() || undefined,
          source: 'manual_input',
          timestamp: new Date().toISOString(),
        },
        user_id: identity.userId ? Number(identity.userId) : null,
        agent_id: identity.agentId ? Number(identity.agentId) : null,
        api_key: identity.apiKey || null,
      });
      documentContent = '';
      documentTitle = '';
      pushToast('文档已添加', '手工录入内容已保存。', 'success');
      await refreshKnowledgeBases({ silent: true });
      if (documentMode === 'manage') await refreshDocuments();
    } catch (error) {
      pushToast('添加失败', error.message, 'error');
    } finally {
      addBusy = false;
    }
  }

  async function runSearch() {
    if (!searchQuery.trim()) {
      pushToast('请输入关键词', '搜索框不能为空。', 'warning');
      return;
    }
    searchBusy = true;
    try {
      const result = await api.search({
        query: searchQuery.trim(),
        ...knowledgeBaseRequest(searchKnowledgeBase),
        limit: Number(searchLimit || 5),
        ...identity,
      });
      searchResults = result?.results ?? [];
      searchSummary = result?.summary ?? '';
      pushToast('搜索完成', `找到 ${searchResults.length} 条结果。`, 'success');
    } catch (error) {
      pushToast('搜索失败', error.message, 'error');
    } finally {
      searchBusy = false;
    }
  }

  async function sendChat() {
    if (!chatInput.trim()) {
      pushToast('请输入问题', '对话消息不能为空。', 'warning');
      return;
    }
    const question = chatInput.trim();
    chatMessages = [
      ...chatMessages,
      {
        role: 'user',
        content: question,
      },
    ];
    chatInput = '';
    chatBusy = true;
    try {
      const result = await api.chat({
        query: question,
        ...knowledgeBaseRequest(chatKnowledgeBase),
        user_id: identity.userId ? Number(identity.userId) : null,
        agent_id: identity.agentId ? Number(identity.agentId) : null,
        api_key: identity.apiKey || null,
      });
      chatMessages = [
        ...chatMessages,
        {
          role: 'assistant',
          content: result?.response ?? '',
          sources: result?.sources ?? [],
        },
      ];
    } catch (error) {
      chatMessages = [
        ...chatMessages,
        {
          role: 'assistant',
          content: `请求失败: ${error.message}`,
          sources: [],
        },
      ];
    } finally {
      chatBusy = false;
    }
  }

  async function nextPage() {
    managedPage += 1;
    await refreshDocuments();
  }

  async function prevPage() {
    if (managedPage === 0) return;
    managedPage -= 1;
    await refreshDocuments();
  }

  async function deleteDocument(documentId) {
    if (!confirm('确定要删除这个文档吗？')) return;
    try {
      await api.deleteDocument({
        document_id: documentId,
        ...knowledgeBaseRequest(manageKnowledgeBase),
        user_id: identity.userId ? Number(identity.userId) : null,
        agent_id: identity.agentId ? Number(identity.agentId) : null,
        api_key: identity.apiKey || null,
      });
      pushToast('文档已删除', `文档 ${documentId} 已移除。`, 'success');
      await refreshDocuments();
    } catch (error) {
      pushToast('删除失败', error.message, 'error');
    }
  }

  async function deleteFile(filename) {
    if (!confirm(`确定要删除文件 "${filename}" 及其片段吗？`)) return;
    try {
      await api.deleteFile({
        filename,
        ...knowledgeBaseRequest(manageKnowledgeBase),
        user_id: identity.userId ? Number(identity.userId) : null,
        agent_id: identity.agentId ? Number(identity.agentId) : null,
        api_key: identity.apiKey || null,
      });
      pushToast('文件已删除', `${filename} 已移除。`, 'success');
      await refreshDocuments();
    } catch (error) {
      pushToast('删除失败', error.message, 'error');
    }
  }

  function selectFileFilter(filename) {
    fileFilter = filename;
    managedPage = 0;
    documentMode = 'manage';
    refreshDocuments();
  }

  function clearFileFilter() {
    fileFilter = '';
    managedPage = 0;
    refreshDocuments();
  }

  function openDocumentPreview(doc) {
    previewDocument = doc;
  }

  function closeDocumentPreview() {
    previewDocument = null;
  }

  async function saveConfig() {
    configBusy = true;
    try {
      let updates = buildConfigPayloadFromDraft();
      if (configMode === 'advanced') {
        updates = parseJsonOr(null, configDraft.full_config_text);
        if (!updates || typeof updates !== 'object' || Array.isArray(updates)) {
          throw new Error('高级配置必须是 JSON 对象');
        }
      }
      await api.updateConfig(updates, identity);
      pushToast('配置已保存', '后台已重新加载运行时。', 'success');
      await refreshOverview();
    } catch (error) {
      pushToast('保存失败', error.message, 'error');
    } finally {
      configBusy = false;
    }
  }

  async function resetConfig() {
    if (!confirm('确定要重置所有配置为默认值吗？')) return;
    try {
      await api.resetConfig(identity);
      pushToast('配置已重置', '默认配置已恢复。', 'success');
      await refreshOverview();
    } catch (error) {
      pushToast('重置失败', error.message, 'error');
    }
  }

  async function reloadConfig() {
    try {
      await api.reloadConfig(identity);
      pushToast('配置已重新加载', '已从磁盘刷新。', 'success');
      await refreshOverview();
    } catch (error) {
      pushToast('重新加载失败', error.message, 'error');
    }
  }

  function collectionSummary() {
    return `${knowledgeBases.length} 个知识库`;
  }

  function readyStatus() {
    return Boolean(ready?.ready);
  }

  function healthStatus() {
    return Boolean(health?.healthy ?? health?.ready);
  }

  function metricSummary() {
    const total = metrics?.total_requests ?? 0;
    const errors = metrics?.error_count ?? metrics?.errors ?? 0;
    const rate = total ? errors / total : 0;
    return `${total} 次请求，错误率 ${pct(rate)}`;
  }

  function refreshActiveView() {
    if (activeSection === 'overview') return refreshOverview();
    if (activeSection === 'documents') return refreshDocumentsSection();
    if (activeSection === 'mcp') return refreshMcpTools({ silent: true });
    return refreshConfig();
  }

  function operationRows() {
    const ops = metrics?.operations || {};
    return Object.entries(ops)
      .map(([name, stats]) => ({ name, stats }))
      .sort((left, right) => right.stats.count - left.stats.count);
  }

  function providerRows() {
    const providers = metrics?.providers || {};
    return Object.entries(providers)
      .map(([name, stats]) => ({ name, stats }))
      .sort((left, right) => right.stats.count - left.stats.count);
  }

  function runtimeRows() {
    const runtime = ready?.runtime || health?.runtime || {};
    return [
      { label: 'Embedding', data: runtime.embedding_model },
      { label: 'LLM', data: runtime.llm_model },
      { label: 'Vector Store', data: runtime.vector_store },
      { label: 'Security', data: runtime.security },
      { label: 'Rate Limit', data: runtime.rate_limit },
      { label: 'Provider Budget', data: runtime.provider_budget },
    ].filter((row) => row.data);
  }

  onMount(() => {
    readState();
    syncLocation(activeSection, { replace: true });
    const handlePopState = () => {
      const routedSection = sectionFromLocation();
      if (routedSection) {
        activeSection = routedSection;
        writeState();
      }
      void refreshActiveView();
    };
    window.addEventListener('popstate', handlePopState);
    void (async () => {
      try {
        await refreshAll();
      } finally {
        loading = false;
      }
    })();
    return () => {
      window.removeEventListener('popstate', handlePopState);
    };
  });
</script>

<svelte:head>
  <title>MCP-RAG 控制台</title>
  <meta
    name="description"
    content="MCP-RAG admin dashboard for documents, config, and service status."
  />
</svelte:head>

{#if loading}
  <div class="toast-stack" aria-live="polite">
    <div class="toast info">
      <strong>加载中</strong>
      <div class="muted">正在读取服务状态。</div>
    </div>
  </div>
{/if}

<div class="toast-stack" aria-live="polite">
  {#each toasts as toast (toast.id)}
    <div class="toast {toast.tone}">
      <strong>{toast.title}</strong>
      <div class="muted">{toast.message}</div>
    </div>
  {/each}
</div>

<div class="app-shell">
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">MR</div>
      <div class="brand-copy">
        <h1>MCP-RAG 控制台</h1>
        <p>统一后台</p>
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">导航</div>
      <div class="nav-list">
        {#each sections as section}
          {#if section.id === 'documents'}
            <div class="nav-group {activeSection === 'documents' ? 'active' : ''}">
              <button
                class="nav-item {activeSection === 'documents' ? 'active' : ''}"
                on:click={() => switchSection('documents')}
              >
                <span class="nav-title">
                  <strong>{section.title}</strong>
                  <span>{section.subtitle}</span>
                </span>
              </button>
              {#if activeSection === 'documents'}
                <div class="nav-sublist">
                  {#each documentModes as mode}
                    <button
                      class="nav-subitem {activeSection === 'documents' && documentMode === mode.id ? 'active' : ''}"
                      on:click={() => {
                        activeSection = 'documents';
                        switchDocumentMode(mode.id);
                        syncLocation('documents');
                        writeState();
                        void refreshDocumentsSection();
                      }}
                    >
                      {mode.title}
                    </button>
                  {/each}
                </div>
              {/if}
            </div>
          {:else if section.id === 'config'}
            <div class="nav-group {activeSection === 'config' ? 'active' : ''}">
              <button
                class="nav-item {activeSection === 'config' ? 'active' : ''}"
                on:click={() => switchSection('config')}
              >
                <span class="nav-title">
                  <strong>{section.title}</strong>
                  <span>{section.subtitle}</span>
                </span>
              </button>
              {#if activeSection === 'config'}
                <div class="nav-sublist">
                  {#each configModes as mode}
                    <button
                      class="nav-subitem {activeSection === 'config' && configMode === mode.id ? 'active' : ''}"
                      on:click={() => {
                        activeSection = 'config';
                        switchConfigMode(mode.id);
                        syncLocation('config');
                        writeState();
                        void refreshConfig();
                      }}
                    >
                      {mode.title}
                    </button>
                  {/each}
                </div>
              {/if}
            </div>
          {:else}
            <button
              class="nav-item {activeSection === section.id ? 'active' : ''}"
              on:click={() => switchSection(section.id)}
            >
              <span class="nav-title">
                <strong>{section.title}</strong>
                <span>{section.subtitle}</span>
              </span>
            </button>
          {/if}
        {/each}
      </div>
    </div>

    <div class="sidebar-section">
      <div class="sidebar-label">请求上下文</div>
      <div class="field">
        <div class="field-label">Knowledge Base</div>
        <SelectField
          bind:value={selectedKnowledgeBase}
          options={knowledgeBaseOptions()}
          ariaLabel="选择知识库"
          on:change={() => setKnowledgeBase(selectedKnowledgeBase)}
        />
      </div>
      <div class="field">
        <div class="field-label">User ID</div>
        <input bind:value={identity.userId} placeholder="1001" />
      </div>
      <div class="field">
        <div class="field-label">Agent ID</div>
        <input bind:value={identity.agentId} placeholder="50" />
      </div>
      <div class="field">
        <div class="field-label">API Key</div>
        <input bind:value={identity.apiKey} placeholder="可选" />
      </div>
      <div class="field-help">这些值会附带到请求里，用来列出并访问公共库或当前用户的 agent 私有知识库。</div>
    </div>
  </aside>

  <main class="content {activeSection === 'config' && configMode === 'provider' ? 'content--provider' : ''}">
    {#if activeSection === 'overview'}
      <PageShell title="总览" subtitle="状态、请求、集合、版本。">
        <svelte:fragment slot="actions">
          <div class="card-actions">
            <button class="button secondary" on:click={refreshOverview} disabled={overviewBusy}>
              {overviewBusy ? '刷新中...' : '刷新总览'}
            </button>
            <span class="pill info"><strong>{collectionSummary()}</strong><span>集合</span></span>
            <span class="pill good"><strong>{healthStatus() ? '正常' : '降级'}</strong><span>健康</span></span>
            <span class="pill {readyStatus() ? 'good' : 'bad'}"><strong>{readyStatus() ? '就绪' : '未就绪'}</strong><span>状态</span></span>
          </div>
        </svelte:fragment>

        <div class="grid-4">
          <div class="card">
            <div class="metric">
              <div class="metric-label">请求总数</div>
              <div class="metric-value">{metrics?.total_requests ?? 0}</div>
              <div class="metric-meta">{metricSummary()}</div>
            </div>
          </div>
          <div class="card">
            <div class="metric">
              <div class="metric-label">平均延迟</div>
              <div class="metric-value">{(metrics?.average_latency_ms ?? 0).toFixed(1)} ms</div>
              <div class="metric-meta">P50/P95/P99 见下方指标明细</div>
            </div>
          </div>
          <div class="card">
            <div class="metric">
              <div class="metric-label">可用知识库</div>
              <div class="metric-value">{knowledgeBases.length}</div>
              <div class="metric-meta">选中: {knowledgeBaseLabel(knowledgeBaseById(selectedKnowledgeBase))}</div>
            </div>
          </div>
          <div class="card">
            <div class="metric">
              <div class="metric-label">配置版本</div>
              <div class="metric-value">{ready?.config_revision ?? health?.config_revision ?? '当前'}</div>
              <div class="metric-meta">当前配置版本</div>
            </div>
          </div>
        </div>

        <div class="grid-3">
          <div class="card">
            <div class="metric">
              <div class="metric-label">健康</div>
              <div class="metric-value">{safeText(health?.status, healthStatus() ? '正常' : '未知')}</div>
              <div class="metric-meta">{health?.reasons?.length ? health.reasons.join(' · ') : '无异常原因'}</div>
            </div>
          </div>
          <div class="card">
            <div class="metric">
              <div class="metric-label">就绪</div>
              <div class="metric-value">{ready?.ready ? '是' : '否'}</div>
              <div class="metric-meta">{ready?.ready ? '依赖满足' : '依赖检查未通过'}</div>
            </div>
          </div>
          <div class="card">
            <div class="metric">
              <div class="metric-label">错误率</div>
              <div class="metric-value">{pct((metrics?.error_count ?? 0) / Math.max(metrics?.total_requests ?? 1, 1))}</div>
              <div class="metric-meta">总请求 {metrics?.total_requests ?? 0}</div>
            </div>
          </div>
        </div>

        <div class="grid-2">
          <PanelCard title="运行时详情" subtitle="依赖状态。">
            <div class="status-stack">
              {#each runtimeRows() as row}
                <div class="status-row">
                  <div>
                    <strong>{row.label}</strong>
                    <div class="muted">
                      {safeText(row.data?.provider || row.data?.name || row.data?.reason || row.data?.status, '未知')}
                    </div>
                  </div>
                  <span class="status-badge {statusTone(row.data?.status || row.data?.state || 'info')}">
                    {safeText(row.data?.status || row.data?.state, '未知')}
                  </span>
                </div>
              {/each}
            </div>
          </PanelCard>

          <PanelCard title="Provider 健康" subtitle="延迟和错误。">
            <div class="status-stack">
              {#each providerRows() as provider}
                <div class="status-row">
                  <div>
                    <strong>{provider.name}</strong>
                    <div class="muted">
                      请求 {provider.stats.count} · 错误 {provider.stats.error_count} · p95 {provider.stats.p95_latency_ms.toFixed(1)} ms
                    </div>
                  </div>
                  <span class="status-badge {provider.stats.error_count > 0 ? 'warn' : 'good'}">
                    {provider.stats.last_latency_ms.toFixed(1)} ms
                  </span>
                </div>
              {:else}
                <div class="empty-state">尚无 provider 观测数据。</div>
              {/each}
            </div>
          </PanelCard>
        </div>

        <PanelCard title="请求指标" subtitle="计数和分位数。">
          <div class="table">
            <div class="table-row header">
              <span>操作</span>
              <span>请求 / 错误</span>
              <span>P50 / P95 / P99</span>
            </div>
            {#each operationRows() as row}
              <div class="table-row">
                <span>{row.name}</span>
                <span>{row.stats.count} / {row.stats.error_count}</span>
                <span>{row.stats.p50_latency_ms.toFixed(1)} / {row.stats.p95_latency_ms.toFixed(1)} / {row.stats.p99_latency_ms.toFixed(1)} ms</span>
              </div>
            {:else}
              <div class="empty-state">还没有操作数据。使用文档页或配置页后这里会开始积累。</div>
            {/each}
          </div>
        </PanelCard>
      </PageShell>
    {/if}

    {#if activeSection === 'documents'}
      <PageShell title="文档管理" subtitle="导入、检索、删除。">
        {#if documentMode === 'ingest'}
          <div class="grid-2">
            <PanelCard title="文件上传" subtitle="上传后自动切片入库。">
              <svelte:fragment slot="actions">
                <button class="button secondary" on:click={openFilePicker}>选择文件</button>
                <button class="button primary" on:click={uploadQueuedFiles} disabled={uploadBusy || queuedFiles.length === 0}>
                  {uploadBusy ? '上传中...' : '开始上传'}
                </button>
              </svelte:fragment>

                <div
                  class="dropzone"
                  role="button"
                  tabindex="0"
                  on:dragover|preventDefault={() => {}}
                  on:drop|preventDefault={handleDrop}
                  on:click={openFilePicker}
                  on:keydown={handleDropzoneKeydown}
                >
                  <h3>拖拽文件到这里</h3>
                  <p>支持 `txt`、`md`、`pdf`、`docx`。静态站点会把请求发给当前同域后端。</p>
                  <input id="file-input" class="hidden-input" type="file" multiple accept=".txt,.md,.pdf,.docx" on:change={(event) => addFiles(event.currentTarget.files)} />
                </div>

                <div class="field mt-16">
                  <div class="field-label">目标集合</div>
                  <SelectField bind:value={documentKnowledgeBase} options={knowledgeBaseOptions()} ariaLabel="选择目标知识库" />
                </div>

                <div class="file-list mt-16">
                  {#each queuedFiles as file, index}
                    <div class="file-chip">
                      <div>
                        <strong>{file.name}</strong>
                        <div class="meta">{toBytes(file.size)} · {file.type || 'unknown'}</div>
                      </div>
                      <button class="button ghost" on:click={() => removeQueuedFile(index)}>移除</button>
                    </div>
                  {:else}
                    <div class="empty-state">还没有选中文件。</div>
                  {/each}
                </div>
            </PanelCard>

            <PanelCard title="手工录入" subtitle="快速补充短文本。">
                <div class="field">
                  <div class="field-label">标题</div>
                  <input bind:value={documentTitle} placeholder="例如：运维说明" />
                </div>
                <div class="field mt-14">
                  <div class="field-label">内容</div>
                  <textarea bind:value={documentContent} placeholder="输入文档内容..."></textarea>
                </div>
                <div class="field-row mt-14">
                  <div class="field">
                    <div class="field-label">集合</div>
                    <SelectField bind:value={documentKnowledgeBase} options={knowledgeBaseOptions()} ariaLabel="选择录入知识库" />
                  </div>
                  <div class="field">
                    <div class="field-label">&nbsp;</div>
                    <button class="button success" on:click={addTextDocument} disabled={addBusy}>
                      {addBusy ? '保存中...' : '添加文档'}
                    </button>
                  </div>
                </div>
            </PanelCard>
          </div>
        {/if}

        {#if documentMode === 'search'}
          <div class="grid-2">
            <PanelCard title="检索" subtitle="查片段并返回摘要。">
                <div class="field">
                  <div class="field-label">搜索关键词</div>
                  <input bind:value={searchQuery} placeholder="输入关键词..." on:keydown={(event) => event.key === 'Enter' && runSearch()} />
                </div>
                <div class="field-row mt-14">
                  <div class="field">
                    <div class="field-label">集合</div>
                    <SelectField bind:value={searchKnowledgeBase} options={knowledgeBaseOptions()} ariaLabel="选择搜索知识库" />
                  </div>
                  <div class="field">
                    <div class="field-label">返回数量</div>
                    <input type="number" min="1" max="20" bind:value={searchLimit} />
                  </div>
                </div>
                <div class="mt-14">
                  <button class="button primary" on:click={runSearch} disabled={searchBusy}>
                    {searchBusy ? '搜索中...' : '执行搜索'}
                  </button>
                </div>
            </PanelCard>

            <PanelCard title="对话" subtitle="基于检索结果问答。">
                <div class="field mb-14">
                  <div class="field-label">对话集合</div>
                  <SelectField bind:value={chatKnowledgeBase} options={knowledgeBaseOptions()} ariaLabel="选择对话知识库" />
                </div>
                <div class="chat-box">
                  <div class="message-list">
                    {#each chatMessages as message}
                      <div class="message-card {message.role}">
                        <div class="message-role">{message.role === 'user' ? 'User' : 'Assistant'}</div>
                        <div class="message-content">{message.content}</div>
                        {#if message.sources?.length}
                          <div class="message-sources">
                            {#each message.sources as source, index}
                              <div class="message-source">
                                <strong>Source {index + 1}:</strong>
                                <div class="source-content">{clampText(source.content, 280)}</div>
                              </div>
                            {/each}
                          </div>
                        {/if}
                      </div>
                    {/each}
                  </div>
                  <div class="chat-input-row">
                    <textarea bind:value={chatInput} placeholder="输入问题后点击发送..." on:keydown={(event) => event.key === 'Enter' && (event.metaKey || event.ctrlKey) && sendChat()}></textarea>
                    <div class="card-actions justify-end">
                      <button class="button ghost" on:click={() => (chatMessages = chatMessages.slice(0, 1))}>清空</button>
                      <button class="button success" on:click={sendChat} disabled={chatBusy}>
                        {chatBusy ? '发送中...' : '发送'}
                      </button>
                    </div>
                  </div>
                </div>
            </PanelCard>
          </div>

          {#if searchResults.length || searchSummary}
            <PanelCard title="搜索结果" subtitle={`${searchResults.length} 条命中`}>
              <div class="result-list">
                {#if searchSummary}
                  <div class="result-card">
                    <span class="result-score">摘要</span>
                    <div class="result-content">{searchSummary}</div>
                  </div>
                {/if}
                {#if searchResults.length}
                  {#each searchResults as result}
                    <div class="result-card">
                      <div class="toolbar">
                        <div>
                          <strong>{safeText(result.filename || result.source || '结果')}</strong>
                          <div class="meta">{JSON.stringify(result.metadata || {})}</div>
                        </div>
                        <span class="result-score">{(Number(result.score || 0) * 100).toFixed(1)}%</span>
                      </div>
                      <div class="result-content">{clampText(result.content, 360)}</div>
                    </div>
                  {/each}
                {:else}
                  <div class="empty-state">没有搜索结果。</div>
                {/if}
              </div>
            </PanelCard>
          {/if}
        {/if}

        {#if documentMode === 'manage'}
          <PanelCard title="内容管理" subtitle="查看文件、片段并删除。">
            <svelte:fragment slot="actions">
                <button class="button secondary" on:click={refreshDocuments} disabled={manageBusy}>{manageBusy ? '刷新中...' : '刷新列表'}</button>
                {#if fileFilter}
                  <button class="button ghost" on:click={clearFileFilter}>清除筛选</button>
                {/if}
            </svelte:fragment>
              <div class="field-row">
                <div class="field">
                  <div class="field-label">集合</div>
                  <SelectField bind:value={manageKnowledgeBase} options={knowledgeBaseOptions()} ariaLabel="选择管理知识库" on:change={refreshDocuments} />
                </div>
                <div class="field">
                  <div class="field-label">筛选文件名</div>
                  <input bind:value={fileFilter} placeholder="可选，留空显示全部" on:keydown={(event) => event.key === 'Enter' && refreshDocuments()} />
                </div>
              </div>

              <div class="table mt-16">
                <div class="table-row header">
                  <span>名称</span>
                  <span>类型 / 片段</span>
                  <span>操作</span>
                </div>
                {#if files.length}
                  {#each files as file}
                    <div class="table-row">
                      <span>{file.filename}</span>
                      <span>{safeText(file.file_type, 'unknown')} · {safeText(file.chunk_count, 0)} chunks</span>
                      <span class="card-actions">
                        <button class="button secondary" on:click={() => selectFileFilter(file.filename)}>查看片段</button>
                        <button class="button danger" on:click={() => deleteFile(file.filename)}>删除</button>
                      </span>
                    </div>
                  {/each}
                {:else}
                  <div class="empty-state">暂无文件，先切到“导入”上传资料。</div>
                {/if}
              </div>

              <div class="divider my-18"></div>

              <div class="table">
                <div class="table-row header">
                  <span>文档 ID</span>
                  <span>来源 / 时间</span>
                  <span>操作</span>
                </div>
                {#if documents.length}
                  {#each documents as doc}
                    <div class="table-row">
                      <span class="mono">{doc.id}</span>
                      <span>
                        {safeText(doc.metadata?.filename, 'manual')}{doc.metadata?.timestamp ? ` · ${formatTime(doc.metadata.timestamp)}` : ''}
                      </span>
                      <span class="card-actions">
                        <button class="button secondary" on:click={() => openDocumentPreview(doc)}>查看内容</button>
                        <button class="button danger" on:click={() => deleteDocument(doc.id)}>删除</button>
                      </span>
                    </div>
                  {/each}
                {:else}
                  <div class="empty-state">当前页没有文档。</div>
                {/if}
              </div>

              <div class="toolbar mt-16">
                <div class="muted small">第 {managedPage + 1} 页 · 共 {documentsTotal} 条</div>
                <div class="toolbar-group">
                  <button class="button secondary" on:click={prevPage} disabled={managedPage === 0}>上一页</button>
                  <button class="button secondary" on:click={nextPage} disabled={documents.length < managedPageSize}>下一页</button>
                </div>
              </div>
          </PanelCard>
        {/if}
      </PageShell>
    {/if}

    {#if activeSection === 'mcp'}
      <PageShell title="MCP 调试" subtitle="列出工具、查看 schema，并直接发起调用。">
        <svelte:fragment slot="actions">
          <div class="card-actions">
            <button class="button secondary" on:click={() => refreshMcpTools()} disabled={mcpBusy}>刷新工具</button>
            <button class="button primary" on:click={runMcpTool} disabled={mcpBusy || !mcpTool}>
              {mcpBusy ? '调用中...' : '执行 MCP 调用'}
            </button>
          </div>
        </svelte:fragment>

        <div class="grid-2">
          <PanelCard title="工具列表" subtitle="当前 MCP Server 暴露的工具。">
            <div class="field">
              <div class="field-label">选择工具</div>
              <SelectField
                bind:value={mcpTool}
                options={mcpTools.map((item) => ({ value: item.name, label: item.name }))}
                ariaLabel="选择 MCP 工具"
              />
            </div>
            <div class="status-stack mt-16">
              {#each mcpTools as tool}
                <div class="status-row">
                  <div>
                    <strong>{tool.name}</strong>
                    <div class="muted">{tool.description}</div>
                  </div>
                  <span class="status-badge {tool.name === mcpTool ? 'good' : 'info'}">
                    {tool.name === mcpTool ? '已选中' : '可用'}
                  </span>
                </div>
              {:else}
                <div class="empty-state">暂无 MCP 工具，请先刷新。</div>
              {/each}
            </div>
          </PanelCard>

          <PanelCard title="输入 Schema" subtitle="当前工具的入参定义。">
            <pre class="code-block">{prettyJson(currentMcpTool()?.input_schema || {})}</pre>
          </PanelCard>
        </div>

        <div class="grid-2">
          <PanelCard title="调用参数" subtitle="直接编辑 JSON 请求体。">
            <div class="field">
              <div class="field-label">Arguments JSON</div>
              <textarea bind:value={mcpArguments} placeholder={"{\"query\":\"FastAPI 是什么？\",\"scope\":\"public\"}"}></textarea>
            </div>
            <div class="field-help">这里填的是 MCP tool `arguments`，不是 HTTP 参数。</div>
          </PanelCard>

          <PanelCard title="调用结果" subtitle="HTTP 调试 facade 返回的标准化结果。">
            <pre class="code-block">{mcpResult || '尚未执行调用。'}</pre>
          </PanelCard>
        </div>
      </PageShell>
    {/if}

    {#if activeSection === 'config'}
        {#if configMode === 'provider'}
          <div class="provider-pane provider-pane--sidebar">
            <div class="provider-pane__header">
              <div>
                <h3>服务商</h3>
                <p>搜索并切换需要管理的模型服务商。</p>
              </div>
            </div>
            <div class="field">
              <input bind:value={providerSearch} placeholder="搜索服务商..." />
            </div>
            <div class="provider-list">
              {#each filteredProviderCatalog() as provider}
                <button
                  class="provider-list-item {providerEditor === provider.id ? 'active' : ''}"
                  type="button"
                  on:click={() => (providerEditor = provider.id)}
                >
                  <div class="provider-list-item__copy">
                    <strong>{provider.title}</strong>
                    <span>{provider.vendor}</span>
                  </div>
                  <div class="provider-list-item__meta">
                    <span>{provider.families.length}</span>
                  </div>
                </button>
              {:else}
                <div class="empty-state">没有匹配到服务商。</div>
              {/each}
            </div>
          </div>

            {#if providerDefinition(providerEditor)}
              <div class="provider-shell__detail">
                <div class="provider-pane provider-pane--summary">
                  <div class="provider-pane__header">
                    <div>
                      <h3>{providerDefinition(providerEditor).title}</h3>
                      <p>{providerDefinition(providerEditor).description}</p>
                    </div>
                    <div class="provider-pane__pill">{providerDefinition(providerEditor).families.length} 项能力</div>
                  </div>
                  <div class="provider-meta-grid">
                    <div class="provider-meta-card">
                      <span class="provider-meta-card__label">厂商</span>
                      <strong>{providerDefinition(providerEditor).vendor}</strong>
                    </div>
                    <div class="provider-meta-card">
                      <span class="provider-meta-card__label">官网</span>
                      {#if providerDefinition(providerEditor).website}
                        <a href={providerDefinition(providerEditor).website} target="_blank" rel="noreferrer">
                          {providerDefinition(providerEditor).website}
                        </a>
                      {:else}
                        <strong>本地 provider</strong>
                      {/if}
                    </div>
                  </div>

                  {#if !providerDefinition(providerEditor).local}
                    <div class="field mt-16">
                      <div class="field-label">API URL</div>
                      <input
                        value={providerConfigValue(providerEditor, 'base_url', providerDefinition(providerEditor).defaults?.base_url ?? '')}
                        placeholder={providerDefinition(providerEditor).defaults?.base_url ?? 'https://your-provider/v1'}
                        on:input={(event) => updateNamedProviderConfig(providerEditor, 'base_url', event.currentTarget.value)}
                      />
                    </div>
                    <div class="field mt-16">
                      <div class="field-label">API Key</div>
                      <input
                        value={providerConfigValue(providerEditor, 'api_key', '')}
                        placeholder="输入该服务商的 API Key"
                        on:input={(event) => updateNamedProviderConfig(providerEditor, 'api_key', event.currentTarget.value)}
                      />
                    </div>
                  {:else}
                    <div class="field mt-16">
                      <div class="field-label">本地说明</div>
                      <div class="field-help">本地 embedding provider 不依赖远端 API Key 或 URL，使用左侧系统选项里的本地运行参数。</div>
                    </div>
                  {/if}
                </div>
                <div class="provider-pane provider-pane--models">
                  <div class="provider-pane__header">
                    <div>
                      <h3>模型列表</h3>
                      <p>只展示你手动添加或同步回来的模型，不再提供默认模型占位。</p>
                    </div>
                    <div class="provider-pane__pill">{filteredProviderModelRows(providerEditor, '').length} 个模型</div>
                  </div>

                  <div class="provider-model-toolbar provider-model-toolbar--compact">
                    <div class="field">
                      <input bind:value={providerModelSearch} placeholder="搜索模型 ID..." />
                    </div>
                    <div class="field field--grow">
                      <div class="field-row">
                        <input
                          value={customModelDrafts[customModelDraftKey(providerEditor)] || ''}
                          placeholder="输入模型 ID 后手动添加"
                          on:input={(event) => {
                            customModelDrafts = {
                              ...customModelDrafts,
                              [customModelDraftKey(providerEditor)]: event.currentTarget.value,
                            };
                          }}
                          on:keydown={(event) => event.key === 'Enter' && addCustomProviderModel(providerEditor)}
                        />
                        <button class="button ghost" type="button" on:click={() => addCustomProviderModel(providerEditor)}>
                          添加
                        </button>
                        <button
                          class="button secondary"
                          type="button"
                          disabled={providerFetchBusy[customModelDraftKey(providerEditor)]}
                          on:click={() => fetchProviderModels(providerEditor)}
                        >
                          {providerFetchBusy[customModelDraftKey(providerEditor)] ? '同步中...' : '获取模型列表'}
                        </button>
                      </div>
                    </div>
                  </div>
                  <div class="field-help">手动添加时根据模型 ID 自动判断对话或向量类型。</div>

                  <div class="provider-model-list mt-16">
                    {#each filteredProviderModelRows(providerEditor, providerModelSearch) as row}
                      <div class="provider-model-row">
                        <div class="provider-model-row__identity">
                          <div class="provider-model-row__icon">{row.family === 'embedding' ? '向' : '话'}</div>
                          <div class="provider-model-row__copy">
                            <strong class="provider-model-row__model" title={row.id}>{row.id}</strong>
                            <div class="provider-model-row__meta">
                              <span class="provider-model-pill">{modelFamilyMeta[row.family].label}</span>
                              <span class="provider-model-pill">{providerModelSourceLabel(row.source)}</span>
                              {#if row.selected}
                                <span class="provider-model-pill provider-model-pill--active">当前默认</span>
                              {/if}
                            </div>
                          </div>
                        </div>
                        <div class="provider-model-row__actions">
                          <button
                            class="button ghost"
                            type="button"
                            disabled={row.selected}
                            on:click={() => updateProviderModel(providerEditor, row.family, row.id)}
                          >
                            {row.selected ? '已选中' : '设为默认'}
                          </button>
                          {#if row.removable}
                            <button
                              class="button danger"
                              type="button"
                              on:click={() => removeCustomProviderModel(providerEditor, row.family, row.id)}
                            >
                              删除
                            </button>
                          {/if}
                        </div>
                      </div>
                    {:else}
                      <div class="empty-state">当前还没有模型。先点击“获取模型列表”，或手动添加模型 ID。</div>
                    {/each}
                  </div>
                </div>
              </div>
            {/if}
        {/if}

        {#if configMode === 'system'}
          <div class="grid-2">
            <PanelCard title="服务与存储" subtitle="服务端口、向量库和持久化目录。">
              <div class="field-row">
                <div class="field">
                  <div class="field-label">Host</div>
                  <input bind:value={configDraft.host} />
                </div>
                <div class="field">
                  <div class="field-label">Port / HTTP Port</div>
                  <div class="field-row">
                    <input type="number" min="1" bind:value={configDraft.port} />
                    <input type="number" min="1" bind:value={configDraft.http_port} />
                  </div>
                </div>
              </div>
              <div class="field-row mt-14">
                <div class="field">
                  <div class="field-label">Vector DB Type</div>
                  <input bind:value={configDraft.vector_db_type} placeholder="chroma / qdrant" />
                </div>
                <div class="field">
                  <div class="field-label">Debug</div>
                  <label class="collection-chip"><input type="checkbox" bind:checked={configDraft.debug} /> 启用调试日志</label>
                </div>
              </div>
              <div class="field mt-14">
                <div class="field-label">Chroma Persist Directory</div>
                <input bind:value={configDraft.chroma_persist_directory} />
              </div>
              <div class="field mt-14">
                <div class="field-label">Knowledge Base DB Path</div>
                <input bind:value={configDraft.knowledge_base_db_path} />
              </div>
              <div class="field mt-14">
                <div class="field-label">Qdrant URL</div>
                <input bind:value={configDraft.qdrant_url} />
              </div>
            </PanelCard>

            <PanelCard title="检索策略" subtitle="检索数量、阈值和功能开关。">
              <div class="field-row">
                <div class="field">
                  <div class="field-label">向量服务商</div>
                  <SelectField
                    value={configDraft.embedding_provider}
                    options={providerSelectOptions('embedding')}
                    ariaLabel="选择向量服务商"
                    on:change={(event) => selectEmbeddingProvider(event.detail.value)}
                  />
                </div>
                <div class="field">
                  <div class="field-label">向量服务商回退</div>
                  <SelectField
                    bind:value={configDraft.embedding_fallback_provider}
                    options={[{ value: '', label: '不启用回退' }, ...providerSelectOptions('embedding')]}
                    ariaLabel="选择向量服务商回退"
                  />
                </div>
              </div>
              <div class="field-row">
                <div class="field">
                  <div class="field-label">向量模型</div>
                  <SelectField
                    value={providerModelValue(configDraft.embedding_provider, 'embedding')}
                    options={providerModelOptions(configDraft.embedding_provider, 'embedding')}
                    ariaLabel="选择向量模型"
                    on:change={(event) => updateProviderModel(configDraft.embedding_provider, 'embedding', event.detail.value)}
                  />
                </div>
                <div class="field">
                  <div class="field-label">LLM 服务商</div>
                  <SelectField
                    value={configDraft.llm_provider}
                    options={providerSelectOptions('chat')}
                    ariaLabel="选择 LLM 服务商"
                    on:change={(event) => selectLlmProvider(event.detail.value)}
                  />
                </div>
              </div>
              <div class="field-row">
                <div class="field">
                  <div class="field-label">LLM 模型</div>
                  <SelectField
                    value={configDraft.llm_model}
                    options={providerModelOptions(configDraft.llm_provider, 'chat')}
                    ariaLabel="选择 LLM 模型"
                    on:change={(event) => updateProviderModel(configDraft.llm_provider, 'chat', event.detail.value)}
                  />
                </div>
                <div class="field">
                  <div class="field-label">LLM 服务商回退</div>
                  <SelectField
                    bind:value={configDraft.llm_fallback_provider}
                    options={[{ value: '', label: '不启用回退' }, ...providerSelectOptions('chat')]}
                    ariaLabel="选择 LLM 服务商回退"
                  />
                </div>
              </div>
              <div class="field-row mt-14">
                <div class="field">
                  <div class="field-label">最大检索结果</div>
                  <input type="number" min="1" bind:value={configDraft.max_retrieval_results} />
                </div>
                <div class="field">
                  <div class="field-label">相似度阈值</div>
                  <input type="number" step="0.01" min="0" max="1" bind:value={configDraft.similarity_threshold} />
                </div>
              </div>
              <div class="field mt-14">
                <div class="field-label">功能开关</div>
                <div class="field-switch">
                  <label class="collection-chip"><input type="checkbox" bind:checked={configDraft.enable_cache} /> 缓存</label>
                  <label class="collection-chip"><input type="checkbox" bind:checked={configDraft.enable_reranker} /> 重排</label>
                  <label class="collection-chip"><input type="checkbox" bind:checked={configDraft.enable_llm_summary} /> 总结</label>
                </div>
              </div>
            </PanelCard>

            <PanelCard title="缓存与安全" subtitle="缓存、鉴权和租户级 key。">
              <div class="field-row">
                <div class="field">
                  <div class="field-label">Cache</div>
                  <label class="collection-chip"><input type="checkbox" bind:checked={configDraft.cache.enabled} /> 启用请求级缓存</label>
                </div>
                <div class="field">
                  <div class="field-label">TTL / Entries</div>
                  <div class="field-row">
                    <input type="number" min="1" bind:value={configDraft.cache.ttl_seconds} />
                    <input type="number" min="1" bind:value={configDraft.cache.max_entries} />
                  </div>
                </div>
              </div>
              <div class="field-row mt-14">
                <div class="field">
                  <div class="field-label">Security</div>
                  <label class="collection-chip"><input type="checkbox" bind:checked={configDraft.security.enabled} /> 启用鉴权</label>
                </div>
                <div class="field">
                  <div class="field-label">Allow Anonymous</div>
                  <label class="collection-chip"><input type="checkbox" bind:checked={configDraft.security.allow_anonymous} /> 允许匿名</label>
                </div>
              </div>
              <div class="field mt-14">
                <div class="field-label">API Keys</div>
                <textarea bind:value={configDraft.security.api_keys_text} placeholder="每行一个 API key"></textarea>
              </div>
              <div class="field mt-14">
                <div class="field-label">Tenant API Keys JSON</div>
                <textarea bind:value={configDraft.security.tenant_api_keys_text} placeholder={'{"u1_default": ["key-1"]}'}></textarea>
              </div>
            </PanelCard>

            <PanelCard title="限流与配额" subtitle="请求窗口和索引上限。">
              <div class="field-row">
                <div class="field">
                  <div class="field-label">Requests / Window</div>
                  <input type="number" min="0" bind:value={configDraft.rate_limit.requests_per_window} />
                </div>
                <div class="field">
                  <div class="field-label">Window Seconds</div>
                  <input type="number" min="1" bind:value={configDraft.rate_limit.window_seconds} />
                </div>
              </div>
              <div class="field mt-14">
                <div class="field-label">Burst</div>
                <input type="number" min="0" bind:value={configDraft.rate_limit.burst} />
              </div>
              <div class="divider my-18"></div>
              <div class="field-row">
                <div class="field">
                  <div class="field-label">Max Upload Files</div>
                  <input type="number" min="1" bind:value={configDraft.quotas.max_upload_files} />
                </div>
                <div class="field">
                  <div class="field-label">Max Upload Bytes</div>
                  <input type="number" min="1" bind:value={configDraft.quotas.max_upload_bytes} />
                </div>
              </div>
              <div class="field-row mt-14">
                <div class="field">
                  <div class="field-label">Max File Bytes</div>
                  <input type="number" min="1" bind:value={configDraft.quotas.max_upload_file_bytes} />
                </div>
                <div class="field">
                  <div class="field-label">Max Index Docs</div>
                  <input type="number" min="1" bind:value={configDraft.quotas.max_index_documents} />
                </div>
              </div>
              <div class="field-row mt-14">
                <div class="field">
                  <div class="field-label">Max Index Chunks</div>
                  <input type="number" min="1" bind:value={configDraft.quotas.max_index_chunks} />
                </div>
                <div class="field">
                  <div class="field-label">Max Index Chars</div>
                  <input type="number" min="1" bind:value={configDraft.quotas.max_index_chars} />
                </div>
              </div>
            </PanelCard>

            <PanelCard title="观测与 Provider Budget" subtitle="健康阈值和 provider 预算。">
              <div class="field-row">
                <div class="field">
                  <div class="field-label">Warning Error Rate</div>
                  <input type="number" step="0.01" min="0" max="1" bind:value={configDraft.observability.warning_error_rate} />
                </div>
                <div class="field">
                  <div class="field-label">Critical Error Rate</div>
                  <input type="number" step="0.01" min="0" max="1" bind:value={configDraft.observability.critical_error_rate} />
                </div>
              </div>
              <div class="field-row mt-14">
                <div class="field">
                  <div class="field-label">Slow Request ms</div>
                  <input type="number" min="0" bind:value={configDraft.observability.slow_request_ms} />
                </div>
                <div class="field">
                  <div class="field-label">Latency Window Size</div>
                  <input type="number" min="1" bind:value={configDraft.observability.latency_window_size} />
                </div>
              </div>
              <div class="divider my-18"></div>
              <div class="field-switch">
                <label class="collection-chip"><input type="checkbox" bind:checked={configDraft.provider_budget.enabled} /> 启用 Provider Budget</label>
              </div>
              <div class="field-row mt-14">
                <div class="field">
                  <div class="field-label">Embedding Window / Burst</div>
                  <div class="field-row">
                    <input type="number" min="1" bind:value={configDraft.provider_budget.embeddings.requests_per_window} />
                    <input type="number" min="0" bind:value={configDraft.provider_budget.embeddings.burst} />
                  </div>
                </div>
                <div class="field">
                  <div class="field-label">Embedding Failure / Cooldown</div>
                  <div class="field-row">
                    <input type="number" min="0" bind:value={configDraft.provider_budget.embeddings.failure_threshold} />
                    <input type="number" min="0" bind:value={configDraft.provider_budget.embeddings.cooldown_seconds} />
                  </div>
                </div>
              </div>
              <div class="field-row mt-14">
                <div class="field">
                  <div class="field-label">LLM Window / Burst</div>
                  <div class="field-row">
                    <input type="number" min="1" bind:value={configDraft.provider_budget.llm.requests_per_window} />
                    <input type="number" min="0" bind:value={configDraft.provider_budget.llm.burst} />
                  </div>
                </div>
                <div class="field">
                  <div class="field-label">LLM Failure / Cooldown</div>
                  <div class="field-row">
                    <input type="number" min="0" bind:value={configDraft.provider_budget.llm.failure_threshold} />
                    <input type="number" min="0" bind:value={configDraft.provider_budget.llm.cooldown_seconds} />
                  </div>
                </div>
              </div>
            </PanelCard>
          </div>
        {/if}

        {#if configMode === 'advanced'}
          <div class="advanced-config-card">
            <PanelCard title="完整配置 JSON" subtitle="直接编辑全部配置；保存时会按完整对象提交。" fill={true}>
              <div class="field advanced-config-field">
                <div class="field-label">Config JSON</div>
                <textarea class="advanced-config-textarea" bind:value={configDraft.full_config_text} spellcheck="false"></textarea>
              </div>
              <div class="field-help">这里是全部配置，不只是 provider_configs。适合直接维护完整配置文件。</div>
            </PanelCard>
          </div>
        {/if}
    {/if}

  </main>
</div>

{#if previewDocument}
  <div class="modal-backdrop" role="presentation" on:click={closeDocumentPreview}>
    <div
      class="modal-card"
      role="dialog"
      aria-modal="true"
      aria-label="片段内容"
      tabindex="-1"
      on:click|stopPropagation
      on:keydown={(event) => event.key === 'Escape' && closeDocumentPreview()}
    >
      <div class="modal-card__header">
        <div>
          <h3>片段内容</h3>
          <p>{safeText(previewDocument.metadata?.filename, 'manual')}{previewDocument.metadata?.timestamp ? ` · ${formatTime(previewDocument.metadata.timestamp)}` : ''}</p>
        </div>
        <button class="button ghost" type="button" on:click={closeDocumentPreview}>关闭</button>
      </div>
      <div class="modal-card__meta">
        <span class="provider-model-pill mono">{previewDocument.id}</span>
        {#if previewDocument.metadata?.collection_variant}
          <span class="provider-model-pill">{previewDocument.metadata.collection_variant}</span>
        {/if}
      </div>
      <pre class="document-preview">{previewDocument.content || '该片段没有内容。'}</pre>
    </div>
  </div>
{/if}
