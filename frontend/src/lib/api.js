const rawBase = (import.meta.env.VITE_API_BASE_URL || '').trim();
export const API_BASE = rawBase.replace(/\/+$/, '');

export class ApiError extends Error {
  constructor(status, message, payload) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.payload = payload;
  }
}

function buildUrl(path, query = {}) {
  const base = API_BASE ? `${API_BASE}${path}` : path;
  const url = new URL(base, window.location.origin);
  for (const [key, value] of Object.entries(query)) {
    if (value === undefined || value === null || value === '') continue;
    url.searchParams.set(key, String(value));
  }
  return url.toString();
}

function identityHeaders(identity = {}) {
  const headers = {};
  const apiKey = identity.apiKey || identity.api_key;
  if (apiKey) {
    headers['x-api-key'] = apiKey;
  }
  return headers;
}

async function request(path, { method = 'GET', query = {}, body, formData, headers = {} } = {}) {
  const requestHeaders = { ...headers };
  const init = { method, headers: requestHeaders };

  if (formData) {
    init.body = formData;
  } else if (body !== undefined) {
    init.body = JSON.stringify(body);
    if (!requestHeaders['Content-Type']) {
      requestHeaders['Content-Type'] = 'application/json';
    }
  }

  const response = await fetch(buildUrl(path, query), init);
  const text = await response.text();
  let payload = null;

  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = text;
    }
  }

  if (!response.ok) {
    const message = typeof payload === 'object' && payload
      ? payload.detail || payload.message || response.statusText
      : text || response.statusText;
    throw new ApiError(response.status, message, payload);
  }

  return payload;
}

function identityQuery(identity = {}) {
  const query = {};
  if (identity.userId) query.user_id = identity.userId;
  if (identity.agentId) query.agent_id = identity.agentId;
  if (identity.kbId) query.kb_id = identity.kbId;
  if (identity.scope) query.scope = identity.scope;
  if (identity.apiKey || identity.api_key) query.api_key = identity.apiKey || identity.api_key;
  return query;
}

export const api = {
  health: (identity = {}) => request('/health', { headers: identityHeaders(identity) }),
  ready: (identity = {}) => request('/ready', { headers: identityHeaders(identity) }),
  metrics: (identity = {}) => request('/metrics', { headers: identityHeaders(identity) }),
  config: (identity = {}) => request('/config', { query: identityQuery(identity), headers: identityHeaders(identity) }),
  updateConfig: (updates, identity = {}) =>
    request('/config/bulk', { method: 'POST', body: { updates }, headers: identityHeaders(identity) }),
  resetConfig: (identity = {}) => request('/config/reset', { method: 'POST', headers: identityHeaders(identity) }),
  reloadConfig: (identity = {}) => request('/config/reload', { method: 'POST', headers: identityHeaders(identity) }),
  knowledgeBases: (identity = {}) =>
    request('/knowledge-bases', { query: identityQuery(identity), headers: identityHeaders(identity) }),
  createKnowledgeBase: (payload) =>
    request('/knowledge-bases', { method: 'POST', body: payload, headers: identityHeaders(payload) }),
  mcpTools: (identity = {}) =>
    request('/debug/mcp/tools', { query: identityQuery(identity), headers: identityHeaders(identity) }),
  mcpCall: (payload) =>
    request('/debug/mcp/call', { method: 'POST', body: payload, headers: identityHeaders(payload) }),
  collections: (identity = {}) =>
    request('/collections', { query: identityQuery(identity), headers: identityHeaders(identity) }),
  uploadFiles: ({ files, collection = 'default', kbId, scope, ...identity }) => {
    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file);
    }
    formData.append('collection', collection);
    if (kbId) formData.append('kb_id', kbId);
    if (scope) formData.append('scope', scope);
    if (identity.userId) formData.append('user_id', identity.userId);
    if (identity.agentId) formData.append('agent_id', identity.agentId);
    if (identity.apiKey) formData.append('api_key', identity.apiKey);
    return request('/upload-files', { method: 'POST', formData, headers: identityHeaders(identity) });
  },
  addDocument: (payload) => request('/add-document', { method: 'POST', body: payload, headers: identityHeaders(payload) }),
  search: ({ query, collection = 'default', limit = 5, kbId, scope, ...identity }) =>
    request('/search', {
      query: {
        query,
        collection,
        limit,
        kb_id: kbId,
        scope,
        ...identityQuery(identity),
      },
      headers: identityHeaders(identity),
    }),
  chat: (payload) => request('/chat', { method: 'POST', body: payload, headers: identityHeaders(payload) }),
  listDocuments: ({ collection = 'default', limit = 100, offset = 0, filename, kbId, scope, ...identity }) =>
    request('/list-documents', {
      query: {
        collection,
        limit,
        offset,
        filename,
        kb_id: kbId,
        scope,
        ...identityQuery(identity),
      },
      headers: identityHeaders(identity),
    }),
  listFiles: ({ collection = 'default', kbId, scope, ...identity }) =>
    request('/list-files', {
      query: { collection, kb_id: kbId, scope, ...identityQuery(identity) },
      headers: identityHeaders(identity),
    }),
  deleteDocument: (payload) => request('/delete-document', { method: 'DELETE', body: payload, headers: identityHeaders(payload) }),
  deleteFile: (payload) => request('/delete-file', { method: 'DELETE', body: payload, headers: identityHeaders(payload) }),
};
