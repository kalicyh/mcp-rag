"""HTTP server for MCP-RAG configuration and document management."""

import logging
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from starlette.responses import PlainTextResponse
import tempfile
import shutil

from .config import config_manager, settings
from .database import get_vector_database
from .embedding import get_embedding_model
from .document_processor import get_document_processor
from .mcp_server import mcp_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

logger = logging.getLogger(__name__)

app = FastAPI(title="MCP-RAG HTTP API", description="API for configuring MCP-RAG and adding documents")

# Mount static files
static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

streamable_http_manager = StreamableHTTPSessionManager(
    app=mcp_server.server,
    json_response=True,
    stateless=True,
)


async def _streamable_http_asgi(scope, receive, send):
    if scope.get("type") != "http":
        response = PlainTextResponse("Streamable HTTP supports only HTTP requests", status_code=405)
        await response(scope, receive, send)
        return

    try:
        await streamable_http_manager.handle_request(scope, receive, send)
    except RuntimeError as err:  # pragma: no cover - defensive
        logger.error("Streamable HTTP transport unavailable: %s", err)
        response = PlainTextResponse("MCP transport unavailable", status_code=503)
        await response(scope, receive, send)


app.mount("/mcp", _streamable_http_asgi, name="streamable-mcp")
app.mount("/mcp/", _streamable_http_asgi, name="streamable-mcp-slash")
app.mount("/sse", _streamable_http_asgi, name="sse")


@app.on_event("startup")
async def _start_streamable_http_manager():
    context = streamable_http_manager.run()
    app.state.streamable_http_context = context
    try:
        await context.__aenter__()
    except Exception:
        logger.exception("Failed to start Streamable HTTP session manager")
        raise


@app.on_event("shutdown")
async def _stop_streamable_http_manager():
    context = getattr(app.state, "streamable_http_context", None)
    if context is None:
        return
    try:
        await context.__aexit__(None, None, None)
    except Exception:
        logger.exception("Error shutting down Streamable HTTP session manager")
    finally:
        app.state.streamable_http_context = None


class ConfigUpdate(BaseModel):
    """Configuration update model."""
    key: str
    value: Any


class BulkConfigUpdate(BaseModel):
    """Bulk configuration update model."""
    updates: Dict[str, Any]


class DocumentAdd(BaseModel):
    """Document addition model."""
    content: str
    collection: str = "default"
    metadata: Dict[str, Any] = {}


class FileUploadResponse(BaseModel):
    """File upload response model."""
    filename: str
    file_type: str
    content_length: int
    processed: bool
    error: str = ""
    preview: str = ""


class BatchUploadResponse(BaseModel):
    """Batch file upload response model."""
    total_files: int
    successful: int
    failed: int
    results: List[FileUploadResponse]


class DeleteDocumentRequest(BaseModel):
    """Delete document request model."""
    document_id: str
    collection: str = "default"


class DeleteFileRequest(BaseModel):
    """Delete file request model."""
    filename: str
    collection: str = "default"


@app.get("/")
async def root():
    """Root endpoint - redirect to documents page."""
    return RedirectResponse(url="/documents-page")


@app.get("/doc")
async def doc_redirect():
    """Redirect to API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/documents-page", response_class=HTMLResponse)
async def documents_page():
    """Serve the documents management page."""
    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP-RAG 资料管理</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .tabs {
            display: flex;
            margin-bottom: 30px;
            border-bottom: 1px solid #ddd;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            background: #f8f9fa;
            border: 1px solid #ddd;
            border-bottom: none;
            margin-right: 5px;
            border-radius: 5px 5px 0 0;
        }
        .tab.active {
            background: white;
            border-bottom: 1px solid white;
            margin-bottom: -1px;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .section h2 {
            color: #555;
            margin-top: 0;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
        }
        .upload-area {
            border: 2px dashed #007acc;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9fa;
            transition: all 0.3s;
        }
        .upload-area:hover {
            background: #e9ecef;
            border-color: #005aa3;
        }
        .upload-area.dragover {
            background: #e3f2fd;
            border-color: #2196f3;
        }
        .file-input {
            display: none;
        }
        .upload-btn {
            background-color: #007acc;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px;
        }
        .upload-btn:hover {
            background-color: #005aa3;
        }
        .file-list {
            margin-top: 20px;
        }
        .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            margin-bottom: 10px;
            background: #f8f9fa;
        }
        .file-info {
            flex: 1;
        }
        .file-name {
            font-weight: bold;
        }
        .file-meta {
            color: #666;
            font-size: 14px;
        }
        .file-status {
            padding: 5px 10px;
            border-radius: 3px;
            font-size: 12px;
            font-weight: bold;
        }
        .status-success {
            background: #d4edda;
            color: #155724;
        }
        .status-error {
            background: #f8d7da;
            color: #721c24;
        }
        .status-processing {
            background: #fff3cd;
            color: #856404;
        }
        .preview-content {
            background: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
            margin-top: 10px;
            max-height: 300px;
            overflow-y: auto;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 14px;
        }
        .btn {
            background-color: #007acc;
            color: white;
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            margin-right: 5px;
        }
        .btn:hover {
            background-color: #005aa3;
        }
        .btn-danger {
            background-color: #dc3545;
        }
        .btn-danger:hover {
            background-color: #c82333;
        }
        .btn-success {
            background-color: #28a745;
        }
        .btn-success:hover {
            background-color: #218838;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: #007acc;
            width: 0%;
            transition: width 0.3s;
        }
        .status-message {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
            display: none;
        }
        .status-success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status-error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .chat-message {
            margin-bottom: 10px;
        }
        .chat-message.user {
            text-align: right;
        }
        .chat-message.assistant {
            text-align: left;
        }
        .view-toggle {
            margin-bottom: 15px;
        }
        .view-toggle .btn {
            margin-right: 5px;
            background-color: #6c757d;
        }
        .view-toggle .btn.active {
            background-color: #007bff;
        }
    </style>
</head>
<body>
    <div class="container">
        <div style="text-align: center; margin-bottom: 20px;">
            <a href="/config-page" style="margin: 0 10px; color: #007acc; text-decoration: none;">配置管理</a> |
            <a href="/documents-page" style="margin: 0 10px; color: #007acc; text-decoration: none;">资料管理</a>
        </div>
        <h1>MCP-RAG 资料管理</h1>

        <div class="tabs">
            <div class="tab active" onclick="switchTab('upload')">资料上传</div>
            <div class="tab" onclick="switchTab('search')">资料查询</div>
            <div class="tab" onclick="switchTab('chat')">知识库对话</div>
            <div class="tab" onclick="switchTab('manage')">内容管理</div>
        </div>

        <div id="upload" class="tab-content active">
            <div class="section">
                <h2>文件上传</h2>
                <div class="collection-select">
                    <label>选择集合: </label>
                    <select id="collectionSelect">
                        <option value="default">默认集合</option>
                    </select>
                </div>

                <div class="upload-area" id="uploadArea">
                    <div>
                        <p>拖拽文件到此处或点击选择文件</p>
                        <p style="color: #666; font-size: 14px;">支持格式: TXT, MD, PDF, DOCX</p>
                        <input type="file" id="fileInput" class="file-input" multiple accept=".txt,.md,.pdf,.docx">
                        <br>
                        <button class="upload-btn" onclick="document.getElementById('fileInput').click()">选择文件</button>
                    </div>
                </div>

                <div class="progress-bar" id="progressBar" style="display: none;">
                    <div class="progress-fill" id="progressFill"></div>
                </div>

                <div class="file-list" id="fileList"></div>

                <div class="status-message" id="statusMessage"></div>
            </div>

            <div class="section">
                <h2>文本输入</h2>
                <div class="collection-select">
                    <label>选择集合: </label>
                    <select id="textCollectionSelect">
                        <option value="default">默认集合</option>
                    </select>
                </div>
                <div style="margin-bottom: 15px;">
                    <label for="documentTitle" style="display: block; margin-bottom: 5px; font-weight: bold;">文档标题 (可选):</label>
                    <input type="text" id="documentTitle" placeholder="输入文档标题..." style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 10px;">
                </div>
                <div style="margin-bottom: 15px;">
                    <textarea id="documentContent" placeholder="输入文档内容..." style="width: 100%; height: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; resize: vertical;"></textarea>
                </div>
                <button class="btn btn-success" onclick="addTextDocument()">添加文档</button>
                <div class="status-message" id="textStatusMessage"></div>
            </div>
        </div>

        <div id="search" class="tab-content">
            <div class="section">
                <h2>资料查询</h2>
                <div style="margin-bottom: 15px;">
                    <input type="text" id="searchQuery" placeholder="输入搜索关键词..." style="width: 60%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;">
                    <select id="searchCollection" style="padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin-left: 10px;">
                        <option value="default">默认集合</option>
                    </select>
                    <button class="btn" onclick="searchDocuments()" style="margin-left: 10px;">搜索</button>
                </div>

                <div id="searchResults"></div>
            </div>
        </div>

        <div id="chat" class="tab-content">
            <div class="section">
                <h2>知识库对话测试</h2>
                <div style="margin-bottom: 15px;">
                    <select id="chatCollection" style="padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin-right: 10px;">
                        <option value="default">默认集合</option>
                    </select>
                    <span style="color: #666; font-size: 14px;">选择要对话的知识库集合</span>
                </div>

                <div id="chatHistory" style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; height: 400px; overflow-y: auto; background: #f8f9fa; margin-bottom: 15px;">
                    <div style="text-align: center; color: #666; margin-top: 150px;">
                        开始与知识库对话吧！
                    </div>
                </div>

                <div style="display: flex; gap: 10px;">
                    <input type="text" id="chatQuery" placeholder="输入您的问题..." style="flex: 1; padding: 10px; border: 1px solid #ddd; border-radius: 4px;" onkeypress="handleChatKeyPress(event)">
                    <button class="btn btn-success" onclick="sendChatMessage()">发送</button>
                    <button class="btn btn-danger" onclick="clearChatHistory()">清空</button>
                </div>

                <div class="status-message" id="chatStatusMessage"></div>
            </div>
        </div>

        <div id="manage" class="tab-content">
            <div class="section">
                <h2>内容管理</h2>
                <div style="margin-bottom: 15px;">
                    <select id="manageCollection" style="padding: 8px; border: 1px solid #ddd; border-radius: 4px; margin-right: 10px;" onchange="loadDocuments()">
                        <option value="default">默认集合</option>
                    </select>
                    <button class="btn" onclick="loadDocuments()">刷新列表</button>
                </div>

                <div class="view-toggle">
                    <button class="btn active" id="btn-view-files" onclick="switchView('files')">文件视图</button>
                    <button class="btn" id="btn-view-docs" onclick="switchView('docs')">片段视图</button>
                </div>

                <div id="fileListContainer"></div>
                <div id="documentList" style="display: none;"></div>
                
                <div style="margin-top: 20px; text-align: center;" id="pagination">
                    <button class="btn" onclick="prevPage()" id="prevPageBtn" disabled>上一页</button>
                    <span id="pageInfo" style="margin: 0 10px;">第 1 页</span>
                    <button class="btn" onclick="nextPage()" id="nextPageBtn" disabled>下一页</button>
                </div>

                <div class="status-message" id="manageStatusMessage"></div>
            </div>
        </div>
    </div>

    <script>
        const API_BASE = '';
        let uploadedFiles = [];

        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));

            // Show selected tab
            document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');
            document.getElementById(tabName).classList.add('active');
            
            // Save current tab to localStorage
            localStorage.setItem('currentTab', tabName);
        }

        function showStatus(message, isSuccess = true) {
            const statusDiv = document.getElementById('statusMessage');
            statusDiv.textContent = message;
            statusDiv.className = `status-message ${isSuccess ? 'status-success' : 'status-error'}`;
            statusDiv.style.display = 'block';
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }

        async function loadCollections() {
            try {
                const response = await fetch(`${API_BASE}/collections`);
                const data = await response.json();
                const selects = ['collectionSelect', 'searchCollection', 'textCollectionSelect', 'chatCollection', 'manageCollection'];

                selects.forEach(selectId => {
                    const select = document.getElementById(selectId);
                    if (select) {
                        select.innerHTML = '';
                        // Always add default collection first
                        const defaultOption = document.createElement('option');
                        defaultOption.value = 'default';
                        defaultOption.textContent = '默认集合';
                        select.appendChild(defaultOption);

                        // Add other collections
                        if (data.collections) {
                            data.collections.forEach(collection => {
                                if (collection !== 'default') {  // Avoid duplicate default
                                    const option = document.createElement('option');
                                    option.value = collection;
                                    option.textContent = collection;
                                    select.appendChild(option);
                                }
                            });
                        }
                    }
                });
            } catch (error) {
                console.error('Failed to load collections:', error);
                // Ensure default collection is always available
                const selects = ['collectionSelect', 'searchCollection', 'textCollectionSelect', 'chatCollection', 'manageCollection'];
                selects.forEach(selectId => {
                    const select = document.getElementById(selectId);
                    if (select && select.children.length === 0) {
                        const defaultOption = document.createElement('option');
                        defaultOption.value = 'default';
                        defaultOption.textContent = '默认集合';
                        select.appendChild(defaultOption);
                    }
                });
            }
        }

        function updateFileList() {
            const fileList = document.getElementById('fileList');
            fileList.innerHTML = '';
            
            if (uploadedFiles.length === 0) {
                fileList.innerHTML = '<div style="text-align: center; color: #666;">暂无文件</div>';
                return;
            }

            uploadedFiles.forEach((file, index) => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                fileItem.innerHTML = `
                    <div class="file-info">
                        <div class="file-name">${file.name}</div>
                        <div class="file-meta">${(file.size / 1024).toFixed(1)} KB</div>
                    </div>
                    <div>
                        <button class="btn btn-danger" onclick="removeFile(${index})">删除</button>
                    </div>
                `;
                fileList.appendChild(fileItem);
            });
        }

        function removeFile(index) {
            uploadedFiles.splice(index, 1);
            updateFileList();
        }

        function handleFileSelect(files) {
            if (!files || files.length === 0) return;
            
            Array.from(files).forEach(file => {
                // Check if file already exists
                if (!uploadedFiles.some(f => f.name === file.name)) {
                    uploadedFiles.push(file);
                }
            });
            
            updateFileList();
        }

        async function uploadFiles() {
            if (uploadedFiles.length === 0) {
                showStatus('请先选择文件', false);
                return;
            }

            const collection = document.getElementById('collectionSelect').value;
            const progressBar = document.getElementById('progressBar');
            const progressFill = document.getElementById('progressFill');

            progressBar.style.display = 'block';
            progressFill.style.width = '0%';

            const formData = new FormData();
            uploadedFiles.forEach(file => {
                formData.append('files', file);
            });
            formData.append('collection', collection);

            try {
                const response = await fetch(`${API_BASE}/upload-files`, {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (response.ok) {
                    progressFill.style.width = '100%';
                    const successCount = typeof result.successful === 'number' ? result.successful : 0;
                    const totalFiles = typeof result.total_files === 'number' ? result.total_files : uploadedFiles.length;

                    if (successCount === totalFiles && totalFiles > 0) {
                        showStatus(`上传完成: ${successCount}/${totalFiles} 个文件成功`, true);
                        uploadedFiles = [];
                        updateFileList();
                    } else if (successCount > 0) {
                        showStatus(`上传部分成功: ${successCount}/${totalFiles} 个文件成功`, false);
                    } else {
                        showStatus('上传失败: 所有文件处理失败', false);
                    }

                    // Show results
                    result.results.forEach(fileResult => {
                        updateFileStatus(fileResult);
                    });
                } else {
                    showStatus('上传失败: ' + result.detail, false);
                }
            } catch (error) {
                showStatus('上传失败: ' + error.message, false);
            } finally {
                setTimeout(() => {
                    progressBar.style.display = 'none';
                }, 2000);
            }
        }

        function updateFileStatus(fileResult) {
            const fileList = document.getElementById('fileList');
            const fileItems = fileList.querySelectorAll('.file-item');

            fileItems.forEach(item => {
                const fileName = item.querySelector('.file-name').textContent;
                if (fileName === fileResult.filename) {
                    let statusClass = 'status-processing';
                    if (fileResult.processed) {
                        statusClass = 'status-success';
                    } else if (fileResult.error) {
                        statusClass = 'status-error';
                    }

                    const statusDiv = document.createElement('div');
                    statusDiv.className = `file-status ${statusClass}`;
                    statusDiv.textContent = fileResult.processed ? '处理成功' : (fileResult.error || '处理中');

                    item.appendChild(statusDiv);

                    if (fileResult.preview) {
                        const previewDiv = document.createElement('div');
                        previewDiv.className = 'preview-content';
                        previewDiv.textContent = fileResult.preview.length > 500 ?
                            fileResult.preview.substring(0, 500) + '...' : fileResult.preview;
                        item.appendChild(previewDiv);
                    }
                }
            });
        }

        async function searchDocuments() {
            const query = document.getElementById('searchQuery').value.trim();
            const collection = document.getElementById('searchCollection').value;

            if (!query) {
                showStatus('请输入搜索关键词', false);
                return;
            }

            try {
                const response = await fetch(`${API_BASE}/search?query=${encodeURIComponent(query)}&collection=${collection}&limit=10`);
                const data = await response.json();

                const resultsDiv = document.getElementById('searchResults');
                resultsDiv.innerHTML = `<h3>搜索结果 (${data.results.length} 个)</h3>`;

                data.results.forEach(result => {
                    const resultDiv = document.createElement('div');
                    resultDiv.className = 'file-item';
                    resultDiv.innerHTML = `
                        <div class="file-info">
                            <div class="file-name">相似度: ${(result.score * 100).toFixed(1)}%</div>
                            <div class="file-meta">${result.metadata ? JSON.stringify(result.metadata) : ''}</div>
                        </div>
                        <div class="preview-content" style="margin-top: 10px;">
                            ${result.content.length > 300 ? result.content.substring(0, 300) + '...' : result.content}
                        </div>
                    `;
                    resultsDiv.appendChild(resultDiv);
                });

                // Display LLM summary if available
                // Display LLM summary if available
                if (data.summary) {
                    const summaryDiv = document.createElement('div');
                    summaryDiv.className = 'file-item';
                    summaryDiv.style.border = '2px solid #007acc';
                    summaryDiv.innerHTML = `
                        <div class="file-info">
                            <div class="file-name" style="color: #007acc;">🤖 LLM 总结</div>
                            <div class="file-meta">基于查询生成的智能总结</div>
                        </div>
                        <div class="preview-content" style="margin-top: 10px; background: #e3f2fd;">
                            ${data.summary}
                        </div>
                    `;
                    resultsDiv.insertBefore(summaryDiv, resultsDiv.firstChild);
                }

            } catch (error) {
                showStatus('搜索失败: ' + error.message, false);
            }
        }

        async function addTextDocument() {
            const title = document.getElementById('documentTitle').value.trim();
            const content = document.getElementById('documentContent').value.trim();
            const collection = document.getElementById('textCollectionSelect').value;

            if (!content) {
                showTextStatus('请输入文档内容', false);
                return;
            }

            try {
                const metadata = {};
                if (title) {
                    metadata.title = title;
                }
                metadata.source = 'manual_input';
                metadata.timestamp = new Date().toISOString();

                const response = await fetch(`${API_BASE}/add-document`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        content: content,
                        collection: collection,
                        metadata: metadata
                    })
                });

                if (response.ok) {
                    showTextStatus('文档添加成功', true);
                    document.getElementById('documentContent').value = '';
                    document.getElementById('documentTitle').value = '';
                } else {
                    const error = await response.json();
                    showTextStatus('添加失败: ' + (error.detail || '未知错误'), false);
                }
            } catch (error) {
                showTextStatus('添加失败: ' + error.message, false);
            }
        }

        function showTextStatus(message, isSuccess = true) {
            const statusDiv = document.getElementById('textStatusMessage');
            statusDiv.textContent = message;
            statusDiv.className = `status-message ${isSuccess ? 'status-success' : 'status-error'}`;
            statusDiv.style.display = 'block';
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }

        async function sendChatMessage() {
            const query = document.getElementById('chatQuery').value.trim();
            const collection = document.getElementById('chatCollection').value;

            if (!query) {
                showChatStatus('请输入问题', false);
                return;
            }

            // Add user message to chat
            addMessageToChat('user', query);
            document.getElementById('chatQuery').value = '';

            try {
                const response = await fetch(`${API_BASE}/chat`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        query: query,
                        collection: collection
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    addMessageToChat('assistant', data.response, data.sources);
                } else {
                    const error = await response.json();
                    addMessageToChat('assistant', '抱歉，处理您的请求时出现错误: ' + (error.detail || '未知错误'));
                }
            } catch (error) {
                addMessageToChat('assistant', '网络错误，请稍后重试: ' + error.message);
            }
        }

        function addMessageToChat(role, content, sources = null) {
            const chatHistory = document.getElementById('chatHistory');
            const messageDiv = document.createElement('div');
            messageDiv.className = `chat-message ${role}`;
            messageDiv.style.marginBottom = '15px';
            messageDiv.style.padding = '10px';
            messageDiv.style.borderRadius = '8px';

            if (role === 'user') {
                messageDiv.style.background = '#007acc';
                messageDiv.style.color = 'white';
                messageDiv.style.textAlign = 'right';
                messageDiv.innerHTML = `<strong>您:</strong> ${content}`;
            } else {
                messageDiv.style.background = '#f0f0f0';
                messageDiv.style.border = '1px solid #ddd';
                messageDiv.innerHTML = `<strong>助手:</strong> ${content}`;

                if (sources && sources.length > 0) {
                    const sourcesDiv = document.createElement('div');
                    sourcesDiv.style.marginTop = '10px';
                    sourcesDiv.style.fontSize = '12px';
                    sourcesDiv.style.color = '#666';
                    sourcesDiv.innerHTML = '<strong>参考来源:</strong>';
                    
                    sources.forEach((source, index) => {
                        const sourceDiv = document.createElement('div');
                        sourceDiv.style.marginTop = '5px';
                        sourceDiv.style.padding = '5px';
                        sourceDiv.style.background = '#f8f9fa';
                        sourceDiv.style.borderRadius = '4px';
                        sourceDiv.innerHTML = `<strong>来源 ${index + 1}:</strong> ${source.content}`;
                        sourcesDiv.appendChild(sourceDiv);
                    });
                    
                    messageDiv.appendChild(sourcesDiv);
                }
            }

            chatHistory.appendChild(messageDiv);
            chatHistory.scrollTop = chatHistory.scrollHeight;
        }

        function clearChatHistory() {
            document.getElementById('chatHistory').innerHTML = '<div style="text-align: center; color: #666; margin-top: 150px;">开始与知识库对话吧！</div>';
        }

        function handleChatKeyPress(event) {
            if (event.key === 'Enter') {
                sendChatMessage();
            }
        }

        function showChatStatus(message, isSuccess = true) {
            const statusDiv = document.getElementById('chatStatusMessage');
            statusDiv.textContent = message;
            statusDiv.className = `status-message ${isSuccess ? 'status-success' : 'status-error'}`;
            statusDiv.style.display = 'block';
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }

        // Content Management Functions
        let currentPage = 0;
        const pageSize = 10;
        let currentView = 'files';
        let currentFileFilter = null;

        async function switchView(view) {
            currentView = view;
            document.getElementById('btn-view-files').className = view === 'files' ? 'btn active' : 'btn';
            document.getElementById('btn-view-docs').className = view === 'docs' ? 'btn active' : 'btn';
            
            document.getElementById('fileListContainer').style.display = view === 'files' ? 'block' : 'none';
            document.getElementById('documentList').style.display = view === 'docs' ? 'block' : 'none';
            
            // Hide pagination in file view for now
            document.getElementById('pagination').style.display = view === 'docs' ? 'block' : 'none';
            
            // Save current view to localStorage
            localStorage.setItem('currentView', view);

            if (view === 'files') {
                // Clear file filter when switching to file view
                currentFileFilter = null;
                const filterInfo = document.getElementById('fileFilterInfo');
                if (filterInfo) {
                    filterInfo.remove();
                }
                await loadFiles();
            } else {
                currentPage = 0;
                await loadDocuments();
            }
        }

        async function loadFiles() {
            const collection = document.getElementById('manageCollection').value || 'default';
            const listDiv = document.getElementById('fileListContainer');
            listDiv.innerHTML = '<div style="text-align: center; padding: 20px;">加载中...</div>';

            try {
                const response = await fetch(`${API_BASE}/list-files?collection=${collection}`);
                const data = await response.json();
                
                listDiv.innerHTML = '';
                
                if (!data.files || data.files.length === 0) {
                    listDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #666;">暂无文件</div>';
                    return;
                }

                data.files.forEach(file => {
                    const itemDiv = document.createElement('div');
                    itemDiv.className = 'file-item';
                    
                    const fileInfo = document.createElement('div');
                    fileInfo.className = 'file-info';
                    fileInfo.innerHTML = `
                        <div class="file-name">文件: ${file.filename}</div>
                        <div class="file-meta">
                            类型: ${file.file_type} | 片段数: ${file.chunk_count} | 总大小: ${(file.total_size / 1024).toFixed(1)} KB
                        </div>
                    `;
                    
                    const buttonContainer = document.createElement('div');
                    
                    const viewChunksBtn = document.createElement('button');
                    viewChunksBtn.className = 'btn';
                    viewChunksBtn.textContent = '查看片段';
                    viewChunksBtn.onclick = () => viewFileChunks(file.filename);
                    
                    const deleteBtn = document.createElement('button');
                    deleteBtn.className = 'btn btn-danger';
                    deleteBtn.textContent = '删除';
                    deleteBtn.onclick = () => deleteFile(file.filename);
                    
                    buttonContainer.appendChild(viewChunksBtn);
                    buttonContainer.appendChild(deleteBtn);
                    
                    itemDiv.appendChild(fileInfo);
                    itemDiv.appendChild(buttonContainer);
                    listDiv.appendChild(itemDiv);
                });
            } catch (error) {
                listDiv.innerHTML = `<div style="color: red; text-align: center;">加载失败: ${error.message}</div>`;
            }
        }

        async function deleteFile(filename) {
            if (!confirm(`确定要删除文件 "${filename}" 及其所有片段吗？`)) {
                return;
            }

            const collection = document.getElementById('manageCollection').value || 'default';
            try {
                const response = await fetch(`${API_BASE}/delete-file`, {
                    method: 'DELETE',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        filename: filename,
                        collection: collection
                    })
                });

                if (response.ok) {
                    showStatus('文件删除成功', true);
                    loadFiles();
                } else {
                    const error = await response.json();
                    showStatus('删除失败: ' + (error.detail || '未知错误'), false);
                }
            } catch (error) {
                showStatus('删除请求失败: ' + error.message, false);
            }
        }

        function viewFileChunks(filename) {
            currentFileFilter = filename;
            switchView('docs');
        }

        async function loadDocuments(page = 0) {
            if (currentView === 'files') {
                await loadFiles();
                return;
            }

            currentPage = page;
            const collection = document.getElementById('manageCollection').value;
            const listDiv = document.getElementById('documentList');
            
            // Remove any existing filter info first
            const existingFilterInfo = document.getElementById('fileFilterInfo');
            if (existingFilterInfo) {
                existingFilterInfo.remove();
            }
            
            // Show filter info if active
            if (currentFileFilter) {
                const filterInfo = document.createElement('div');
                filterInfo.id = 'fileFilterInfo';
                filterInfo.style.marginBottom = '15px';
                filterInfo.style.padding = '10px';
                filterInfo.style.background = '#e3f2fd';
                filterInfo.style.border = '1px solid #007acc';
                filterInfo.style.borderRadius = '5px';
                filterInfo.innerHTML = `
                    正在显示文件的片段: <strong>${currentFileFilter}</strong>
                    <button class="btn" onclick="clearFileFilter()" style="margin-left: 10px;">显示全部</button>
                `;
                listDiv.parentElement.insertBefore(filterInfo, listDiv);
            }
            
            listDiv.innerHTML = '<div style="text-align: center; padding: 20px;">加载中...</div>';
            
            try {
                // Add filename parameter if filtering by file
                const filenameParam = currentFileFilter ? `&filename=${encodeURIComponent(currentFileFilter)}` : '';
                const response = await fetch(`${API_BASE}/list-documents?collection=${collection}&limit=${pageSize}&offset=${page * pageSize}${filenameParam}`);
                if (!response.ok) throw new Error('Failed to load documents');
                
                const data = await response.json();
                
                listDiv.innerHTML = '';
                
                // No need to filter on frontend anymore - backend handles it
                const filteredDocs = data.documents;
                
                if (filteredDocs.length === 0) {
                    listDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #666;">暂无文档</div>';
                } else {
                    filteredDocs.forEach(doc => {
                        const item = document.createElement('div');
                        item.className = 'file-item';
                        item.innerHTML = `
                            <div class="file-info">
                                <div class="file-name">ID: ${doc.id}</div>
                                <div class="file-meta">
                                    ${doc.metadata.filename ? `文件: ${doc.metadata.filename}` : ''}
                                    ${doc.metadata.timestamp ? ` | 时间: ${new Date(doc.metadata.timestamp).toLocaleString()}` : ''}
                                </div>
                                <div class="preview-content" style="margin-top: 5px; max-height: 100px;">
                                    ${doc.content.substring(0, 200)}${doc.content.length > 200 ? '...' : ''}
                                </div>
                            </div>
                            <div>
                                <button class="btn btn-danger" onclick="deleteDocument('${doc.id}')">删除</button>
                            </div>
                        `;
                        listDiv.appendChild(item);
                    });
                }
                
                // Update pagination
                document.getElementById('pageInfo').textContent = `第 ${currentPage + 1} 页`;
                document.getElementById('prevPageBtn').disabled = currentPage === 0;
                document.getElementById('nextPageBtn').disabled = filteredDocs.length < pageSize;
                
            } catch (error) {
                listDiv.innerHTML = `<div style="color: red; text-align: center;">加载失败: ${error.message}</div>`;
            }
        }

        async function deleteDocument(docId) {
            if (!confirm('确定要删除这个文档吗？')) return;
            
            const collection = document.getElementById('manageCollection').value;
            
            try {
                const response = await fetch(`${API_BASE}/delete-document`, {
                    method: 'DELETE',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        document_id: docId,
                        collection: collection
                    })
                });
                
                if (response.ok) {
                    showManageStatus('删除成功', true);
                    loadDocuments(currentPage);
                } else {
                    const error = await response.json();
                    showManageStatus('删除失败: ' + (error.detail || '未知错误'), false);
                }
            } catch (error) {
                showManageStatus('删除失败: ' + error.message, false);
            }
        }

        function prevPage() {
            if (currentPage > 0) {
                loadDocuments(currentPage - 1);
            }
        }

        function nextPage() {
            loadDocuments(currentPage + 1);
        }

        function showManageStatus(message, isSuccess = true) {
            const statusDiv = document.getElementById('manageStatusMessage');
            statusDiv.textContent = message;
            statusDiv.className = `status-message ${isSuccess ? 'status-success' : 'status-error'}`;
            statusDiv.style.display = 'block';
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }

        function clearFileFilter() {
            currentFileFilter = null;
            // Remove filter info element by ID
            const filterInfo = document.getElementById('fileFilterInfo');
            if (filterInfo) {
                filterInfo.remove();
            }
            loadDocuments(0);
        }

        // Initialize
        document.addEventListener('DOMContentLoaded', async function() {
            await loadCollections();
            
            // Restore tab state from localStorage
            const savedTab = localStorage.getItem('currentTab');
            if (savedTab && ['upload', 'search', 'chat', 'manage'].includes(savedTab)) {
                switchTab(savedTab);
            }
            
            // Restore view state from localStorage
            const savedView = localStorage.getItem('currentView');
            if (savedView && ['files', 'docs'].includes(savedView)) {
                switchView(savedView);
            } else {
                switchView('files');
            }

            // File input handling
            document.getElementById('fileInput').addEventListener('change', function(e) {
                handleFileSelect(e.target.files);
            });

            // Drag and drop handling
            const uploadArea = document.getElementById('uploadArea');
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            uploadArea.addEventListener('dragleave', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
            });
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                handleFileSelect(e.dataTransfer.files);
            });

            // Auto upload when files are selected
            document.getElementById('fileInput').addEventListener('change', function() {
                if (uploadedFiles.length > 0) {
                    uploadFiles();
                }
            });
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.get("/config-page", response_class=HTMLResponse)
async def config_page():
    """Serve the configuration page."""
    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MCP-RAG 配置管理</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .section h2 {
            color: #555;
            margin-top: 0;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
        }
        .form-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }
        input, select {
            width: 100%;
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 14px;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
        }
        .checkbox-group input {
            width: auto;
            margin-right: 10px;
        }
        .btn {
            background-color: #007acc;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin-right: 10px;
        }
        .btn:hover {
            background-color: #005aa3;
        }
        .btn-danger {
            background-color: #dc3545;
        }
        .btn-danger:hover {
            background-color: #c82333;
        }
        .btn-success {
            background-color: #28a745;
        }
        .btn-success:hover {
            background-color: #218838;
        }
        .status {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
            display: none;
        }
        .status.success {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        .status.error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        .current-config {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .config-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }
        .config-key {
            font-weight: bold;
        }
        .config-value {
            color: #007acc;
        }
    </style>
</head>
<body>
    <div class="container">
        <div style="text-align: center; margin-bottom: 20px;">
            <a href="/config-page" style="margin: 0 10px; color: #007acc; text-decoration: none;">配置管理</a> |
            <a href="/documents-page" style="margin: 0 10px; color: #007acc; text-decoration: none;">资料管理</a>
        </div>
        <h1>MCP-RAG 配置管理</h1>

        <div id="currentConfig" class="current-config">
            <h3>当前配置</h3>
            <div id="configDisplay"></div>
        </div>

        <div class="section">
            <h2>服务器设置</h2>
            <div class="form-group">
                <label for="host">主机地址:</label>
                <input type="text" id="host" placeholder="0.0.0.0">
            </div>
            <div class="form-group">
                <label for="http_port">HTTP端口:</label>
                <input type="number" id="http_port" placeholder="8060">
            </div>
            <div class="form-group">
                <div class="checkbox-group">
                    <input type="checkbox" id="debug">
                    <label for="debug">调试模式</label>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>向量数据库设置</h2>
            <div class="form-group">
                <label for="vector_db_type">数据库类型:</label>
                <select id="vector_db_type">
                    <option value="chroma">ChromaDB</option>
                    <option value="qdrant">Qdrant</option>
                </select>
            </div>
            <div class="form-group">
                <label for="chroma_persist_directory">ChromaDB 数据目录:</label>
                <input type="text" id="chroma_persist_directory" placeholder="./data/chroma">
            </div>
            <div class="form-group">
                <label for="qdrant_url">Qdrant 服务器地址:</label>
                <input type="text" id="qdrant_url" placeholder="http://localhost:6333">
            </div>
        </div>

        <div class="section">
            <h2>嵌入模型设置</h2>
            <div class="form-group">
                <label for="embedding_provider">嵌入提供商:</label>
                <select id="embedding_provider">
                    <option value="doubao">豆包 (Doubao)</option>
                    <option value="zhipu">智谱 (Zhipu)</option>
                    <option value="local">本地模型</option>
                </select>
            </div>

            <!-- Provider Tabs -->
            <div id="provider-tabs" class="tabs" style="margin-top: 20px; border-bottom: 1px solid #ddd; display: flex;">
                 <div class="tab active" onclick="switchProviderTab('doubao')" id="tab-doubao" style="padding: 10px 20px; cursor: pointer; background: white; border: 1px solid #ddd; border-bottom: 1px solid white; margin-bottom: -1px; border-radius: 5px 5px 0 0;">Doubao 设置</div>
                 <div class="tab" onclick="switchProviderTab('zhipu')" id="tab-zhipu" style="padding: 10px 20px; cursor: pointer; background: #f5f5f5; border: 1px solid #ddd; border-bottom: none; margin-bottom: -1px; border-radius: 5px 5px 0 0; margin-left: 5px;">Zhipu 设置</div>
            </div>

            <!-- Doubao Config -->
            <div id="content-doubao" class="provider-content" style="padding: 20px; border: 1px solid #ddd; border-top: none;">
                <div class="form-group">
                    <label>Doubao API地址:</label>
                    <input type="text" id="doubao_base_url" placeholder="https://ark.cn-beijing.volces.com/api/v3">
                </div>
                <div class="form-group">
                    <label>Doubao 模型:</label>
                    <input type="text" id="doubao_model" placeholder="doubao-embedding-text-240715">
                </div>
                <div class="form-group">
                    <label>Doubao API密钥:</label>
                    <input type="text" id="doubao_api_key" placeholder="您的豆包API密钥">
                </div>
            </div>

            <!-- Zhipu Config -->
            <div id="content-zhipu" class="provider-content" style="display: none; padding: 20px; border: 1px solid #ddd; border-top: none;">
                <div class="form-group">
                    <label>Zhipu API地址:</label>
                    <input type="text" id="zhipu_base_url" placeholder="https://open.bigmodel.cn/api/paas/v4">
                </div>
                <div class="form-group">
                    <label>Zhipu 模型:</label>
                    <input type="text" id="zhipu_model" placeholder="embedding-3">
                </div>
                <div class="form-group">
                    <label>Zhipu API密钥:</label>
                    <input type="text" id="zhipu_api_key" placeholder="您的智谱API密钥">
                </div>
            </div>
            <!-- LLM Settings Section -->
            <div style="margin-top: 30px; border-top: 1px dashed #ddd; padding-top: 20px;">
                <h3>LLM 设置 (用于总结)</h3>
                <div class="form-group">
                    <label for="llm_api_key">LLM API 密钥:</label>
                    <input type="text" id="llm_api_key" placeholder="可选 (一般不用写)">
                </div>
                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="enable_llm_summary">
                        <label for="enable_llm_summary">启用LLM总结</label>
                    </div>
                </div>
                <div class="form-group">
                    <div class="checkbox-group">
                        <input type="checkbox" id="enable_thinking">
                        <label for="enable_thinking">启用深度思考</label>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>RAG 设置</h2>
            <div class="form-group">
                <label for="max_retrieval_results">最大检索结果数:</label>
                <input type="number" id="max_retrieval_results" min="1" max="20" placeholder="5">
            </div>
            <div class="form-group">
                <label for="similarity_threshold">相似度阈值:</label>
                <input type="number" id="similarity_threshold" min="0" max="1" step="0.1" placeholder="0.7">
            </div>
            <div class="form-group">
                <div class="checkbox-group">
                    <input type="checkbox" id="enable_reranker">
                    <label for="enable_reranker">启用重排序</label>
                </div>
            </div>
            <div class="form-group">
                <div class="checkbox-group">
                    <input type="checkbox" id="enable_cache">
                    <label for="enable_cache">启用缓存</label>
                </div>
            </div>
        </div>

        <div class="section">
            <button class="btn btn-success" onclick="loadConfig()">加载配置</button>
            <button class="btn" onclick="saveAllConfig()">保存所有配置</button>
            <button class="btn btn-danger" onclick="resetConfig()">重置为默认</button>
        </div>

        <div id="status" class="status"></div>
    </div>

    <script>
        const API_BASE = '';

        async function showStatus(message, isSuccess = true) {
            const statusDiv = document.getElementById('status');
            statusDiv.textContent = message;
            statusDiv.className = `status ${isSuccess ? 'success' : 'error'}`;
            statusDiv.style.display = 'block';
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 5000);
        }

        function switchProviderTab(provider) {
            // Update tabs
            document.querySelectorAll('.tab').forEach(el => {
                el.classList.remove('active');
                el.style.background = '#f5f5f5';
                el.style.borderBottom = 'none';
            });
            const activeTab = document.getElementById(`tab-${provider}`);
            if(activeTab) {
                activeTab.classList.add('active');
                activeTab.style.background = 'white';
                activeTab.style.borderBottom = '1px solid white';
            }

            // Update content
            document.querySelectorAll('.provider-content').forEach(el => el.style.display = 'none');
            const content = document.getElementById(`content-${provider}`);
            if(content) content.style.display = 'block';
        }

        async function loadConfig() {
            try {
                const response = await fetch(`${API_BASE}/config`);
                const config = await response.json();

                // Fill form fields
                Object.keys(config).forEach(key => {
                    const element = document.getElementById(key);
                    if (element) {
                        if (element.type === 'checkbox') {
                            element.checked = config[key];
                        } else {
                            element.value = config[key] || '';
                        }
                    }
                });
                
                // Load provider configs
                if (config.provider_configs) {
                    if (config.provider_configs.doubao) {
                        document.getElementById('doubao_base_url').value = config.provider_configs.doubao.base_url || '';
                        document.getElementById('doubao_model').value = config.provider_configs.doubao.model || '';
                        document.getElementById('doubao_api_key').value = config.provider_configs.doubao.api_key || '';
                    }
                    if (config.provider_configs.zhipu) {
                        document.getElementById('zhipu_base_url').value = config.provider_configs.zhipu.base_url || '';
                        document.getElementById('zhipu_model').value = config.provider_configs.zhipu.model || '';
                        document.getElementById('zhipu_api_key').value = config.provider_configs.zhipu.api_key || '';
                    }
                }

                // Display current config
                displayCurrentConfig(config);
                showStatus('配置加载成功', true);
            } catch (error) {
                showStatus('加载配置失败: ' + error.message, false);
            }
        }

        function displayCurrentConfig(config) {
            const displayDiv = document.getElementById('configDisplay');
            displayDiv.innerHTML = '';

            Object.entries(config).forEach(([key, value]) => {
                const itemDiv = document.createElement('div');
                itemDiv.className = 'config-item';
                itemDiv.innerHTML = `
                    <span class="config-key">${key}:</span>
                    <span class="config-value">${value}</span>
                `;
                displayDiv.appendChild(itemDiv);
            });
        }

        async function saveAllConfig() {
            const updates = {};
            
            // Build provider_configs object
            const provider_configs = {
                doubao: {
                    base_url: document.getElementById('doubao_base_url').value,
                    model: document.getElementById('doubao_model').value,
                    api_key: document.getElementById('doubao_api_key').value || null
                },
                zhipu: {
                    base_url: document.getElementById('zhipu_base_url').value,
                    model: document.getElementById('zhipu_model').value,
                    api_key: document.getElementById('zhipu_api_key').value || null
                }
            };
            updates['provider_configs'] = provider_configs;

            // Collect all other form values
            const inputs = document.querySelectorAll('input, select');
            inputs.forEach(input => {
                if (input.id && !input.id.startsWith('doubao_') && !input.id.startsWith('zhipu_')) {
                    if (input.type === 'checkbox') {
                        updates[input.id] = input.checked;
                    } else if (input.type === 'number') {
                        updates[input.id] = parseFloat(input.value) || 0;
                    } else {
                        updates[input.id] = input.value || null;
                    }
                }
            });

            try {
                const response = await fetch(`${API_BASE}/config/bulk`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ updates })
                });

                if (response.ok) {
                    showStatus('配置保存成功', true);
                    loadConfig(); // Reload to show updated config
                } else {
                    const error = await response.json();
                    showStatus('保存失败: ' + (error.detail || '未知错误'), false);
                }
            } catch (error) {
                showStatus('保存配置失败: ' + error.message, false);
            }
        }

        async function resetConfig() {
            if (!confirm('确定要重置所有配置为默认值吗？')) {
                return;
            }

            try {
                const response = await fetch(`${API_BASE}/config/reset`, {
                    method: 'POST'
                });

                if (response.ok) {
                    showStatus('配置已重置为默认值', true);
                    loadConfig();
                } else {
                    const error = await response.json();
                    showStatus('重置失败: ' + (error.detail || '未知错误'), false);
                }
            } catch (error) {
                showStatus('重置配置失败: ' + error.message, false);
            }
        }

        // Load config on page load
        window.onload = loadConfig;
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.get("/config")
async def get_config():
    """Get current configuration."""
    return config_manager.get_all_settings()


@app.post("/config")
async def update_config(config: ConfigUpdate):
    """Update a single configuration setting."""
    success = config_manager.update_setting(config.key, config.value)
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to update config {config.key}")
    return {"message": f"Config {config.key} updated successfully"}


@app.post("/config/bulk")
async def update_config_bulk(config: BulkConfigUpdate):
    """Update multiple configuration settings."""
    success = config_manager.update_settings(config.updates)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to update config")
    return {"message": "Config updated successfully"}


@app.post("/config/reset")
async def reset_config():
    """Reset configuration to defaults."""
    success = config_manager.reset_to_defaults()
    if not success:
        raise HTTPException(status_code=400, detail="Failed to reset config")
    return {"message": "Config reset to defaults successfully"}


@app.post("/add-document")
async def add_document(doc: DocumentAdd):
    """Add a single document."""
    try:
        vector_db = await get_vector_database()
        await vector_db.add_document(
            content=doc.content,
            collection_name=doc.collection,
            metadata=doc.metadata
        )
        return {"message": "Document added successfully"}
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-files")
async def upload_files(
    files: List[UploadFile] = File(...),
    collection: str = Form("default")
):
    """Upload and process multiple files."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    processor = get_document_processor()
    results = []

    for file in files:
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as temp_file:
                shutil.copyfileobj(file.file, temp_file)
                temp_path = Path(temp_file.name)

            # Process the file
            processed_doc = processor.process_file(temp_path, file.filename)

            # Clean up temp file
            temp_path.unlink()

            # Add to vector database if processing was successful
            if not processed_doc.error and processed_doc.content.strip():
                try:
                    vector_db = await get_vector_database()
                    await vector_db.add_document(
                        content=processed_doc.content,
                        collection_name=collection,
                        metadata={
                            **processed_doc.metadata,
                            "filename": processed_doc.filename,
                            "file_type": processed_doc.file_type,
                            "source": "upload"
                        }
                    )
                    processed = True
                    error = ""
                except Exception as e:
                    processed = False
                    error = f"Failed to add to database: {str(e)}"
            else:
                processed = False
                error = processed_doc.error or "No content extracted"

            # Create preview (first 500 characters)
            preview = processed_doc.content[:500] + "..." if len(processed_doc.content) > 500 else processed_doc.content

            result = FileUploadResponse(
                filename=file.filename,
                file_type=processed_doc.file_type,
                content_length=len(processed_doc.content),
                processed=processed,
                error=error,
                preview=preview if processed else ""
            )
            results.append(result)

        except Exception as e:
            result = FileUploadResponse(
                filename=file.filename,
                file_type="unknown",
                content_length=0,
                processed=False,
                error=str(e),
                preview=""
            )
            results.append(result)

    return BatchUploadResponse(
        total_files=len(files),
        successful=len([r for r in results if r.processed]),
        failed=len([r for r in results if not r.processed]),
        results=results
    )


@app.get("/collections")
async def list_collections():
    """List all collections."""
    try:
        vector_db = await get_vector_database()
        collections = await vector_db.list_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat")
async def chat_with_knowledge_base(chat_request: dict):
    """Chat with knowledge base using LLM."""
    try:
        query = chat_request.get("query", "")
        collection = chat_request.get("collection", "default")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")

        # Get components
        vector_db = await get_vector_database()
        embedding_model = await get_embedding_model()

        # Encode query and search
        query_embedding = await embedding_model.encode_single(query)
        search_results = await vector_db.search(
            query_embedding=query_embedding,
            collection_name=collection,
            limit=5
        )

        # Combine retrieved content
        context = "\n\n".join([
            f"文档 {i+1}: {r.document.content}"
            for i, r in enumerate(search_results)
        ])

        # Generate response using LLM
        from .llm import get_llm_model
        llm_model = await get_llm_model()
        
        prompt = f"""基于以下知识库内容回答用户的问题。如果知识库内容不足以回答问题，请说明无法找到相关信息。

知识库内容:
{context}

用户问题: {query}

请提供准确、简洁的回答:"""
        
        response = await llm_model.generate(prompt)

        return {
            "query": query,
            "response": response,
            "sources": [
                {
                    "content": r.document.content[:200] + "..." if len(r.document.content) > 200 else r.document.content,
                    "score": r.score,
                    "metadata": r.document.metadata
                } for r in search_results
            ]
        }
    except Exception as e:
        logger.error(f"Failed to chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_documents(query: str, collection: str = "default", limit: int = 5):
    """Search documents."""
    try:
        logger.info(f"Searching for '{query}' in collection '{collection}'")

        # Get components
        vector_db = await get_vector_database()
        embedding_model = await get_embedding_model()

        # Encode query
        query_embedding = await embedding_model.encode_single(query)

        # Search
        results = await vector_db.search(
            query_embedding=query_embedding,
            collection_name=collection,
            limit=limit
        )

        # Check if LLM summary is enabled
        if settings.enable_llm_summary:
            try:
                from .llm import get_llm_model
                llm_model = await get_llm_model()

                # Combine all retrieved content
                combined_content = "\n\n".join([
                    f"文档 {i+1} (相似度: {r.score:.3f}):\n{r.document.content}"
                    for i, r in enumerate(results)
                ])

                # Generate summary using LLM
                summary = await llm_model.summarize(combined_content, query)

                return {
                    "query": query,
                    "collection": collection,
                    "summary": summary,
                    "results": [
                        {
                            "content": r.document.content,
                            "score": r.score,
                            "metadata": r.document.metadata
                        } for r in results
                    ]
                }
            except Exception as llm_error:
                logger.warning(f"LLM summary failed, falling back to direct results: {llm_error}")
                # Fall back to direct results if LLM fails

        # Return direct search results
        return {
            "query": query,
            "collection": collection,
            "results": [
                {
                    "content": r.document.content,
                    "score": r.score,
                    "metadata": r.document.metadata
                } for r in results
            ]
        }
    except Exception as e:
        logger.error(f"Failed to search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-documents")
async def list_documents(collection: str = "default", limit: int = 100, offset: int = 0, filename: str = None):
    """List documents in a collection."""
    try:
        db = await get_vector_database()
        result = await db.list_documents(collection_name=collection, limit=limit, offset=offset, filename=filename)
        return result
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-document")
async def delete_document(request: DeleteDocumentRequest):
    """Delete a document."""
    try:
        db = await get_vector_database()
        success = await db.delete_document(document_id=request.document_id, collection_name=request.collection)
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found or failed to delete")
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-files")
async def list_files(collection: str = "default"):
    """List files in a collection."""
    try:
        db = await get_vector_database()
        result = await db.list_files(collection_name=collection)
        return {"files": result}
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-file")
async def delete_file(request: DeleteFileRequest):
    """Delete a file."""
    try:
        db = await get_vector_database()
        success = await db.delete_file(filename=request.filename, collection_name=request.collection)
        if success:
            return {"message": "File deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="File not found or failed to delete")
    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
