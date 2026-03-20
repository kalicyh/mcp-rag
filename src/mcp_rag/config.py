"""Configuration management for MCP-RAG service."""

from __future__ import annotations

import json
from contextlib import suppress
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for a specific model provider."""

    base_url: str
    model: str
    api_key: Optional[str] = None


class SecuritySettings(BaseModel):
    """Auth and tenant-scoped API key settings."""

    enabled: bool = Field(default=False, description="启用安全校验")
    allow_anonymous: bool = Field(default=True, description="允许匿名访问")
    api_keys: list[str] = Field(default_factory=list, description="全局 API Key 列表")
    tenant_api_keys: Dict[str, list[str]] = Field(default_factory=dict, description="按 tenant key 绑定的 API Key 列表")


class RateLimitSettings(BaseModel):
    """In-memory request rate limit settings."""

    requests_per_window: int = Field(default=120, description="窗口内允许的请求数")
    window_seconds: int = Field(default=60, description="限流窗口长度")
    burst: int = Field(default=30, description="预留突发额度")


class QuotaSettings(BaseModel):
    """Upload and indexing quota settings."""

    max_upload_files: int = Field(default=20, description="单次上传最大文件数")
    max_upload_bytes: int = Field(default=50 * 1024 * 1024, description="单次上传最大总字节数")
    max_upload_file_bytes: int = Field(default=10 * 1024 * 1024, description="单文件最大字节数")
    max_index_documents: int = Field(default=500, description="单次索引最大文档数")
    max_index_chunks: int = Field(default=2000, description="单次索引最大 chunk 数")
    max_index_chars: int = Field(default=500_000, description="单次索引最大字符数")


class ObservabilitySettings(BaseModel):
    """Thresholds for health summaries."""

    warning_error_rate: float = Field(default=0.05, description="健康摘要警告错误率")
    critical_error_rate: float = Field(default=0.2, description="健康摘要严重错误率")
    slow_request_ms: float = Field(default=1000.0, description="慢请求阈值（毫秒）")
    latency_window_size: int = Field(default=512, description="分位数统计保留的最近样本数量")


class CacheSettings(BaseModel):
    """Request-level retrieval cache settings."""

    enabled: bool = Field(default=False, description="启用请求级检索缓存")
    max_entries: int = Field(default=256, description="缓存最大条目数")
    ttl_seconds: int = Field(default=300, description="缓存 TTL（秒）")


class ProviderBudgetRule(BaseModel):
    """Budget and failure policy for one provider family."""

    requests_per_window: int = Field(default=60, description="窗口内允许的提供商请求数")
    window_seconds: int = Field(default=60, description="提供商预算窗口长度")
    burst: int = Field(default=10, description="提供商突发额度")
    failure_threshold: int = Field(default=3, description="连续失败后打开熔断")
    cooldown_seconds: int = Field(default=30, description="熔断冷却时间")


class ProviderBudgetSettings(BaseModel):
    """Provider-side budgets and circuit breaker controls."""

    enabled: bool = Field(default=True, description="启用 provider 预算和熔断")
    embeddings: ProviderBudgetRule = Field(
        default_factory=lambda: ProviderBudgetRule(requests_per_window=300, burst=60),
        description="Embedding provider 预算",
    )
    llm: ProviderBudgetRule = Field(
        default_factory=lambda: ProviderBudgetRule(requests_per_window=120, burst=20),
        description="LLM provider 预算",
    )


class Settings(BaseModel):
    """Application settings with JSON persistence."""

    # Server settings
    host: str = Field(default="0.0.0.0", description="服务器主机")
    port: int = Field(default=8060, description="服务器端口")
    http_port: int = Field(default=8060, description="HTTP API 服务器端口")
    debug: bool = Field(default=False, description="调试模式")

    # Vector database settings
    vector_db_type: str = Field(default="chroma", description="向量数据库类型")  # chroma or qdrant
    chroma_persist_directory: str = Field(default="./data/chroma", description="ChromaDB 数据目录")
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant 服务器地址")

    # Embedding model settings
    embedding_provider: str = Field(default="zhipu", description="嵌入提供商 (doubao, zhipu, m3e-small, e5-small)")
    embedding_fallback_provider: Optional[str] = Field(default=None, description="嵌入回退提供商")
    embedding_device: str = Field(default="cpu", description="嵌入设备")  # cpu or cuda (仅本地模型使用)
    embedding_cache_dir: Optional[str] = Field(default=None, description="嵌入缓存目录 (仅本地模型使用)")

    provider_configs: Dict[str, ProviderConfig] = Field(
        default_factory=lambda: {
            "doubao": ProviderConfig(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                model="doubao-embedding-text-240715",
                api_key=None
            ),
            "zhipu": ProviderConfig(
                base_url="https://open.bigmodel.cn/api/paas/v4",
                model="embedding-3",
                api_key=None
            )
        },
        description="各提供商的特定配置"
    )

    # LLM settings for summary mode
    llm_provider: str = Field(default="doubao", description="LLM 提供商")  # ollama, doubao, chatglm
    llm_fallback_provider: Optional[str] = Field(default=None, description="LLM 回退提供商")
    llm_model: str = Field(default="doubao-seed-1.6-250615", description="LLM 模型")
    llm_base_url: str = Field(default="https://ark.cn-beijing.volces.com/api/v3", description="LLM API 基础地址")
    llm_api_key: Optional[str] = Field(default=None, description="LLM API 密钥")
    enable_llm_summary: bool = Field(default=False, description="启用LLM总结")
    enable_thinking: bool = Field(default=True, description="启用深度思考")

    # RAG settings
    max_retrieval_results: int = Field(default=5, description="最大检索结果数")
    similarity_threshold: float = Field(default=0.7, description="相似度阈值")
    enable_reranker: bool = Field(default=False, description="启用重排序")
    enable_cache: bool = Field(default=False, description="启用缓存")
    cache: CacheSettings = Field(default_factory=CacheSettings, description="缓存配置")

    # Security and observability guardrails
    security: SecuritySettings = Field(default_factory=SecuritySettings, description="安全配置")
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings, description="限流配置")
    quotas: QuotaSettings = Field(default_factory=QuotaSettings, description="配额配置")
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings, description="观测配置")
    provider_budget: ProviderBudgetSettings = Field(default_factory=ProviderBudgetSettings, description="Provider 预算配置")


class ConfigManager:
    """Configuration manager with JSON persistence."""

    def __init__(self, config_file: str = "./data/config.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self._settings: Settings | None = None
        self._lock = RLock()
        self._last_mtime_ns: int | None = None
        self._revision = 0

    @property
    def settings(self) -> Settings:
        """Get current settings."""
        with self._lock:
            if self._settings is None:
                self._set_settings(self._load_settings())
            return self._settings

    @property
    def revision(self) -> int:
        """Expose the current in-memory revision."""

        with self._lock:
            return self._revision

    def _load_settings(self) -> Settings:
        """Load settings from JSON file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return Settings(**data)
            except Exception as e:
                print(f"Failed to load config from {self.config_file}: {e}")
                return Settings()
        return Settings()

    def _set_settings(self, settings_obj: Settings) -> Settings:
        self._settings = settings_obj
        self._last_mtime_ns = self._read_mtime_ns()
        self._revision += 1
        return settings_obj

    def _save_settings(self, settings: Settings) -> None:
        """Save settings to JSON file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(settings.model_dump(), f, ensure_ascii=False, indent=2)
            self._last_mtime_ns = self._read_mtime_ns()
        except Exception as e:
            print(f"Failed to save config to {self.config_file}: {e}")

    def _read_mtime_ns(self) -> int | None:
        with suppress(FileNotFoundError):
            return self.config_file.stat().st_mtime_ns
        return None

    def ensure_config_file(self) -> Settings:
        """Persist a default config file when the service boots without one."""

        with self._lock:
            if not self.config_file.exists():
                defaults = self._settings or Settings()
                self._save_settings(defaults)
                return self._set_settings(defaults)
            if self._settings is None:
                return self._set_settings(self._load_settings())
            self._last_mtime_ns = self._read_mtime_ns()
            return self._settings

    def update_setting(self, key: str, value) -> bool:
        """Update a single setting and save to file."""
        try:
            with self._lock:
                current_data = self.settings.model_dump()
                current_data[key] = value
                new_settings = Settings(**current_data)
                self._save_settings(new_settings)
                self._set_settings(new_settings)
                return True
        except Exception as e:
            print(f"Failed to update setting {key}: {e}")
            return False

    def update_settings(self, updates: dict) -> bool:
        """Update multiple settings and save to file."""
        try:
            with self._lock:
                current_data = self.settings.model_dump()
                current_data.update(updates)
                new_settings = Settings(**current_data)
                self._save_settings(new_settings)
                self._set_settings(new_settings)
                return True
        except Exception as e:
            print(f"Failed to update settings: {e}")
            return False

    def get_all_settings(self) -> dict:
        """Get all settings as dictionary."""
        with self._lock:
            settings_obj = self.reload_if_changed() or self.settings
            return settings_obj.model_dump()

    def reload(self) -> Settings:
        """Force reload settings from disk."""
        with self._lock:
            return self._set_settings(self._load_settings())

    def reload_if_changed(self) -> Settings | None:
        """Reload settings when the on-disk file timestamp changed."""

        with self._lock:
            current_mtime_ns = self._read_mtime_ns()
            if self._settings is None or current_mtime_ns != self._last_mtime_ns:
                return self._set_settings(self._load_settings())
            return None

    def reset_to_defaults(self) -> bool:
        """Reset all settings to defaults."""
        try:
            with self._lock:
                default_settings = Settings()
                self._save_settings(default_settings)
                self._set_settings(default_settings)
                return True
        except Exception as e:
            print(f"Failed to reset settings: {e}")
            return False


class SettingsProxy:
    """Live proxy that always reads the current manager settings."""

    def __init__(self, manager: ConfigManager):
        self._manager = manager

    @property
    def current(self) -> Settings:
        return self._manager.settings

    def reload(self) -> Settings:
        return self._manager.reload()

    def reload_if_changed(self) -> Settings | None:
        return self._manager.reload_if_changed()

    def ensure_config_file(self) -> Settings:
        return self._manager.ensure_config_file()

    @property
    def revision(self) -> int:
        return self._manager.revision

    def model_dump(self, *args, **kwargs) -> dict[str, Any]:
        return self.current.model_dump(*args, **kwargs)

    def model_copy(self, *args, **kwargs) -> Settings:
        return self.current.model_copy(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.current, name)


# Global config manager instance
config_manager = ConfigManager()

# Backward compatibility - global settings instance
settings = SettingsProxy(config_manager)
