"""Configuration management for MCP-RAG service."""

import json
import os
from pathlib import Path
from typing import Optional, Dict
from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for a specific model provider."""
    base_url: str
    model: str
    api_key: Optional[str] = None


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


class ConfigManager:
    """Configuration manager with JSON persistence."""

    def __init__(self, config_file: str = "./data/config.json"):
        self.config_file = Path(config_file)
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        self._settings = None

    @property
    def settings(self) -> Settings:
        """Get current settings."""
        if self._settings is None:
            self._settings = self._load_settings()
        return self._settings

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
        else:
            # Create default config
            default_settings = Settings()
            self._save_settings(default_settings)
            return default_settings

    def _save_settings(self, settings: Settings) -> None:
        """Save settings to JSON file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(settings.model_dump(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save config to {self.config_file}: {e}")

    def update_setting(self, key: str, value) -> bool:
        """Update a single setting and save to file."""
        try:
            current_data = self.settings.model_dump()
            current_data[key] = value
            new_settings = Settings(**current_data)
            self._save_settings(new_settings)
            self._settings = new_settings
            return True
        except Exception as e:
            print(f"Failed to update setting {key}: {e}")
            return False

    def update_settings(self, updates: dict) -> bool:
        """Update multiple settings and save to file."""
        try:
            current_data = self.settings.model_dump()
            current_data.update(updates)
            new_settings = Settings(**current_data)
            self._save_settings(new_settings)
            self._settings = new_settings
            return True
        except Exception as e:
            print(f"Failed to update settings: {e}")
            return False

    def get_all_settings(self) -> dict:
        """Get all settings as dictionary."""
        return self.settings.model_dump()

    def reset_to_defaults(self) -> bool:
        """Reset all settings to defaults."""
        try:
            default_settings = Settings()
            self._save_settings(default_settings)
            self._settings = default_settings
            return True
        except Exception as e:
            print(f"Failed to reset settings: {e}")
            return False


# Global config manager instance
config_manager = ConfigManager()

# Backward compatibility - global settings instance
settings = config_manager.settings