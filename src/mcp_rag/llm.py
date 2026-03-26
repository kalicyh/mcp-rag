"""LLM model management for MCP-RAG."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from .config import canonical_provider_name

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available, Doubao LLM will not work")

DOUBAO_AVAILABLE = HTTPX_AVAILABLE


class LLMModel(ABC):
    """Abstract base class for LLM models."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        pass

    @abstractmethod
    async def summarize(self, content: str, query: str) -> str:
        """Summarize content based on query."""
        pass


class OllamaModel(LLMModel):
    """Ollama-based LLM model."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "qwen2:7b"):
        try:
            import ollama
            self.client = ollama.Client(host=base_url)
            self.model = model
            self.available = True
        except ImportError:
            logger.warning("ollama package not available")
            self.available = False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        if not self.available:
            raise RuntimeError("Ollama not available")

        try:
            import ollama
            response = self.client.generate(model=self.model, prompt=prompt, **kwargs)
            return response['response']
        except Exception as e:
            logger.error(f"Failed to generate with Ollama: {e}")
            raise

    async def summarize(self, content: str, query: str) -> str:
        """Summarize content based on query."""
        prompt = f"""基于以下查询：{query}

请总结以下内容的相关信息：

{content}

请提供简洁准确的总结："""
        return await self.generate(prompt)


class DoubaoLLMModel(LLMModel):
    """Doubao (豆包) LLM model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://ark.cn-beijing.volces.com/api/v3",
        model: str = "doubao-seed-1.6-250615",
        *,
        enable_thinking: bool = True,
    ):
        if not DOUBAO_AVAILABLE:
            raise RuntimeError("httpx not available. Please install it with: pip install httpx")

        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.enable_thinking = enable_thinking
        self.client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        """Initialize the Doubao client."""
        try:
            if not self.api_key:
                raise ValueError("Doubao API key is required. Please set the ARK_API_KEY environment variable or configure it in the web interface.")

            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=60.0
            )
            logger.info(f"Initialized Doubao LLM model: {self.model}")

        except Exception as e:
            logger.error(f"Failed to initialize Doubao LLM model: {e}")
            raise

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from prompt."""
        if not self.client:
            raise RuntimeError("Client not initialized")

        try:
            thinking_config = {}
            if not self.enable_thinking:
                thinking_config = {"thinking": {"type": "disabled"}}

            response = await self.client.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    **thinking_config
                }
            )

            if response.status_code != 200:
                raise RuntimeError(f"Doubao API error: {response.status_code} - {response.text}")

            data = response.json()
            return data["choices"][0]["message"]["content"]

        except Exception as e:
            logger.error(f"Failed to generate with Doubao: {e}")
            raise

    async def summarize(self, content: str, query: str) -> str:
        """Summarize content based on query."""
        prompt = f"""基于以下查询：{query}

请总结以下内容的相关信息：

{content}

请提供简洁准确的总结："""
        return await self.generate(prompt)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()


class OpenAICompatibleLLMModel(LLMModel):
    """Generic OpenAI-compatible LLM model."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ):
        if not HTTPX_AVAILABLE:
            raise RuntimeError("httpx not available. Please install it with: pip install httpx")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client: Optional[httpx.AsyncClient] = None

    async def initialize(self) -> None:
        if not self.api_key:
            raise ValueError("API key is required for OpenAI-compatible LLM providers.")

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )
        logger.info("Initialized OpenAI-compatible LLM model %s at %s", self.model, self.base_url)

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self.client:
            raise RuntimeError("Client not initialized")

        response = await self.client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                **kwargs,
            },
        )

        if response.status_code != 200:
            raise RuntimeError(f"LLM API error: {response.status_code} - {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def summarize(self, content: str, query: str) -> str:
        prompt = f"""基于以下查询：{query}

请总结以下内容的相关信息：

{content}

请提供简洁准确的总结："""
        return await self.generate(prompt)

    async def close(self) -> None:
        if self.client:
            await self.client.aclose()


def _provider_llm_config(settings_obj: Any, provider: str) -> tuple[str, str, Optional[str]]:
    provider_name = canonical_provider_name(provider)
    provider_configs = getattr(settings_obj, "provider_configs", {}) or {}
    provider_config = provider_configs.get(provider_name)

    model = str(
        getattr(provider_config, "llm_model", None)
        or getattr(provider_config, "model", None)
        or getattr(settings_obj, "llm_model", "")
        or ""
    )
    base_url = str(
        getattr(provider_config, "base_url", None)
        or getattr(settings_obj, "llm_base_url", "https://ark.cn-beijing.volces.com/api/v3")
        or ""
    )
    api_key = getattr(provider_config, "api_key", None) or getattr(settings_obj, "llm_api_key", None)
    return model, base_url, api_key


async def get_llm_model(settings_obj: Any) -> LLMModel:
    """Build an LLM model for the provided settings."""

    provider = canonical_provider_name(getattr(settings_obj, "llm_provider", "doubao") or "doubao")
    model_name, base_url, api_key = _provider_llm_config(settings_obj, provider)

    if provider == "doubao":
        if not api_key:
            raise ValueError("Doubao API key is required for LLM. Please configure it in the web interface.")
        llm_model = DoubaoLLMModel(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
            enable_thinking=bool(getattr(settings_obj, "enable_thinking", True)),
        )
        await llm_model.initialize()
        return llm_model

    if provider == "ollama":
        return OllamaModel(
            base_url=base_url,
            model=model_name,
        )

    provider_configs = getattr(settings_obj, "provider_configs", {}) or {}
    if provider in provider_configs:
        llm_model = OpenAICompatibleLLMModel(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
        )
        await llm_model.initialize()
        return llm_model

    raise ValueError(f"Unsupported LLM provider: {provider}")
