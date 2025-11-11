"""LLM model management for MCP-RAG."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

from .config import settings

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

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://ark.cn-beijing.volces.com/api/v3", model: str = "doubao-seed-1.6-250615"):
        if not DOUBAO_AVAILABLE:
            raise RuntimeError("httpx not available. Please install it with: pip install httpx")

        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
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
            if not settings.enable_thinking:
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


# Global LLM model instance
llm_model: Optional[LLMModel] = None


async def get_llm_model() -> LLMModel:
    """Get the global LLM model instance."""
    global llm_model
    if llm_model is None:
        if settings.llm_provider == "doubao":
            # Check if API key is available for Doubao
            if settings.llm_api_key:
                llm_model = DoubaoLLMModel(
                    api_key=settings.llm_api_key,
                    base_url=settings.llm_base_url,
                    model=settings.llm_model
                )
                await llm_model.initialize()
            else:
                raise ValueError("Doubao API key is required for LLM. Please configure it in the web interface.")
        elif settings.llm_provider == "ollama":
            llm_model = OllamaModel(
                base_url=settings.llm_base_url,
                model=settings.llm_model
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
    return llm_model