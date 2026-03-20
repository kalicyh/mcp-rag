"""Embedding providers for MCP-RAG indexing."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

try:
    import httpx
except ImportError:  # pragma: no cover - dependency guard
    httpx = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


class EmbeddingModel(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    async def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    async def encode(self, texts: Sequence[str]) -> List[List[float]]:
        raise NotImplementedError

    async def encode_single(self, text: str) -> List[float]:
        vectors = await self.encode([text])
        return vectors[0] if vectors else []

    async def close(self) -> None:
        return None


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    """Local sentence-transformers embedding model."""

    def __init__(
        self,
        model_name: str = "m3e-small",
        device: str = "cpu",
        cache_dir: str | None = None,
    ):
        if SentenceTransformer is None:  # pragma: no cover - dependency guard
            raise RuntimeError("sentence-transformers is not installed")

        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model = None

    async def initialize(self) -> None:
        model_mapping = {
            "m3e-small": "moka-ai/m3e-small",
            "e5-small": "intfloat/e5-small-v2",
        }
        actual_model = model_mapping.get(self.model_name, self.model_name)
        self.model = SentenceTransformer(
            actual_model,
            device=self.device,
            cache_folder=self.cache_dir,
        )
        logger.info("Initialized sentence-transformers model %s on %s", actual_model, self.device)

    async def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if self.model is None:
            raise RuntimeError("Embedding model not initialized")

        def _encode() -> List[List[float]]:
            return self.model.encode(texts, convert_to_numpy=True).tolist()

        return await asyncio.to_thread(_encode)


class OpenAICompatibleEmbeddingModel(EmbeddingModel):
    """OpenAI-compatible embeddings provider used by Doubao/Zhipu/OpenAI."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str,
        model: str,
        dimensions: int | None = None,
    ):
        if httpx is None:  # pragma: no cover - dependency guard
            raise RuntimeError("httpx is not installed")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dimensions = dimensions
        self.client: Optional["httpx.AsyncClient"] = None

    async def initialize(self) -> None:
        if not self.api_key:
            raise ValueError("API key is required for embedding service")

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        logger.info("Initialized OpenAI-compatible embedding model %s at %s", self.model, self.base_url)

    async def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if self.client is None:
            raise RuntimeError("Embedding client not initialized")

        payload: dict[str, object] = {
            "model": self.model,
            "input": list(texts),
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        response = await self.client.post("/embeddings", json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Embedding API error: {response.status_code} - {response.text}")

        data = response.json()
        return [item["embedding"] for item in data["data"]]

    async def close(self) -> None:
        if self.client is not None:
            await self.client.aclose()
            self.client = None


DoubaoEmbeddingModel = OpenAICompatibleEmbeddingModel

