"""Embedding model management for MCP-RAG."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from sentence_transformers import SentenceTransformer

from .config import settings

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available, Doubao embedding will not work")

DOUBAO_AVAILABLE = HTTPX_AVAILABLE


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings."""
        pass

    @abstractmethod
    async def encode_single(self, text: str) -> List[float]:
        """Encode single text to embedding."""
        pass


class SentenceTransformerModel(EmbeddingModel):
    """SentenceTransformer-based embedding model."""

    def __init__(self, model_name: str = "m3e-small", device: str = "cpu", cache_dir: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.model: Optional[SentenceTransformer] = None

    async def initialize(self) -> None:
        """Initialize the embedding model."""
        try:
            # Map model names to actual model identifiers
            model_mapping = {
                "m3e-small": "moka-ai/m3e-small",
                "e5-small": "intfloat/e5-small-v2"
            }

            actual_model_name = model_mapping.get(self.model_name, self.model_name)

            self.model = SentenceTransformer(
                actual_model_name,
                device=self.device,
                cache_folder=self.cache_dir
            )
            logger.info(f"Initialized embedding model: {actual_model_name} on {self.device}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model {self.model_name}: {e}")
            raise

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to embeddings."""
        if not self.model:
            raise RuntimeError("Model not initialized")

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise

    async def encode_single(self, text: str) -> List[float]:
        """Encode single text to embedding."""
        embeddings = await self.encode([text])
        return embeddings[0]


class DoubaoEmbeddingModel(EmbeddingModel):
    """Doubao (豆包) embedding model."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://ark.cn-beijing.volces.com/api/v3", model: str = "doubao-embedding-text-240715"):
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
                raise ValueError("API key is required for Doubao embedding")

            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                timeout=30.0
            )
            logger.info(f"Initialized Doubao embedding model: {self.model}")

        except Exception as e:
            logger.error(f"Failed to initialize Doubao embedding model: {e}")
            raise

    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode multiple texts to embeddings."""
        if not self.client:
            raise RuntimeError("Client not initialized")

        try:
            response = await self.client.post(
                "/embeddings",
                json={
                    "model": self.model,
                    "input": texts,
                    "encoding_format": "float"
                }
            )

            if response.status_code != 200:
                raise RuntimeError(f"Doubao API error: {response.status_code} - {response.text}")

            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings

        except Exception as e:
            logger.error(f"Failed to encode texts with Doubao: {e}")
            raise

    async def encode_single(self, text: str) -> List[float]:
        """Encode single text to embedding."""
        embeddings = await self.encode([text])
        return embeddings[0]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()


# Global embedding model instance
embedding_model: Optional[EmbeddingModel] = None


async def get_embedding_model() -> EmbeddingModel:
    """Get the global embedding model instance."""
    global embedding_model
    if embedding_model is None:
        if settings.embedding_provider == "doubao":
            embedding_model = DoubaoEmbeddingModel(
                api_key=settings.embedding_api_key,
                base_url=settings.embedding_base_url,
                model=settings.embedding_model
            )
        else:
            # Local SentenceTransformer models
            embedding_model = SentenceTransformerModel(
                model_name=settings.embedding_model,
                device=settings.embedding_device,
                cache_dir=settings.embedding_cache_dir
            )
        await embedding_model.initialize()
    return embedding_model