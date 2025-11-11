"""Vector database management for MCP-RAG."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document data structure."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Search result data structure."""
    document: Document
    score: float


class VectorDatabase(ABC):
    """Abstract base class for vector databases."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector database."""
        pass

    @abstractmethod
    async def add_document(self, content: str, collection_name: str = "default", metadata: Dict[str, Any] = None) -> None:
        """Add a single document to the collection."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        collection_name: str = "default",
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str = "default") -> None:
        """Delete a collection."""
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all collections."""
        pass


class ChromaDatabase(VectorDatabase):
    """Chroma vector database implementation."""

    def __init__(self, embedding_function=None):
        self.client: Optional[chromadb.Client] = None
        self.collections: Dict[str, chromadb.Collection] = {}
        self.embedding_function = embedding_function

    async def initialize(self) -> None:
        """Initialize Chroma client."""
        try:
            chroma_settings = ChromaSettings(
                persist_directory=settings.chroma_persist_directory,
                is_persistent=True
            )
            self.client = chromadb.PersistentClient(path=settings.chroma_persist_directory)
            logger.info(f"Chroma database initialized at {settings.chroma_persist_directory}")

            # Ensure default collection exists
            await self._ensure_default_collection()
        except Exception as e:
            logger.error(f"Failed to initialize Chroma database: {e}")
            raise

    async def _ensure_default_collection(self) -> None:
        """Ensure the default collection exists with correct configuration."""
        if not self.client:
            return

        try:
            # Try to get the default collection
            try:
                collection = self.client.get_collection(name="default")
                # Check if collection uses the correct distance metric
                current_space = collection.metadata.get("hnsw:space") if collection.metadata else None
                if current_space != "cosine":
                    logger.warning(f"Default collection uses distance metric '{current_space}'. Recreating with cosine.")
                    self.client.delete_collection(name="default")
                    collection = self.client.create_collection(
                        name="default",
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info("Recreated default collection with cosine similarity")
            except Exception:
                # Collection doesn't exist, create it with cosine similarity
                collection = self.client.create_collection(
                    name="default",
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info("Created default collection with cosine similarity")

        except Exception as e:
            logger.error(f"Failed to ensure default collection: {e}")
            # Don't raise here as this is not critical for initialization

    async def add_document(self, content: str, collection_name: str = "default", metadata: Dict[str, Any] = None) -> None:
        """Add a single document to Chroma collection."""
        if metadata is None:
            metadata = {}

        document = Document(
            id=f"doc_{len(content)}_{hash(content)}",  # Simple ID generation
            content=content,
            metadata=metadata
        )

        await self.add_documents([document], collection_name)

    async def add_documents(self, documents: List[Document], collection_name: str = "default") -> None:
        """Add multiple documents to Chroma collection."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        try:
            # Get or create collection
            try:
                collection = self.client.get_collection(name=collection_name)
                # Check if collection uses the correct distance metric
                current_space = collection.metadata.get("hnsw:space") if collection.metadata else None
                if current_space != "cosine":
                    logger.warning(f"Collection '{collection_name}' uses distance metric '{current_space}'. Recreating with cosine.")
                    self.client.delete_collection(name=collection_name)
                    collection = self.client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Recreated collection '{collection_name}' with cosine similarity")
            except Exception:
                # Collection doesn't exist, create it with cosine similarity
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created collection '{collection_name}' with cosine similarity")

            # Prepare data for Chroma
            ids = [doc.id for doc in documents]
            contents = [doc.content for doc in documents]
            metadatas = [doc.metadata for doc in documents]

            # Calculate embeddings for documents
            from .embedding import get_embedding_model
            embedding_model = await get_embedding_model()
            embeddings = await embedding_model.encode(contents)

            # Add documents to collection with embeddings
            collection.add(
                documents=contents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )

            logger.info(f"Added {len(documents)} documents to collection '{collection_name}'")

        except Exception as e:
            logger.error(f"Failed to add documents to collection '{collection_name}': {e}")
            raise

    async def search(
        self,
        query_embedding: List[float],
        collection_name: str = "default",
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[SearchResult]:
        """Search Chroma collection using built-in vector search."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        try:
            collection = self.client.get_collection(name=collection_name)
            if not collection:
                logger.warning(f"Collection '{collection_name}' not found")
                return []

            # Use ChromaDB's built-in vector search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                include=["documents", "metadatas", "distances"]
            )

            # Create search results
            search_results = []
            if results["distances"] and results["documents"] and len(results["distances"]) > 0:
                distances = results["distances"][0]
                documents = results["documents"][0]
                ids = results["ids"][0] if results["ids"] else []
                metadatas = results["metadatas"][0] if results["metadatas"] else []

                for i, distance in enumerate(distances):
                    # For cosine distance, similarity = 1 - distance
                    similarity = 1 - distance

                    if similarity >= threshold:
                        document = Document(
                            id=ids[i] if i < len(ids) else f"result_{i}",
                            content=documents[i],
                            metadata=metadatas[i] if i < len(metadatas) else {}
                        )
                        search_results.append(SearchResult(document=document, score=float(similarity)))

            logger.info(f"Found {len(search_results)} results above threshold {threshold}")
            return search_results

        except Exception as e:
            logger.error(f"Failed to search collection '{collection_name}': {e}")
            raise

    async def delete_collection(self, collection_name: str = "default") -> None:
        """Delete Chroma collection."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
            logger.info(f"Deleted collection '{collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection '{collection_name}': {e}")
            raise

    async def list_collections(self) -> List[str]:
        """List all Chroma collections."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise


# Global database instance
vector_db: Optional[VectorDatabase] = None


async def get_vector_database() -> VectorDatabase:
    """Get the global vector database instance."""
    global vector_db
    if vector_db is None:
        if settings.vector_db_type == "chroma":
            vector_db = ChromaDatabase()
        else:
            raise ValueError(f"Unsupported vector database type: {settings.vector_db_type}")
        await vector_db.initialize()
    return vector_db