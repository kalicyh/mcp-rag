"""Vector database management for MCP-RAG."""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re

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
    async def add_document(self, content: str, collection_name: str = "default", metadata: Dict[str, Any] = None, user_id: Optional[int] = None, agent_id: Optional[int] = None) -> None:
        """Add a single document to the collection."""
        pass

    @abstractmethod
    async def search(
        self,
        query_embedding: List[float],
        collection_name: str = "default",
        limit: int = 5,
        threshold: float = 0.7,
        user_id: Optional[int] = None, 
        agent_id: Optional[int] = None
    ) -> List[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str = "default", user_id: Optional[int] = None, agent_id: Optional[int] = None) -> None:
        """Delete a collection."""
        pass

    @abstractmethod
    async def list_collections(self, user_id: Optional[int] = None, agent_id: Optional[int] = None) -> List[str]:
        """List all collections."""
        pass

    @abstractmethod
    async def list_documents(self, collection_name: str = "default", limit: int = 100, offset: int = 0, filename: Optional[str] = None, user_id: Optional[int] = None, agent_id: Optional[int] = None) -> Dict[str, Any]:
        """List documents in a collection."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str, collection_name: str = "default", user_id: Optional[int] = None, agent_id: Optional[int] = None) -> bool:
        """Delete a document from a collection."""
        pass

    @abstractmethod
    async def list_files(self, collection_name: str = "default", user_id: Optional[int] = None, agent_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all files in a collection."""
        pass

    @abstractmethod
    async def delete_file(self, filename: str, collection_name: str = "default") -> bool:
        """Delete all documents associated with a file from a Chroma collection."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Get all documents
            result = collection.get(include=["metadatas"])
            
            # Find IDs to delete
            ids_to_delete = []
            if result["ids"] and result["metadatas"]:
                for idx, metadata in enumerate(result["metadatas"]):
                    doc_id = result["ids"][idx]
                    
                    # Check if metadata has filename
                    if metadata.get("filename") == filename:
                        ids_to_delete.append(doc_id)
                    # For legacy chunks without filename metadata, check ID
                    elif "_chunk_" in doc_id:
                        base_id = doc_id.rsplit("_chunk_", 1)[0]
                        if base_id == filename:
                            ids_to_delete.append(doc_id)
                    elif doc_id == filename:
                        ids_to_delete.append(doc_id)
            
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for file '{filename}' from '{actual_collection_name}'")
            else:
                logger.warning(f"No chunks found for file '{filename}' in '{actual_collection_name}'")
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete file '{filename}' from '{actual_collection_name}': {e}")
            return False

    def _get_collection_name(self, base_name: str, user_id: Optional[int], agent_id: Optional[int]) -> str:
        """
        Generate collection name based on user_id and agent_id.
        Format: u{user_id}_a{agent_id}_{base_name} or u{user_id}_{base_name} or {base_name}
        """
        if user_id is not None:
            if agent_id is not None:
                return f"u{user_id}_a{agent_id}_{base_name}"
            return f"u{user_id}_{base_name}"
        return base_name

    def _parse_collection_name(self, actual_name: str) -> Dict[str, Any]:
        """
        Parse actual collection name to extract user_id, agent_id and original name.
        Returns: {"user_id": int|None, "agent_id": int|None, "base_name": str}
        """
        # Regex for u{uid}_a{aid}_{name}
        ua_match = re.match(r"^u(\d+)_a(\d+)_(.+)$", actual_name)
        if ua_match:
            return {
                "user_id": int(ua_match.group(1)),
                "agent_id": int(ua_match.group(2)),
                "base_name": ua_match.group(3)
            }
        
        # Regex for u{uid}_{name}
        u_match = re.match(r"^u(\d+)_(.+)$", actual_name)
        if u_match:
            return {
                "user_id": int(u_match.group(1)),
                "agent_id": None,
                "base_name": u_match.group(2)
            }
            
        return {
            "user_id": None,
            "agent_id": None,
            "base_name": actual_name
        }

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

    async def add_document(self, content: str, collection_name: str = "default", metadata: Dict[str, Any] = None, user_id: Optional[int] = None, agent_id: Optional[int] = None) -> None:
        """Add a single document to Chroma collection."""
        if metadata is None:
            metadata = {}

        # Auto-generate filename if not provided
        if "filename" not in metadata:
            if "title" in metadata and metadata["title"]:
                # Use title as filename (sanitized)
                filename = metadata["title"].strip()[:50]  # Limit length
                # Replace invalid filename characters
                for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
                    filename = filename.replace(char, '_')
            else:
                # Generate default filename with timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"手动输入_{timestamp}"
            
            metadata["filename"] = filename

        document = Document(
            id=f"doc_{len(content)}_{hash(content)}",  # Simple ID generation
            content=content,
            metadata=metadata
        )

        await self.add_documents([document], collection_name, user_id, agent_id)

    async def add_documents(self, documents: List[Document], collection_name: str = "default", user_id: Optional[int] = None, agent_id: Optional[int] = None) -> None:
        """Add multiple documents to Chroma collection."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        # Resolve actual collection name
        actual_collection_name = self._get_collection_name(collection_name, user_id, agent_id)

        try:
            # Get or create collection
            try:
                collection = self.client.get_collection(name=actual_collection_name)
                # Check if collection uses the correct distance metric
                current_space = collection.metadata.get("hnsw:space") if collection.metadata else None
                if current_space != "cosine":
                    logger.warning(f"Collection '{actual_collection_name}' uses distance metric '{current_space}'. Recreating with cosine.")
                    self.client.delete_collection(name=actual_collection_name)
                    collection = self.client.create_collection(
                        name=actual_collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"Recreated collection '{actual_collection_name}' with cosine similarity")
            except Exception:
                # Collection doesn't exist, create it with cosine similarity
                collection = self.client.create_collection(
                    name=actual_collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created collection '{actual_collection_name}' with cosine similarity")

            # Prepare data for Chroma
            ids = []
            contents = []
            metadatas = []

            from .text_splitter import split_text

            for doc in documents:
                # Split text into chunks
                chunks = split_text(doc.content)
                
                for i, chunk in enumerate(chunks):
                    # Create unique ID for chunk
                    chunk_id = f"{doc.id}_chunk_{i}"
                    
                    # Create metadata for chunk
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "original_id": doc.id
                    })
                    
                    ids.append(chunk_id)
                    contents.append(chunk)
                    metadatas.append(chunk_metadata)

            if not ids:
                logger.warning(f"No content to add to collection '{actual_collection_name}'")
                return

            # Calculate embeddings for documents
            from .embedding import get_embedding_model
            embedding_model = await get_embedding_model()
            
            # Process in batches to avoid API limits
            batch_size = 10
            for i in range(0, len(contents), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_contents = contents[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                
                batch_embeddings = await embedding_model.encode(batch_contents)

                # Add documents to collection with embeddings
                collection.add(
                    documents=batch_contents,
                    metadatas=batch_metadatas,
                    ids=batch_ids,
                    embeddings=batch_embeddings
                )

            logger.info(f"Added {len(documents)} documents ({len(contents)} chunks) to collection '{actual_collection_name}'")

        except Exception as e:
            logger.error(f"Failed to add documents to collection '{actual_collection_name}': {e}")
            raise

    async def search(
        self,
        query_embedding: List[float],
        collection_name: str = "default",
        limit: int = 5,
        threshold: float = 0.7,
        user_id: Optional[int] = None, 
        agent_id: Optional[int] = None
    ) -> List[SearchResult]:
        """Search Chroma collection using built-in vector search."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        # Resolve actual collection name
        actual_collection_name = self._get_collection_name(collection_name, user_id, agent_id)

        try:
            collection = self.client.get_collection(name=actual_collection_name)
            if not collection:
                logger.warning(f"Collection '{actual_collection_name}' not found")
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
            logger.error(f"Failed to search collection '{actual_collection_name}': {e}")
            raise

    async def delete_collection(self, collection_name: str = "default", user_id: Optional[int] = None, agent_id: Optional[int] = None) -> None:
        """Delete Chroma collection."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        # Resolve actual collection name
        actual_collection_name = self._get_collection_name(collection_name, user_id, agent_id)

        try:
            self.client.delete_collection(name=actual_collection_name)
            if actual_collection_name in self.collections:
                del self.collections[actual_collection_name]
            logger.info(f"Deleted collection '{actual_collection_name}'")
        except Exception as e:
            logger.error(f"Failed to delete collection '{actual_collection_name}': {e}")
            raise

    async def list_collections(self, user_id: Optional[int] = None, agent_id: Optional[int] = None) -> List[str]:
        """List all Chroma collections."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        try:
            collections = self.client.list_collections()
            all_collections = [col.name for col in collections]
            
            # Filter and map back to base names if filtering is needed/desired
            # For now, we return full names or filtered list if user_id is provided
            
            if user_id is None:
                 return all_collections
                 
            filtered = []
            for col_name in all_collections:
                info = self._parse_collection_name(col_name)
                
                # If specific user requested, only show their collections
                if user_id is not None:
                    if info["user_id"] != user_id:
                        continue
                        
                    # If specific agent requested, further filter
                    if agent_id is not None and info["agent_id"] != agent_id:
                        continue
                
                filtered.append(col_name)
                
            return filtered
            
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    async def list_documents(self, collection_name: str = "default", limit: int = 100, offset: int = 0, filename: Optional[str] = None, user_id: Optional[int] = None, agent_id: Optional[int] = None) -> Dict[str, Any]:
        """List documents in a Chroma collection."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        # Resolve actual collection name
        actual_collection_name = self._get_collection_name(collection_name, user_id, agent_id)

        try:
            collection = self.client.get_collection(name=actual_collection_name)
            # Get all documents with metadata
            where_clause = {"filename": filename} if filename else None
            result = collection.get(
                where=where_clause,
                limit=limit,
                offset=offset,
                include=["metadatas", "documents"]
            )
            
            documents = []
            if result["ids"]:
                for i, doc_id in enumerate(result["ids"]):
                    documents.append({
                        "id": doc_id,
                        "content": result["documents"][i] if result["documents"] else "",
                        "metadata": result["metadatas"][i] if result["metadatas"] else {}
                    })
            
            # Get total count (approximation as Chroma doesn't have a direct count method efficiently exposed in all versions)
            # For now, we'll just return what we have. In a real app, we might want a separate count query.
            total = collection.count()
            
            return {
                "total": total,
                "documents": documents,
                "limit": limit,
                "offset": offset
            }
        except Exception as e:
            logger.error(f"Failed to list documents in '{actual_collection_name}': {e}")
            # Return empty result if collection doesn't exist or other error
            return {"total": 0, "documents": [], "limit": limit, "offset": offset}

    async def delete_document(self, document_id: str, collection_name: str = "default", user_id: Optional[int] = None, agent_id: Optional[int] = None) -> bool:
        """Delete a document from a Chroma collection."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        # Resolve actual collection name
        actual_collection_name = self._get_collection_name(collection_name, user_id, agent_id)

        try:
            collection = self.client.get_collection(name=actual_collection_name)
            collection.delete(ids=[document_id])
            logger.info(f"Deleted document '{document_id}' from '{actual_collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document '{document_id}' from '{actual_collection_name}': {e}")
            return False

    async def list_files(self, collection_name: str = "default", user_id: Optional[int] = None, agent_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all files in a Chroma collection."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        # Resolve actual collection name
        actual_collection_name = self._get_collection_name(collection_name, user_id, agent_id)

        try:
            collection = self.client.get_collection(name=actual_collection_name)
            # Get all metadatas and IDs
            result = collection.get(include=["metadatas"])
            
            files = {}
            if result["metadatas"] and result["ids"]:
                for idx, metadata in enumerate(result["metadatas"]):
                    doc_id = result["ids"][idx]
                    
                    # Try to get filename from metadata first
                    filename = metadata.get("filename")
                    
                    # If no filename in metadata, try to extract from ID
                    # For legacy chunks like "doc_150000_-9093687922038414764_chunk_34"
                    # or new chunks like "some_id_chunk_0"
                    if not filename:
                        # Check if this is a chunk (has _chunk_ in ID)
                        if "_chunk_" in doc_id:
                            # Remove the chunk suffix to get the base ID
                            filename = doc_id.rsplit("_chunk_", 1)[0]
                        else:
                            # Not a chunk, use the ID as filename
                            filename = doc_id
                    
                    if filename not in files:
                        files[filename] = {
                            "filename": filename,
                            "chunk_count": 0,
                            "total_size": 0,
                            "file_type": metadata.get("file_type", "unknown"),
                            "upload_time": metadata.get("timestamp", "")
                        }
                    
                    files[filename]["chunk_count"] += 1
                    # Approximate size if available, otherwise just count chunks
                    files[filename]["total_size"] += metadata.get("size", 0)

            return list(files.values())
        except Exception as e:
            logger.error(f"Failed to list files in '{actual_collection_name}': {e}")
            return []

    async def delete_file(self, filename: str, collection_name: str = "default") -> bool:
        """Delete all documents associated with a file from a Chroma collection."""
        if not self.client:
            raise RuntimeError("Database not initialized")

        try:
            collection = self.client.get_collection(name=collection_name)
            
            # Delete where metadata['filename'] == filename
            collection.delete(where={"filename": filename})
            
            logger.info(f"Deleted file '{filename}' from '{collection_name}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file '{filename}' from '{collection_name}': {e}")
            return False


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