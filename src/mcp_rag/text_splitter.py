"""Text splitting utilities for MCP-RAG."""

import logging
from typing import List

logger = logging.getLogger(__name__)


class RecursiveCharacterTextSplitter:
    """Splits text into chunks recursively based on separators."""

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        final_chunks = []
        
        # Initial split
        if not text:
            return []
            
        self._split_text_recursive(text, self.separators, final_chunks)
        return final_chunks

    def _split_text_recursive(self, text: str, separators: List[str], final_chunks: List[str]):
        """Recursive helper to split text."""
        
        # If text is small enough, add it
        if len(text) <= self.chunk_size:
            final_chunks.append(text)
            return

        # If no more separators, force split
        if not separators:
            # Just split by char
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                final_chunks.append(text[i:i + self.chunk_size])
            return

        # Try to split by current separator
        separator = separators[0]
        next_separators = separators[1:]
        
        if separator == "":
            # Special case for empty separator (character split)
            self._split_text_recursive(text, [], final_chunks)
            return

        splits = text.split(separator)
        
        # Re-merge splits that are too small to form chunks
        current_chunk = []
        current_length = 0
        
        for split in splits:
            # Add separator length if not the first split
            sep_len = len(separator) if current_length > 0 else 0
            
            if current_length + sep_len + len(split) <= self.chunk_size:
                current_chunk.append(split)
                current_length += sep_len + len(split)
            else:
                # Process current chunk if it exists
                if current_chunk:
                    merged_text = separator.join(current_chunk)
                    if len(merged_text) > self.chunk_size:
                        # If still too big (single split too big), recurse
                        self._split_text_recursive(merged_text, next_separators, final_chunks)
                    else:
                        final_chunks.append(merged_text)
                
                # Start new chunk
                current_chunk = [split]
                current_length = len(split)
        
        # Process remaining chunk
        if current_chunk:
            merged_text = separator.join(current_chunk)
            if len(merged_text) > self.chunk_size:
                self._split_text_recursive(merged_text, next_separators, final_chunks)
            else:
                final_chunks.append(merged_text)


def split_text(text: str, chunk_size: int = 4000, chunk_overlap: int = 200) -> List[str]:
    """Convenience function to split text."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
