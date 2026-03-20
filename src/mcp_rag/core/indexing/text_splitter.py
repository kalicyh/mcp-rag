"""Lightweight recursive text splitter for MCP-RAG indexing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(slots=True)
class RecursiveCharacterTextSplitter:
    """Split text by recursive separators with a char-window fallback."""

    chunk_size: int = 4000
    chunk_overlap: int = 200
    separators: Sequence[str] = ("\n\n", "\n", " ", "")

    def split_text(self, text: str) -> List[str]:
        if not text:
            return []

        chunks: list[str] = []
        self._split_recursive(text, list(self.separators), chunks)
        return [chunk for chunk in chunks if chunk]

    def _split_recursive(self, text: str, separators: list[str], chunks: list[str]) -> None:
        if len(text) <= self.chunk_size:
            chunks.append(text)
            return

        if not separators:
            chunks.extend(self._split_by_window(text))
            return

        separator = separators[0]
        rest = separators[1:]

        if separator == "":
            chunks.extend(self._split_by_window(text))
            return

        pieces = text.split(separator)
        current_parts: list[str] = []
        current_len = 0

        for piece in pieces:
            separator_len = len(separator) if current_parts else 0
            piece_len = len(piece)
            projected = current_len + separator_len + piece_len

            if projected <= self.chunk_size:
                current_parts.append(piece)
                current_len = projected
                continue

            if current_parts:
                merged = separator.join(current_parts)
                if len(merged) > self.chunk_size:
                    self._split_recursive(merged, rest, chunks)
                else:
                    chunks.append(merged)

            current_parts = [piece]
            current_len = piece_len

        if current_parts:
            merged = separator.join(current_parts)
            if len(merged) > self.chunk_size:
                self._split_recursive(merged, rest, chunks)
            else:
                chunks.append(merged)

    def _split_by_window(self, text: str) -> List[str]:
        if self.chunk_size <= 0:
            return [text]

        step = max(1, self.chunk_size - max(0, self.chunk_overlap))
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunks.append(text[start : start + self.chunk_size])
        return chunks


def split_text(
    text: str,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    separators: Sequence[str] | None = None,
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or ("\n\n", "\n", " ", ""),
    )
    return splitter.split_text(text)

