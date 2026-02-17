"""Text chunking with configurable size and overlap."""

import hashlib
from dataclasses import dataclass
from typing import List

from .settings import ChunkingConfig


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    chunk_id: str
    document_id: str
    text: str
    start_char: int
    end_char: int


class TextChunker:
    """Splits text into overlapping chunks of configurable size."""

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        if self.config.chunk_overlap >= self.config.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

    def _generate_chunk_id(self, document_id: str, chunk_index: int, text: str) -> str:
        """Generate a stable, deterministic chunk ID."""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{document_id}_chunk_{chunk_index}_{content_hash}"

    def chunk_text(self, text: str, document_id: str) -> List[Chunk]:
        """
        Split text into chunks with overlap.

        Args:
            text: The raw text to chunk (UTF-8 compatible)
            document_id: Identifier for the source document

        Returns:
            List of Chunk objects with metadata
        """
        if not text:
            return []

        chunks: List[Chunk] = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        step = chunk_size - overlap

        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end]

            chunk_id = self._generate_chunk_id(document_id, chunk_index, chunk_text)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                text=chunk_text,
                start_char=start,
                end_char=end
            ))

            if end >= len(text):
                break

            start += step
            chunk_index += 1

        return chunks

    def chunk_file(self, file_path: str, document_id: str | None = None) -> List[Chunk]:
        """
        Read and chunk a text file.

        Args:
            file_path: Path to the text file
            document_id: Optional document ID (defaults to file path)

        Returns:
            List of Chunk objects
        """
        doc_id = document_id or file_path

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return self.chunk_text(text, doc_id)
