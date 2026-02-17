"""JSON document index management."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from .settings import IndexConfig
from .chunker import Chunk


@dataclass
class IndexedChunk:
    """A chunk with its embedding stored in the index."""
    chunk_id: str
    text: str
    embedding: List[float]
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": {
                "start_char": self.start_char,
                "end_char": self.end_char
            }
        }


@dataclass
class IndexedDocument:
    """A document with its chunks in the index."""
    document_id: str
    source: str
    chunks: List[IndexedChunk] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "chunks": [chunk.to_dict() for chunk in self.chunks]
        }


class DocumentIndex:
    """Manages the JSON document index."""

    def __init__(self, config: IndexConfig | None = None):
        self.config = config or IndexConfig()
        self.documents: dict[str, IndexedDocument] = {}
        self.embedding_model: str = ""
        self.created_at: str = ""

    def add_document(
        self,
        document_id: str,
        source: str,
        chunks: List[Chunk],
        embeddings: List[List[float]]
    ) -> None:
        """
        Add a document with its chunks and embeddings to the index.

        Args:
            document_id: Unique identifier for the document
            source: Source path or name
            chunks: List of Chunk objects
            embeddings: List of embedding vectors (same order as chunks)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")

        indexed_chunks = [
            IndexedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding,
                start_char=chunk.start_char,
                end_char=chunk.end_char
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]

        self.documents[document_id] = IndexedDocument(
            document_id=document_id,
            source=source,
            chunks=indexed_chunks
        )

    def set_embedding_model(self, model_name: str) -> None:
        """Set the embedding model used for this index."""
        self.embedding_model = model_name

    def to_dict(self) -> dict[str, Any]:
        """Convert index to dictionary format."""
        return {
            "schema_version": self.config.schema_version,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
            "embedding_model": self.embedding_model,
            "documents": [doc.to_dict() for doc in self.documents.values()]
        }

    def save(self, output_path: str | None = None) -> str:
        """
        Save the index to a JSON file.

        Args:
            output_path: Path to save the index (uses config default if not provided)

        Returns:
            Path where the index was saved
        """
        path = output_path or self.config.output_path
        self.created_at = datetime.now(timezone.utc).isoformat()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

        return path

    @classmethod
    def load(cls, path: str) -> "DocumentIndex":
        """
        Load an index from a JSON file.

        Args:
            path: Path to the JSON index file

        Returns:
            DocumentIndex instance
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        config = IndexConfig(schema_version=data.get("schema_version", "1.0"))
        index = cls(config)
        index.embedding_model = data.get("embedding_model", "")
        index.created_at = data.get("created_at", "")

        for doc_data in data.get("documents", []):
            chunks = [
                IndexedChunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    embedding=c["embedding"],
                    start_char=c["metadata"]["start_char"],
                    end_char=c["metadata"]["end_char"]
                )
                for c in doc_data.get("chunks", [])
            ]
            index.documents[doc_data["document_id"]] = IndexedDocument(
                document_id=doc_data["document_id"],
                source=doc_data.get("source", ""),
                chunks=chunks
            )

        return index

    def get_all_chunks(self) -> List[IndexedChunk]:
        """Get all chunks from all documents."""
        all_chunks = []
        for doc in self.documents.values():
            all_chunks.extend(doc.chunks)
        return all_chunks

    def __len__(self) -> int:
        """Return total number of chunks in the index."""
        return sum(len(doc.chunks) for doc in self.documents.values())
