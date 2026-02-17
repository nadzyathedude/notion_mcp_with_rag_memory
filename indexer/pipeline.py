"""Document indexing pipeline - combines chunking, embedding, and index storage."""

from pathlib import Path
from typing import List

from .settings import PipelineConfig, ChunkingConfig, EmbeddingConfig, IndexConfig
from .chunker import TextChunker, Chunk
from .embeddings import EmbeddingProvider, OpenAIEmbeddings
from .index import DocumentIndex


class IndexingPipeline:
    """
    Main pipeline that orchestrates document indexing.

    Combines text chunking, embedding generation, and JSON index storage.
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None
    ):
        """
        Initialize the indexing pipeline.

        Args:
            config: Pipeline configuration (uses defaults if not provided)
            embedding_provider: Custom embedding provider (uses OpenAI if not provided)
        """
        self.config = config or PipelineConfig()
        self.chunker = TextChunker(self.config.chunking)
        self.embedder = embedding_provider or OpenAIEmbeddings(self.config.embedding)
        self.index = DocumentIndex(self.config.index)
        self.index.set_embedding_model(self.embedder.model_name)

    def add_text(self, text: str, document_id: str, source: str | None = None) -> int:
        """
        Process raw text and add to the index.

        Args:
            text: Raw text content
            document_id: Unique identifier for the document
            source: Optional source description (defaults to document_id)

        Returns:
            Number of chunks created
        """
        chunks = self.chunker.chunk_text(text, document_id)

        if not chunks:
            return 0

        embeddings = self.embedder.embed_texts([c.text for c in chunks])
        self.index.add_document(
            document_id=document_id,
            source=source or document_id,
            chunks=chunks,
            embeddings=embeddings
        )

        return len(chunks)

    def add_file(self, file_path: str, document_id: str | None = None) -> int:
        """
        Process a text file and add to the index.

        Args:
            file_path: Path to the text file
            document_id: Optional document ID (defaults to filename)

        Returns:
            Number of chunks created
        """
        path = Path(file_path)
        doc_id = document_id or path.stem

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        return self.add_text(text, doc_id, source=str(path))

    def add_files(self, file_paths: List[str]) -> int:
        """
        Process multiple files and add to the index.

        Args:
            file_paths: List of file paths

        Returns:
            Total number of chunks created
        """
        total_chunks = 0
        for path in file_paths:
            total_chunks += self.add_file(path)
        return total_chunks

    def add_directory(
        self,
        directory: str,
        pattern: str = "*.txt",
        recursive: bool = False
    ) -> int:
        """
        Process all matching files in a directory.

        Args:
            directory: Directory path
            pattern: Glob pattern for file matching (default: *.txt)
            recursive: Whether to search recursively

        Returns:
            Total number of chunks created
        """
        path = Path(directory)
        glob_method = path.rglob if recursive else path.glob
        files = list(glob_method(pattern))

        return self.add_files([str(f) for f in files])

    def save(self, output_path: str | None = None) -> str:
        """
        Save the index to a JSON file.

        Args:
            output_path: Output path (uses config default if not provided)

        Returns:
            Path where index was saved
        """
        return self.index.save(output_path)

    def get_stats(self) -> dict:
        """Get statistics about the current index."""
        return {
            "total_documents": len(self.index.documents),
            "total_chunks": len(self.index),
            "embedding_model": self.embedder.model_name,
            "chunk_size": self.config.chunking.chunk_size,
            "chunk_overlap": self.config.chunking.chunk_overlap
        }
