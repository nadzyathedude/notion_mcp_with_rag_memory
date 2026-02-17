"""Retrieval module: cosine similarity search over indexed chunks."""

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .index import DocumentIndex, IndexedChunk


@dataclass
class RetrievalResult:
    """A chunk with its similarity score."""
    chunk: IndexedChunk
    score: float
    document_id: str
    source: str = ""


@dataclass
class RetrievedChunk:
    """Enriched chunk representation for the filtering/reranking pipeline."""
    chunk_id: str
    text: str
    similarity: float
    rerank_score: Optional[float] = None
    document_id: str = "unknown"
    source: str = ""

    @property
    def effective_score(self) -> float:
        """Return rerank_score if available, otherwise the cosine similarity."""
        return self.rerank_score if self.rerank_score is not None else self.similarity


def retrieval_result_to_chunk(result: RetrievalResult) -> RetrievedChunk:
    """Convert a RetrievalResult into a RetrievedChunk."""
    return RetrievedChunk(
        chunk_id=result.chunk.chunk_id,
        text=result.chunk.text,
        similarity=result.score,
        document_id=result.document_id,
        source=result.source,
    )


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Pure Python implementation — no numpy required.

    Args:
        a: First embedding vector
        b: Second embedding vector

    Returns:
        Cosine similarity in range [-1, 1]

    Raises:
        ValueError: If vectors have different dimensions
    """
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimension mismatch: {len(a)} vs {len(b)}"
        )

    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


class Retriever:
    """Searches indexed chunks by embedding similarity."""

    def __init__(self, index: DocumentIndex):
        """
        Args:
            index: A loaded DocumentIndex instance
        """
        self.index = index
        self._chunk_doc_map: dict[str, str] = {}
        self._chunk_source_map: dict[str, str] = {}
        self._build_chunk_maps()

    def _build_chunk_maps(self) -> None:
        """Map chunk IDs to their parent document IDs and sources."""
        for doc in self.index.documents.values():
            for chunk in doc.chunks:
                self._chunk_doc_map[chunk.chunk_id] = doc.document_id
                self._chunk_source_map[chunk.chunk_id] = doc.source

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Find the top-k most similar chunks to the query embedding.

        Args:
            query_embedding: Embedding vector of the user question
            top_k: Number of results to return

        Returns:
            List of RetrievalResult sorted by similarity (descending)
        """
        all_chunks = self.index.get_all_chunks()

        if not all_chunks:
            return []

        scored: List[RetrievalResult] = []
        for chunk in all_chunks:
            score = cosine_similarity(query_embedding, chunk.embedding)
            scored.append(RetrievalResult(
                chunk=chunk,
                score=score,
                document_id=self._chunk_doc_map.get(chunk.chunk_id, "unknown"),
                source=self._chunk_source_map.get(chunk.chunk_id, ""),
            ))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
