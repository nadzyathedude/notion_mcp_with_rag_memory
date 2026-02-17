"""RAG adapter: bridges Notion-synced index with the RAG query pipeline."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from indexer.embeddings import EmbeddingProvider, OpenAIEmbeddings
from indexer.index import DocumentIndex
from indexer.llm import LLMProvider, OpenAILLM
from indexer.rag import (
    RAGResult,
    answer_question_filtered,
    answer_question_cited,
)
from indexer.retriever import Retriever, RetrievedChunk, cosine_similarity
from indexer.settings import EmbeddingConfig, FallbackStrategy

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a semantic search over Notion content."""
    chunk_id: str
    text: str
    score: float
    document_id: str
    source: str
    notion_page_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "score": round(self.score, 4),
            "document_id": self.document_id,
            "source": self.source,
            "notion_page_id": self.notion_page_id,
        }


class NotionRAGAdapter:
    """Provides search and QA over Notion-synced RAG index."""

    def __init__(
        self,
        index_path: str = "notion_index.json",
        embedding_provider: EmbeddingProvider | None = None,
        llm_provider: LLMProvider | None = None,
    ):
        self.index_path = index_path
        self._embedding_provider = embedding_provider
        self._llm_provider = llm_provider

    def _get_embedding_provider(self, index: DocumentIndex) -> EmbeddingProvider:
        if self._embedding_provider:
            return self._embedding_provider
        config = EmbeddingConfig(model=index.embedding_model)
        return OpenAIEmbeddings(config)

    def _get_llm_provider(self) -> LLMProvider:
        if self._llm_provider:
            return self._llm_provider
        return OpenAILLM()

    def _extract_notion_page_id(self, document_id: str) -> str:
        """Extract the Notion page ID from a document_id like 'notion_abc123'."""
        if document_id.startswith("notion_"):
            return document_id[len("notion_"):]
        return document_id

    def search(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[SearchResult]:
        """
        Semantic search over indexed Notion content.

        Args:
            query: Search query text
            top_k: Max results to return
            threshold: Minimum similarity score (0.0 = no filtering)

        Returns:
            List of SearchResult objects
        """
        path = Path(self.index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Notion index not found at {self.index_path}. "
                f"Run sync_notion first."
            )

        index = DocumentIndex.load(self.index_path)
        if len(index) == 0:
            return []

        embedder = self._get_embedding_provider(index)
        query_embedding = embedder.embed_texts([query])[0]

        retriever = Retriever(index)
        results = retriever.search(query_embedding, top_k=top_k)

        search_results: list[SearchResult] = []
        for r in results:
            if r.score < threshold:
                continue
            search_results.append(SearchResult(
                chunk_id=r.chunk.chunk_id,
                text=r.chunk.text,
                score=r.score,
                document_id=r.document_id,
                source=r.source,
                notion_page_id=self._extract_notion_page_id(r.document_id),
            ))

        return search_results

    def ask(
        self,
        question: str,
        top_k: int = 5,
        threshold: float = 0.3,
        enforce_citations: bool = False,
    ) -> dict[str, Any]:
        """
        Run the full RAG pipeline over Notion-derived content.

        Args:
            question: The question to answer
            top_k: Number of chunks to retrieve
            threshold: Similarity threshold for filtering
            enforce_citations: Whether to require citations in the answer

        Returns:
            Dict with answer, chunks, and metadata
        """
        path = Path(self.index_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Notion index not found at {self.index_path}. "
                f"Run sync_notion first."
            )

        if enforce_citations:
            rag_result = answer_question_cited(
                question=question,
                index_path=self.index_path,
                top_k=top_k,
                threshold=threshold,
                fallback_strategy=FallbackStrategy.TOP_1,
                embedding_provider=self._embedding_provider,
                llm_provider=self._llm_provider,
            )
        else:
            rag_result = answer_question_filtered(
                question=question,
                index_path=self.index_path,
                top_k=top_k,
                threshold=threshold,
                fallback_strategy=FallbackStrategy.TOP_1,
                embedding_provider=self._embedding_provider,
                llm_provider=self._llm_provider,
            )

        # Build response
        response: dict[str, Any] = {
            "answer": rag_result.answer,
            "chunks_retrieved": rag_result.chunks_retrieved,
            "chunks_after_filter": rag_result.chunks_after_filter,
            "avg_similarity": round(rag_result.avg_similarity, 4),
            "used_fallback": rag_result.used_fallback,
            "sources": [],
        }

        for chunk in rag_result.chunks:
            notion_page_id = self._extract_notion_page_id(chunk.document_id)
            response["sources"].append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "score": round(chunk.effective_score, 4),
                "notion_page_id": notion_page_id,
                "source": chunk.source,
            })

        if rag_result.citation_validation:
            response["citation_validation"] = {
                "is_valid": rag_result.citation_validation.is_valid,
                "citations_found": rag_result.citation_validation.citations_found,
                "invalid_citations": rag_result.citation_validation.invalid_citations,
            }

        return response

    def get_index_stats(self) -> dict[str, Any]:
        """Get statistics about the Notion index."""
        path = Path(self.index_path)
        if not path.exists():
            return {
                "exists": False,
                "message": "No Notion index found. Run sync_notion first.",
            }

        index = DocumentIndex.load(self.index_path)

        notion_docs = [
            doc_id for doc_id in index.documents
            if doc_id.startswith("notion_")
        ]

        return {
            "exists": True,
            "total_documents": len(index.documents),
            "notion_documents": len(notion_docs),
            "total_chunks": len(index),
            "embedding_model": index.embedding_model,
            "index_path": self.index_path,
        }
