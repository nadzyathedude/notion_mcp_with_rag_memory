"""Document Indexing Pipeline - Chunking + Embeddings + RAG"""

from .chunker import TextChunker, Chunk
from .citations import CitationValidationResult, validate_citations, extract_citations
from .embeddings import EmbeddingProvider, OpenAIEmbeddings
from .index import DocumentIndex, IndexedChunk, IndexedDocument
from .llm import LLMProvider, OpenAILLM
from .pipeline import IndexingPipeline
from .rag import answer_question, answer_question_filtered, answer_question_cited, RAGResult
from .reranker import Reranker, LLMReranker
from .retriever import Retriever, RetrievalResult, RetrievedChunk, cosine_similarity
from .settings import (
    PipelineConfig, ChunkingConfig, EmbeddingConfig, IndexConfig, LLMConfig,
    RetrievalConfig, FallbackStrategy,
)

__all__ = [
    "TextChunker",
    "Chunk",
    "CitationValidationResult",
    "validate_citations",
    "extract_citations",
    "EmbeddingProvider",
    "OpenAIEmbeddings",
    "DocumentIndex",
    "IndexedChunk",
    "IndexedDocument",
    "LLMProvider",
    "OpenAILLM",
    "IndexingPipeline",
    "answer_question",
    "answer_question_filtered",
    "answer_question_cited",
    "RAGResult",
    "Reranker",
    "LLMReranker",
    "RetrievedChunk",
    "Retriever",
    "RetrievalResult",
    "cosine_similarity",
    "PipelineConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "IndexConfig",
    "LLMConfig",
    "RetrievalConfig",
    "FallbackStrategy",
]
