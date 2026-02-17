"""RAG pipeline: Question -> Embed -> Retrieve -> Augment -> LLM -> Answer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .citations import CitationValidationResult, validate_citations
from .embeddings import EmbeddingProvider, OpenAIEmbeddings
from .index import DocumentIndex
from .llm import LLMProvider, OpenAILLM
from .reranker import Reranker
from .retriever import Retriever, RetrievalResult, RetrievedChunk, retrieval_result_to_chunk
from .settings import EmbeddingConfig, FallbackStrategy, LLMConfig

# Default context char limit (~12k chars ≈ ~3k tokens, safe for most models)
DEFAULT_MAX_CONTEXT_CHARS = 12000


def _build_prompt(
    question: str,
    results: List[RetrievalResult],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """
    Build an augmented prompt from retrieved chunks and the user question.

    Args:
        question: The user's question
        results: Ranked retrieval results
        max_context_chars: Max total characters for context section

    Returns:
        Full prompt string ready for LLM
    """
    context_parts: list[str] = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        chunk_text = result.chunk.text.strip()
        header = f"[Chunk {i} | doc: {result.document_id} | score: {result.score:.3f}]"
        section = f"{header}\n{chunk_text}"

        if total_chars + len(section) > max_context_chars:
            remaining = max_context_chars - total_chars
            if remaining > 100:
                context_parts.append(section[:remaining] + "...")
            break

        context_parts.append(section)
        total_chars += len(section)

    context = "\n\n".join(context_parts)

    return (
        "You are given the following context:\n\n"
        f"{context}\n\n"
        "Answer the question based only on the provided context. "
        "If the context does not contain enough information, say so.\n\n"
        f"Question:\n{question}"
    )


def answer_question(
    question: str,
    index_path: str,
    top_k: int = 5,
    embedding_provider: EmbeddingProvider | None = None,
    llm_provider: LLMProvider | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """
    Full RAG pipeline: embed question -> retrieve -> augment -> generate answer.

    Args:
        question: User question text
        index_path: Path to the JSON index file
        top_k: Number of chunks to retrieve
        embedding_provider: Custom embedding provider (auto-configured if None)
        llm_provider: Custom LLM provider (auto-configured if None)
        max_context_chars: Max characters for context section

    Returns:
        LLM-generated answer string
    """
    # 1. Load index
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    index = DocumentIndex.load(index_path)

    if len(index) == 0:
        raise ValueError("Index is empty — no chunks to search")

    # 2. Create embedding provider matching the index model
    if embedding_provider is None:
        embedding_config = EmbeddingConfig(model=index.embedding_model)
        embedding_provider = OpenAIEmbeddings(embedding_config)

    # 3. Embed the question
    query_embedding = embedding_provider.embed_texts([question])[0]

    # 4. Validate dimension match
    sample_chunk = index.get_all_chunks()[0]
    if len(query_embedding) != len(sample_chunk.embedding):
        raise ValueError(
            f"Embedding dimension mismatch: query has {len(query_embedding)}, "
            f"index has {len(sample_chunk.embedding)}. "
            f"Ensure the same embedding model is used."
        )

    # 5. Retrieve top-k chunks
    retriever = Retriever(index)
    results = retriever.search(query_embedding, top_k=top_k)

    if not results:
        return "No relevant chunks found in the index."

    # 6. Build augmented prompt
    prompt = _build_prompt(question, results, max_context_chars)

    # 7. Call LLM
    if llm_provider is None:
        llm_provider = OpenAILLM()

    return llm_provider.generate(prompt)


# ---------------------------------------------------------------------------
# Enhanced pipeline with threshold filtering and reranking
# ---------------------------------------------------------------------------

@dataclass
class RAGResult:
    """Structured result from the filtered RAG pipeline."""
    answer: str
    chunks_retrieved: int
    chunks_after_filter: int
    avg_similarity: float
    avg_rerank_score: Optional[float]
    used_fallback: bool
    chunks: List[RetrievedChunk] = field(default_factory=list)
    citation_validation: Optional[CitationValidationResult] = None


def _convert_results(results: List[RetrievalResult]) -> List[RetrievedChunk]:
    """Convert a list of RetrievalResult into RetrievedChunk objects."""
    return [retrieval_result_to_chunk(r) for r in results]


def _apply_threshold_filter(
    chunks: List[RetrievedChunk],
    threshold: float,
    fallback: FallbackStrategy,
) -> tuple[List[RetrievedChunk], bool]:
    """
    Filter chunks below the threshold using effective_score.

    Returns:
        (filtered_chunks, used_fallback)
    """
    filtered = [c for c in chunks if c.effective_score >= threshold]

    if filtered:
        return filtered, False

    # All chunks were below threshold
    if fallback == FallbackStrategy.TOP_1 and chunks:
        return [chunks[0]], True  # chunks are already sorted desc
    return [], False


def _build_prompt_from_chunks(
    question: str,
    chunks: List[RetrievedChunk],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    used_fallback: bool = False,
) -> str:
    """Build an augmented prompt from RetrievedChunk objects."""
    context_parts: list[str] = []
    total_chars = 0

    for i, chunk in enumerate(chunks, 1):
        text = chunk.text.strip()
        score_info = f"sim: {chunk.similarity:.3f}"
        if chunk.rerank_score is not None:
            score_info += f" | rerank: {chunk.rerank_score:.3f}"
        header = f"[Chunk {i} | doc: {chunk.document_id} | {score_info}]"
        section = f"{header}\n{text}"

        if total_chars + len(section) > max_context_chars:
            remaining = max_context_chars - total_chars
            if remaining > 100:
                context_parts.append(section[:remaining] + "...")
            break

        context_parts.append(section)
        total_chars += len(section)

    context = "\n\n".join(context_parts)

    fallback_note = ""
    if used_fallback:
        fallback_note = (
            "\nNote: No chunks met the relevance threshold. "
            "The best available chunk is shown, but it may not be highly relevant.\n"
        )

    return (
        "You are given the following context:\n\n"
        f"{context}\n\n"
        f"{fallback_note}"
        "Answer the question based only on the provided context. "
        "If the context does not contain enough information, say so.\n\n"
        f"Question:\n{question}"
    )


def answer_question_filtered(
    question: str,
    index_path: str,
    top_k: int = 5,
    threshold: float = 0.75,
    reranker: Optional[Reranker] = None,
    fallback_strategy: FallbackStrategy = FallbackStrategy.TOP_1,
    embedding_provider: Optional[EmbeddingProvider] = None,
    llm_provider: Optional[LLMProvider] = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> RAGResult:
    """
    Enhanced RAG pipeline with threshold filtering and optional reranking.

    Steps: load index -> embed -> retrieve top-k -> (optional) rerank
           -> threshold filter -> build prompt -> LLM -> RAGResult
    """
    # 1. Load index
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    index = DocumentIndex.load(index_path)
    if len(index) == 0:
        raise ValueError("Index is empty — no chunks to search")

    # 2. Create embedding provider
    if embedding_provider is None:
        embedding_config = EmbeddingConfig(model=index.embedding_model)
        embedding_provider = OpenAIEmbeddings(embedding_config)

    # 3. Embed the question
    query_embedding = embedding_provider.embed_texts([question])[0]

    # 4. Validate dimension match
    sample_chunk = index.get_all_chunks()[0]
    if len(query_embedding) != len(sample_chunk.embedding):
        raise ValueError(
            f"Embedding dimension mismatch: query has {len(query_embedding)}, "
            f"index has {len(sample_chunk.embedding)}."
        )

    # 5. Retrieve top-k
    retriever = Retriever(index)
    results = retriever.search(query_embedding, top_k=top_k)

    if not results:
        return RAGResult(
            answer="No relevant chunks found in the index.",
            chunks_retrieved=0, chunks_after_filter=0,
            avg_similarity=0.0, avg_rerank_score=None,
            used_fallback=False,
        )

    # 6. Convert to RetrievedChunk
    chunks = _convert_results(results)
    chunks_retrieved = len(chunks)
    avg_similarity = sum(c.similarity for c in chunks) / len(chunks)

    # 7. Optional reranking
    avg_rerank: Optional[float] = None
    if reranker is not None:
        chunks = reranker.rerank(question, chunks)
        scored = [c for c in chunks if c.rerank_score is not None]
        if scored:
            avg_rerank = sum(c.rerank_score for c in scored) / len(scored)

    # 8. Threshold filter
    filtered, used_fallback = _apply_threshold_filter(
        chunks, threshold, fallback_strategy
    )

    if not filtered:
        return RAGResult(
            answer="Insufficient context — no chunks met the relevance threshold.",
            chunks_retrieved=chunks_retrieved,
            chunks_after_filter=0,
            avg_similarity=avg_similarity,
            avg_rerank_score=avg_rerank,
            used_fallback=False,
            chunks=chunks,
        )

    # 9. Build prompt and generate
    prompt = _build_prompt_from_chunks(
        question, filtered, max_context_chars, used_fallback
    )

    if llm_provider is None:
        llm_provider = OpenAILLM()

    answer = llm_provider.generate(prompt)

    return RAGResult(
        answer=answer,
        chunks_retrieved=chunks_retrieved,
        chunks_after_filter=len(filtered),
        avg_similarity=avg_similarity,
        avg_rerank_score=avg_rerank,
        used_fallback=used_fallback,
        chunks=filtered,
    )


# ---------------------------------------------------------------------------
# Citation-enforced pipeline
# ---------------------------------------------------------------------------

_CITATION_MAX_RETRIES = 2


def _build_cited_prompt(
    question: str,
    chunks: List[RetrievedChunk],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    used_fallback: bool = False,
    is_retry: bool = False,
) -> tuple[str, List[str]]:
    """
    Build a prompt that labels each chunk with Source + Chunk ID
    and instructs the LLM to cite chunk IDs in its answer.

    Returns:
        (prompt_text, list_of_allowed_chunk_ids)
    """
    context_parts: list[str] = []
    allowed_ids: list[str] = []
    total_chars = 0

    for chunk in chunks:
        text = chunk.text.strip()
        source_label = chunk.source or chunk.document_id
        header = f"[Source: {source_label} | Chunk ID: {chunk.chunk_id}]"
        section = f"{header}\n{text}"

        if total_chars + len(section) > max_context_chars:
            remaining = max_context_chars - total_chars
            if remaining > 100:
                context_parts.append(section[:remaining] + "...")
                allowed_ids.append(chunk.chunk_id)
            break

        context_parts.append(section)
        allowed_ids.append(chunk.chunk_id)
        total_chars += len(section)

    context = "\n\n".join(context_parts)

    fallback_note = ""
    if used_fallback:
        fallback_note = (
            "\nNote: No chunks met the relevance threshold. "
            "The best available chunk is shown, but it may not be highly relevant.\n"
        )

    retry_note = ""
    if is_retry:
        retry_note = (
            "\nIMPORTANT: Your previous answer was rejected because it did not "
            "contain proper citations. You MUST include chunk ID citations in "
            "square brackets this time.\n"
        )

    return (
        "You are given the following context.\n\n"
        f"{context}\n\n"
        f"{fallback_note}"
        f"{retry_note}"
        "You must:\n"
        "- Answer ONLY using the provided context\n"
        "- Cite chunk IDs in square brackets after each factual claim, "
        "e.g. [chunk_id] or [id1, id2]\n"
        "- Only use chunk IDs listed above — do NOT fabricate sources\n"
        "- If the context does not contain enough information, respond: "
        '"Insufficient information in the provided sources."\n\n'
        f"Question:\n{question}"
    ), allowed_ids


def answer_question_cited(
    question: str,
    index_path: str,
    top_k: int = 5,
    threshold: float = 0.75,
    reranker: Optional[Reranker] = None,
    fallback_strategy: FallbackStrategy = FallbackStrategy.TOP_1,
    embedding_provider: Optional[EmbeddingProvider] = None,
    llm_provider: Optional[LLMProvider] = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    strict: bool = False,
) -> RAGResult:
    """
    Citation-enforced RAG pipeline.

    Same as answer_question_filtered but:
    - Context labels each chunk with Source + Chunk ID
    - Prompt instructs LLM to cite chunk IDs
    - Post-generation validation checks citations
    - On failure: retries once with stricter prompt, or returns error in strict mode

    Args:
        question: User question text
        index_path: Path to the JSON index file
        top_k: Number of chunks to retrieve
        threshold: Similarity threshold for filtering
        reranker: Optional reranker instance
        fallback_strategy: What to do when all chunks are below threshold
        embedding_provider: Custom embedding provider
        llm_provider: Custom LLM provider
        max_context_chars: Max characters for context section
        strict: If True, fail on invalid citations instead of returning best effort
    """
    # 1. Load index
    path = Path(index_path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    index = DocumentIndex.load(index_path)
    if len(index) == 0:
        raise ValueError("Index is empty — no chunks to search")

    # 2. Create embedding provider
    if embedding_provider is None:
        embedding_config = EmbeddingConfig(model=index.embedding_model)
        embedding_provider = OpenAIEmbeddings(embedding_config)

    # 3. Embed the question
    query_embedding = embedding_provider.embed_texts([question])[0]

    # 4. Validate dimension match
    sample_chunk = index.get_all_chunks()[0]
    if len(query_embedding) != len(sample_chunk.embedding):
        raise ValueError(
            f"Embedding dimension mismatch: query has {len(query_embedding)}, "
            f"index has {len(sample_chunk.embedding)}."
        )

    # 5. Retrieve top-k
    retriever = Retriever(index)
    results = retriever.search(query_embedding, top_k=top_k)

    if not results:
        return RAGResult(
            answer="No relevant chunks found in the index.",
            chunks_retrieved=0, chunks_after_filter=0,
            avg_similarity=0.0, avg_rerank_score=None,
            used_fallback=False,
        )

    # 6. Convert to RetrievedChunk
    chunks = _convert_results(results)
    chunks_retrieved = len(chunks)
    avg_similarity = sum(c.similarity for c in chunks) / len(chunks)

    # 7. Optional reranking
    avg_rerank: Optional[float] = None
    if reranker is not None:
        chunks = reranker.rerank(question, chunks)
        scored = [c for c in chunks if c.rerank_score is not None]
        if scored:
            avg_rerank = sum(c.rerank_score for c in scored) / len(scored)

    # 8. Threshold filter
    filtered, used_fallback = _apply_threshold_filter(
        chunks, threshold, fallback_strategy
    )

    if not filtered:
        return RAGResult(
            answer="Insufficient context — no chunks met the relevance threshold.",
            chunks_retrieved=chunks_retrieved,
            chunks_after_filter=0,
            avg_similarity=avg_similarity,
            avg_rerank_score=avg_rerank,
            used_fallback=False,
            chunks=chunks,
        )

    # 9. Build citation-enforced prompt and generate with validation
    if llm_provider is None:
        llm_provider = OpenAILLM()

    answer = ""
    validation = CitationValidationResult(is_valid=False)

    for attempt in range(_CITATION_MAX_RETRIES):
        is_retry = attempt > 0
        prompt, allowed_ids = _build_cited_prompt(
            question, filtered, max_context_chars, used_fallback, is_retry
        )
        answer = llm_provider.generate(prompt)
        validation = validate_citations(answer, allowed_ids)

        if validation.is_valid:
            break

    # 10. Handle validation failure
    if not validation.is_valid and strict:
        answer = (
            "Error: LLM failed to produce valid citations after "
            f"{_CITATION_MAX_RETRIES} attempts. "
            f"Citations found: {validation.citations_found}. "
            f"Invalid citations: {validation.invalid_citations}."
        )

    return RAGResult(
        answer=answer,
        chunks_retrieved=chunks_retrieved,
        chunks_after_filter=len(filtered),
        avg_similarity=avg_similarity,
        avg_rerank_score=avg_rerank,
        used_fallback=used_fallback,
        chunks=filtered,
        citation_validation=validation,
    )
