"""Reranking module: second-stage relevance scoring for retrieved chunks."""

import json
import re
from typing import List, runtime_checkable

from typing import Protocol

from .llm import LLMProvider
from .retriever import RetrievedChunk

# Max chars of chunk text sent to the LLM for scoring
_MAX_CHUNK_CHARS = 2000


@runtime_checkable
class Reranker(Protocol):
    """Protocol for chunk rerankers."""

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Score and re-sort chunks by relevance to the query."""
        ...


class LLMReranker:
    """Reranks chunks by asking an LLM to score each one for relevance."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    def _score_single_chunk(self, query: str, chunk: RetrievedChunk) -> float:
        """Ask the LLM to rate relevance of a single chunk on a 0-1 scale."""
        text = chunk.text[:_MAX_CHUNK_CHARS]
        prompt = (
            "Rate how relevant the following text passage is to the given query.\n"
            "Respond with ONLY a JSON object: {\"score\": <float between 0.0 and 1.0>}\n\n"
            f"Query: {query}\n\n"
            f"Passage:\n{text}\n\n"
            "Relevance score:"
        )
        response = self.llm.generate(prompt)
        return self._parse_score(response)

    @staticmethod
    def _parse_score(response: str) -> float:
        """Extract a 0-1 score from the LLM response. JSON first, regex fallback."""
        # Try JSON parse
        try:
            data = json.loads(response)
            score = float(data["score"])
            return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

        # Regex fallback: find a decimal number
        match = re.search(r"(\d+\.?\d*)", response)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))

        return 0.0

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Score each chunk with the LLM and re-sort by effective_score descending."""
        for chunk in chunks:
            try:
                chunk.rerank_score = self._score_single_chunk(query, chunk)
            except Exception:
                # On failure, keep original similarity as the effective score
                chunk.rerank_score = None

        chunks.sort(key=lambda c: c.effective_score, reverse=True)
        return chunks
