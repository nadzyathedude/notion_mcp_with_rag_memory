"""Citation extraction and validation for RAG answers."""

import re
from dataclasses import dataclass, field
from typing import List, Set

# Pattern matches chunk IDs inside square brackets, e.g. [doc1_chunk_2]
# Supports comma-separated lists: [doc1_chunk_2, doc2_chunk_0]
_CITATION_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_CHUNK_ID_RE = re.compile(r"[a-zA-Z0-9_]+-?_chunk_\d+_[a-f0-9]+")


@dataclass
class CitationValidationResult:
    """Result of citation validation against allowed chunk IDs."""
    is_valid: bool
    citations_found: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    has_citations: bool = False


def extract_citations(text: str) -> List[str]:
    """
    Extract all chunk ID citations from a text.

    Looks for patterns like [chunk_id] or [id1, id2] in the text.

    Args:
        text: The LLM-generated answer text

    Returns:
        List of unique chunk IDs found (order preserved)
    """
    seen: Set[str] = set()
    result: List[str] = []

    for bracket_match in _CITATION_BRACKET_RE.finditer(text):
        inner = bracket_match.group(1)
        # Split on commas for multi-citation brackets
        parts = [p.strip() for p in inner.split(",")]
        for part in parts:
            # Validate it looks like a chunk ID
            if _CHUNK_ID_RE.match(part) and part not in seen:
                seen.add(part)
                result.append(part)

    return result


def validate_citations(
    answer: str,
    allowed_chunk_ids: List[str],
) -> CitationValidationResult:
    """
    Validate that an LLM answer contains proper citations.

    Checks:
    - At least one citation is present
    - All cited IDs are from the allowed set
    - No fabricated/unknown IDs

    Args:
        answer: The LLM-generated answer
        allowed_chunk_ids: List of chunk IDs that were in the context

    Returns:
        CitationValidationResult with validation details
    """
    found = extract_citations(answer)
    allowed_set = set(allowed_chunk_ids)

    invalid = [cid for cid in found if cid not in allowed_set]

    return CitationValidationResult(
        is_valid=len(found) > 0 and len(invalid) == 0,
        citations_found=found,
        invalid_citations=invalid,
        has_citations=len(found) > 0,
    )
