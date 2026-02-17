"""Embedding generation with configurable providers."""

import time
from abc import ABC, abstractmethod
from typing import List

import openai

from .settings import EmbeddingConfig
from .chunker import Chunk


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass

    @abstractmethod
    def embed_chunks(self, chunks: List[Chunk]) -> List[tuple[Chunk, List[float]]]:
        """Generate embeddings for chunks, returning (chunk, embedding) pairs."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name used for embeddings."""
        pass


class OpenAIEmbeddings(EmbeddingProvider):
    """OpenAI embedding provider with batching and retry logic."""

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()

        if not self.config.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide api_key in EmbeddingConfig."
            )

        self.client = openai.OpenAI(api_key=self.config.api_key)

    @property
    def model_name(self) -> str:
        return self.config.model

    def _embed_batch_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts with retry logic for transient errors."""
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.config.model,
                    input=texts
                )
                return [item.embedding for item in response.data]
            except openai.RateLimitError as e:
                last_error = e
                wait_time = self.config.retry_delay * (2 ** attempt)
                time.sleep(wait_time)
            except openai.APIConnectionError as e:
                last_error = e
                wait_time = self.config.retry_delay * (2 ** attempt)
                time.sleep(wait_time)
            except openai.APIStatusError as e:
                if e.status_code >= 500:
                    last_error = e
                    wait_time = self.config.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    raise

        raise RuntimeError(
            f"Failed to generate embeddings after {self.config.max_retries} attempts: {last_error}"
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts with batching.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors (List[float])
        """
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        batch_size = self.config.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self._embed_batch_with_retry(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_chunks(self, chunks: List[Chunk]) -> List[tuple[Chunk, List[float]]]:
        """
        Generate embeddings for chunks.

        Args:
            chunks: List of Chunk objects

        Returns:
            List of (chunk, embedding) tuples
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embed_texts(texts)

        return list(zip(chunks, embeddings))
