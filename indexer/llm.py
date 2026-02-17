"""LLM provider abstraction for text generation."""

import time
from abc import ABC, abstractmethod

import openai

from .settings import LLMConfig


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate a text response from the given prompt."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name used for generation."""
        ...


class OpenAILLM(LLMProvider):
    """OpenAI Chat Completions provider with retry logic."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

        if not self.config.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide api_key in LLMConfig."
            )

        self.client = openai.OpenAI(api_key=self.config.api_key)

    @property
    def model_name(self) -> str:
        return self.config.model

    def generate(self, prompt: str) -> str:
        """
        Generate a response using OpenAI Chat Completions.

        Args:
            prompt: The full prompt (system context + user question)

        Returns:
            Generated text response
        """
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                content = response.choices[0].message.content
                return content or ""
            except openai.RateLimitError as e:
                last_error = e
                time.sleep(self.config.retry_delay * (2 ** attempt))
            except openai.APIConnectionError as e:
                last_error = e
                time.sleep(self.config.retry_delay * (2 ** attempt))
            except openai.APIStatusError as e:
                if e.status_code >= 500:
                    last_error = e
                    time.sleep(self.config.retry_delay * (2 ** attempt))
                else:
                    raise

        raise RuntimeError(
            f"Failed to generate response after {self.config.max_retries} attempts: {last_error}"
        )
