"""LLM Provider Abstraction Layer

Supports multiple LLM backends: Gemini, OpenAI, DeepSeek.
"""
from abc import ABC, abstractmethod
from typing import Generator


class LLMProvider(ABC):
    """Abstract base class for LLM providers.

    Implementations must handle:
    - Embedding generation (for vector search)
    - Chat/streaming generation (for responses)
    """

    name: str = "abstract"

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of texts.

        Returns:
            List of embedding vectors (list of floats), one per input text.
        """
        ...

    @abstractmethod
    def generate_stream(
        self, prompt: str, model: str, system_prompt: str = None
    ) -> Generator[str, None, None]:
        """Generate a streaming response from the LLM.

        Args:
            prompt: The user prompt / question.
            model: Model name/identifier to use.
            system_prompt: Optional system instruction.

        Yields:
            Text chunks as they are generated.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured (API key present)."""
        ...