import os
from rag_tutor.llm.providers.base import LLMProvider
from rag_tutor.llm.providers.gemini import GeminiProvider
from rag_tutor.llm.providers.openai import OpenAIProvider
from rag_tutor.llm.providers.deepseek import DeepSeekProvider

# Registry of all providers
_PROVIDER_REGISTRY: dict[str, type[LLMProvider]] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "deepseek": DeepSeekProvider,
}


def list_providers() -> list[str]:
    """Return a list of supported provider names."""
    return list(_PROVIDER_REGISTRY.keys())


def get_provider(name: str = None, api_key: str = None) -> LLMProvider:
    """Get an LLM provider instance by name.

    Args:
        name: Provider name (gemini, openai, deepseek).
              Falls back to LLM_PROVIDER env var, then gemini.
        api_key: Optional explicit API key. If omitted, provider reads from env.

    Returns:
        Configured LLMProvider instance.

    Raises:
        ValueError: If the provider name is unknown or API key is missing.
    """
    provider_name = (name or os.getenv("LLM_PROVIDER", "gemini")).lower().strip()

    if provider_name not in _PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Supported: {', '.join(list_providers())}"
        )

    provider_cls = _PROVIDER_REGISTRY[provider_name]
    provider = provider_cls(api_key=api_key)

    if not provider.is_available():
        raise ValueError(
            f"Provider '{provider_name}' is not available. "
            f"Please set the corresponding API key in your .env file or sidebar."
        )

    return provider


def get_embedding_provider(name: str = None, api_key: str = None) -> LLMProvider:
    """Get the provider to use for embeddings.

    Defaults to the value of EMBEDDING_PROVIDER env var, or falls back
    to the main LLM provider. This enables hybrid setups like
    DeepSeek for chat + OpenAI for embeddings.
    """
    embedding_provider = os.getenv("EMBEDDING_PROVIDER")
    if embedding_provider:
        return get_provider(name=embedding_provider, api_key=api_key)
    return get_provider(name=name, api_key=api_key)