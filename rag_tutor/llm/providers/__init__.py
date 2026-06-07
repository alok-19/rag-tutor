from rag_tutor.llm.providers.base import LLMProvider
from rag_tutor.llm.providers.gemini import GeminiProvider
from rag_tutor.llm.providers.openai import OpenAIProvider
from rag_tutor.llm.providers.deepseek import DeepSeekProvider
from rag_tutor.llm.providers.factory import get_provider, get_embedding_provider, list_providers

# BGE is an optional, lazy-loaded provider. We expose it via module-level
# __getattr__ so that ``import rag_tutor.llm.providers`` does not pull in
# sentence-transformers / transformers / torchvision at app startup.
_LAZY_BGE_NAMES = ("BgeM3Provider", "BgeUnavailableError")


def __getattr__(name):
    if name in _LAZY_BGE_NAMES:
        from rag_tutor.llm.providers import bge as _bge

        value = getattr(_bge, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'rag_tutor.llm.providers' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_BGE_NAMES))


__all__ = [
    "LLMProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "get_provider",
    "get_embedding_provider",
    "list_providers",
    *_LAZY_BGE_NAMES,
]