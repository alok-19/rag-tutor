import os
from rag_tutor.llm.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek provider (OpenAI-compatible API).

    DeepSeek does not offer an embedding API as of mid-2025,
    so users should set EMBEDDING_PROVIDER=openai or gemini alongside
    LLM_PROVIDER=deepseek for a hybrid setup.
    """

    name = "deepseek"

    def __init__(self, api_key: str = None):
        base_url = "https://api.deepseek.com"
        # Prefer env keys; fallback to passed key for manual sidebar entry
        super().__init__(api_key=os.getenv("DEEPSEEK_API_KEY") or api_key, base_url=base_url)

    def is_available(self) -> bool:
        return bool(os.getenv("DEEPSEEK_API_KEY") or (hasattr(self, "_api_key") and self._api_key))