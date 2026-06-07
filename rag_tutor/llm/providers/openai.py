import os
import time
import random
from typing import Generator
from openai import OpenAI
from rag_tutor.llm.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    """OpenAI provider for embeddings and generation."""

    name = "openai"

    def __init__(self, api_key: str = None, base_url: str = None):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url  # None for native OpenAI
        self._client = None
        self._last_key = None

    def _get_client(self):
        if self._client is None or self._api_key != self._last_key:
            if not self._api_key:
                raise ValueError(
                    "OpenAI API key not set. Provide OPENAI_API_KEY in .env or sidebar."
                )
            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
            self._last_key = self._api_key
        return self._client

    def is_available(self) -> bool:
        return bool(self._api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings with exponential backoff retry."""
        client = self._get_client()
        model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            delay = 2.0
            max_retries = 6
            response = None

            for attempt in range(max_retries):
                try:
                    response = client.embeddings.create(model=model, input=batch)
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "rate_limit" in err_str.lower():
                        sleep_time = delay + random.uniform(0.1, 0.5)
                        print(
                            f"Rate limited on batch. Retrying in {sleep_time:.2f}s... "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        delay *= 2
                    else:
                        raise e
            else:
                raise RuntimeError(
                    f"Failed to generate embeddings after {max_retries} retries."
                )

            if response:
                all_embeddings.extend([d.embedding for d in response.data])
            time.sleep(0.5)

        return all_embeddings

    def generate_stream(
        self, prompt: str, model: str, system_prompt: str = None
    ) -> Generator[str, None, None]:
        """Stream generation with OpenAI."""
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta