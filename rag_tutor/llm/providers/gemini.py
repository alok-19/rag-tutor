import os
import time
import random
from typing import Generator
from google import genai
from google.genai import types as genai_types
from google.genai.errors import APIError
from rag_tutor.llm.providers.base import LLMProvider


class GeminiProvider(LLMProvider):
    """Google Gemini provider for embeddings and generation."""

    name = "gemini"

    def __init__(self, api_key: str = None):
        # Prefer env keys; fallback to passed key for manual sidebar entry
        env_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        # Simple heuristic: reject short or placeholder-looking keys
        if env_key and len(env_key) >= 10 and "your_" not in env_key.lower():
            self._api_key = env_key
        else:
            self._api_key = api_key
        self._client = None
        self._last_key = None

    def _get_client(self):
        if self._client is None or self._api_key != self._last_key:
            if not self._api_key:
                raise ValueError(
                    "Gemini API key not set. Provide GEMINI_API_KEY in .env or sidebar."
                )
            self._client = genai.Client(api_key=self._api_key)
            self._last_key = self._api_key
        return self._client

    def is_available(self) -> bool:
        return bool(self._api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings with exponential backoff retry."""
        client = self._get_client()
        model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
        batch_size = 64
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            contents = [
                genai_types.Content(parts=[genai_types.Part.from_text(text=txt)])
                for txt in batch
            ]

            delay = 2.0
            max_retries = 6
            response = None

            for attempt in range(max_retries):
                try:
                    response = client.models.embed_content(
                        model=model, contents=contents
                    )
                    break
                except APIError as e:
                    if e.code == 429:
                        sleep_time = delay + random.uniform(0.1, 0.5)
                        print(
                            f"Rate limited (429) on batch. Retrying in {sleep_time:.2f}s... "
                            f"(Attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(sleep_time)
                        delay *= 2
                    else:
                        raise e
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
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
                all_embeddings.extend([e.values for e in response.embeddings])
            time.sleep(1.2)

        return all_embeddings

    def generate_stream(
        self, prompt: str, model: str, system_prompt: str = None
    ) -> Generator[str, None, None]:
        """Stream generation with Gemini."""
        client = self._get_client()

        contents = []
        if system_prompt:
            contents.append(
                genai_types.Content(
                    role="system",
                    parts=[genai_types.Part.from_text(text=system_prompt)],
                )
            )
        contents.append(
            genai_types.Content(
                role="user", parts=[genai_types.Part.from_text(text=prompt)]
            )
        )

        response_stream = client.models.generate_content_stream(
            model=model, contents=contents
        )
        for chunk in response_stream:
            yield chunk.text