"""BGE-M3 local embedding provider.

This is an **opt-in** alternative to the cloud embedding providers (Gemini,
OpenAI). It runs the ``BAAI/bge-m3`` model locally via sentence-transformers.

Key properties:
    - Multilingual (100+ languages), strong on academic / technical text.
    - Matryoshka Representation Learning: you can truncate the dense vector
      to any prefix dimension (we default to 1024) and still get
      competitive retrieval quality.
    - No API key, no network calls, no per-token cost.
    - Requires the ``[hybrid]`` extras (``sentence-transformers`` + ``torch``).

The class implements the existing ``LLMProvider`` interface so it slots
into the existing factory/embedding dispatch. ``generate_stream`` raises
``NotImplementedError`` because BGE is an embedding-only model.
"""
from __future__ import annotations

import os
from typing import Generator

from rag_tutor.config import BGE_M3_MODEL_NAME, BGE_M3_DIM, BGE_M3_DEVICE
from rag_tutor.llm.providers.base import LLMProvider


class BgeUnavailableError(RuntimeError):
    """Raised when the BGE-M3 provider is used without the [hybrid] extras."""


def _try_import_sentence_transformer():
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer
    except Exception:
        return None


class BgeM3Provider(LLMProvider):
    """Local BGE-M3 embedding provider.

    Args:
        model_name: HF model id. Defaults to ``BAAI/bge-m3``.
        dim: Output dimension after Matryoshka truncation. Defaults to 1024
            (the BGE-M3 paper's recommended dense-only setting). The model
            produces 1024-dim vectors by default; if you change ``dim`` you
            must also set the matching dimension in your Chroma collection.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
    """

    name = "bge"

    def __init__(
        self,
        api_key: str = None,  # accepted for interface compat; unused
        model_name: str = None,
        dim: int = None,
        device: str = None,
    ) -> None:
        self._api_key = api_key  # never used
        self._model_name = model_name or BGE_M3_MODEL_NAME
        self._dim = dim if dim is not None else BGE_M3_DIM
        self._device = device or BGE_M3_DEVICE
        self._model = None  # lazy

    @staticmethod
    def is_available() -> bool:
        """True if the local model can be loaded in this environment."""
        return _try_import_sentence_transformer() is not None

    def _load(self):
        if self._model is not None:
            return self._model
        SentenceTransformer = _try_import_sentence_transformer()
        if SentenceTransformer is None:
            raise BgeUnavailableError(
                "BGE-M3 requires the 'sentence-transformers' package. "
                "Install it with: pip install -e \".[hybrid]\" "
                "(or: pip install sentence-transformers torch)."
            )
        self._model = SentenceTransformer(self._model_name, device=self._device)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode ``texts`` to dense vectors, truncated to ``self._dim`` dims.

        We ask ``sentence-transformers`` to L2-normalize for us, then
        re-normalize after Matryoshka truncation (slicing breaks the unit
        length). The final vectors are unit length so cosine similarity
        reduces to a dot product, matching the Chroma
        ``hnsw:space=cosine`` collection setting.
        """
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        import numpy as np

        # Matryoshka truncation: keep the first ``self._dim`` dimensions.
        if self._dim and vectors.shape[1] > self._dim:
            vectors = vectors[:, : self._dim]
        # Always (re-)normalize so we are robust to fakes that ignore
        # ``normalize_embeddings`` and so that slicing does not break cosine.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        return vectors.tolist()

    def generate_stream(
        self, prompt: str, model: str = None, system_prompt: str = None
    ) -> Generator[str, None, None]:
        """BGE-M3 is an embedding-only model; generation is not supported."""
        raise NotImplementedError(
            "BgeM3Provider is embeddings-only. Choose a chat provider "
            "(gemini, openai, deepseek) via LLM_PROVIDER for generation."
        )
