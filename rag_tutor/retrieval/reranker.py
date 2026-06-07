"""BGE cross-encoder reranker for two-stage retrieval.

This module is an optional, additive enhancement. It is **only** used when
``ENABLE_RERANKER=true`` is set in the environment. When the ``[hybrid]``
extras are not installed, every public function/class on this module raises
a clear, actionable error rather than crashing the import of the package.

Two-stage retrieval flow:
    1. Vector store returns top-N candidates (e.g. n=20) — fast but noisy.
    2. Cross-encoder reranks them with a precise query-document relevance
       score and returns the top-K (e.g. k=4) for the LLM prompt.

This is the single highest-impact quality improvement available for a
typical RAG pipeline. It is **off by default** to avoid a forced
~1-2 GB dependency install for users who do not want it.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

from rag_tutor.retrieval.vector_store import RetrievedSource


_RERANKER_MODEL_ENV = "RERANKER_MODEL"
_DEFAULT_MODEL = "BAAI/bge-reranker-base"


class RerankerUnavailableError(RuntimeError):
    """Raised when the reranker is requested but its dependencies are missing."""


def _try_import_cross_encoder():
    """Import the CrossEncoder class lazily.

    Returns the class, or ``None`` if sentence-transformers is not installed.
    We deliberately avoid importing at module level so the rest of the
    package loads cleanly for users who have not installed ``[hybrid]``.
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        return CrossEncoder
    except Exception:
        return None


class BgeReranker:
    """BGE cross-encoder reranker. Loads the model lazily on first use.

    Args:
        model_name: Hugging Face model id. Defaults to
            ``BAAI/bge-reranker-base`` (lightweight, English + code).
            Use ``BAAI/bge-reranker-v2-m3`` for multilingual / academic text.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self._model_name = (
            model_name
            or os.getenv(_RERANKER_MODEL_ENV, _DEFAULT_MODEL).strip()
            or _DEFAULT_MODEL
        )
        self._model = None  # lazy

    @staticmethod
    def is_available() -> bool:
        """Return True if sentence-transformers is importable in this env."""
        return _try_import_cross_encoder() is not None

    def _load(self):
        if self._model is not None:
            return self._model
        CrossEncoder = _try_import_cross_encoder()
        if CrossEncoder is None:
            raise RerankerUnavailableError(
                "bge-reranker requires the 'sentence-transformers' package. "
                "Install it with: pip install -e \".[hybrid]\" "
                "(or: pip install sentence-transformers torch)."
            )
        self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
        self,
        query: str,
        documents: list[RetrievedSource],
        top_k: Optional[int] = None,
    ) -> list[RetrievedSource]:
        """Rerank ``documents`` by query relevance and return the top ``top_k``.

        If ``top_k`` is None or larger than ``len(documents)``, all documents
        are returned in reranked order. If the input is empty, it is returned
        unchanged.
        """
        if not documents:
            return documents
        if not query or not query.strip():
            return documents[:top_k] if top_k else documents

        model = self._load()
        pairs = [[query, doc.text] for doc in documents]
        scores = model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda pair: float(pair[1]),
            reverse=True,
        )
        ordered = [doc for doc, _score in ranked]
        if top_k is not None and top_k >= 0:
            return ordered[:top_k]
        return ordered


def rerank_enabled() -> bool:
    """Return True if the user opted in via ``ENABLE_RERANKER=true``."""
    flag = os.getenv("ENABLE_RERANKER", "false").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def get_reranker() -> BgeReranker:
    """Factory for the default reranker. Honors the ``RERANKER_MODEL`` env var."""
    return BgeReranker()
