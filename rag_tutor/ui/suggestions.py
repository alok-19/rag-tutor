"""Content-aware quick-start suggestion generation.

Instead of hardcoded suggestions, derive 3 starter questions from the actual
ingested material for the active subject. Pure helper + an LLM-backed generator;
both degrade gracefully to sensible defaults.
"""
from __future__ import annotations

import json
import random
from typing import Callable

from rag_tutor.retrieval.vector_store import get_collection
from rag_tutor.llm import generate_response_stream

DEFAULT_SUGGESTIONS = [
    "Summarize the key takeaways from the study materials.",
    "Explain the most complex concept described in these files.",
    "What are the most important terms I should memorize?",
]

# Max chars collected from the LLM when generating suggestions.
_MAX_SUGGEST_CHARS = 4000


def sample_chunks(subject: str, n: int = 6) -> list[str]:
    """Return a small sample of ingested text chunks for ``subject``.

    Pulls via Chroma ``get`` and samples deterministically-ish across the
    available chunks. Returns an empty list if nothing is stored.
    """
    try:
        collection = get_collection()
        result = collection.get(where={"subject": subject}, limit=100)
    except Exception:
        return []
    docs = result.get("documents", []) if isinstance(result, dict) else []
    if not docs:
        return []
    # Spread the sample across the document set rather than taking the head.
    if len(docs) > n:
        step = len(docs) // n
        docs = [docs[i] for i in range(0, len(docs), step)][:n]
    return [d[:300] for d in docs if isinstance(d, str) and d.strip()]


def build_suggestion_prompt(subject: str, chunks: list[str]) -> str:
    joined = "\n---\n".join(chunks)
    return f"""You are a study assistant. Based on these excerpts from '{subject}' study
materials, suggest exactly 3 concise, specific questions a student would benefit
from asking. Make them grounded in the actual content (not generic).

Respond with STRICT JSON only:
{{"suggestions": ["question 1", "question 2", "question 3"]}}

Excerpts:
{joined}
"""


def parse_suggestions(raw: str) -> list[str]:
    """Parse the model output into up to 3 suggestion strings. Never raises."""
    if not raw or not raw.strip():
        return []
    # Find the first {...} block.
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        for key in ("suggestions", "questions", "items"):
            val = data.get(key)
            if isinstance(val, list):
                out = [str(x).strip() for x in val if str(x).strip()]
                return out[:3]
    return []


def generate_suggestions(
    subject: str,
    api_key: str | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> list[str]:
    """Generate 3 content-aware suggestions for ``subject``.

    Falls back to ``DEFAULT_SUGGESTIONS`` on any failure (no chunks, parse
    error, LLM error). Never raises.
    """
    chunks = sample_chunks(subject)
    if not chunks:
        return list(DEFAULT_SUGGESTIONS)

    if status_callback:
        status_callback("💡 Finding good starting questions...")
    prompt = build_suggestion_prompt(subject, chunks)

    text = ""
    try:
        for chunk, _ in generate_response_stream(prompt=prompt, api_key=api_key, status_callback=status_callback):
            text += chunk
            if len(text) > _MAX_SUGGEST_CHARS:
                break
    except Exception:
        return list(DEFAULT_SUGGESTIONS)
    finally:
        if status_callback:
            status_callback("")

    parsed = parse_suggestions(text)
    return parsed if parsed else list(DEFAULT_SUGGESTIONS)
