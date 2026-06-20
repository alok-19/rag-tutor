"""Helpers that turn study-material context into structured study artifacts.

These reuse the existing streaming ``generate_response_stream`` (no provider
changes required) by collecting the stream into a single string, then parsing
it into validated dataclasses. A cap on collected tokens guards against a
runaway model that never emits JSON.
"""
from __future__ import annotations

from typing import Callable

from rag_tutor.llm import generate_response_stream
from rag_tutor.retrieval import RetrievedSource
from rag_tutor.study.quiz import build_quiz_prompt, parse_quiz, Question
from rag_tutor.study.flashcards import build_flashcard_prompt, parse_flashcards, Flashcard

# Hard cap on collected output characters. A 5-question quiz or 8-card deck in
# JSON is well under this; anything larger is a model gone off-script.
_MAX_OUTPUT_CHARS = 16000


def _sources_to_context(sources: list[RetrievedSource]) -> str:
    """Render retrieved sources into the ``Context:`` block used by generators."""
    if not sources:
        return "(No context available.)"
    parts = []
    for i, s in enumerate(sources, 1):
        parts.append(f"[{i}] Source: {s.source}, Page: {s.page}\n{s.text}")
    return "\n\n".join(parts)


def _collect_stream(prompt: str, api_key: str | None, status_callback: Callable[[str], None] | None) -> str:
    """Run the streaming generator to completion and return the full text.

    Stops early if the accumulated output exceeds ``_MAX_OUTPUT_CHARS`` to
    avoid unbounded buffering on a misbehaving model.
    """
    text = ""
    for chunk, _fallback in generate_response_stream(
        prompt=prompt,
        api_key=api_key,
        status_callback=status_callback,
    ):
        text += chunk
        if len(text) > _MAX_OUTPUT_CHARS:
            break
    return text


def generate_quiz(
    subject: str,
    sources: list[RetrievedSource],
    api_key: str | None = None,
    num_questions: int = 5,
    status_callback: Callable[[str], None] | None = None,
) -> list[Question]:
    """Generate a multiple-choice quiz grounded in ``sources``."""
    if status_callback:
        status_callback("✍️ Writing quiz questions...")
    context = _sources_to_context(sources)
    prompt = build_quiz_prompt(subject, context, num_questions=num_questions)
    raw = _collect_stream(prompt, api_key, status_callback)
    if status_callback:
        status_callback("")
    return parse_quiz(raw)


def generate_flashcards(
    subject: str,
    sources: list[RetrievedSource],
    api_key: str | None = None,
    num_cards: int = 8,
    status_callback: Callable[[str], None] | None = None,
) -> list[Flashcard]:
    """Generate flashcards grounded in ``sources``."""
    if status_callback:
        status_callback("🗂️ Creating flashcards...")
    context = _sources_to_context(sources)
    prompt = build_flashcard_prompt(subject, context, num_cards=num_cards)
    raw = _collect_stream(prompt, api_key, status_callback)
    if status_callback:
        status_callback("")
    return parse_flashcards(raw)
