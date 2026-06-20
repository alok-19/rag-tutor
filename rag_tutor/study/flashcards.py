"""Flashcard generation from retrieved study-material context.

Same defensive-parsing philosophy as ``quiz.py``: the LLM is asked for strict
JSON, and any malformed output degrades to an empty list instead of crashing.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag_tutor.study.quiz import _extract_json_object, _strip_trailing_commas


@dataclass
class Flashcard:
    front: str
    back: str


def build_flashcard_prompt(subject: str, context_text: str, num_cards: int = 8) -> str:
    """Construct the prompt asking the LLM for flashcards as JSON."""
    return f"""You are an expert tutor for '{subject}'. Using ONLY the study-material
context below, create {num_cards} high-quality flashcards. Each card's 'front' is
a concise question or term, and 'back' is a clear, accurate answer grounded in
the context.

Respond with STRICT JSON and nothing else — no markdown fences, no commentary.
The exact schema:

{{
  "cards": [
    {{ "front": "...", "back": "..." }}
  ]
}}

Rules:
- Front and back must both be non-empty.
- Base answers on the provided context.

Context:
{context_text}
"""


def parse_flashcards(raw: str) -> list[Flashcard]:
    """Parse the model's raw output into validated Flashcards.

    Returns an empty list if nothing valid can be recovered. Never raises.
    """
    if not raw or not raw.strip():
        return []

    candidate = _strip_trailing_commas(_extract_json_object(raw))
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    cards_raw = _coerce_card_list(data)
    cards: list[Flashcard] = []
    for item in cards_raw:
        card = _build_card(item)
        if card is not None:
            cards.append(card)
    return cards


def _coerce_card_list(data) -> list:
    """Accept {"cards": [...]} or a bare list; return the inner list."""
    if isinstance(data, dict):
        for key in ("cards", "flashcards", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        # A single card at top level.
        if {"front", "back"} <= set(data.keys()):
            return [data]
        return []
    if isinstance(data, list):
        return data
    return []


def _build_card(item) -> Flashcard | None:
    """Validate one card dict; return None if unusable."""
    if not isinstance(item, dict):
        return None
    front = str(item.get("front") or item.get("question") or item.get("term") or "").strip()
    back = str(item.get("back") or item.get("answer") or item.get("definition") or "").strip()
    if not front or not back:
        return None
    return Flashcard(front=front, back=back)
