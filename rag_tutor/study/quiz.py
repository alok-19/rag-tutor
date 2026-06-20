"""Quiz generation from retrieved study-material context.

The LLM is asked to emit strict JSON; ``parse_quiz`` defends against the usual
flavors of malformed output (prose preamble, trailing commas, truncated text,
missing fields) so a bad model response degrades to a friendly error instead of
crashing the UI.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field


@dataclass
class Question:
    question: str
    options: list[str]
    correct_index: int
    explanation: str = ""


@dataclass
class Score:
    correct: int
    total: int

    @property
    def percent(self) -> float:
        return (self.correct / self.total * 100.0) if self.total else 0.0


def build_quiz_prompt(subject: str, context_text: str, num_questions: int = 5) -> str:
    """Construct the prompt asking the LLM for a multiple-choice quiz as JSON."""
    return f"""You are an expert tutor for '{subject}'. Using ONLY the study-material
context below, create a {num_questions}-question multiple-choice quiz that tests
real understanding (not rote recall). Each question must have exactly 4 options.

Respond with STRICT JSON and nothing else — no markdown fences, no commentary.
The exact schema:

{{
  "quiz": [
    {{
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "correct_index": 0,
      "explanation": "why the correct option is right, citing the source"
    }}
  ]
}}

Rules:
- "correct_index" is a zero-based integer in [0, 3].
- "options" must contain exactly 4 non-empty, distinct strings.
- Base every question on the provided context. If the context is too thin for a
  full quiz, return as many valid questions as you can.

Context:
{context_text}
"""


def _extract_json_object(text: str) -> str:
    """Pull the first balanced ``{...}`` block out of ``text``."""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]  # unbalanced — best effort


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` or ``]`` (common LLM JSON error)."""
    return re.sub(r",(\s*[}\]])", r"\1", text)


def parse_quiz(raw: str) -> list[Question]:
    """Parse the model's raw output into a list of validated Questions.

    Returns an empty list if no valid question can be recovered. Never raises.
    """
    if not raw or not raw.strip():
        return []

    candidate = _strip_trailing_commas(_extract_json_object(raw))
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return []

    questions_raw = _coerce_question_list(data)
    questions: list[Question] = []
    for item in questions_raw:
        q = _build_question(item)
        if q is not None:
            questions.append(q)
    return questions


def _coerce_question_list(data) -> list:
    """Accept {"quiz": [...]} or a bare list; return the inner list."""
    if isinstance(data, dict):
        for key in ("quiz", "questions", "items", "data"):
            if isinstance(data.get(key), list):
                return data[key]
        # A single question object at the top level.
        if {"question"} <= set(data.keys()):
            return [data]
        return []
    if isinstance(data, list):
        return data
    return []


def _build_question(item) -> Question | None:
    """Validate one question dict; return None if unusable."""
    if not isinstance(item, dict):
        return None
    question = (item.get("question") or item.get("q") or "").strip()
    if not question:
        return None

    options = item.get("options") or item.get("choices") or []
    if not isinstance(options, list):
        return None
    options = [str(o).strip() for o in options if str(o).strip()]
    if len(options) < 2:
        return None
    # Keep at most 4 options, padding nothing — grading adapts to len.
    options = options[:4]

    correct_index = _resolve_correct_index(item, len(options))
    if correct_index is None:
        return None

    explanation = str(item.get("explanation") or item.get("rationale") or "").strip()
    return Question(
        question=question,
        options=options,
        correct_index=correct_index,
        explanation=explanation,
    )


def _resolve_correct_index(item: dict, n_options: int) -> int | None:
    """Find the correct option index from any of the common key shapes."""
    raw = item.get("correct_index")
    if raw is None:
        raw = item.get("answer_index", item.get("correct", item.get("answer")))
    try:
        if isinstance(raw, str) and raw.strip().isdigit():
            idx = int(raw.strip())
        elif isinstance(raw, int):
            idx = raw
        else:
            # Letter form: "A"/"a" -> 0.
            letter = str(raw).strip().upper()
            if len(letter) == 1 and "A" <= letter <= "Z":
                idx = ord(letter) - ord("A")
            else:
                return None
    except (TypeError, ValueError):
        return None
    return idx if 0 <= idx < n_options else None


def grade_quiz(questions: list[Question], answers: list[int]) -> Score:
    """Score a quiz given the user's selected option indices.

    Missing/None answers count as incorrect. ``answers`` shorter than
    ``questions`` is handled safely.
    """
    if not questions:
        return Score(correct=0, total=0)
    correct = 0
    for i, q in enumerate(questions):
        if i < len(answers) and answers[i] is not None and answers[i] == q.correct_index:
            correct += 1
    return Score(correct=correct, total=len(questions))
