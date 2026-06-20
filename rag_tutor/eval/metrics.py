"""Pure retrieval-quality metrics for the eval harness.

All functions here are deterministic and side-effect free: they take ranked
retrieved results and a set of relevant items and return a float score. They
work on plain ``(source, page)`` tuples so they have no dependency on the
retrieval or LLM layers, making them trivially unit-testable.

Conventions:
- ``retrieved`` is an ordered list of ``(source, page)`` tuples, best-first.
- ``relevant`` is a list (or set) of ``(source, page)`` tuples considered
  ground-truth relevant for the query.
- All scores are in ``[0.0, 1.0]``.
"""
from __future__ import annotations

from typing import Iterable

from rag_tutor.retrieval.vector_store import RetrievedSource


def _relevant_set(relevant: Iterable) -> set[tuple[str, int]]:
    """Normalize the relevant collection into a set of (source, page) tuples."""
    out: set[tuple[str, int]] = set()
    for item in relevant or []:
        if isinstance(item, dict):
            src = item.get("source")
            pg = item.get("page")
        elif isinstance(item, tuple):
            src, pg = item
        else:
            # Assume a RetrievedSource-like dataclass.
            src = getattr(item, "source", None)
            pg = getattr(item, "page", None)
        if src is not None and pg is not None:
            out.add((str(src), int(pg)))
    return out


def _retrieved_pairs(retrieved: Iterable) -> list[tuple[str, int]]:
    """Normalize the retrieved collection into an ordered list of tuples."""
    out: list[tuple[str, int]] = []
    for item in retrieved or []:
        if isinstance(item, dict):
            src = item.get("source")
            pg = item.get("page")
        elif isinstance(item, tuple):
            src, pg = item
        else:
            src = getattr(item, "source", None)
            pg = getattr(item, "page", None)
        if src is not None and pg is not None:
            out.append((str(src), int(pg)))
    return out


def hit_rate(retrieved: Iterable, relevant: Iterable) -> float:
    """1.0 if any relevant item appears anywhere in retrieved, else 0.0."""
    pairs = _retrieved_pairs(retrieved)
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    return 1.0 if any(p in rel for p in pairs) else 0.0


def reciprocal_rank(retrieved: Iterable, relevant: Iterable) -> float:
    """Reciprocal rank of the first relevant hit (1/rank). 0.0 if none."""
    pairs = _retrieved_pairs(retrieved)
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    for i, p in enumerate(pairs, start=1):
        if p in rel:
            return 1.0 / i
    return 0.0


def precision_at_k(retrieved: Iterable, relevant: Iterable, k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    pairs = _retrieved_pairs(retrieved)[:k]
    rel = _relevant_set(relevant)
    if not pairs:
        return 0.0
    hits = sum(1 for p in pairs if p in rel)
    return hits / len(pairs)


def recall_at_k(retrieved: Iterable, relevant: Iterable, k: int) -> float:
    """Fraction of relevant items that appear in the top-k retrieved.

    Defined as 0.0 when there are no relevant items.
    """
    if k <= 0:
        return 0.0
    pairs = _retrieved_pairs(retrieved)[:k]
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for p in pairs if p in rel)
    return hits / len(rel)


def mrr(items: Iterable[tuple[Iterable, Iterable]]) -> float:
    """Mean Reciprocal Rank across many (retrieved, relevant) pairs."""
    items = list(items or [])
    if not items:
        return 0.0
    total = sum(reciprocal_rank(r, rel) for r, rel in items)
    return total / len(items)


def citation_coverage(answer_text: str, sources: list[RetrievedSource] | list[dict]) -> float:
    """Heuristic grounding check: fraction of cited sources referenced in text.

    A source counts as "referenced" if either its filename (without extension)
    or its page number (as "page N" / "p. N" / "pN") appears in the answer.
    This is a deliberate heuristic — it does not require an LLM judge.

    Returns 0.0 when there are no sources to cover.
    """
    if not sources:
        return 0.0
    text = (answer_text or "").lower()
    if not text:
        return 0.0
    covered = 0
    for s in sources:
        src = s.get("source") if isinstance(s, dict) else getattr(s, "source", "")
        page = s.get("page") if isinstance(s, dict) else getattr(s, "page", None)
        name = str(src or "")
        # filename without extension, lowercased
        stem = name.rsplit(".", 1)[0].lower() if "." in name else name.lower()
        referenced = False
        if stem and len(stem) >= 4 and stem in text:
            referenced = True
        if page is not None:
            pg = str(page)
            if f"page {pg}" in text or f"p. {pg}" in text or f"p{pg}" in text or f"pg {pg}" in text:
                referenced = True
        if referenced:
            covered += 1
    return covered / len(sources)
