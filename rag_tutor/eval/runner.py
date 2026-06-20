"""Eval orchestration: run retrieval over a dataset and aggregate metrics.

The runner is the only piece that touches the real retrieval layer. It accepts
an injectable ``retrieve`` callable (defaulting to ``retrieve_context``) so it
can be unit-tested with a deterministic stub and so the eval path is decoupled
from the live embedding/provider stack.

Resilience contract: a single failing retrieval (network error, empty store,
provider issue) is recorded as a zero-score item with an error note and the run
continues. The aggregate report is always produced.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

from rag_tutor.retrieval.vector_store import RetrievedSource
from rag_tutor.eval.dataset import EvalItem
from rag_tutor.eval.metrics import (
    hit_rate,
    reciprocal_rank,
    precision_at_k,
    recall_at_k,
    citation_coverage,
)


# Signature of the retrieval callable the runner consumes.
# Returns a list of RetrievedSource for a given (query, subject, k).
RetrieveFn = Callable[[str, str | None, int], List[RetrievedSource]]


@dataclass
class ItemResult:
    query: str
    hit: float
    rr: float
    precision: float
    recall: float
    n_retrieved: int
    subject: str | None = None
    error: str | None = None


@dataclass
class EvalReport:
    items: List[ItemResult] = field(default_factory=list)
    hit_rate: float = 0.0
    mrr: float = 0.0
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    n_items: int = 0
    n_errors: int = 0
    k: int = 4

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "n_items": self.n_items,
            "n_errors": self.n_errors,
            "hit_rate": round(self.hit_rate, 4),
            "mrr": round(self.mrr, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "recall_at_k": round(self.recall_at_k, 4),
        }


def _keyword_relevant_pairs(item: EvalItem, retrieved: List[RetrievedSource]) -> list[tuple[str, int]]:
    """Fallback relevance when a dataset item has no explicit relevant_sources.

    Treats a retrieved passage as relevant if its text contains any of the
    item's ``expected_keywords`` (case-insensitive). Returns the matching
    (source, page) pairs. Returns the item's explicit pairs when provided.
    """
    explicit = item.relevant_pairs()
    if explicit:
        return explicit
    keywords = [k.lower() for k in item.expected_keywords if k]
    if not keywords:
        return []
    pairs = []
    for r in retrieved:
        text = (r.text or "").lower()
        if any(kw in text for kw in keywords):
            pairs.append((r.source, r.page))
    return pairs


def run_eval(
    dataset: List[EvalItem],
    retrieve: RetrieveFn,
    k: int = 4,
    subject: str | None = None,
) -> EvalReport:
    """Run retrieval over ``dataset`` and return an aggregated EvalReport.

    Args:
        dataset: list of EvalItem to evaluate.
        retrieve: callable(query, subject, k) -> list[RetrievedSource]. Inject
            a stub for tests; defaults to the live retriever at the call site.
        k: retrieval depth and the ``@k`` for precision/recall.
        subject: optional subject override; otherwise each item's own subject
            is used (or None).
    """
    report = EvalReport(k=k)
    if not dataset:
        return report

    rr_scores: list[float] = []
    hit_scores: list[float] = []
    p_scores: list[float] = []
    rec_scores: list[float] = []

    for item in dataset:
        target_subject = subject or item.subject
        try:
            retrieved = retrieve(item.query, target_subject, k) or []
        except Exception as e:
            res = ItemResult(
                query=item.query,
                hit=0.0, rr=0.0, precision=0.0, recall=0.0,
                n_retrieved=0, subject=target_subject, error=str(e),
            )
            report.items.append(res)
            report.n_errors += 1
            continue

        retrieved_pairs = [(r.source, r.page) for r in retrieved]
        relevant_pairs = _keyword_relevant_pairs(item, retrieved)

        h = hit_rate(retrieved_pairs, relevant_pairs)
        rr = reciprocal_rank(retrieved_pairs, relevant_pairs)
        p = precision_at_k(retrieved_pairs, relevant_pairs, k)
        rec = recall_at_k(retrieved_pairs, relevant_pairs, k)

        res = ItemResult(
            query=item.query,
            hit=h, rr=rr, precision=p, recall=rec,
            n_retrieved=len(retrieved), subject=target_subject,
        )
        report.items.append(res)
        hit_scores.append(h)
        rr_scores.append(rr)
        p_scores.append(p)
        rec_scores.append(rec)

    n = len(report.items)
    report.n_items = n
    if n:
        report.hit_rate = sum(hit_scores) / n
        report.mrr = sum(rr_scores) / n
        report.precision_at_k = sum(p_scores) / n
        report.recall_at_k = sum(rec_scores) / n
    return report


def evaluate_answer_grounding(
    answer_text: str,
    sources: List[RetrievedSource] | List[dict],
) -> float:
    """Convenience wrapper exposing citation coverage for the eval CLI."""
    return citation_coverage(answer_text, sources)


@dataclass
class LeakageReport:
    """Cross-subject contamination report.

    ``leakage_rate`` is the fraction of retrieved chunks (averaged over all
    queries) that came from a DIFFERENT subject than the query's intended
    subject. Lower is better; 0.0 means no cross-subject contamination.
    ``on_subject_rate`` is its complement (higher is better).
    """
    n_items: int = 0
    n_errors: int = 0
    leakage_rate: float = 0.0
    on_subject_rate: float = 0.0
    items: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "n_errors": self.n_errors,
            "leakage_rate": round(self.leakage_rate, 4),
            "on_subject_rate": round(self.on_subject_rate, 4),
            "items": self.items,
        }


def _retrieved_subjects(retrieved: List[RetrievedSource]) -> list[str | None]:
    """Best-effort subject extraction from retrieved sources.

    ``RetrievedSource`` does not carry a subject field (it's filtered out at
    query time), so for the leakage probe we rely on a retrieve function that
    returns sources annotated with a ``subject`` attribute via the source dict.
    In the live CLI we issue an unfiltered query and attach subjects from the
    raw Chroma metadata. Here we just read whatever attribute is present.
    """
    out = []
    for r in retrieved or []:
        out.append(getattr(r, "subject", None))
    return out


def run_leakage_eval(
    dataset: List[EvalItem],
    retrieve_cross_subject: Callable[[str, int], List[tuple]],
    k: int = 4,
) -> LeakageReport:
    """Measure cross-subject contamination when subject filtering is OFF.

    ``retrieve_cross_subject(query, k)`` must return a list of ``(source, page,
    subject)`` tuples — i.e. the raw retrieval result WITHOUT a subject filter,
    with each chunk's own subject attached. The report then measures how often
    a query intended for subject A pulls chunks from subject B.

    A query is skipped (not counted) if its own ``subject`` is unset.
    """
    report = LeakageReport()
    eligible = [it for it in dataset if it.subject]
    if not eligible:
        return report

    on_subject_rates: list[float] = []
    for item in eligible:
        try:
            rows = retrieve_cross_subject(item.query, k) or []
        except Exception as e:
            report.items.append({"query": item.query, "subject": item.subject, "error": str(e)})
            report.n_errors += 1
            continue
        if not rows:
            on_subject_rates.append(0.0)
            report.items.append({
                "query": item.query, "subject": item.subject,
                "retrieved_subjects": [], "on_subject_rate": 0.0,
            })
            continue
        correct = sum(1 for r in rows if len(r) >= 3 and r[2] == item.subject)
        rate = correct / len(rows)
        on_subject_rates.append(rate)
        report.items.append({
            "query": item.query,
            "subject": item.subject,
            "retrieved_subjects": [r[2] if len(r) >= 3 else None for r in rows],
            "on_subject_rate": round(rate, 4),
        })

    report.n_items = len(on_subject_rates)
    if on_subject_rates:
        report.on_subject_rate = sum(on_subject_rates) / len(on_subject_rates)
        report.leakage_rate = 1.0 - report.on_subject_rate
    return report
