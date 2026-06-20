"""Self-contained RAG evaluation harness.

Measures retrieval quality (Hit Rate, MRR, Precision@k, Recall@k) and answer
grounding (citation coverage) against a ground-truth dataset. No external
dependencies and no LLM tokens required.

Entry point: ``python -m rag_tutor eval`` (wired in rag_tutor/__main__.py).
"""
from .metrics import (
    hit_rate,
    reciprocal_rank,
    mrr,
    precision_at_k,
    recall_at_k,
    citation_coverage,
)
from .dataset import EvalItem, load_dataset, DEFAULT_DATASET
from .runner import run_eval, EvalReport, ItemResult, evaluate_answer_grounding, run_leakage_eval, LeakageReport

__all__ = [
    "hit_rate",
    "reciprocal_rank",
    "mrr",
    "precision_at_k",
    "recall_at_k",
    "citation_coverage",
    "EvalItem",
    "load_dataset",
    "DEFAULT_DATASET",
    "run_eval",
    "EvalReport",
    "ItemResult",
    "evaluate_answer_grounding",
    "run_leakage_eval",
    "LeakageReport",
]
