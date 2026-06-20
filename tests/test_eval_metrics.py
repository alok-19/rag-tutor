import pytest
from rag_tutor.eval.metrics import (
    hit_rate,
    reciprocal_rank,
    precision_at_k,
    recall_at_k,
    mrr,
    citation_coverage,
)
from rag_tutor.retrieval.vector_store import RetrievedSource

REL = [("a.pdf", 1), ("b.pdf", 3)]


# ============================================================
# hit_rate
# ============================================================

def test_hit_rate_match_present():
    assert hit_rate([("a.pdf", 1), ("c.pdf", 2)], REL) == 1.0


def test_hit_rate_no_match():
    assert hit_rate([("c.pdf", 2), ("d.pdf", 9)], REL) == 0.0


def test_hit_rate_empty_retrieved():
    assert hit_rate([], REL) == 0.0


def test_hit_rate_empty_relevant():
    assert hit_rate([("a.pdf", 1)], []) == 0.0


def test_hit_rate_accepts_dataclass():
    retrieved = [RetrievedSource("a.pdf", 1, "x")]
    assert hit_rate(retrieved, REL) == 1.0


def test_hit_rate_accepts_dicts():
    retrieved = [{"source": "b.pdf", "page": 3}]
    assert hit_rate(retrieved, [{"source": "b.pdf", "page": 3}]) == 1.0


# ============================================================
# reciprocal_rank
# ============================================================

def test_reciprocal_rank_first_position():
    assert reciprocal_rank([("a.pdf", 1), ("c.pdf", 2)], REL) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank([("c.pdf", 2), ("b.pdf", 3)], REL) == 0.5


def test_reciprocal_rank_third_position():
    assert reciprocal_rank([("c.pdf", 2), ("d.pdf", 9), ("a.pdf", 1)], REL) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank([("c.pdf", 2)], REL) == 0.0


# ============================================================
# precision_at_k / recall_at_k
# ============================================================

def test_precision_at_k_basic():
    # 2 relevant in top-4 of 4 retrieved
    assert precision_at_k([("a.pdf", 1), ("b.pdf", 3), ("c.pdf", 2), ("d.pdf", 4)], REL, k=4) == 0.5


def test_precision_at_k_none_relevant():
    assert precision_at_k([("c.pdf", 2), ("d.pdf", 4)], REL, k=2) == 0.0


def test_precision_at_k_truncates():
    # Only consider top 2 even if more retrieved
    assert precision_at_k([("a.pdf", 1), ("x.pdf", 9), ("b.pdf", 3)], REL, k=2) == 0.5


def test_precision_at_k_zero_k():
    assert precision_at_k([("a.pdf", 1)], REL, k=0) == 0.0


def test_recall_at_k_all_found():
    assert recall_at_k([("a.pdf", 1), ("b.pdf", 3)], REL, k=4) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k([("a.pdf", 1), ("c.pdf", 2)], REL, k=4) == 0.5


def test_recall_at_k_none_found():
    assert recall_at_k([("c.pdf", 2)], REL, k=4) == 0.0


def test_recall_at_k_empty_relevant():
    assert recall_at_k([("a.pdf", 1)], [], k=4) == 0.0


# ============================================================
# mrr
# ============================================================

def test_mrr_aggregates():
    # rank1 (1.0) + rank2 (0.5) -> mean 0.75
    items = [
        ([("a.pdf", 1)], REL),          # rr=1.0
        ([("c.pdf", 2), ("b.pdf", 3)], REL),  # rr=0.5
    ]
    assert mrr(items) == 0.75


def test_mrr_empty():
    assert mrr([]) == 0.0


def test_mrr_with_miss():
    items = [([("c.pdf", 2)], REL)]  # miss -> rr=0
    assert mrr(items) == 0.0


# ============================================================
# citation_coverage
# ============================================================

def test_citation_coverage_filename_match():
    answer = "According to mynotes.pdf, page 4, the answer is yes."
    sources = [{"source": "mynotes.pdf", "page": 4}]
    assert citation_coverage(answer, sources) == 1.0


def test_citation_coverage_page_only_match():
    answer = "See page 12 for details."
    sources = [{"source": "obscure.pdf", "page": 12}]
    assert citation_coverage(answer, sources) == 1.0


def test_citation_coverage_partial():
    answer = "Source A says page 3."
    sources = [
        {"source": "a.pdf", "page": 3},   # matched by page
        {"source": "z.pdf", "page": 99},  # no match
    ]
    assert citation_coverage(answer, sources) == 0.5


def test_citation_coverage_no_sources():
    assert citation_coverage("some text", []) == 0.0


def test_citation_coverage_empty_answer():
    assert citation_coverage("", [{"source": "a.pdf", "page": 1}]) == 0.0


def test_citation_coverage_accepts_dataclass():
    answer = "mynotes says it clearly."
    sources = [RetrievedSource("mynotes.pdf", 4, "x")]
    assert citation_coverage(answer, sources) == 1.0


def test_citation_coverage_short_stem_not_matched():
    """Very short filename stems (<4 chars) shouldn't false-match common words."""
    answer = "The answer is ab."  # 'ab' would match 'ab.pdf' stem naively
    sources = [{"source": "ab.pdf", "page": 99}]
    assert citation_coverage(answer, sources) == 0.0
