from rag_tutor.eval.runner import run_eval, EvalReport, ItemResult, run_leakage_eval, LeakageReport
from rag_tutor.eval.dataset import EvalItem
from rag_tutor.retrieval.vector_store import RetrievedSource


def _src(source, page, text=""):
    return RetrievedSource(source=source, page=page, text=text)


def _make_retrieve(mapping):
    """Build a deterministic retrieve stub: query -> list[RetrievedSource]."""
    def _retrieve(query, subject, k):
        return mapping.get(query, [])[:k]
    return _retrieve


# ============================================================
# Basic aggregation
# ============================================================

def test_run_eval_perfect_retrieval():
    items = [
        EvalItem(query="q1", relevant_sources=[{"source": "a.pdf", "page": 1}]),
        EvalItem(query="q2", relevant_sources=[{"source": "b.pdf", "page": 2}]),
    ]
    retrieve = _make_retrieve({
        "q1": [_src("a.pdf", 1)],
        "q2": [_src("b.pdf", 2)],
    })
    report = run_eval(items, retrieve, k=4)
    assert report.n_items == 2
    assert report.hit_rate == 1.0
    assert report.mrr == 1.0
    assert report.precision_at_k == 1.0
    assert report.recall_at_k == 1.0
    assert report.n_errors == 0


def test_run_eval_all_misses():
    items = [EvalItem(query="q1", relevant_sources=[{"source": "a.pdf", "page": 1}])]
    retrieve = _make_retrieve({"q1": [_src("z.pdf", 9)]})
    report = run_eval(items, retrieve, k=4)
    assert report.hit_rate == 0.0
    assert report.mrr == 0.0
    assert report.precision_at_k == 0.0
    assert report.recall_at_k == 0.0


def test_run_eval_partial():
    items = [
        EvalItem(query="hit", relevant_sources=[{"source": "a.pdf", "page": 1}]),
        EvalItem(query="miss", relevant_sources=[{"source": "a.pdf", "page": 1}]),
    ]
    retrieve = _make_retrieve({
        "hit": [_src("a.pdf", 1)],
        "miss": [_src("z.pdf", 9)],
    })
    report = run_eval(items, retrieve, k=4)
    assert report.hit_rate == 0.5
    assert report.mrr == 0.5


def test_run_eval_empty_dataset():
    report = run_eval([], _make_retrieve({}), k=4)
    assert report.n_items == 0
    assert report.hit_rate == 0.0


# ============================================================
# Keyword-based relevance fallback
# ============================================================

def test_keyword_relevance_when_no_explicit_sources():
    items = [EvalItem(query="q1", expected_keywords=["deadlock", "circular wait"])]
    retrieve = _make_retrieve({
        "q1": [
            _src("a.pdf", 1, text="A deadlock involves circular wait conditions."),
            _src("b.pdf", 2, text="Totally unrelated content about scheduling."),
        ],
    })
    report = run_eval(items, retrieve, k=4)
    # a.pdf (rank 1) is keyword-relevant -> hit, rr=1, precision=0.25, recall=1
    assert report.hit_rate == 1.0
    assert report.mrr == 1.0
    assert report.recall_at_k == 1.0


def test_explicit_sources_take_precedence_over_keywords():
    items = [EvalItem(
        query="q1",
        relevant_sources=[{"source": "c.pdf", "page": 5}],
        expected_keywords=["should_be_ignored"],
    )]
    retrieve = _make_retrieve({"q1": [_src("a.pdf", 1, text="should_be_ignored match")]})
    report = run_eval(items, retrieve, k=4)
    # Explicit source c.pdf/5 not retrieved -> miss despite keyword match.
    assert report.hit_rate == 0.0


# ============================================================
# Resilience
# ============================================================

def test_run_eval_records_error_without_crashing():
    items = [
        EvalItem(query="ok", relevant_sources=[{"source": "a.pdf", "page": 1}]),
        EvalItem(query="boom", relevant_sources=[{"source": "a.pdf", "page": 1}]),
    ]

    def retrieve(query, subject, k):
        if query == "boom":
            raise RuntimeError("provider exploded")
        return [_src("a.pdf", 1)]

    report = run_eval(items, retrieve, k=4)
    assert report.n_items == 2
    assert report.n_errors == 1
    # The errored item is recorded with zeros and an error string.
    err_item = [i for i in report.items if i.error][0]
    assert err_item.hit == 0.0
    assert "exploded" in err_item.error
    # The successful item still scores.
    assert report.hit_rate == 0.5


def test_run_eval_retrieval_returns_none_handled():
    items = [EvalItem(query="q1", relevant_sources=[{"source": "a.pdf", "page": 1}])]
    def retrieve(query, subject, k):
        return None
    report = run_eval(items, retrieve, k=4)
    assert report.n_items == 1
    assert report.hit_rate == 0.0
    assert report.n_errors == 0  # None is tolerated, not an error


# ============================================================
# Report shape
# ============================================================

def test_report_to_dict():
    report = EvalReport(k=4, n_items=3, hit_rate=0.6667, mrr=0.5, precision_at_k=0.4, recall_at_k=0.7)
    d = report.to_dict()
    assert d["k"] == 4
    assert d["n_items"] == 3
    assert d["hit_rate"] == 0.6667
    assert "precision_at_k" in d


def test_item_result_defaults():
    r = ItemResult(query="q", hit=0.0, rr=0.0, precision=0.0, recall=0.0, n_retrieved=0)
    assert r.error is None


def test_subject_override_used():
    """A subject override should be passed to retrieve, ignoring item.subject."""
    seen = {}
    def retrieve(query, subject, k):
        seen["subject"] = subject
        return []
    items = [EvalItem(query="q", subject="Original")]
    run_eval(items, retrieve, k=4, subject="Override")
    assert seen["subject"] == "Override"


# ============================================================
# Cross-subject leakage
# ============================================================

def test_item_result_records_subject():
    """ItemResult should carry the subject the query was evaluated against."""
    items = [EvalItem(query="q", subject="DB", relevant_sources=[{"source": "a.pdf", "page": 1}])]
    retrieve = _make_retrieve({"q": [_src("a.pdf", 1)]})
    report = run_eval(items, retrieve, k=4)
    assert report.items[0].subject == "DB"


def test_leakage_no_contamination():
    """All retrieved chunks from the correct subject -> leakage 0."""
    items = [EvalItem(query="q1", subject="A")]
    def cross(query, k):
        return [("a.pdf", 1, "A"), ("a.pdf", 2, "A")]
    report = run_leakage_eval(items, cross, k=4)
    assert report.on_subject_rate == 1.0
    assert report.leakage_rate == 0.0
    assert report.n_items == 1


def test_leakage_full_contamination():
    """All retrieved chunks from a different subject -> leakage 1."""
    items = [EvalItem(query="q1", subject="A")]
    def cross(query, k):
        return [("b.pdf", 1, "B"), ("b.pdf", 2, "B")]
    report = run_leakage_eval(items, cross, k=4)
    assert report.on_subject_rate == 0.0
    assert report.leakage_rate == 1.0


def test_leakage_partial():
    items = [EvalItem(query="q1", subject="A")]
    def cross(query, k):
        return [("a.pdf", 1, "A"), ("b.pdf", 2, "B"), ("b.pdf", 3, "B"), ("a.pdf", 4, "A")]
    report = run_leakage_eval(items, cross, k=4)
    assert report.on_subject_rate == 0.5
    assert report.leakage_rate == 0.5


def test_leakage_averaged_over_items():
    items = [
        EvalItem(query="clean", subject="A"),
        EvalItem(query="dirty", subject="A"),
    ]
    def cross(query, k):
        if query == "clean":
            return [("a.pdf", 1, "A")]
        return [("b.pdf", 1, "B")]
    report = run_leakage_eval(items, cross, k=4)
    # (1.0 + 0.0) / 2
    assert report.on_subject_rate == 0.5


def test_leakage_skips_items_without_subject():
    items = [EvalItem(query="q1")]  # no subject
    def cross(query, k):
        return [("a.pdf", 1, "A")]
    report = run_leakage_eval(items, cross, k=4)
    assert report.n_items == 0
    assert report.on_subject_rate == 0.0


def test_leakage_empty_retrieval_recorded():
    items = [EvalItem(query="q1", subject="A")]
    def cross(query, k):
        return []
    report = run_leakage_eval(items, cross, k=4)
    assert report.on_subject_rate == 0.0
    assert report.items[0]["on_subject_rate"] == 0.0


def test_leakage_error_does_not_crash():
    items = [EvalItem(query="q1", subject="A")]
    def cross(query, k):
        raise RuntimeError("boom")
    report = run_leakage_eval(items, cross, k=4)
    assert report.n_errors == 1
    assert report.n_items == 0
    assert "boom" in report.items[0]["error"]


def test_leakage_report_to_dict():
    report = LeakageReport(n_items=2, leakage_rate=0.25, on_subject_rate=0.75)
    d = report.to_dict()
    assert d["leakage_rate"] == 0.25
    assert d["on_subject_rate"] == 0.75
    assert "items" in d
