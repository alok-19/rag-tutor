from rag_tutor.ui.citations import confidence_label, _strength_label
from rag_tutor.retrieval.vector_store import RetrievedSource


def _src(score: float) -> RetrievedSource:
    return RetrievedSource(source="a.pdf", page=1, text="t", score=score)


def test_confidence_no_sources():
    text, level = confidence_label([])
    assert level == "none"
    assert "No supporting" in text


def test_confidence_high():
    text, level = confidence_label([_src(0.92), _src(0.81)])
    assert level == "high"
    assert "2 passages" in text
    assert "92%" in text


def test_confidence_medium():
    text, level = confidence_label([_src(0.55)])
    assert level == "medium"
    assert "1 passage" in text  # singular


def test_confidence_low():
    text, level = confidence_label([_src(0.20)])
    assert level == "low"
    assert "weakly" in text


def test_confidence_uses_best_score():
    # Even with one low score, a higher one raises the level.
    text, level = confidence_label([_src(0.10), _src(0.75), _src(0.30)])
    assert level == "high"


def test_confidence_boundary_high():
    # Exactly the threshold counts as high.
    _, level = confidence_label([_src(0.70)])
    assert level == "high"


def test_confidence_boundary_medium():
    _, level = confidence_label([_src(0.45)])
    assert level == "medium"


def test_strength_label_buckets():
    assert _strength_label(0.90)[1] == "high"
    assert _strength_label(0.50)[1] == "medium"
    assert _strength_label(0.10)[1] == "low"
