import json
from pathlib import Path

from rag_tutor.eval.dataset import load_dataset, EvalItem, DEFAULT_DATASET


def test_default_dataset_loads():
    items = load_dataset(None)
    assert len(items) > 0
    assert all(isinstance(i, EvalItem) for i in items)
    assert all(i.query for i in items)


def test_default_dataset_matches_source():
    assert len(load_dataset(None)) == len(DEFAULT_DATASET)


def test_missing_file_returns_empty(tmp_path):
    assert load_dataset(tmp_path / "nope.json") == []


def test_malformed_json_returns_empty(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    assert load_dataset(p) == []


def test_non_dict_non_list_returns_empty(tmp_path):
    p = tmp_path / "weird.json"
    p.write_text('"just a string"', encoding="utf-8")
    assert load_dataset(p) == []


def test_dict_with_items_wrapper(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps({"items": [
        {"query": "Q1", "expected_keywords": ["a"]},
    ]}), encoding="utf-8")
    items = load_dataset(p)
    assert len(items) == 1
    assert items[0].query == "Q1"


def test_bare_list(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps([
        {"query": "Q1", "relevant_sources": [{"source": "a.pdf", "page": 1}]},
    ]), encoding="utf-8")
    items = load_dataset(p)
    assert len(items) == 1
    assert items[0].relevant_pairs() == [("a.pdf", 1)]


def test_skips_items_without_query(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps([
        {"query": "", "expected_keywords": ["a"]},
        {"query": "   "},
        {"question": "Q via alt key"},
        {"no_query": "x"},
    ]), encoding="utf-8")
    items = load_dataset(p)
    assert len(items) == 1
    assert items[0].query == "Q via alt key"


def test_alt_keys_recognized(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps([{
        "question": "Q",
        "relevant": [{"source": "b.pdf", "page": 2}],
        "keywords": ["k1"],
        "subject": "DB",
    }]), encoding="utf-8")
    items = load_dataset(p)
    assert items[0].relevant_sources[0]["page"] == 2
    assert items[0].expected_keywords == ["k1"]
    assert items[0].subject == "DB"


def test_relevant_pairs_filters_invalid():
    item = EvalItem(query="Q", relevant_sources=[
        {"source": "a.pdf", "page": 1},
        {"source": "b.pdf"},           # no page -> filtered
        {"page": 5},                   # no source -> filtered
        "not a dict",                  # ignored
    ])
    assert item.relevant_pairs() == [("a.pdf", 1)]


def test_non_list_relevant_treated_empty(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps([{"query": "Q", "relevant_sources": "bad"}]), encoding="utf-8")
    items = load_dataset(p)
    assert items[0].relevant_pairs() == []


def test_corrupt_relevant_dict_entries_skipped(tmp_path):
    p = tmp_path / "ds.json"
    p.write_text(json.dumps([{
        "query": "Q",
        "relevant_sources": [{"source": "a.pdf", "page": 1}, "bad", 123],
    }]), encoding="utf-8")
    items = load_dataset(p)
    assert items[0].relevant_pairs() == [("a.pdf", 1)]
