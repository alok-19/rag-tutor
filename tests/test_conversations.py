import json
from pathlib import Path

from rag_tutor.conversations import (
    load_conversations,
    save_conversation,
    clear_conversation,
    list_saved_subjects,
    _subject_path,
)


def test_load_missing_subject_returns_empty(tmp_path):
    assert load_conversations("Ghost", base_dir=tmp_path) == []


def test_save_and_load_roundtrip(tmp_path):
    msgs = [
        {"role": "user", "content": "What is paging?"},
        {"role": "assistant", "content": "Paging is...", "sources": []},
    ]
    save_conversation("Operating System", msgs, base_dir=tmp_path)

    loaded = load_conversations("Operating System", base_dir=tmp_path)
    assert loaded == msgs
    assert "Operating System" in list_saved_subjects(base_dir=tmp_path)


def test_save_preserves_message_metadata(tmp_path):
    """Sources, feedback, and query fields survive the roundtrip."""
    msgs = [{
        "role": "assistant",
        "content": "Answer",
        "sources": [{"source": "a.pdf", "page": 3, "text": "snippet"}],
        "user_query": "explain X",
        "feedback": "thumbs_up",
    }]
    save_conversation("DB", msgs, base_dir=tmp_path)
    loaded = load_conversations("DB", base_dir=tmp_path)
    assert loaded[0]["sources"][0]["page"] == 3
    assert loaded[0]["feedback"] == "thumbs_up"


def test_corrupt_file_returns_empty(tmp_path):
    path = _subject_path("Broken", base_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not valid json", encoding="utf-8")
    assert load_conversations("Broken", base_dir=tmp_path) == []


def test_non_list_content_returns_empty(tmp_path):
    """A JSON file holding a dict or primitive must not crash the loader."""
    path = _subject_path("Weird", base_dir=tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"not": "a list"}', encoding="utf-8")
    assert load_conversations("Weird", base_dir=tmp_path) == []


def test_clear_removes_file(tmp_path):
    save_conversation("Gone", [{"role": "user", "content": "hi"}], base_dir=tmp_path)
    assert _subject_path("Gone", base_dir=tmp_path).exists()
    clear_conversation("Gone", base_dir=tmp_path)
    assert not _subject_path("Gone", base_dir=tmp_path).exists()
    # Clearing a non-existent subject is a no-op (no error).
    clear_conversation("Never Existed", base_dir=tmp_path)


def test_subject_path_is_safe(tmp_path):
    """Tricky subject names cannot escape the conversations directory."""
    p = _subject_path("../escape", base_dir=tmp_path)
    assert tmp_path in p.parents
    assert ".." not in p.name

    p2 = _subject_path("a/b\\c:d*e", base_dir=tmp_path)
    assert tmp_path in p2.parents
    assert p2.parent == tmp_path


def test_save_creates_base_dir(tmp_path):
    nested = tmp_path / "deep" / "nested"
    save_conversation("OS", [{"role": "user", "content": "hi"}], base_dir=nested)
    assert nested.exists()
    assert load_conversations("OS", base_dir=nested) == [{"role": "user", "content": "hi"}]


def test_overwrite_replaces_history(tmp_path):
    save_conversation("S", [{"role": "user", "content": "old"}], base_dir=tmp_path)
    save_conversation("S", [{"role": "user", "content": "new"}], base_dir=tmp_path)
    loaded = load_conversations("S", base_dir=tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["content"] == "new"


def test_non_serializable_fallback(tmp_path):
    """Objects without a JSON representation should not crash save."""
    class Weird:
        pass

    msgs = [{"role": "user", "content": Weird()}]
    save_conversation("NS", msgs, base_dir=tmp_path)
    loaded = load_conversations("NS", base_dir=tmp_path)
    assert isinstance(loaded[0]["content"], str)


def test_list_saved_subjects_format(tmp_path):
    save_conversation("Operating System", [], base_dir=tmp_path)
    save_conversation("Maths", [], base_dir=tmp_path)
    names = list_saved_subjects(base_dir=tmp_path)
    assert "Operating System" in names
    assert "Maths" in names
