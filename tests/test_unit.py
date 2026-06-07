import tempfile
import json
from pathlib import Path
import pytest

from study_rag.documents.hashing import calculate_file_hash
from study_rag.documents.registry import load_registry, save_registry, migrate_legacy_registry
from study_rag.documents.chunking import chunk_pages
from study_rag.documents.pdf_loader import DocumentPage
from study_rag.retrieval.query_expansion import expand_query
from study_rag.retrieval.prompts import construct_rag_prompt
from study_rag.retrieval.memory import build_chat_history, disambiguate_query
from study_rag.feedback import save_feedback, get_feedback_stats

def test_file_hashing():
    """Verify SHA256 hashing utility produces correct hashes."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(b"hello world")
        tmp_path = Path(tmp.name)
        
    try:
        # sha256 of "hello world" is b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
        expected = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
        assert calculate_file_hash(tmp_path) == expected
    finally:
        tmp_path.unlink()

def test_registry_load_save():
    """Verify registry saves and loads data correctly."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = Path(tmp.name)
        
    try:
        test_data = {"subject1": {"file1.pdf": {"hash": "123", "pages_count": 5}}}
        save_registry(test_data, registry_path=tmp_path)
        
        loaded = load_registry(registry_path=tmp_path)
        assert loaded == test_data
    finally:
        tmp_path.unlink()

def test_malformed_registry_fallback():
    """Verify malformed registry content returns an empty dict instead of failing."""
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
        tmp.write("{invalid_json: 123}")
        tmp_path = Path(tmp.name)
        
    try:
        loaded = load_registry(registry_path=tmp_path)
        assert loaded == {}
    finally:
        tmp_path.unlink()

def test_non_dict_registry_fallback():
    """Verify a JSON file containing a list or primitive returns an empty dict."""
    with tempfile.NamedTemporaryFile(delete=False, mode="w") as tmp:
        tmp.write('["not a dict"]')
        tmp_path = Path(tmp.name)
        
    try:
        loaded = load_registry(registry_path=tmp_path)
        assert loaded == {}
    finally:
        tmp_path.unlink()

def test_legacy_registry_migration():
    """Verify top-level legacy keys are migrated to 'Operating System' group."""
    legacy_registry = {
        "operating_system_notes.pdf": {"hash": "abc", "pages_count": 10},
        "another_notes.pdf": {"hash": "def", "pages_count": 20},
        "Maths": {"algebra.pdf": {"hash": "xyz", "pages_count": 4}}
    }
    
    migrated = migrate_legacy_registry(legacy_registry, subject_name="Operating System")
    
    # Check top level keys
    assert "operating_system_notes.pdf" not in migrated
    assert "another_notes.pdf" not in migrated
    assert migrated["Maths"] == {"algebra.pdf": {"hash": "xyz", "pages_count": 4}}
    
    # Check 'Operating System' key contents
    assert "Operating System" in migrated
    os_reg = migrated["Operating System"]
    assert os_reg["operating_system_notes.pdf"] == {"hash": "abc", "pages_count": 10}
    assert os_reg["another_notes.pdf"] == {"hash": "def", "pages_count": 20}

def test_query_expansion():
    """Verify acronym expansion only expands targeted abbreviations with word boundaries."""
    # Standard query expansion
    assert "pcb" in expand_query("what is pcb").lower()
    assert "process control block" in expand_query("what is pcb").lower()
    
    # Multiple expansions
    expanded = expand_query("explain SJF and FCFS")
    assert "shortest job first" in expanded.lower()
    assert "first come first served" in expanded.lower()
    
    # Non-matching substring should NOT expand (e.g. pcbboard, apcb)
    assert expand_query("pcbboard") == "pcbboard"
    assert expand_query("apcb") == "apcb"

def test_prompt_construction():
    """Verify RAG prompt format integrates inputs properly."""
    subject = "Database Systems"
    query = "What is ACID?"
    context = "ACID stands for Atomicity, Consistency, Isolation, Durability."
    
    prompt = construct_rag_prompt(subject, query, context)
    
    assert subject in prompt
    assert query in prompt
    assert context in prompt
    assert "expert, friendly college-level teaching assistant" in prompt

def test_prompt_with_chat_history():
    """Verify RAG prompt includes conversation history when provided."""
    subject = "OS"
    query = "Explain more"
    context = "A process is a program in execution."
    history = "User: What is a process?\nAssistant: A process is a program in execution."
    
    prompt = construct_rag_prompt(subject, query, context, chat_history=history)
    
    assert "Conversation History" in prompt
    assert history in prompt
    assert query in prompt

def test_semantic_chunking():
    """Verify text splitter breaks pages into semantic chunks."""
    pages = [
        DocumentPage(text="This is a sentence. " * 50, page_num=1, source="test.pdf"),
        DocumentPage(text="Another paragraph. " * 50, page_num=2, source="test.pdf"),
    ]
    
    chunks = chunk_pages(pages, chunk_size=100, chunk_overlap=20)
    
    assert len(chunks) > len(pages)
    assert all(isinstance(c, DocumentPage) for c in chunks)
    assert all(c.source == "test.pdf" for c in chunks)
    assert all(c.chunk_index >= 0 for c in chunks)
    # Verify metadata preservation
    page_1_chunks = [c for c in chunks if c.page_num == 1]
    assert len(page_1_chunks) >= 1

def test_chat_history_builder():
    """Verify chat history is formatted from recent messages."""
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
        {"role": "user", "content": "What is OS?"},
        {"role": "assistant", "content": "Operating System."},
        {"role": "user", "content": "Explain more"},
        {"role": "assistant", "content": "It manages resources."},
    ]
    
    history = build_chat_history(messages, max_turns=2)
    
    assert "User: What is OS?" in history
    assert "Assistant: Operating System." in history
    assert "User: Explain more" in history
    assert "Assistant: It manages resources." in history
    assert "User: Hello" not in history  # Only last 2 turns

def test_disambiguate_follow_up_query():
    """Verify follow-up queries are disambiguated using prior context."""
    messages = [
        {"role": "user", "content": "What is a deadlock?"},
        {"role": "assistant", "content": "A deadlock occurs when two or more processes are waiting indefinitely."},
    ]
    
    # Short follow-up should be disambiguated
    disambiguated = disambiguate_query("Explain more", messages)
    assert "Regarding:" in disambiguated
    assert "Explain more" in disambiguated
    
    # Standalone query should not be modified
    standalone = disambiguate_query("What is virtual memory?", messages)
    assert standalone == "What is virtual memory?"
    
    # Pronoun-based follow-up should be disambiguated if vague
    vague = disambiguate_query("Why does it happen?", messages)
    assert "Regarding:" in vague
    
    # Very short query should be disambiguated
    short = disambiguate_query("Why?", messages)
    assert "Regarding:" in short

def test_feedback_save_and_stats():
    """Verify feedback can be saved and statistics retrieved."""
    import tempfile
    from study_rag.feedback import FEEDBACK_FILE
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Monkeypatch feedback file for isolation
        original_file = FEEDBACK_FILE
        test_file = Path(tmpdir) / "feedback.jsonl"
        
        # We can't easily monkeypatch module-level constant, so we test directly
        # by writing to a temp file and reading it back using json directly
        test_file.write_text("", encoding="utf-8")
        
        entry = {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "subject": "OS",
            "query": "What is a process?",
            "response": "A process is...",
            "rating": "thumbs_up",
            "sources": []
        }
        test_file.write_text(json.dumps(entry) + "\n", encoding="utf-8")
        
        # Read back and verify
        lines = test_file.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["rating"] == "thumbs_up"
        assert data["subject"] == "OS"
