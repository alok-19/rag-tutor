"""Tests for the optional local-first RAG stack: reranker, BGE-M3, LlamaIndex chunking.

All optional dependencies are imported lazily and mocked here so the
suite runs without the ``[hybrid]`` extras installed. The tests
exercise the wiring, fallback logic, and integration points.
"""
import importlib
import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rag_tutor.documents.pdf_loader import DocumentPage
from rag_tutor.retrieval.vector_store import RetrievedSource


# Modules we sometimes ``importlib.reload`` to pick up new env values.
# Re-reloading them in teardown restores the baseline state for any
# subsequent test that did not opt in to a reload.
_RELOADED_MODULES = (
    "rag_tutor.config",
    "rag_tutor.retrieval.rag_service",
    "rag_tutor.retrieval.memory",
    "rag_tutor.retrieval.reranker",
    "rag_tutor.ingestion.service",
    "rag_tutor.llm.providers.factory",
    "rag_tutor.llm.providers.bge",
)


@pytest.fixture(autouse=True)
def _reset_module_state():
    """After each test, re-reload modules that may have been reloaded
    during the test so that the next test (e.g. integration) sees a
    clean baseline regardless of what env vars the previous test set.
    """
    yield
    for mod_name in _RELOADED_MODULES:
        mod = sys.modules.get(mod_name)
        if mod is not None:
            try:
                importlib.reload(mod)
            except Exception:
                # If a reload fails (e.g. a fake was uninstalled), skip it.
                pass


# ============================================================
# 1. bge-reranker
# ============================================================

def _install_fake_sentence_transformers(monkeypatch, predict_return=None):
    """Install a fake ``sentence_transformers`` module on sys.modules.

    Lets us exercise the reranker code path without the real model.
    Returns the fake module so tests can introspect / override behavior.
    """
    fake = types.ModuleType("sentence_transformers")
    captured = {"name": None, "calls": 0}

    class FakeCrossEncoder:
        def __init__(self, name):
            captured["name"] = name

        def predict(self, pairs):
            captured["calls"] += 1
            if predict_return is not None:
                return list(predict_return)
            # Default: descending scores in input order so we can test ordering.
            return [1.0 - i * 0.1 for i in range(len(pairs))]

    fake.CrossEncoder = FakeCrossEncoder  # type: ignore[attr-defined]
    fake._captured = captured  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    return fake


def test_reranker_unavailable_without_dependencies(monkeypatch):
    """If sentence-transformers is missing, is_available() returns False."""
    # Ensure the module is "absent" for the duration of this test.
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    # The reranker import is a function call, not a real import; the
    # underlying function will see ``None`` and report unavailable.
    from rag_tutor.retrieval import reranker as r_mod

    # Force a fresh re-evaluation by clearing the cached class.
    monkeypatch.setattr(r_mod, "_try_import_cross_encoder", lambda: None)
    assert r_mod.BgeReranker.is_available() is False


def test_reranker_raises_clear_error_when_loading_without_deps(monkeypatch):
    """BgeReranker.rerank() should raise a helpful error if deps missing."""
    from rag_tutor.retrieval import reranker as r_mod
    from rag_tutor.retrieval.reranker import RerankerUnavailableError

    monkeypatch.setattr(r_mod, "_try_import_cross_encoder", lambda: None)
    rr = r_mod.BgeReranker()
    docs = [RetrievedSource(source="a.pdf", page=1, text="hello")]
    with pytest.raises(RerankerUnavailableError) as exc:
        rr.rerank("q", docs)
    assert "pip install -e" in str(exc.value)


def test_reranker_reranks_by_predicted_score(monkeypatch):
    """Reranker should sort candidates by the cross-encoder score."""
    fake = _install_fake_sentence_transformers(
        monkeypatch, predict_return=[0.1, 0.9, 0.4]
    )
    from rag_tutor.retrieval.reranker import BgeReranker

    docs = [
        RetrievedSource(source="a.pdf", page=1, text="first"),
        RetrievedSource(source="b.pdf", page=2, text="second"),
        RetrievedSource(source="c.pdf", page=3, text="third"),
    ]
    rr = BgeReranker()
    assert rr.is_available() is True

    ranked = rr.rerank("q", docs, top_k=3)
    # Highest score (0.9) was index 1 -> "second"; expect it first.
    assert [d.text for d in ranked] == ["second", "third", "first"]

    # Model was loaded lazily on first rerank call.
    assert fake._captured["name"] == "BAAI/bge-reranker-base"


def test_reranker_top_k_truncates(monkeypatch):
    """top_k limits the returned list size."""
    _install_fake_sentence_transformers(monkeypatch, predict_return=[0.5, 0.9, 0.1])
    from rag_tutor.retrieval.reranker import BgeReranker

    docs = [
        RetrievedSource(source="a.pdf", page=1, text="a"),
        RetrievedSource(source="b.pdf", page=1, text="b"),
        RetrievedSource(source="c.pdf", page=1, text="c"),
    ]
    rr = BgeReranker()
    out = rr.rerank("q", docs, top_k=2)
    assert len(out) == 2


def test_reranker_handles_empty_input(monkeypatch):
    """Empty input -> empty output, no model load attempted."""
    _install_fake_sentence_transformers(monkeypatch)
    from rag_tutor.retrieval.reranker import BgeReranker

    rr = BgeReranker()
    assert rr.rerank("q", []) == []


def test_rerank_enabled_flag():
    """rerank_enabled() honors ENABLE_RERANKER env var."""
    from rag_tutor.retrieval import reranker as r_mod

    with patch.dict(os.environ, {"ENABLE_RERANKER": "true"}, clear=False):
        assert r_mod.rerank_enabled() is True
    with patch.dict(os.environ, {"ENABLE_RERANKER": "false"}, clear=False):
        assert r_mod.rerank_enabled() is False
    # When unset, default is off.
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("ENABLE_RERANKER", None)
        assert r_mod.rerank_enabled() is False


def test_rag_service_uses_reranker_when_enabled(monkeypatch, tmp_path):
    """End-to-end: retrieve_context with reranker fetches initial_k then reranks."""
    _install_fake_sentence_transformers(
        monkeypatch, predict_return=[0.1, 0.9, 0.2, 0.8, 0.3]
    )

    # Build a minimal fake chroma collection.
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [[f"doc {i}" for i in range(5)]],
        "metadatas": [
            [{"source": "s.pdf", "page": i + 1} for i in range(5)]
        ],
    }

    monkeypatch.setattr(
        "rag_tutor.retrieval.vector_store.get_collection",
        lambda db_path, collection_name: fake_collection,
    )
    monkeypatch.setattr(
        "rag_tutor.llm.embeddings.get_embedding",
        lambda text, api_key=None, provider_name=None: [0.0] * 4,
    )
    monkeypatch.setenv("ENABLE_RERANKER", "true")
    monkeypatch.setenv("RERANKER_INITIAL_K", "5")

    # Reload config to pick up env changes.
    import importlib
    from rag_tutor import config as cfg_mod
    importlib.reload(cfg_mod)
    from rag_tutor.retrieval import rag_service as svc
    importlib.reload(svc)

    results = svc.retrieve_context(
        query="what is a process?",
        subject="OS",
        n_results=2,
        db_path=tmp_path,
        collection_name="c",
    )

    # We asked for n_results=2, so we should get exactly 2.
    assert len(results) == 2
    # And the reranker should have run on all 5 candidates.
    assert fake_collection.query.call_args.kwargs["n_results"] == 5


def test_rag_service_reranker_silent_fallback_when_deps_missing(monkeypatch, tmp_path):
    """If deps are absent, retrieval works normally without reranking."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(
        "rag_tutor.retrieval.reranker._try_import_cross_encoder", lambda: None
    )
    fake_collection = MagicMock()
    fake_collection.query.return_value = {
        "documents": [["a", "b"]],
        "metadatas": [[{"source": "s.pdf", "page": 1}, {"source": "s.pdf", "page": 2}]],
    }
    monkeypatch.setattr(
        "rag_tutor.retrieval.vector_store.get_collection",
        lambda db_path, collection_name: fake_collection,
    )
    monkeypatch.setattr(
        "rag_tutor.llm.embeddings.get_embedding",
        lambda text, api_key=None, provider_name=None: [0.0] * 4,
    )
    monkeypatch.setenv("ENABLE_RERANKER", "true")

    import importlib
    from rag_tutor import config as cfg_mod
    importlib.reload(cfg_mod)
    from rag_tutor.retrieval import rag_service as svc
    importlib.reload(svc)

    results = svc.retrieve_context(
        query="q", subject="OS", n_results=2, db_path=tmp_path, collection_name="c"
    )
    # Should fall back to plain top-2 from the vector store.
    assert len(results) == 2
    assert [r.text for r in results] == ["a", "b"]


# ============================================================
# 2. BGE-M3 provider
# ============================================================

def _install_fake_sentence_transformers_st(monkeypatch, dim: int = 4):
    fake = types.ModuleType("sentence_transformers")
    captured = {"name": None, "device": None, "normalize": None, "batches": []}

    class FakeSentenceTransformer:
        def __init__(self, name, device=None):
            captured["name"] = name
            captured["device"] = device

        def encode(self, texts, batch_size=32, normalize_embeddings=False,
                   convert_to_numpy=False, show_progress_bar=False):
            captured["normalize"] = normalize_embeddings
            captured["batches"].append(list(texts))
            # Return a simple vector per text, length = `dim`, with
            # distinct values so normalization is observable.
            import numpy as np
            arr = np.arange(len(list(texts)) * dim, dtype=float).reshape(len(list(texts)), dim)
            return arr

    fake.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    fake._captured = captured  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    return fake


def test_bge_provider_not_registered_when_deps_missing(monkeypatch):
    """Without sentence-transformers, 'bge' is not in the factory registry.

    We force the bge module's import to fail by replacing it with a sentinel
    module that raises ImportError on attribute access, then reload the
    factory so its ``try/except`` re-evaluates.
    """
    sentinel = types.ModuleType("rag_tutor.llm.providers.bge")

    def _raise(*a, **kw):
        raise ImportError("simulated missing sentence-transformers")

    sentinel.__getattr__ = _raise  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "rag_tutor.llm.providers.bge", sentinel)

    import importlib
    from rag_tutor.llm.providers import factory
    importlib.reload(factory)
    assert "bge" not in factory.list_providers()

    # And calling get_provider("bge") should fail cleanly.
    with pytest.raises(ValueError) as exc:
        factory.get_provider(name="bge", api_key="anything")
    assert "Unknown LLM provider" in str(exc.value)


def test_bge_provider_is_available_reflects_import(monkeypatch):
    """is_available() returns True iff sentence-transformers is importable."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    import importlib
    import rag_tutor.llm.providers.bge as bge_mod
    importlib.reload(bge_mod)
    monkeypatch.setattr(bge_mod, "_try_import_sentence_transformer", lambda: None)
    assert bge_mod.BgeM3Provider.is_available() is False


def test_bge_provider_embed_returns_correct_dim(monkeypatch):
    """Embed should return L2-normalized vectors of the requested dimension."""
    _install_fake_sentence_transformers_st(monkeypatch, dim=4)
    import importlib
    from rag_tutor.llm.providers import bge as bge_mod
    importlib.reload(bge_mod)

    provider = bge_mod.BgeM3Provider(dim=4)
    vectors = provider.embed(["hello", "world"])

    assert len(vectors) == 2
    for v in vectors:
        assert len(v) == 4
        # Vector should be roughly unit length (cosine-ready).
        import math
        norm = math.sqrt(sum(x * x for x in v))
        assert 0.99 <= norm <= 1.01


def test_bge_provider_matryoshka_truncation(monkeypatch):
    """Vectors are truncated to the requested dim even if model outputs more."""
    # Fake produces 8-dim vectors; we ask for 3.
    _install_fake_sentence_transformers_st(monkeypatch, dim=8)
    import importlib
    from rag_tutor.llm.providers import bge as bge_mod
    importlib.reload(bge_mod)

    provider = bge_mod.BgeM3Provider(dim=3)
    vectors = provider.embed(["a"])
    assert len(vectors[0]) == 3
    import math
    norm = math.sqrt(sum(x * x for x in vectors[0]))
    assert 0.99 <= norm <= 1.01


def test_bge_provider_generate_stream_not_implemented(monkeypatch):
    """BGE is embeddings-only; generation must raise NotImplementedError."""
    _install_fake_sentence_transformers_st(monkeypatch)
    from rag_tutor.llm.providers.bge import BgeM3Provider

    provider = BgeM3Provider()
    with pytest.raises(NotImplementedError):
        list(provider.generate_stream("prompt"))


# ============================================================
# 3. LlamaIndex chunking
# ============================================================

def _install_fake_llama_index(monkeypatch, chunked_texts=None):
    """Install a tiny fake llama_index.core with the symbols we use."""
    if chunked_texts is None:
        chunked_texts = ["chunk1 from p1", "chunk2 from p1", "chunk1 from p2"]

    # Remove any previously installed real (or fake) llama_index modules so
    # our fake actually wins.
    for mod in list(sys.modules):
        if mod == "llama_index" or mod.startswith("llama_index."):
            del sys.modules[mod]

    fake_root = types.ModuleType("llama_index")
    fake_core = types.ModuleType("llama_index.core")
    fake_ingestion = types.ModuleType("llama_index.core.ingestion")
    fake_node_parser = types.ModuleType("llama_index.core.node_parser")
    fake_node_parser_base = types.ModuleType("llama_index.core.node_parser.base")
    fake_schema = types.ModuleType("llama_index.core.schema")

    class FakeDocument:
        def __init__(self, text, doc_id=None, metadata=None):
            self.text = text
            self.doc_id = doc_id
            self.metadata = metadata or {}

    class FakeTextNode:
        def __init__(self, text="", metadata=None):
            self.text = text
            self.metadata = metadata or {}

        def get_content(self):
            return self.text

    class FakeBaseNode:
        pass

    class FakeNodeParser:
        def __init__(self, *args, **kwargs):
            pass

        def _parse_nodes(self, nodes, show_progress=False, **kwargs):
            return list(nodes)

        def _retrieve_nodes(self, nodes, show_progress=False, **kwargs):
            return list(nodes)

    class FakeSentenceSplitter:
        def __init__(self, chunk_size=1024, chunk_overlap=200, **kwargs):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

    class FakeIngestionPipeline:
        def __init__(self, transformations=None, **kwargs):
            self.transformations = transformations or []

        def run(self, documents=None, **kwargs):
            nodes = []
            for d in documents or []:
                for ct in chunked_texts:
                    nodes.append(
                        FakeTextNode(text=ct, metadata=dict(d.metadata))
                    )
            return nodes

    fake_core.Document = FakeDocument  # type: ignore[attr-defined]
    fake_ingestion.IngestionPipeline = FakeIngestionPipeline  # type: ignore[attr-defined]
    fake_node_parser.SentenceSplitter = FakeSentenceSplitter  # type: ignore[attr-defined]
    fake_node_parser_base.NodeParser = FakeNodeParser  # type: ignore[attr-defined]
    fake_schema.BaseNode = FakeBaseNode  # type: ignore[attr-defined]
    fake_schema.TextNode = FakeTextNode  # type: ignore[attr-defined]

    sys.modules["llama_index"] = fake_root
    sys.modules["llama_index.core"] = fake_core
    sys.modules["llama_index.core.ingestion"] = fake_ingestion
    sys.modules["llama_index.core.node_parser"] = fake_node_parser
    sys.modules["llama_index.core.node_parser.base"] = fake_node_parser_base
    sys.modules["llama_index.core.schema"] = fake_schema
    monkeypatch.setitem(sys.modules, "llama_index", fake_root)
    monkeypatch.setitem(sys.modules, "llama_index.core", fake_core)
    monkeypatch.setitem(sys.modules, "llama_index.core.ingestion", fake_ingestion)
    monkeypatch.setitem(sys.modules, "llama_index.core.node_parser", fake_node_parser)
    monkeypatch.setitem(sys.modules, "llama_index.core.node_parser.base", fake_node_parser_base)
    monkeypatch.setitem(sys.modules, "llama_index.core.schema", fake_schema)


def test_llama_index_chunking_requires_package(monkeypatch):
    """If llama-index-core is missing, raise a clear, actionable error."""
    # Make the import fail.
    for mod in list(sys.modules):
        if mod.startswith("llama_index"):
            del sys.modules[mod]
    monkeypatch.setitem(sys.modules, "llama_index", None)
    monkeypatch.setitem(sys.modules, "llama_index.core", None)

    from rag_tutor.documents import chunking as chunk_mod
    pages = [DocumentPage(text="hello", page_num=1, source="a.pdf")]
    with pytest.raises(RuntimeError) as exc:
        chunk_mod.chunk_pages_llama_index(pages)
    assert "pip install -e" in str(exc.value)


def test_llama_index_chunking_returns_document_pages(monkeypatch):
    """When llama-index-core is available, returns our DocumentPage type."""
    _install_fake_llama_index(
        monkeypatch,
        chunked_texts=["alpha text", "beta text"],
    )
    from rag_tutor.documents import chunking as chunk_mod

    pages = [
        DocumentPage(text="source text page 1", page_num=1, source="book.pdf"),
    ]
    chunks = chunk_mod.chunk_pages_llama_index(pages)

    assert all(isinstance(c, DocumentPage) for c in chunks)
    assert all(c.source == "book.pdf" for c in chunks)
    assert len(chunks) >= 1


def test_llama_index_chunking_preserves_page_metadata(monkeypatch):
    """chunks report the correct page number from the source DocumentPage."""
    _install_fake_llama_index(
        monkeypatch, chunked_texts=["only chunk"]
    )
    from rag_tutor.documents import chunking as chunk_mod

    pages = [
        DocumentPage(text="x", page_num=1, source="a.pdf"),
        DocumentPage(text="y", page_num=5, source="a.pdf"),
    ]
    chunks = chunk_mod.chunk_pages_llama_index(pages)
    pages_seen = sorted({c.page_num for c in chunks})
    assert pages_seen == [1, 5]


def test_ingest_pdfs_dispatches_to_llama_index(monkeypatch, tmp_path):
    """When CHUNKING_STRATEGY=llama_index, ingestion uses the LI chunker."""
    _install_fake_llama_index(monkeypatch, chunked_texts=["li chunk"])

    # Set env + reload config and service first so the service picks up
    # CHUNKING_STRATEGY=llama_index. We then patch the service's *local*
    # bindings because ``from rag_tutor.documents import ...`` re-binds at
    # import time and survives in the reloaded service's namespace.
    monkeypatch.setenv("CHUNKING_STRATEGY", "llama_index")
    import importlib
    from rag_tutor import config as cfg_mod
    importlib.reload(cfg_mod)
    from rag_tutor.ingestion import service as ingest_svc
    importlib.reload(ingest_svc)

    monkeypatch.setattr(
        ingest_svc, "extract_pages_from_pdf",
        lambda fp: [DocumentPage(text="x", page_num=1, source=fp.name)],
    )
    monkeypatch.setattr(
        ingest_svc, "get_embeddings_batch",
        lambda texts, api_key=None: [[0.1, 0.2, 0.3] for _ in texts],
    )
    monkeypatch.setattr(
        ingest_svc, "add_documents_to_store",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        ingest_svc, "delete_source_documents",
        lambda **kwargs: None,
    )

    pdf_dir = tmp_path / "subj"
    pdf_dir.mkdir()
    (pdf_dir / "book.pdf").write_text("dummy")

    processed = ingest_svc.ingest_pdfs(
        pdf_dir=pdf_dir,
        subject_name="subj",
        api_key="x",
        db_path=tmp_path,
        collection_name="c",
        registry_file=tmp_path / "reg.json",
    )
    assert processed == 1
