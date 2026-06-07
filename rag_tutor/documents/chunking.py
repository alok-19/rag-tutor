"""Document chunking strategies.

Two strategies are supported:

* ``recursive`` (default): the original ``RecursiveCharacterTextSplitter``
  from ``langchain-text-splitters``. Zero additional dependencies, fast, and
  predictable. This is what every existing user has been running.

* ``llama_index``: LlamaIndex's ``SentenceSplitter`` wrapped inside an
  ``IngestionPipeline`` that also performs deterministic chunk-level
  deduplication via a small hash-based filter transform. Used when
  ``CHUNKING_STRATEGY=llama_index`` is set. Requires the ``[hybrid]``
  extras.

Both strategies return the same ``DocumentPage`` objects so the rest of the
pipeline (embedding, vector store, retrieval, UI) is identical.
"""
from rag_tutor.documents.pdf_loader import DocumentPage


def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[DocumentPage]:
    """Split page-level text into overlapping chunks while preserving metadata.

    Args:
        pages: List of DocumentPage objects with raw page text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of characters of overlap between consecutive chunks.

    Returns:
        List of DocumentPage objects where each represents a chunk.
        chunk_index is incremented per page.
    """
    # Imported lazily here — importing langchain_text_splitters at module top
    # level eagerly pulls in ``transformers`` and ``sentence_transformers``
    # (via langchain_text_splitters/__init__.py), which transitively triggers
    # torchvision image-processing imports. None of that is needed for chunking.
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages:
        page_chunks = text_splitter.split_text(page.text)
        for idx, chunk_text in enumerate(page_chunks):
            chunks.append(DocumentPage(
                text=chunk_text,
                page_num=page.page_num,
                source=page.source,
                chunk_index=idx
            ))

    return chunks


def _require_hybrid_extras():
    """Raise a clear error if the [hybrid] extras are not installed."""
    try:
        import llama_index.core  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "CHUNKING_STRATEGY=llama_index requires the 'llama-index-core' "
            "package. Install it with: pip install -e \".[hybrid]\" "
            "(or: pip install llama-index-core). "
            f"Underlying import error: {e}"
        )


def _make_hash_dedup_transform():
    """Build a small LlamaIndex node transform that drops duplicate chunks.

    LlamaIndex does not ship a built-in 'HashedNodeParser' in 0.14.x, so we
    implement the equivalent behavior with a thin custom transform. It
    hashes each node's text and returns only the first occurrence.
    """
    from llama_index.core.schema import BaseNode, TextNode
    from llama_index.core.node_parser.base import NodeParser

    class _HashDedupNodeParser(NodeParser):
        """Drop TextNode instances whose content has been seen before."""

        def _parse_nodes(self, nodes, show_progress=False, **kwargs):
            seen: set[str] = set()
            kept: list[BaseNode] = []
            for node in nodes:
                if not isinstance(node, TextNode):
                    kept.append(node)
                    continue
                digest = hash(node.get_content())
                if digest in seen:
                    continue
                seen.add(digest)
                kept.append(node)
            return kept

        def _retrieve_nodes(self, nodes, show_progress=False, **kwargs):
            return self._parse_nodes(nodes, show_progress=show_progress, **kwargs)

    return _HashDedupNodeParser()


def chunk_pages_llama_index(
    pages: list[DocumentPage],
    chunk_size: int = 1024,
    chunk_overlap: int = 200,
    dedup: bool = True,
) -> list[DocumentPage]:
    """Chunk pages using LlamaIndex's ``SentenceSplitter`` with optional dedup.

    The ``IngestionPipeline`` orchestrates:
        1. ``SentenceSplitter`` — splits each page's text into sentences and
           greedily packs them up to ``chunk_size`` tokens.
        2. ``_HashDedupNodeParser`` (if ``dedup=True``) — drops chunks whose
           text hash collides with an earlier one, providing deterministic
           exact-match deduplication. This is the LlamaIndex-idiomatic
           equivalent of a "HashedNodeParser" for 0.14.x.

    Output ``DocumentPage.text`` may differ from the recursive splitter
    (LlamaIndex's tokenizer vs. ``len()``), but the downstream storage
    format is unchanged.
    """
    _require_hybrid_extras()

    from llama_index.core import Document
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core.node_parser import SentenceSplitter

    docs = []
    for page in pages:
        doc_id = f"{page.source}::p{page.page_num}"
        docs.append(
            Document(
                text=page.text,
                doc_id=doc_id,
                metadata={
                    "source": page.source,
                    "page": page.page_num,
                },
            )
        )

    transforms = [SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)]
    if dedup:
        transforms.append(_make_hash_dedup_transform())

    pipeline = IngestionPipeline(transforms=transforms)
    nodes = pipeline.run(documents=docs)

    # Convert LlamaIndex TextNode objects back into our internal format.
    # chunk_index is assigned per (source, page) in emission order so
    # downstream filters by chunk_index keep working.
    chunks: list[DocumentPage] = []
    per_page_index: dict[tuple[str, int], int] = {}
    for node in nodes:
        meta = node.metadata or {}
        source = meta.get("source", "unknown")
        try:
            page_num = int(meta.get("page", 0))
        except (TypeError, ValueError):
            page_num = 0
        key = (source, page_num)
        idx = per_page_index.get(key, 0)
        chunks.append(
            DocumentPage(
                text=node.get_content(),
                page_num=page_num,
                source=source,
                chunk_index=idx,
            )
        )
        per_page_index[key] = idx + 1

    return chunks
