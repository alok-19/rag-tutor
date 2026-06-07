from pathlib import Path
from rag_tutor.config import (
    DB_PATH,
    COLLECTION_NAME,
    ENABLE_RERANKER,
    RERANKER_INITIAL_K,
)
from rag_tutor.llm.embeddings import get_embedding
from rag_tutor.retrieval.query_expansion import expand_query
from rag_tutor.retrieval.vector_store import query_vector_store, RetrievedSource


def _maybe_rerank(
    query: str,
    sources: list[RetrievedSource],
    use_reranker: bool | None,
    top_k: int,
) -> list[RetrievedSource]:
    """Optionally rerank ``sources`` using the BGE cross-encoder.

    Args:
        query: The original user query (post-expansion is fine).
        sources: The vector-store candidates.
        use_reranker: Explicit override. If ``None``, falls back to the
            ``ENABLE_RERANKER`` env var. If the reranker dependencies are
            not installed, we silently fall back to the original order so
            retrieval still works.
        top_k: Final number of sources to return.
    """
    if not sources:
        return sources
    if not (use_reranker if use_reranker is not None else ENABLE_RERANKER):
        return sources[:top_k]

    try:
        from rag_tutor.retrieval.reranker import BgeReranker, RerankerUnavailableError
    except Exception:
        return sources[:top_k]

    if not BgeReranker.is_available():
        return sources[:top_k]

    try:
        reranker = BgeReranker()
        return reranker.rerank(query, sources, top_k=top_k)
    except RerankerUnavailableError:
        return sources[:top_k]
    except Exception:
        # Never let a reranker failure break retrieval.
        return sources[:top_k]


def retrieve_context(
    query: str,
    subject: str,
    api_key: str = None,
    n_results: int = 4,
    db_path: Path = DB_PATH,
    collection_name: str = COLLECTION_NAME,
    use_reranker: bool | None = None,
) -> list[RetrievedSource]:
    """Perform full retrieval pipeline:
    1. Expand query terms (acronyms)
    2. Embed expanded query
    3. Query vector store with subject filter
    4. (Optional) Rerank candidates with BGE cross-encoder
    """
    expanded = expand_query(query)

    # When reranking is on, we over-fetch to give the cross-encoder room
    # to find the truly best passages. Cap at 50 for sanity.
    initial_k = max(n_results, RERANKER_INITIAL_K) if (use_reranker if use_reranker is not None else ENABLE_RERANKER) else n_results
    initial_k = min(initial_k, 50)

    query_embedding = get_embedding(expanded, api_key=api_key)
    candidates = query_vector_store(
        query_embedding=query_embedding,
        subject=subject,
        n_results=initial_k,
        db_path=db_path,
        collection_name=collection_name,
    )

    return _maybe_rerank(expanded, candidates, use_reranker, top_k=n_results)
