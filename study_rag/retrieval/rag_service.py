from pathlib import Path
from study_rag.config import DB_PATH, COLLECTION_NAME
from study_rag.llm.embeddings import get_embedding
from study_rag.retrieval.query_expansion import expand_query
from study_rag.retrieval.vector_store import query_vector_store, RetrievedSource

def retrieve_context(
    query: str,
    subject: str,
    api_key: str = None,
    n_results: int = 4,
    db_path: Path = DB_PATH,
    collection_name: str = COLLECTION_NAME
) -> list[RetrievedSource]:
    """Perform full retrieval pipeline:
    1. Expand query terms (acronyms)
    2. Embed expanded query
    3. Query vector store with subject filter
    """
    expanded = expand_query(query)
    query_embedding = get_embedding(expanded, api_key=api_key)
    return query_vector_store(
        query_embedding=query_embedding,
        subject=subject,
        n_results=n_results,
        db_path=db_path,
        collection_name=collection_name
    )
