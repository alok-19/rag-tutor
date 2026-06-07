from dataclasses import dataclass
from pathlib import Path
import chromadb
from rag_tutor.config import DB_PATH, COLLECTION_NAME

@dataclass
class RetrievedSource:
    source: str
    page: int
    text: str

def get_chroma_client(db_path: Path = DB_PATH) -> chromadb.PersistentClient:
    """Get the persistent ChromaDB client."""
    return chromadb.PersistentClient(path=str(db_path))

def get_collection(db_path: Path = DB_PATH, collection_name: str = COLLECTION_NAME):
    """Get or create the ChromaDB collection."""
    client = get_chroma_client(db_path=db_path)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

def query_vector_store(
    query_embedding: list[float],
    subject: str,
    n_results: int = 4,
    db_path: Path = DB_PATH,
    collection_name: str = COLLECTION_NAME
) -> list[RetrievedSource]:
    """Query ChromaDB for relevant source chunks for a specific subject."""
    collection = get_collection(db_path=db_path, collection_name=collection_name)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where={"subject": subject}
    )
    
    retrieved_sources = []
    if results and "documents" in results and results["documents"]:
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        for doc, meta in zip(documents, metadatas):
            retrieved_sources.append(RetrievedSource(
                source=meta.get("source", "Unknown"),
                page=meta.get("page", 0),
                text=doc
            ))
            
    return retrieved_sources

def delete_source_documents(
    filename: str,
    subject_name: str,
    db_path: Path = DB_PATH,
    collection_name: str = COLLECTION_NAME
):
    """Delete all indexed chunks for a specific source file within a subject."""
    collection = get_collection(db_path=db_path, collection_name=collection_name)
    try:
        collection.delete(where={
            "$and": [
                {"source": {"$eq": filename}},
                {"subject": {"$eq": subject_name}}
            ]
        })
    except Exception:
        # Fallback to simple query if nested query is not supported
        collection.delete(where={"source": filename})

def add_documents_to_store(
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    ids: list[str],
    db_path: Path = DB_PATH,
    collection_name: str = COLLECTION_NAME
):
    """Insert text chunks and embeddings into ChromaDB."""
    collection = get_collection(db_path=db_path, collection_name=collection_name)
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

def has_subject_documents(
    subject: str,
    db_path: Path = DB_PATH,
    collection_name: str = COLLECTION_NAME
) -> bool:
    """Check if the active subject has any indexed documents in the store."""
    if not db_path.exists():
        return False
    try:
        client = get_chroma_client(db_path=db_path)
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        if collection_name not in collection_names:
            return False
            
        collection = client.get_collection(collection_name)
        count_results = collection.get(where={"subject": subject}, limit=1)
        if count_results and count_results["ids"]:
            return True
    except Exception:
        pass
    return False
