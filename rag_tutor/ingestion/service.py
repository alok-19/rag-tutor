from pathlib import Path
from rag_tutor.config import DB_PATH, COLLECTION_NAME, REGISTRY_FILE, CHUNKING_STRATEGY
from rag_tutor.documents import (
    calculate_file_hash,
    load_registry,
    save_registry,
    migrate_legacy_registry,
    extract_pages_from_pdf,
    chunk_pages,
)
from rag_tutor.documents.chunking import chunk_pages_llama_index
from rag_tutor.llm.embeddings import get_embeddings_batch
from rag_tutor.retrieval.vector_store import delete_source_documents, add_documents_to_store


def _chunk_for_strategy(pages, strategy: str):
    """Dispatch to the configured chunking strategy.

    Args:
        pages: List of DocumentPage objects.
        strategy: ``"recursive"`` (default) or ``"llama_index"``.

    Returns:
        List of DocumentPage chunks.
    """
    if strategy == "llama_index":
        return chunk_pages_llama_index(pages)
    return chunk_pages(pages)


def ingest_pdfs(
    pdf_dir: Path,
    subject_name: str,
    api_key: str = None,
    db_path: Path = DB_PATH,
    collection_name: str = COLLECTION_NAME,
    registry_file: Path = REGISTRY_FILE,
    chunking_strategy: str = None
) -> int:
    """Ingest all PDFs in the directory into Chroma DB for a specific subject.
    Returns the number of files successfully processed or updated.
    """
    if not pdf_dir.exists():
        print(f"Error: Source directory {pdf_dir} does not exist.")
        return 0

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return 0

    strategy = (chunking_strategy or CHUNKING_STRATEGY or "recursive").lower().strip()
    print(f"Found {len(pdf_files)} PDF(s) to process for subject '{subject_name}' (chunking: {strategy}).")

    registry = load_registry(registry_path=registry_file)
    registry = migrate_legacy_registry(registry, subject_name=subject_name)
    subject_registry = registry.get(subject_name, {})
    if not isinstance(subject_registry, dict):
        subject_registry = {}

    files_processed = 0

    for filepath in pdf_files:
        file_hash = calculate_file_hash(filepath)
        filename = filepath.name

        # Check if file is already ingested and has not changed
        if filename in subject_registry and subject_registry[filename].get("hash") == file_hash:
            print(f"Skipping '{filename}' (already ingested and unchanged).")
            continue

        print(f"\nProcessing '{filename}'...")

        # Extract pages
        try:
            pages = extract_pages_from_pdf(filepath)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        if not pages:
            print(f"No readable text found in {filename}.")
            continue

        # Split pages into chunks (strategy dispatched here)
        try:
            chunks = _chunk_for_strategy(pages, strategy)
        except RuntimeError as e:
            # Strategy-specific dependency missing; surface a clear message.
            print(f"Chunking failed: {e}")
            continue
        print(f"Extracted {len(pages)} pages -> {len(chunks)} chunks. Generating embeddings...")
        
        # Prepare content for embedding
        texts = [c.text for c in chunks]
        
        # If the file was previously ingested but changed, delete old entries
        if filename in subject_registry:
            print(f"Updating '{filename}': deleting old collection entries...")
            delete_source_documents(
                filename=filename,
                subject_name=subject_name,
                db_path=db_path,
                collection_name=collection_name
            )
            
        try:
            # Generate embeddings
            embeddings = get_embeddings_batch(texts, api_key=api_key)
            
            # Prepare metadata and IDs
            metadatas = [{
                "source": c.source, 
                "page": c.page_num,
                "chunk_index": c.chunk_index,
                "subject": subject_name
            } for c in chunks]
            
            # Unique ID including subject and chunk index to prevent collisions
            ids = [f"{subject_name}_{filename}_page_{c.page_num}_chunk_{c.chunk_index}" for c in chunks]
            
            # Add to Chroma DB
            add_documents_to_store(
                texts=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids,
                db_path=db_path,
                collection_name=collection_name
            )
            
            # Record in registry
            subject_registry[filename] = {
                "hash": file_hash,
                "pages_count": len(pages),
                "chunks_count": len(chunks),
                "timestamp": filepath.stat().st_mtime
            }
            files_processed += 1
            print(f"Successfully ingested '{filename}'.")
            
        except Exception as e:
            print(f"Failed to ingest '{filename}' due to error: {e}")
            
    # Save the updated registry
    registry[subject_name] = subject_registry
    save_registry(registry, registry_path=registry_file)
    print(f"\nIngestion completed. Processed {files_processed} new/updated files.")
    return files_processed
