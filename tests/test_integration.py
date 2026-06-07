import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch
import pytest

from study_rag.documents import DocumentPage
from study_rag.ingestion import ingest_pdfs
from study_rag.retrieval import retrieve_context
from study_rag.retrieval.vector_store import get_collection, has_subject_documents

@pytest.fixture
def temp_dirs():
    """Fixture to manage temporary DB and study materials directories."""
    db_dir = Path(tempfile.mkdtemp())
    materials_dir = Path(tempfile.mkdtemp())
    yield db_dir, materials_dir
    # Cleanup
    shutil.rmtree(db_dir)
    shutil.rmtree(materials_dir)

@patch("study_rag.ingestion.service.extract_pages_from_pdf")
@patch("study_rag.ingestion.service.get_embeddings_batch")
def test_ingestion_and_retrieval_flow(mock_embeds, mock_extract, temp_dirs):
    db_dir, materials_dir = temp_dirs
    registry_file = db_dir / "ingestion_registry.json"
    subject = "Computer Networks"
    
    # Setup mock PDF file in study_materials
    subj_dir = materials_dir / subject
    subj_dir.mkdir()
    pdf_file = subj_dir / "chapter1.pdf"
    pdf_file.write_text("dummy pdf contents") # write some text to hash
    
    # Mock PDF extraction
    mock_extract.return_value = [
        DocumentPage(text="TCP stands for Transmission Control Protocol.", page_num=1, source="chapter1.pdf"),
        DocumentPage(text="UDP is a connectionless transport protocol.", page_num=2, source="chapter1.pdf")
    ]
    
    # Mock Embedding generation (return 2 embedding vectors of length 768)
    mock_embeds.return_value = [[0.1] * 768, [0.2] * 768]
    
    # 1. Ingest a subject with one PDF
    processed_count = ingest_pdfs(
        pdf_dir=subj_dir,
        subject_name=subject,
        api_key="mock_key",
        db_path=db_dir,
        collection_name="test_collection",
        registry_file=registry_file
    )
    
    assert processed_count == 1
    assert has_subject_documents(subject, db_path=db_dir, collection_name="test_collection")
    
    # 2. Skip unchanged PDFs
    processed_count_second = ingest_pdfs(
        pdf_dir=subj_dir,
        subject_name=subject,
        api_key="mock_key",
        db_path=db_dir,
        collection_name="test_collection",
        registry_file=registry_file
    )
    assert processed_count_second == 0 # skipped because hash is unchanged
    
    # 3. Update changed PDFs and delete old Chroma entries
    # Change PDF contents to trigger hash change
    pdf_file.write_text("updated dummy pdf contents")
    mock_extract.return_value = [
        DocumentPage(text="TCP stands for Transmission Control Protocol.", page_num=1, source="chapter1.pdf"),
        DocumentPage(text="IP stands for Internet Protocol.", page_num=2, source="chapter1.pdf")
    ]
    mock_embeds.return_value = [[0.1] * 768, [0.3] * 768]
    
    processed_count_third = ingest_pdfs(
        pdf_dir=subj_dir,
        subject_name=subject,
        api_key="mock_key",
        db_path=db_dir,
        collection_name="test_collection",
        registry_file=registry_file
    )
    assert processed_count_third == 1 # re-ingested because hash changed
    
    # Verify collection contents: old UDP page should be deleted, new IP page should exist
    coll = get_collection(db_path=db_dir, collection_name="test_collection")
    all_docs = coll.get(where={"subject": subject})
    assert len(all_docs["documents"]) == 2
    assert "IP stands for Internet Protocol." in all_docs["documents"]
    assert "UDP is a connectionless transport protocol." not in all_docs["documents"]


@patch("study_rag.retrieval.rag_service.get_embedding")
@patch("study_rag.ingestion.service.extract_pages_from_pdf")
@patch("study_rag.ingestion.service.get_embeddings_batch")
def test_subject_isolated_retrieval(mock_embeds, mock_extract, mock_get_embedding, temp_dirs):
    db_dir, materials_dir = temp_dirs
    registry_file = db_dir / "ingestion_registry.json"
    
    # Setup subjects
    subj_os = materials_dir / "Operating System"
    subj_os.mkdir()
    pdf_os = subj_os / "os.pdf"
    pdf_os.write_text("os pdf content")
    
    subj_cn = materials_dir / "Computer Networks"
    subj_cn.mkdir()
    pdf_cn = subj_cn / "cn.pdf"
    pdf_cn.write_text("cn pdf content")
    
    # Ingest OS
    mock_extract.return_value = [
        DocumentPage(text="Deadlock occurs when processes block each other.", page_num=1, source="os.pdf")
    ]
    mock_embeds.return_value = [[0.9] * 768]
    ingest_pdfs(subj_os, "Operating System", "mock_key", db_dir, "test_collection", registry_file)
    
    # Ingest CN
    mock_extract.return_value = [
        DocumentPage(text="Deadlocks are not common in network routing.", page_num=1, source="cn.pdf")
    ]
    mock_embeds.return_value = [[0.8] * 768]
    ingest_pdfs(subj_cn, "Computer Networks", "mock_key", db_dir, "test_collection", registry_file)
    
    # Query: we search for "deadlock" but filter by "Operating System"
    mock_get_embedding.return_value = [0.9] * 768
    
    results = retrieve_context(
        query="what is deadlock?",
        subject="Operating System",
        api_key="mock_key",
        n_results=5,
        db_path=db_dir,
        collection_name="test_collection"
    )
    
    # 4. Retrieve only from active subject
    assert len(results) == 1
    assert results[0].source == "os.pdf"
    assert results[0].page == 1
    assert "Deadlock occurs when processes block each other." in results[0].text
    
    # 5. Return the same citation shape used by the UI
    assert hasattr(results[0], "source")
    assert hasattr(results[0], "page")
    assert hasattr(results[0], "text")
