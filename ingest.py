from pathlib import Path
from study_rag.config import DB_PATH
from study_rag.documents import load_registry
from study_rag.ingestion import ingest_pdfs

if __name__ == "__main__":
    from study_rag.config import STUDY_DIR
    # Fallback to Operating System subject if run as script
    pdf_directory = STUDY_DIR / "Operating System"
    pdf_directory.mkdir(parents=True, exist_ok=True)
    print(f"Running CLI ingestion for subject 'Operating System' in: {pdf_directory}")
    try:
        ingest_pdfs(pdf_directory, "Operating System")
    except Exception as e:
        print(f"Ingestion failed: {e}")
