import sys
import argparse
from pathlib import Path

def run_ui():
    """Launches the Streamlit UI programmatically."""
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        print("Error: Streamlit is not installed. Please install it using: pip install streamlit")
        sys.exit(1)
        
    current_dir = Path(__file__).parent
    app_path = current_dir / "ui" / "streamlit_app.py"
    
    # Configure arguments to pass to streamlit
    sys.argv = ["streamlit", "run", str(app_path)]
    sys.exit(stcli.main())

def run_ingest(subject_name: str):
    """Executes the document ingestion CLI pipeline."""
    from study_rag.config import STUDY_DIR
    from study_rag.ingestion import ingest_pdfs
    
    pdf_directory = STUDY_DIR / subject_name
    pdf_directory.mkdir(parents=True, exist_ok=True)
    print(f"Running CLI ingestion for subject '{subject_name}' in: {pdf_directory}")
    try:
        processed = ingest_pdfs(pdf_directory, subject_name)
        print(f"Ingestion completed. Processed {processed} files.")
    except Exception as e:
        print(f"Ingestion failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Study RAG Assistant CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run UI command
    subparsers.add_parser("run", help="Start the Streamlit Web UI interface")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest study notes PDF files for a subject")
    ingest_parser.add_argument(
        "--subject", 
        default="Operating System",
        help="Subject name to ingest (default: Operating System)"
    )
    
    args = parser.parse_args()
    
    if args.command == "run":
        run_ui()
    elif args.command == "ingest":
        run_ingest(args.subject)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
