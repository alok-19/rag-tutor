import os
import shutil
from pathlib import Path
import streamlit as st
from study_rag.config import STUDY_DIR
from study_rag.documents import load_registry
from study_rag.ingestion import ingest_pdfs

def get_subjects() -> list[str]:
    """Retrieve list of subjects from the study materials directory."""
    dirs = [d.name for d in STUDY_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not dirs:
        default_dir = STUDY_DIR / "Operating System"
        default_dir.mkdir(parents=True, exist_ok=True)
        dirs = ["Operating System"]
    return sorted(dirs)

def render_sidebar() -> tuple[str, str]:
    """Renders the Streamlit sidebar options.
    Returns:
        tuple[str, str]: (api_key, selected_subject)
    """
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/books.png", width=60)
        st.title("Study Buddy AI")
        st.caption("General Purpose RAG Assistant")
        st.write("---")

        # API Key Configuration
        st.subheader("Configure API Key")
        env_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if env_api_key:
            st.success("API Key loaded from environment.")
            api_key = env_api_key
        else:
            api_key = st.text_input("Enter your Gemini API Key:", type="password")
            if not api_key:
                st.warning("Please provide a Gemini/Google API Key to run queries.")
            else:
                os.environ["GEMINI_API_KEY"] = api_key

        st.write("---")

        # Subject Selector / Creator
        st.subheader("Subjects Workspace")
        subjects = get_subjects()
        
        if "selected_subject" not in st.session_state:
            st.session_state.selected_subject = subjects[0] if subjects else "Operating System"
            
        selected_subject = st.selectbox(
            "Select Active Subject:",
            options=subjects,
            index=subjects.index(st.session_state.selected_subject) if st.session_state.selected_subject in subjects else 0
        )
        
        if selected_subject != st.session_state.selected_subject:
            st.session_state.selected_subject = selected_subject
            st.rerun()

        # Create New Subject Dialog
        with st.expander("➕ Create New Subject"):
            new_subj = st.text_input("Subject Name:", key="new_subj_input")
            if st.button("Create", use_container_width=True):
                if new_subj.strip():
                    new_subj_clean = new_subj.strip()
                    new_subj_dir = STUDY_DIR / new_subj_clean
                    new_subj_dir.mkdir(parents=True, exist_ok=True)
                    st.success(f"Created '{new_subj_clean}'")
                    st.session_state.selected_subject = new_subj_clean
                    st.rerun()
                else:
                    st.error("Please enter a valid name.")

        st.write("---")

        # Document Management for Active Subject
        st.subheader(f"Manage Material: {selected_subject}")
        subject_dir = STUDY_DIR / selected_subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        pdf_files = list(subject_dir.glob("*.pdf"))
        
        # File Uploader
        uploaded_files = st.file_uploader(
            "Upload study PDFs:",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            files_saved = 0
            for f in uploaded_files:
                target_path = subject_dir / f.name
                if not target_path.exists():
                    with open(target_path, "wb") as buffer:
                        shutil.copyfileobj(f, buffer)
                    files_saved += 1
            if files_saved > 0:
                st.success(f"Saved {files_saved} file(s) to '{selected_subject}'")
                st.rerun()

        # Registry Status check
        registry = load_registry()
        subject_registry = registry.get(selected_subject, {})

        if pdf_files:
            with st.expander(f"Documents list ({len(pdf_files)})", expanded=True):
                for pdf in pdf_files:
                    name = pdf.name
                    if isinstance(subject_registry, dict) and name in subject_registry:
                        pages = subject_registry[name].get("pages_count", "?")
                        st.markdown(f"🟢 **{name}**  \n`{pages} pages ingested`")
                    else:
                        st.markdown(f"⚪ **{name}**  \n`Pending ingestion`")
            
            # Trigger Ingestion
            if api_key:
                if st.button("🚀 Ingest / Update Subject", use_container_width=True):
                    with st.spinner("Parsing PDFs and generating embeddings..."):
                        try:
                            os.environ["GEMINI_API_KEY"] = api_key
                            ingest_pdfs(subject_dir, selected_subject, api_key=api_key)
                            st.success(f"Subject '{selected_subject}' updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ingestion failed: {e}")
            else:
                st.info("Provide an API key to enable document ingestion.")
        else:
            st.info("No study materials in this subject yet. Upload some PDFs above!")

        st.write("---")
        
        # Utilities
        st.subheader("Actions")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
    return api_key, selected_subject
