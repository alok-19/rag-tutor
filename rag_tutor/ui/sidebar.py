import os
import shutil
from pathlib import Path
import streamlit as st
from rag_tutor.config import STUDY_DIR, LLM_PROVIDER, EMBEDDING_PROVIDER
from rag_tutor.documents import load_registry
from rag_tutor.ingestion import ingest_pdfs


def get_subjects() -> list[str]:
    """Retrieve list of subjects from the study materials directory."""
    dirs = [d.name for d in STUDY_DIR.iterdir() if d.is_dir() and not d.name.startswith(".")]
    if not dirs:
        default_dir = STUDY_DIR / "Operating System"
        default_dir.mkdir(parents=True, exist_ok=True)
        dirs = ["Operating System"]
    return sorted(dirs)


def _get_provider_key_config() -> list[dict]:
    """Return list of API key configs needed for current provider setup."""
    configs = []

    # Determine which providers need keys
    chat_provider = LLM_PROVIDER
    emb_provider = EMBEDDING_PROVIDER

    providers_needed = {chat_provider}
    if emb_provider != chat_provider:
        providers_needed.add(emb_provider)

    # Map provider -> env var, label, setter
    provider_map = {
        "gemini": {
            "env_vars": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
            "label": "Gemini API Key",
            "setter_env": "GEMINI_API_KEY",
        },
        "openai": {
            "env_vars": ["OPENAI_API_KEY"],
            "label": "OpenAI API Key",
            "setter_env": "OPENAI_API_KEY",
        },
        "deepseek": {
            "env_vars": ["DEEPSEEK_API_KEY"],
            "label": "DeepSeek API Key",
            "setter_env": "DEEPSEEK_API_KEY",
        },
    }

    for pname in sorted(providers_needed):
        cfg = provider_map.get(pname)
        if not cfg:
            continue
        # Check if key exists in env
        env_value = None
        for ev in cfg["env_vars"]:
            env_value = env_value or os.getenv(ev)
        configs.append({
            "provider": pname,
            "label": cfg["label"],
            "env_vars": cfg["env_vars"],
            "setter_env": cfg["setter_env"],
            "has_key": bool(env_value),
            "key_value": env_value,
        })

    return configs


def render_sidebar() -> tuple[str, str]:
    """Renders the Streamlit sidebar options.
    Returns:
        tuple[str, str]: (api_key, selected_subject)
        api_key is the key for the primary LLM provider (for backwards compat).
    """
    subjects = get_subjects()
    if "selected_subject" not in st.session_state:
        st.session_state.selected_subject = subjects[0] if subjects else "Operating System"
    selected_subject = st.session_state.selected_subject

    with st.sidebar:
        st.image("https://img.icons8.com/color/96/books.png", width=60)
        st.title("RAG Tutor")
        st.caption("Multi-Provider RAG Assistant")
        st.write("---")

        # Conversation Actions at the top
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ New Chat", use_container_width=True):
                if "subject_messages" not in st.session_state:
                    st.session_state.subject_messages = {}
                st.session_state.subject_messages[selected_subject] = []
                from rag_tutor.conversations import save_conversation
                save_conversation(selected_subject, [])
                st.rerun()
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                from rag_tutor.conversations import clear_conversation
                st.session_state.subject_messages = {}
                for subj in [selected_subject]:
                    clear_conversation(subj)
                st.rerun()

        st.write("---")

        # Provider badge
        chat_provider = LLM_PROVIDER.upper()
        emb_provider = EMBEDDING_PROVIDER.upper()
        if chat_provider == emb_provider:
            st.info(f"**Provider:** {chat_provider}")
        else:
            st.info(f"**Chat:** {chat_provider}  |  **Embeddings:** {emb_provider}")

        # API Key Configuration
        st.subheader("Configure API Keys")

        key_configs = _get_provider_key_config()
        all_keys_ready = True
        primary_api_key = None

        for cfg in key_configs:
            if cfg["has_key"]:
                st.success(f"✅ {cfg['label']} loaded from environment.")
                if cfg["provider"] == LLM_PROVIDER:
                    primary_api_key = cfg["key_value"]
            else:
                all_keys_ready = False
                user_key = st.text_input(
                    f"Enter your {cfg['label']}:",
                    type="password",
                    key=f"api_key_{cfg['provider']}"
                )
                if user_key:
                    os.environ[cfg["setter_env"]] = user_key
                    if cfg["provider"] == LLM_PROVIDER:
                        primary_api_key = user_key
                    st.success(f"{cfg['label']} set!")
                else:
                    st.warning(f"Please provide {cfg['label']} to run queries.")

        if not all_keys_ready and not any(c["has_key"] for c in key_configs):
            st.error("Please configure at least one API key above.")

        st.write("---")

        # Subject Selector / Creator
        st.subheader("Subjects Workspace")

        selected_subject_ui = st.selectbox(
            "Select Active Subject:",
            options=subjects,
            index=subjects.index(selected_subject) if selected_subject in subjects else 0
        )

        if selected_subject_ui != selected_subject:
            st.session_state.selected_subject = selected_subject_ui
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
                        chunks = subject_registry[name].get("chunks_count", "?")
                        st.markdown(f"🟢 **{name}**  \n`{pages} pages / {chunks} chunks ingested`")
                    else:
                        st.markdown(f"⚪ **{name}**  \n`Pending ingestion`")

            # Trigger Ingestion
            if all_keys_ready:
                if st.button("🚀 Ingest / Update Subject", use_container_width=True):
                    with st.spinner("Parsing PDFs and generating embeddings..."):
                        try:
                            # Pass embedding provider key explicitly for ingestion
                            ingest_pdfs(subject_dir, selected_subject)
                            st.success(f"Subject '{selected_subject}' updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Ingestion failed: {e}")
            else:
                st.info("Provide all required API keys to enable document ingestion.")
        else:
            st.info("No study materials in this subject yet. Upload some PDFs above!")

    return primary_api_key or "", selected_subject