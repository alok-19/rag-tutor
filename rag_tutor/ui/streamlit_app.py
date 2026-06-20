import streamlit as st
from rag_tutor.ui.styles import inject_styles
from rag_tutor.ui.sidebar import render_sidebar
from rag_tutor.ui.chat import render_chat_interface
from rag_tutor.ui.study_modes import render_study_modes
from rag_tutor.config import STUDY_DIR


def _render_reader_pane(selected_subject: str) -> None:
    """Render the PDF reader pane when a citation has been opened.

    Driven by session_state set in ``citations._open_in_reader``. Shows the
    source page as a rendered image with prev/next navigation and closes via a
    button. Failures (missing/changed PDF) degrade to a friendly message.
    """
    from rag_tutor.ui.reader import render_pdf_page_highlighted, get_page_count
    from pathlib import Path

    source = st.session_state.get("reader_source")
    page = st.session_state.get("reader_page", 1)
    snippet = st.session_state.get("reader_snippet", "")

    subject_dir = STUDY_DIR / selected_subject
    pdf_path = subject_dir / source if source else None

    header_col, close_col = st.columns([5, 1])
    with header_col:
        st.markdown(f"📖 **{source or 'Reader'}**")
    with close_col:
        if st.button("✕ Close", key="reader_close_btn", use_container_width=True):
            st.session_state["reader_open"] = False
            st.rerun()

    if not pdf_path or not pdf_path.exists():
        st.warning(f"Source file `{source}` not found on disk. It may have been removed.")
        st.caption(f"Looked in: `{subject_dir}`")
        return

    total = get_page_count(pdf_path)
    if total == 0:
        st.error("This PDF could not be read.")
        return

    # Clamp page within bounds.
    page = max(1, min(page, total))
    st.session_state["reader_page"] = page

    # Navigation row.
    pc, nc, info = st.columns([1, 1, 3])
    with pc:
        if st.button("◀ Prev", key="reader_prev", use_container_width=True) and page > 1:
            st.session_state["reader_page"] = page - 1
            st.rerun()
    with nc:
        if st.button("Next ▶", key="reader_next", use_container_width=True) and page < total:
            st.session_state["reader_page"] = page + 1
            st.rerun()
    with info:
        st.caption(f"Page {page} of {total}")
        if snippet:
            st.caption(f"🔍 Highlighting: \"{snippet[:90]}…\"")

    try:
        img_bytes = render_pdf_page_highlighted(pdf_path, page_num=page, snippet=snippet)
        st.image(img_bytes, use_container_width=True)
    except Exception as e:
        st.error(f"Could not render this page: {e}")


def main():
    # Streamlit requires set_page_config to be called first
    st.set_page_config(
        page_title="Personal Study Buddy - RAG Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Inject CSS
    inject_styles()

    # Render sidebar and get configuration
    api_key, selected_subject = render_sidebar()

    # If a citation opened the reader, show chat|reader split layout.
    reader_open = st.session_state.get("reader_open", False)

    if reader_open:
        chat_col, reader_col = st.columns([1.7, 1.0])
        with chat_col:
            _render_main_area(api_key, selected_subject)
        with reader_col:
            _render_reader_pane(selected_subject)
    else:
        _render_main_area(api_key, selected_subject)


def _render_main_area(api_key: str, selected_subject: str) -> None:
    """Top-level mode tabs: Chat (with header) + Study modes."""
    chat_tab, study_tab = st.tabs(["💬 Chat", "🎓 Study"])

    with chat_tab:
        st.markdown(f"""
        <div class="hero-banner">
            <h1>🎓 Personal Study Buddy</h1>
            <p style="font-size: 1.1rem; color: #94a3b8; max-width: 800px; margin: 8px auto 0 auto;">
                Currently studying: <b>{selected_subject}</b>.
                Ask questions, request summaries, or generate quizzes based on your uploaded PDFs.
            </p>
        </div>
        """, unsafe_allow_html=True)
        render_chat_interface(api_key, selected_subject)

    with study_tab:
        st.markdown(f"""
        <div class="hero-banner">
            <h1>🎓 Active Study Modes</h1>
            <p style="font-size: 1.1rem; color: #94a3b8; max-width: 800px; margin: 8px auto 0 auto;">
                Subject: <b>{selected_subject}</b>. Generate quizzes and flashcards from your materials.
            </p>
        </div>
        """, unsafe_allow_html=True)
        render_study_modes(api_key, selected_subject)


if __name__ == "__main__":
    main()
