import streamlit as st
from study_rag.ui.styles import inject_styles
from study_rag.ui.sidebar import render_sidebar
from study_rag.ui.chat import render_chat_interface

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
    
    # Render Header Banner
    st.markdown(f"""
    <div class="hero-banner">
        <h1>🎓 Personal Study Buddy</h1>
        <p style="font-size: 1.1rem; color: #94a3b8; max-width: 800px; margin: 8px auto 0 auto;">
            Currently studying: <b>{selected_subject}</b>. 
            Ask questions, request summaries, or generate quizzes based on your uploaded PDFs.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Render main chat UI
    render_chat_interface(api_key, selected_subject)

if __name__ == "__main__":
    main()
