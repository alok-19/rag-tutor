import streamlit as st
from rag_tutor.retrieval import has_subject_documents, retrieve_context_with_memory, build_chat_history, construct_rag_prompt
from rag_tutor.llm import generate_response_stream
from rag_tutor.llm.providers import get_provider, list_providers
from rag_tutor.config import LLM_PROVIDER, EMBEDDING_PROVIDER
from rag_tutor.feedback import save_feedback

def render_citations(sources: list[dict]):
    """Renders the sources panel beneath a chat message."""
    if sources:
        with st.expander("📚 View Cited Sources", expanded=False):
            cols = st.columns(2)
            for index, src in enumerate(sources):
                col_index = index % 2
                with cols[col_index]:
                    st.markdown(f"""
                    <div class="source-card">
                        <span class="source-badge">{src['source']}</span>
                        <span class="source-badge" style="background:rgba(99,102,241,0.1); border-color:rgba(99,102,241,0.2); color:#a5b4fc;">Page {src['page']}</span>
                        <div style="font-size:0.9rem; color:#94a3b8; margin-top:8px; line-height:1.4;">
                            "{src['text'][:220]}..."
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def render_feedback_buttons(msg: dict, msg_index: int, subject: str):
    """Render Copilot-style ghost feedback buttons at the bottom-right of an assistant message.
    Uses st.button() for BOTH states (active via disabled=True) to guarantee zero DOM/layout shift."""
    current_feedback = msg.get("feedback")
    
    # Large spacer pushes the two tiny button columns tightly to the far right edge
    spacer, up_col, down_col = st.columns([20, 0.35, 0.35])
    
    with up_col:
        up_clicked = st.button(
            "↑",
            key=f"fb_up_{subject}_{msg_index}",
            help="Helpful",
            disabled=(current_feedback == "thumbs_up"),
            use_container_width=True
        )
        if up_clicked and current_feedback != "thumbs_up":
            msg["feedback"] = "thumbs_up"
            save_feedback(
                subject=subject,
                query=msg.get("user_query", ""),
                response=msg["content"],
                rating="thumbs_up",
                sources=msg.get("sources", [])
            )
            st.rerun()
    
    with down_col:
        down_clicked = st.button(
            "↓",
            key=f"fb_down_{subject}_{msg_index}",
            help="Not helpful",
            disabled=(current_feedback == "thumbs_down"),
            use_container_width=True
        )
        if down_clicked and current_feedback != "thumbs_down":
            msg["feedback"] = "thumbs_down"
            save_feedback(
                subject=subject,
                query=msg.get("user_query", ""),
                response=msg["content"],
                rating="thumbs_down",
                sources=msg.get("sources", [])
            )
            st.rerun()

def render_chat_interface(api_key: str, selected_subject: str):
    """Renders the main chat message area, quickstart buttons, and query execution."""
    
    # Check if subject is empty
    has_docs = has_subject_documents(selected_subject)
    if not has_docs:
        st.warning(f"⚠️ **Subject '{selected_subject}' is empty in the database.** Please upload PDF textbook files in the sidebar and click **'🚀 Ingest / Update Subject'** to process them.")
        st.markdown(f"""
        ### Quick Setup Guide:
        1. Paste your **Gemini API Key** in the sidebar.
        2. Select **{selected_subject}** (or create a new subject).
        3. Upload one or more study materials (PDF format) in the sidebar.
        4. Click **🚀 Ingest / Update Subject** to parse, index, and activate your study buddy.
        """)
        st.stop()
        
    # Initialize message history
    if "subject_messages" not in st.session_state:
        st.session_state.subject_messages = {}
        
    if selected_subject not in st.session_state.subject_messages:
        st.session_state.subject_messages[selected_subject] = []
        
    current_messages = st.session_state.subject_messages[selected_subject]
        
    # Render historical messages
    for msg_index, msg in enumerate(current_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                render_citations(msg["sources"])
            if msg["role"] == "assistant":
                render_feedback_buttons(msg, msg_index, selected_subject)
                
    # Suggested / Quick-start questions
    selected_suggestion = None
    if not current_messages:
        st.markdown("### 💡 Quick-start suggestions:")
        col1, col2, col3 = st.columns(3)
        
        # Tailored suggestions based on active subject
        if selected_subject == "Operating System":
            suggestions = [
                "What are the four necessary conditions for a Deadlock?",
                "Explain the difference between Paging and Segmentation.",
                "Explain Process Control Block (PCB)."
            ]
        else:
            suggestions = [
                "Summarize the key takeaways from the study materials.",
                "Generate a 5-question multiple-choice quiz based on these notes.",
                "Explain the most complex concept described in these files."
            ]
        
        if col1.button(suggestions[0], use_container_width=True):
            selected_suggestion = suggestions[0]
        if col2.button(suggestions[1], use_container_width=True):
            selected_suggestion = suggestions[1]
        if col3.button(suggestions[2], use_container_width=True):
            selected_suggestion = suggestions[2]
            
    # Query input
    user_query = st.chat_input(f"Ask a question about {selected_subject}...")
    if selected_suggestion:
        user_query = selected_suggestion
        
    if user_query:
        if not api_key:
            provider_label = LLM_PROVIDER.upper()
            st.error(f"Please provide a {provider_label} API Key in the sidebar to run queries.")
            st.stop()
            
        # Append and display user message
        current_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)
            
        # Capture message history BEFORE current query for memory/context
        previous_messages = current_messages[:-1]
        
        # 1. Retrieval with memory support
        with st.spinner("Searching textbook resources..."):
            try:
                retrieved_sources, disambiguated_query = retrieve_context_with_memory(
                    query=user_query,
                    subject=selected_subject,
                    messages=previous_messages,
                    api_key=api_key
                )
            except Exception as e:
                st.error(f"Error querying the database: {e}")
                st.stop()
                
        # Show disambiguation hint if query was expanded
        if disambiguated_query != user_query:
            st.caption(f"🔍 Searched: *{disambiguated_query}*")
                
        # Format sources for generation prompt and UI
        context_parts = []
        ui_sources = []
        for src in retrieved_sources:
            context_parts.append(f"Source: {src.source}, Page: {src.page}\nContent: {src.text}")
            ui_sources.append({
                "source": src.source,
                "page": src.page,
                "text": src.text
            })
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Build conversation history for the prompt
        chat_history = build_chat_history(previous_messages, max_turns=3)
        
        # 2. Generation & Streaming
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            status_placeholder = st.empty()
            
            prompt = construct_rag_prompt(
                subject=selected_subject,
                query=user_query,
                context_text=context_text,
                chat_history=chat_history
            )
            
            full_response = ""
            success = False
            fallback_used = False
            
            def status_callback(msg: str):
                if msg:
                    status_placeholder.warning(msg)
                else:
                    status_placeholder.empty()
                    
            try:
                stream = generate_response_stream(
                    prompt=prompt,
                    api_key=api_key,
                    status_callback=status_callback
                )
                for chunk, fall_used in stream:
                    full_response += chunk
                    fallback_used = fall_used
                    message_placeholder.markdown(full_response + "▌")
                success = True
                
                if success:
                    message_placeholder.markdown(full_response)
                    if fallback_used:
                        st.caption("ℹ️ *Response generated using backup model due to high demand.*")
                        
                    assistant_msg = {
                        "role": "assistant",
                        "content": full_response + ("\n\n*(Generated using backup model)*" if fallback_used else ""),
                        "sources": ui_sources,
                        "user_query": user_query,
                        "feedback": None
                    }
                    current_messages.append(assistant_msg)
                    st.rerun()
                    
            except Exception as final_err:
                status_placeholder.empty()
                error_message = (
                    "⚠️ **We are experiencing temporary connection issues with our AI provider.**  \n"
                    "Please try again in a moment. In the meantime, you can review the retrieved textbook passages "
                    "from your study materials below:"
                )
                message_placeholder.error(error_message)
                
                if ui_sources:
                    st.markdown("### 📚 Retrieved Textbook Passages:")
                    for src in ui_sources:
                        st.markdown(f"**Source**: `{src['source']}` (Page {src['page']})")
                        st.info(f"\"{src['text']}\"")
                        
                assistant_msg = {
                    "role": "assistant",
                    "content": error_message + "\n\n*(Failed to connect to API, displayed matching segments directly)*",
                    "sources": ui_sources,
                    "user_query": user_query,
                    "feedback": None
                }
                current_messages.append(assistant_msg)
                st.rerun()
