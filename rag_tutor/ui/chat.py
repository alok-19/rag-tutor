import streamlit as st
from rag_tutor.retrieval import (
    has_subject_documents,
    retrieve_context_with_memory,
    build_chat_history,
    construct_rag_prompt,
    RetrievedSource,
)
from rag_tutor.llm import generate_response_stream
from rag_tutor.config import LLM_PROVIDER, EMBEDDING_PROVIDER
from rag_tutor.feedback import save_feedback
from rag_tutor.conversations import load_conversations, save_conversation, clear_conversation
from rag_tutor.ui.citations import render_source_pills, render_interactive_sources, confidence_label
from rag_tutor.ui.streaming import stream_to_placeholder, render_stop_control
from rag_tutor.ui.suggestions import generate_suggestions


def _sources_to_ui(sources: list[RetrievedSource]) -> list[dict]:
    """Convert RetrievedSource dataclasses to JSON-safe dicts for the UI/state."""
    return [
        {"source": s.source, "page": s.page, "text": s.text, "score": getattr(s, "score", 0.0)}
        for s in sources
    ]


def render_citations(sources: list[dict]):
    """Legacy compact citation renderer (kept for back-compat).

    New UIs prefer ``render_interactive_sources`` in citations.py.
    """
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
            save_conversation(subject, st.session_state.subject_messages[subject])
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
            save_conversation(subject, st.session_state.subject_messages[subject])
            st.rerun()


def _load_subject_messages(subject: str) -> list[dict]:
    """Hydrate session_state for ``subject`` from disk if not already loaded."""
    if "subject_messages" not in st.session_state:
        st.session_state.subject_messages = {}
    if subject not in st.session_state.subject_messages:
        st.session_state.subject_messages[subject] = load_conversations(subject)
    return st.session_state.subject_messages[subject]


def _persist(subject: str) -> None:
    """Save the current subject's messages to disk."""
    if "subject_messages" in st.session_state and subject in st.session_state.subject_messages:
        save_conversation(subject, st.session_state.subject_messages[subject])


def _regenerate_last(subject: str, current_messages: list[dict]) -> bool:
    """Handle a regenerate click: drop the last assistant answer, re-ask.

    Returns True if a regeneration was requested (caller should continue to the
    query path using the preserved user query).
    """
    # Pop the trailing assistant message; keep the user query to re-run.
    if current_messages and current_messages[-1]["role"] == "assistant":
        last_assistant = current_messages.pop()
        # The user query that produced it is now the last message.
        if current_messages and current_messages[-1]["role"] == "user":
            return True
        # Edge case: nothing to regenerate against — restore.
        current_messages.append(last_assistant)
    return False


def _build_assistant_msg(content: str, sources: list[dict], user_query: str, fallback_used: bool, stopped: bool) -> dict:
    suffix = ""
    if stopped:
        suffix = "\n\n*(stopped)*"
    elif fallback_used:
        suffix = "\n\n*(Generated using backup model)*"
    return {
        "role": "assistant",
        "content": content + suffix,
        "sources": sources,
        "user_query": user_query,
        "feedback": None,
        "stopped": stopped,
    }


def _handle_query(
    user_query: str,
    api_key: str,
    selected_subject: str,
    current_messages: list[dict],
    is_regeneration: bool = False,
) -> None:
    """Run retrieval + streaming generation for ``user_query`` and persist.

    Preserves the existing fallback/error behavior. Renders live source pills
    during retrieval and uses the interruptible streaming session.
    """
    if not is_regeneration:
        current_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

    previous_messages = current_messages[:-1] if current_messages else []

    # 1. Retrieval with live source-pill transparency.
    retrieval_box = st.container()
    with retrieval_box:
        status = st.spinner("Searching textbook resources...")
        status.__enter__()
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
        # Show retrieved source pills before the answer streams.
        render_source_pills(retrieved_sources)
        if disambiguated_query != user_query:
            st.caption(f"🔍 Searched: *{disambiguated_query}*")

    ui_sources = _sources_to_ui(retrieved_sources)
    context_parts = [
        f"Source: {s.source}, Page: {s.page}\nContent: {s.text}" for s in retrieved_sources
    ]
    context_text = "\n\n---\n\n".join(context_parts)
    chat_history = build_chat_history(previous_messages, max_turns=3)
    prompt = construct_rag_prompt(
        subject=selected_subject,
        query=user_query,
        context_text=context_text,
        chat_history=chat_history,
    )

    # 2. Generation: synchronous streaming into a live placeholder.
    #    Robust by design: single script run, no threads, no sleep/rerun loop.
    with st.chat_message("assistant"):
        placeholder = st.empty()
        # Stop control sets a session flag that stream_to_placeholder polls
        # between chunks; rendering it before the stream so it is interactive.
        render_stop_control(key_suffix=f"_{selected_subject}")

        full_response, fallback_used, gen_error = stream_to_placeholder(
            prompt=prompt,
            placeholder=placeholder,
            api_key=api_key,
        )

        if gen_error is not None:
            _render_generation_error(gen_error, ui_sources)
            # Still record what we have so the turn isn't lost.
            current_messages.append(_build_assistant_msg(
                content=full_response or "",
                sources=ui_sources,
                user_query=user_query,
                fallback_used=False,
                stopped=False,
            ))
            _persist(selected_subject)
            st.stop()

        if not full_response.strip():
            placeholder.info("(No response returned. Try rephrasing your question.)")
            st.stop()

        if fallback_used:
            st.caption("ℹ️ *Response generated using backup model due to high demand.*")

        assistant_msg = _build_assistant_msg(
            content=full_response,
            sources=ui_sources,
            user_query=user_query,
            fallback_used=fallback_used,
            stopped=False,
        )
        current_messages.append(assistant_msg)
        _persist(selected_subject)
        st.rerun()


def _render_generation_error(error, ui_sources: list[dict]) -> None:
    """Surface a friendly error and show the retrieved passages as a fallback."""
    error_message = (
        "⚠️ **We are experiencing temporary connection issues with our AI provider.**  \n"
        "Please try again in a moment. In the meantime, you can review the retrieved textbook passages "
        "from your study materials below:"
    )
    st.error(error_message)
    if ui_sources:
        st.markdown("### 📚 Retrieved Textbook Passages:")
        for src in ui_sources:
            st.markdown(f"**Source**: `{src['source']}` (Page {src['page']})")
            st.info(f"\"{src['text']}\"")


def render_chat_interface(api_key: str, selected_subject: str):
    """Renders the main chat message area, quickstart buttons, and query execution."""

    # Check if subject is empty
    has_docs = has_subject_documents(selected_subject)
    if not has_docs:
        st.warning(f"⚠️ **Subject '{selected_subject}' is empty in the database.** Please upload PDF textbook files in the sidebar and click **'🚀 Ingest / Update Subject'** to process them.")
        st.markdown(f"""
        ### Quick Setup Guide:
        1. Paste your **{LLM_PROVIDER.upper()} API Key** in the sidebar.
        2. Select **{selected_subject}** (or create a new subject).
        3. Upload one or more study materials (PDF format) in the sidebar.
        4. Click **🚀 Ingest / Update Subject** to parse, index, and activate your study buddy.
        """)
        st.stop()

    current_messages = _load_subject_messages(selected_subject)

    # --- Regenerate handling for the last assistant message -----------------
    if st.session_state.get("_pending_regenerate"):
        st.session_state["_pending_regenerate"] = False
        if current_messages and current_messages[-1]["role"] == "assistant":
            # Drop the last assistant answer; its user query becomes the rerun target.
            current_messages.pop()
        if current_messages and current_messages[-1]["role"] == "user":
            user_query = current_messages[-1]["content"]
            # Re-render prior history without the just-removed answer.
            for msg in current_messages[:-1]:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            _handle_query(
                user_query=user_query,
                api_key=api_key,
                selected_subject=selected_subject,
                current_messages=current_messages,
                is_regeneration=True,
            )

    # Render historical messages.
    for msg_index, msg in enumerate(current_messages):
        is_last = (msg_index == len(current_messages) - 1)
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                sources = msg.get("sources", [])
                if sources:
                    # Confidence line + interactive sources.
                    conf_text, _level = confidence_label(
                        [RetrievedSource(**_coerce_src(s)) for s in sources]
                    )
                    st.caption(conf_text)
                    render_interactive_sources(
                        [RetrievedSource(**_coerce_src(s)) for s in sources],
                        subject=selected_subject,
                        key_prefix=f"hist_{msg_index}",
                    )
                    if msg.get("stopped"):
                        st.caption("⏹ *(generation stopped)*")
                    # Regenerate button only on the last assistant message.
                    if is_last:
                        if st.button("🔁 Regenerate", key=f"regen_{selected_subject}_{msg_index}"):
                            st.session_state["_pending_regenerate"] = True
                            st.rerun()
                render_feedback_buttons(msg, msg_index, selected_subject)

    # --- Suggested / Quick-start questions ---------------------------------
    selected_suggestion = None
    if not current_messages:
        st.markdown("### 💡 Quick-start suggestions:")
        # Dynamic suggestions: cached in session, generated on first view.
        cache_key = f"_suggestions_{selected_subject}"
        if cache_key not in st.session_state:
            if api_key:
                st.session_state[cache_key] = generate_suggestions(selected_subject, api_key=api_key)
            else:
                from rag_tutor.ui.suggestions import DEFAULT_SUGGESTIONS
                st.session_state[cache_key] = list(DEFAULT_SUGGESTIONS)
        suggestions = st.session_state[cache_key]

        col1, col2, col3 = st.columns(3)
        cols = [col1, col2, col3]
        for i, col in enumerate(cols):
            if i < len(suggestions) and col.button(suggestions[i], use_container_width=True):
                selected_suggestion = suggestions[i]

    # --- Query input -------------------------------------------------------
    # Process a submitted query exactly once. Streamlit's chat_input returns
    # its value until the run completes and clears it on the next natural
    # rerun; we guard with a pending token so the reruns triggered during
    # streaming/generation never re-process the same query. The token resets
    # automatically once chat_input returns None (a fresh, empty input).
    pending_key = "_pending_query"
    raw_query = st.chat_input(f"Ask a question about {selected_subject}...")
    if selected_suggestion:
        raw_query = selected_suggestion

    user_query = None
    if raw_query:
        if st.session_state.get(pending_key) != raw_query:
            # Fresh query — claim it.
            st.session_state[pending_key] = raw_query
            user_query = raw_query
        # else: same query seen on a prior rerun → skip (already processed).

    # If the input was cleared (new empty submit), release the token.
    if not raw_query and st.session_state.get(pending_key) is not None:
        st.session_state[pending_key] = None

    if user_query is not None:
        if not api_key:
            provider_label = LLM_PROVIDER.upper()
            st.error(f"Please provide a {provider_label} API Key in the sidebar to run queries.")
            st.stop()

        _handle_query(
            user_query=user_query,
            api_key=api_key,
            selected_subject=selected_subject,
            current_messages=current_messages,
        )


def _coerce_src(s: dict) -> dict:
    """Coerce a serialized source dict back to RetrievedSource kwargs."""
    return {
        "source": s.get("source", "Unknown"),
        "page": s.get("page", 0),
        "text": s.get("text", ""),
        "score": s.get("score", 0.0),
    }
