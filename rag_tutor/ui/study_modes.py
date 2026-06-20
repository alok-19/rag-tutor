"""Quiz and Flashcard study modes.

Both modes reuse the retrieval pipeline to gather context from the active
subject, then ask the LLM for structured JSON (parsed defensively). The Quiz
mode is fully interactive (select options, grade, see explanations); the
Flashcard mode supports a flip interaction.

State is kept in ``st.session_state`` keyed by subject so switching subjects
or tabs preserves a generated deck until explicitly regenerated.
"""
from __future__ import annotations

import streamlit as st

from rag_tutor.retrieval import retrieve_context
from rag_tutor.study import generate_quiz, generate_flashcards, grade_quiz
from rag_tutor.ui.citations import render_interactive_sources, confidence_label
from rag_tutor.retrieval.vector_store import RetrievedSource


def _fetch_context(subject: str, api_key: str | None, n: int = 8):
    """Retrieve broad context for study-mode generation."""
    try:
        return retrieve_context(query="key concepts definitions terms", subject=subject, api_key=api_key, n_results=n)
    except Exception:
        return []


def _ctx_to_dict(sources):
    return [{"source": s.source, "page": s.page, "text": s.text, "score": getattr(s, "score", 0.0)} for s in sources]


def render_study_modes(api_key: str, selected_subject: str) -> None:
    """Top-level entry: tabbed Quiz / Flashcards UI."""
    quiz_tab, cards_tab = st.tabs(["📝 Quiz", "🗂️ Flashcards"])
    with quiz_tab:
        _render_quiz_tab(api_key, selected_subject)
    with cards_tab:
        _render_flashcard_tab(api_key, selected_subject)


# ============================================================
# Quiz tab
# ============================================================

def _render_quiz_tab(api_key: str, selected_subject: str) -> None:
    st.markdown("#### 📝 Quiz Yourself")
    st.caption("Questions are generated from your study materials and graded instantly.")

    quiz_key = f"_quiz_{selected_subject}"
    col_gen, col_meta = st.columns([1, 2])
    with col_gen:
        num_q = st.number_input("Questions", min_value=3, max_value=10, value=5, step=1, key=f"quiz_n_{selected_subject}")
        if st.button("✨ Generate Quiz", use_container_width=True, type="primary"):
            if not api_key:
                st.error("Please provide an API key in the sidebar to generate a quiz.")
            else:
                with st.spinner("Retrieving context and writing questions..."):
                    sources = _fetch_context(selected_subject, api_key)
                    questions = generate_quiz(selected_subject, sources, api_key=api_key, num_questions=int(num_q))
                    st.session_state[quiz_key] = {"questions": questions, "answers": [None] * len(questions), "sources": _ctx_to_dict(sources), "graded": False}
                if not questions:
                    st.warning("Couldn't generate a quiz from the current materials. Try again or add more PDFs.")
                st.rerun()

    data = st.session_state.get(quiz_key)
    if not data or not data.get("questions"):
        st.info("Click **Generate Quiz** to create a practice quiz from your materials.")
        return

    questions = data["questions"]
    answers = data["answers"]
    # Ensure answers list length matches questions.
    if len(answers) != len(questions):
        answers = [None] * len(questions)

    st.markdown(f"**{len(questions)} question{'s' if len(questions) != 1 else ''}** ready.")

    for i, q in enumerate(questions):
        st.markdown(f"**Q{i+1}. {q.question}**")
        choice = st.radio(
            "Choose",
            options=list(range(len(q.options))),
            format_func=lambda idx, opts=q.options: opts[idx],
            index=answers[i] if answers[i] is not None else None,
            key=f"quiz_q_{selected_subject}_{i}",
            label_visibility="collapsed",
        )
        answers[i] = choice
        if data.get("graded"):
            _render_grading_feedback(q, answers[i])

    data["answers"] = answers

    gc, rc = st.columns(2)
    with gc:
        if not data.get("graded"):
            if st.button("✅ Submit Answers", use_container_width=True, type="primary"):
                data["graded"] = True
                st.rerun()
        else:
            score = grade_quiz(questions, answers)
            st.success(f"You scored **{score.correct}/{score.total}** ({score.percent:.0f}%)")
    with rc:
        if st.button("🔄 New Quiz"):
            st.session_state.pop(quiz_key, None)
            st.rerun()

    # Show sources used to build the quiz.
    if data.get("sources"):
        with st.expander("📚 Sources used for this quiz"):
            conf_text, _ = confidence_label([RetrievedSource(**s) for s in data["sources"]])
            st.caption(conf_text)
            render_interactive_sources([RetrievedSource(**s) for s in data["sources"]], subject=selected_subject, key_prefix="quiz_srcs")


def _render_grading_feedback(q, chosen) -> None:
    """Show per-question correctness + explanation after grading."""
    correct = (chosen is not None and chosen == q.correct_index)
    if correct:
        st.markdown("✅ **Correct!**")
    else:
        correct_text = q.options[q.correct_index] if 0 <= q.correct_index < len(q.options) else "?"
        st.markdown(f"❌ **Incorrect.** Correct answer: **{correct_text}**")
    if q.explanation:
        st.caption(f"💡 {q.explanation}")


# ============================================================
# Flashcard tab
# ============================================================

def _render_flashcard_tab(api_key: str, selected_subject: str) -> None:
    st.markdown("#### 🗂️ Flashcards")
    st.caption("Click a card to flip between term and definition.")

    cards_key = f"_cards_{selected_subject}"
    col_gen, col_meta = st.columns([1, 2])
    with col_gen:
        num_c = st.number_input("Cards", min_value=3, max_value=15, value=8, step=1, key=f"cards_n_{selected_subject}")
        if st.button("✨ Generate Cards", use_container_width=True, type="primary"):
            if not api_key:
                st.error("Please provide an API key in the sidebar to generate flashcards.")
            else:
                with st.spinner("Retrieving context and creating cards..."):
                    sources = _fetch_context(selected_subject, api_key)
                    cards = generate_flashcards(selected_subject, sources, api_key=api_key, num_cards=int(num_c))
                    st.session_state[cards_key] = {"cards": cards, "flipped": [False] * len(cards), "index": 0, "sources": _ctx_to_dict(sources)}
                if not cards:
                    st.warning("Couldn't generate flashcards from the current materials. Try again or add more PDFs.")
                st.rerun()

    data = st.session_state.get(cards_key)
    if not data or not data.get("cards"):
        st.info("Click **Generate Cards** to create a flashcard deck from your materials.")
        return

    cards = data["cards"]
    idx = data.get("index", 0) % len(cards)
    flipped = data.get("flipped", [False] * len(cards))
    if len(flipped) != len(cards):
        flipped = [False] * len(cards)

    # Single-card view with navigation.
    card = cards[idx]
    face = "back" if flipped[idx] else "front"
    label = "Definition" if flipped[idx] else "Term"
    content = card.back if flipped[idx] else card.front

    st.markdown(
        f"""
        <div class="flashcard {'flipped' if flipped[idx] else ''}">
            <div class="flashcard-face-label">{label}</div>
            <div class="flashcard-content">{content}</div>
            <div class="flashcard-progress">Card {idx + 1} of {len(cards)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    bc, pc, nc, fc = st.columns([1, 1, 1, 1])
    if bc.button("⏮ First", use_container_width=True, key=f"fc_first_{selected_subject}"):
        data["index"] = 0
        st.rerun()
    if pc.button("◀ Prev", use_container_width=True, key=f"fc_prev_{selected_subject}"):
        data["index"] = (idx - 1) % len(cards)
        st.rerun()
    if nc.button("Next ▶", use_container_width=True, key=f"fc_next_{selected_subject}"):
        data["index"] = (idx + 1) % len(cards)
        st.rerun()
    if fc.button("🔄 Flip", use_container_width=True, type="primary", key=f"fc_flip_{selected_subject}"):
        flipped[idx] = not flipped[idx]
        data["flipped"] = flipped
        st.rerun()

    if st.button("♻️ Regenerate Deck", key=f"fc_regen_{selected_subject}"):
        st.session_state.pop(cards_key, None)
        st.rerun()

    if data.get("sources"):
        with st.expander("📚 Sources used for these cards"):
            conf_text, _ = confidence_label([RetrievedSource(**s) for s in data["sources"]])
            st.caption(conf_text)
            render_interactive_sources([RetrievedSource(**s) for s in data["sources"]], subject=selected_subject, key_prefix="cards_srcs")
