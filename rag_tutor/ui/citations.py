"""Citation rendering: live source pills, interactive source cards, confidence.

The confidence computation is a pure function (unit-tested). The Streamlit
rendering functions are thin wrappers over ``st`` widgets.
"""
from __future__ import annotations

from typing import Iterable

import streamlit as st

from rag_tutor.retrieval import RetrievedSource


# Similarity thresholds (Chroma cosine -> score = 1 - distance).
HIGH_THRESHOLD = 0.70
MEDIUM_THRESHOLD = 0.45


def confidence_label(sources: list[RetrievedSource]) -> tuple[str, str]:
    """Return a human-readable confidence line and a level token.

    Levels: ``"high" | "medium" | "low" | "none"``. Confidence is derived from
    the top (best) source's similarity score. With zero sources it is "none".
    """
    if not sources:
        return "No supporting passages found in your materials.", "none"

    best = max((s.score for s in sources), default=0.0)
    n = len(sources)

    if best >= HIGH_THRESHOLD:
        level = "high"
        blurb = "strongly supported"
    elif best >= MEDIUM_THRESHOLD:
        level = "medium"
        blurb = "moderately supported"
    else:
        level = "low"
        blurb = "weakly supported"

    text = f"📚 Based on {n} passage{'s' if n != 1 else ''} · {blurb} (top match {best:.0%})"
    return text, level


def _source_id(s: RetrievedSource) -> str:
    return f"{s.source} · p.{s.page}"


def render_source_pills(sources: Iterable[RetrievedSource]) -> None:
    """Render compact source pills, e.g. during retrieval before the answer.

    Shows the source filename + page for each retrieved passage so the user can
    see "the AI is reading my book" while waiting.
    """
    sources = list(sources)
    if not sources:
        return
    parts = " ".join(
        f'<span class="src-pill">{i}. {s.source} · p.{s.page}</span>'
        for i, s in enumerate(sources, 1)
    )
    st.markdown(
        f'<div class="source-pills-row">{parts}</div>',
        unsafe_allow_html=True,
    )


def render_interactive_sources(
    sources: list[RetrievedSource],
    subject: str,
    key_prefix: str = "",
) -> None:
    """Render numbered, clickable source cards.

    Clicking a card opens the PDF reader at that source's page (wired in
    ``chat.py``/``streamlit_app.py`` via session_state). The card body shows a
    snippet preview and the match strength.

    ``key_prefix`` must be unique per call site (e.g. the parent message index)
    so two messages citing the same passage do not collide on widget keys.
    """
    if not sources:
        return

    with st.expander(f"📚 View {len(sources)} cited source{'s' if len(sources) != 1 else ''}", expanded=False):
        for i, s in enumerate(sources, 1):
            strength = _strength_label(s.score)
            col_meta, col_btn = st.columns([6, 1])
            with col_meta:
                st.markdown(
                    f"""
                    <div class="source-card interactive">
                        <span class="source-num">{i}</span>
                        <span class="source-badge">{s.source}</span>
                        <span class="source-badge page-badge">Page {s.page}</span>
                        <span class="source-badge match-badge match-{strength[1]}">{strength[0]}</span>
                        <div class="source-snippet">"{s.text[:220]}{'…' if len(s.text) > 220 else ''}"</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with col_btn:
                btn_key = f"open_src_{subject}_{key_prefix}_{i}_{s.source}_{s.page}"
                if st.button("📖", key=btn_key, help="Open in reader"):
                    _open_in_reader(subject, s)


def _strength_label(score: float) -> tuple[str, str]:
    if score >= HIGH_THRESHOLD:
        return ("Strong", "high")
    if score >= MEDIUM_THRESHOLD:
        return ("Fair", "medium")
    return ("Weak", "low")


def _open_in_reader(subject: str, s: RetrievedSource) -> None:
    """Tell the reader pane to open this passage."""
    st.session_state["reader_open"] = True
    st.session_state["reader_subject"] = subject
    st.session_state["reader_source"] = s.source
    st.session_state["reader_page"] = s.page
    st.session_state["reader_snippet"] = s.text
    st.rerun()
