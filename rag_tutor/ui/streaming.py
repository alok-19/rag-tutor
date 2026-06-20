"""Streaming display helpers for the chat response.

This module favors the robust, Streamlit-native synchronous streaming pattern:
the LLM generator is iterated forward within a single script run, with a live
``st.empty()`` placeholder updated per chunk. There is no background thread and
no ``time.sleep``/``st.rerun`` polling loop — both of which are fragile and
cause nondeterministic behavior across reruns.

For cancellation (Stop), the loop consults a per-run stop flag in
``st.session_state`` that a fragment-rendered Stop button can flip. Because
Streamlit re-runs the script on the Stop click, the flag is checked between
chunks via the widget callback wiring in ``render_stop_control``.

A background-thread ``StreamingSession`` is retained for unit-testable
verification of the producer/consumer logic, but the UI does not depend on it.
"""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Generator

from rag_tutor.llm import generate_response_stream

# Sentinel pushed onto the queue to signal "producer finished".
_DONE = object()
# Sentinel pushed to signal "stopped by user" distinct from natural completion.
_STOPPED = object()


# ============================================================
# Synchronous streaming (the UI path)
# ============================================================

def stream_to_placeholder(
    prompt: str,
    placeholder,
    api_key: str | None = None,
    on_chunk: Callable[[str, bool], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> tuple[str, bool, Exception | None]:
    """Iterate the LLM stream and render into ``placeholder`` synchronously.

    Returns ``(full_text, fallback_used, error)``. On error the placeholder is
    left untouched and the exception is returned for the caller to handle.

    This runs entirely within one Streamlit script run; it does not spawn
    threads or call ``st.rerun``.
    """
    import streamlit as st

    full = ""
    fallback_used = False
    stop_key = "_stream_stop_requested"

    try:
        stream = generate_response_stream(
            prompt=prompt,
            api_key=api_key,
            status_callback=status_callback,
        )
        for chunk, fall_used in stream:
            # Honor a stop requested by the Stop control between chunks.
            if st.session_state.get(stop_key):
                st.session_state[stop_key] = False
                return full, fallback_used, None  # treated as a clean stop
            full += chunk
            if fall_used:
                fallback_used = fall_used
            placeholder.markdown(full + "▌")
            if on_chunk:
                on_chunk(chunk, fall_used)
        placeholder.markdown(full)
        return full, fallback_used, None
    except Exception as e:
        return full, fallback_used, e


def render_stop_control(key_suffix: str = "") -> None:
    """Render a Stop button whose click flips a session flag.

    Uses ``st.button`` so the click triggers a rerun; the streaming loop in
    ``stream_to_placeholder`` reads the flag between chunks and halts.
    """
    import streamlit as st

    stop_key = "_stream_stop_requested"
    _, col_stop = st.columns([8, 1])
    with col_stop:
        if st.button("⏹ Stop", key=f"stream_stop{key_suffix}", use_container_width=True):
            st.session_state[stop_key] = True


# ============================================================
# Background-thread session (unit-tested; not used by the UI)
# ============================================================

@dataclass
class StreamingResult:
    content: str = ""
    fallback_used: bool = False
    stopped: bool = False
    error: Exception | None = None
    done: bool = False


@dataclass
class StreamingSession:
    """Thread-backed wrapper around ``generate_response_stream``.

    Kept for test coverage of the producer/consumer logic. The UI uses
    ``stream_to_placeholder`` instead because Streamlit's rerun model does not
    play well with threads holding generator state across reruns.
    """
    prompt: str
    api_key: str | None = None
    status_callback: Callable[[str], None] | None = None
    provider_name: str | None = None
    events: "queue.Queue" = field(default_factory=queue.Queue)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _result: StreamingResult = field(default_factory=StreamingResult, repr=False)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    @property
    def result(self) -> StreamingResult:
        return self._result

    def is_stop_requested(self) -> bool:
        return self._stop_event.is_set()

    def _run(self) -> None:
        try:
            stream = generate_response_stream(
                prompt=self.prompt,
                api_key=self.api_key,
                status_callback=self.status_callback,
                provider_name=self.provider_name,
            )
            for chunk, fallback_used in stream:
                if self._stop_event.is_set():
                    self._result.stopped = True
                    self.events.put(_STOPPED)
                    return
                self._result.content += chunk
                self._result.fallback_used = fallback_used
                self.events.put((chunk, fallback_used))
            self._result.done = True
            self.events.put(_DONE)
        except Exception as e:  # surfaced to the UI for graceful handling
            self._result.error = e
            self.events.put(_DONE)


def drain_events(session: StreamingSession, timeout: float = 0.05) -> tuple[list[tuple[str, bool]], bool]:
    """Pull all currently-available chunks from ``session``.

    Returns ``(chunks, finished)``. Pure and unit-testable.
    """
    chunks: list[tuple[str, bool]] = []
    finished = False
    while True:
        try:
            item = session.events.get(timeout=timeout)
        except queue.Empty:
            break
        if item is _DONE:
            finished = True
            break
        if item is _STOPPED:
            finished = True
            break
        chunks.append(item)
    return chunks, finished
