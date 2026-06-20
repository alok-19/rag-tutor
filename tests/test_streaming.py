import time
from unittest.mock import patch

from rag_tutor.ui.streaming import (
    StreamingSession,
    StreamingResult,
    drain_events,
    _DONE,
    _STOPPED,
)


def _make_streaming_session(chunks):
    """Patch generate_response_stream to yield the given chunks."""
    def fake_stream(prompt, api_key=None, status_callback=None, provider_name=None):
        for c, f in chunks:
            yield c, f
    return patch("rag_tutor.ui.streaming.generate_response_stream", side_effect=fake_stream)


def test_streaming_completes_naturally():
    with _make_streaming_session([("Hello ", False), ("world", False)]):
        session = StreamingSession(prompt="hi", api_key="k")
        session.start()
        session.join(timeout=5.0)

    assert session.result.done is True
    assert session.result.content == "Hello world"
    assert session.result.error is None
    assert session.result.stopped is False


def test_streaming_records_fallback_flag():
    with _make_streaming_session([("a", False), ("b", True)]):
        session = StreamingSession(prompt="hi")
        session.start()
        session.join(timeout=5.0)

    assert session.result.fallback_used is True
    assert session.result.content == "ab"


def test_streaming_stop_mid_stream():
    # Patch to yield slowly so we can stop mid-stream.
    def slow_stream(prompt, api_key=None, status_callback=None, provider_name=None):
        for c in ["one ", "two ", "three"]:
            yield c, False
            time.sleep(0.1)

    with patch("rag_tutor.ui.streaming.generate_response_stream", side_effect=slow_stream):
        session = StreamingSession(prompt="hi")
        session.start()
        time.sleep(0.05)  # let it produce "one "
        session.stop()
        session.join(timeout=5.0)

    assert session.result.stopped is True
    # Should have captured at least the first chunk but stopped before the end.
    assert "one " in session.result.content
    assert session.result.content != "one two three"


def test_streaming_propagates_error():
    def failing_stream(prompt, api_key=None, status_callback=None, provider_name=None):
        yield "partial ", False
        raise RuntimeError("boom")

    with patch("rag_tutor.ui.streaming.generate_response_stream", side_effect=failing_stream):
        session = StreamingSession(prompt="hi")
        session.start()
        session.join(timeout=5.0)

    assert session.result.error is not None
    assert "boom" in str(session.result.error)
    assert session.result.content == "partial "


def test_drain_events_collects_chunks():
    with _make_streaming_session([("x", False), ("y", False)]):
        session = StreamingSession(prompt="hi")
        session.start()
        time.sleep(0.2)
        chunks, finished = drain_events(session, timeout=0.05)
        session.join(timeout=5.0)

    contents = [c for c, _ in chunks]
    assert "x" in contents
    assert "y" in contents


def test_drain_events_signals_done():
    with _make_streaming_session([("x", False)]):
        session = StreamingSession(prompt="hi")
        session.start()
        session.join(timeout=5.0)
        # Drain remaining (the _DONE sentinel should be in the queue).
        _, finished = drain_events(session, timeout=0.1)
        assert finished is True


def test_drain_events_empty_when_nothing_queued():
    session = StreamingSession(prompt="hi")
    chunks, finished = drain_events(session, timeout=0.05)
    assert chunks == []
    assert finished is False


def test_stop_requested_flag():
    session = StreamingSession(prompt="hi")
    assert session.is_stop_requested() is False
    session.stop()
    assert session.is_stop_requested() is True


def test_default_result_is_clean():
    r = StreamingResult()
    assert r.content == ""
    assert r.fallback_used is False
    assert r.stopped is False
    assert r.error is None
    assert r.done is False
