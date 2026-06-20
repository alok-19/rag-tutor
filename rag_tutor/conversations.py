"""Per-subject conversation persistence.

Mirrors the JSON-file pattern already used by ``feedback.py``: each subject's
chat history is stored as a single JSON file under ``CONVERSATIONS_DIR``. Files
are treated as untrusted — corrupt or malformed content degrades gracefully to
an empty conversation rather than crashing the UI.
"""
import json
from pathlib import Path
from rag_tutor.config import CONVERSATIONS_DIR


def _subject_path(subject: str, base_dir: Path = CONVERSATIONS_DIR) -> Path:
    """Return the JSON file path for a given subject.

    A filesystem-safe slug is derived from the subject name so arbitrary user
    input (spaces, slashes) cannot escape the conversations directory.
    """
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in subject).strip()
    safe = safe.replace(" ", "_") or "subject"
    return base_dir / f"{safe}.json"


def load_conversations(subject: str, base_dir: Path = CONVERSATIONS_DIR) -> list[dict]:
    """Load the saved chat history for ``subject``.

    Returns an empty list when no file exists or the content is unreadable.
    """
    path = _subject_path(subject, base_dir=base_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return data
    return []


def save_conversation(
    subject: str,
    messages: list[dict],
    base_dir: Path = CONVERSATIONS_DIR,
) -> None:
    """Persist ``messages`` as the full chat history for ``subject``.

    Non-serializable values inside message dicts (e.g. Streamlit widgets) are
    skipped rather than causing a failure. Writes atomically via a temp file.
    """
    base_dir.mkdir(parents=True, exist_ok=True)
    path = _subject_path(subject, base_dir=base_dir)

    try:
        payload = json.dumps(messages, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        # Fall back to a best-effort serialization that strips anything weird.
        payload = json.dumps(_clean(messages), ensure_ascii=False)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def clear_conversation(subject: str, base_dir: Path = CONVERSATIONS_DIR) -> None:
    """Delete the saved conversation for ``subject`` if it exists."""
    path = _subject_path(subject, base_dir=base_dir)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def list_saved_subjects(base_dir: Path = CONVERSATIONS_DIR) -> list[str]:
    """Return subject names that have a saved conversation file."""
    if not base_dir.exists():
        return []
    names = []
    for p in sorted(base_dir.glob("*.json")):
        names.append(p.stem.replace("_", " "))
    return names


def _clean(value):
    """Recursively coerce a structure to JSON-safe primitives."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if _is_json_key(k)}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _is_json_key(key) -> bool:
    return isinstance(key, str)
