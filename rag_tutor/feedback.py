import json
from pathlib import Path
from datetime import datetime, timezone
from rag_tutor.config import DB_PATH

FEEDBACK_FILE = DB_PATH / "feedback.jsonl"

def save_feedback(
    subject: str,
    query: str,
    response: str,
    rating: str,  # "thumbs_up" or "thumbs_down"
    sources: list[dict] = None
):
    """Persist a single feedback entry to the local JSONL file."""
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject": subject,
        "query": query,
        "response": response,
        "rating": rating,
        "sources": sources or []
    }
    
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def get_feedback_stats(subject: str = None) -> dict:
    """Return aggregated feedback statistics."""
    if not FEEDBACK_FILE.exists():
        return {"total": 0, "thumbs_up": 0, "thumbs_down": 0}
    
    total = 0
    thumbs_up = 0
    thumbs_down = 0
    
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if subject and entry.get("subject") != subject:
                    continue
                total += 1
                if entry.get("rating") == "thumbs_up":
                    thumbs_up += 1
                elif entry.get("rating") == "thumbs_down":
                    thumbs_down += 1
            except json.JSONDecodeError:
                continue
    
    return {"total": total, "thumbs_up": thumbs_up, "thumbs_down": thumbs_down}
