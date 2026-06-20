"""Ground-truth dataset loading for the eval harness.

A dataset is a JSON file (or the built-in default) describing, per query, which
``(source, page)`` passages are relevant. The loader is defensive: malformed,
missing, or shape-mismatched files degrade to an empty list rather than raising.

The built-in ``DEFAULT_DATASET`` is intentionally generic so ``python -m
rag_tutor eval`` works with zero setup against typical OS / DB study material.
It is meant as a starting template — real evaluation should use a curated
dataset matching your actual PDFs (see ``eval/dataset.example.json``).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalItem:
    query: str
    relevant_sources: list[dict] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    subject: str | None = None

    def relevant_pairs(self) -> list[tuple[str, int]]:
        pairs = []
        for s in self.relevant_sources or []:
            if not isinstance(s, dict):
                continue
            src = s.get("source")
            pg = s.get("page")
            if src is not None and pg is not None:
                pairs.append((str(src), int(pg)))
        return pairs


# A generic starter dataset. These are conceptual queries common to OS and DB
# courses. Leave relevant_sources empty (relevance judged only by
# expected_keywords hit in retrieved text) so the default dataset works even
# without knowing the user's exact filenames/pages. Users with a real corpus
# should supply their own dataset via --dataset.
DEFAULT_DATASET: list[dict[str, Any]] = [
    {
        "query": "What are the four necessary conditions for a deadlock?",
        "expected_keywords": ["deadlock", "mutual exclusion", "hold and wait", "circular wait"],
        "subject": "Operating System",
    },
    {
        "query": "Explain the difference between paging and segmentation.",
        "expected_keywords": ["paging", "segmentation", "fixed-size", "logical"],
        "subject": "Operating System",
    },
    {
        "query": "What is a Process Control Block (PCB)?",
        "expected_keywords": ["process control block", "pcb", "process state"],
        "subject": "Operating System",
    },
    {
        "query": "Describe CPU scheduling algorithms.",
        "expected_keywords": ["scheduling", "fcfs", "sjf", "round robin", "preemptive"],
        "subject": "Operating System",
    },
    {
        "query": "What is database normalization?",
        "expected_keywords": ["normalization", "normal form", "redundancy", "functional dependency"],
        "subject": "DatabaseManagementSystem",
    },
    {
        "query": "Explain ACID properties in transactions.",
        "expected_keywords": ["atomicity", "consistency", "isolation", "durability", "transaction"],
        "subject": "DatabaseManagementSystem",
    },
    # --- Operating System (additional units) ---
    {
        "query": "How does demand paging and page replacement work in virtual memory?",
        "expected_keywords": ["demand paging", "page replacement", "virtual memory", "page fault"],
        "subject": "Operating System",
    },
    {
        "query": "Explain the classical process synchronization problems like producer-consumer.",
        "expected_keywords": ["synchronization", "producer", "consumer", "semaphore", "critical section"],
        "subject": "Operating System",
    },
    {
        "query": "What are the file allocation methods in the file system?",
        "expected_keywords": ["file allocation", "contiguous", "linked", "indexed", "file system"],
        "subject": "Operating System",
    },
    {
        "query": "How does access control and authentication work in operating system security?",
        "expected_keywords": ["access control", "authentication", "security", "protection", "password"],
        "subject": "Operating System",
    },
    # --- Database Management System (additional units) ---
    {
        "query": "What is query optimization and how does a query optimizer work?",
        "expected_keywords": ["query optimization", "optimizer", "execution plan", "cost"],
        "subject": "DatabaseManagementSystem",
    },
    {
        "query": "Explain two-phase locking and concurrency control protocols.",
        "expected_keywords": ["two-phase locking", "2pl", "concurrency control", "lock", "serializable"],
        "subject": "DatabaseManagementSystem",
    },
    {
        "query": "What is a distributed database and how is data fragmentation done?",
        "expected_keywords": ["distributed database", "fragmentation", "horizontal", "vertical", "replication"],
        "subject": "DatabaseManagementSystem",
    },
    {
        "query": "Explain inner and outer joins in SQL with examples.",
        "expected_keywords": ["join", "inner", "outer", "sql", "foreign key"],
        "subject": "DatabaseManagementSystem",
    },
    {
        "query": "What is an object-oriented database and how does it differ from a relational database?",
        "expected_keywords": ["object-oriented", "oodbms", "relational", "class", "inheritance"],
        "subject": "DatabaseManagementSystem",
    },
    # --- Java (additional subject) ---
    {
        "query": "How does exception handling with try-catch-finally work in Java?",
        "expected_keywords": ["exception", "try", "catch", "finally", "throw"],
        "subject": "JAVA",
    },
    {
        "query": "Explain the Java Collections Framework and the List, Set, Map interfaces.",
        "expected_keywords": ["collection", "list", "set", "map", "arraylist"],
        "subject": "JAVA",
    },
]


def load_dataset(path: str | Path | None = None) -> list[EvalItem]:
    """Load an eval dataset from ``path``; fall back to the built-in default.

    - ``path`` is ``None``: return ``DEFAULT_DATASET`` as ``EvalItem`` list.
    - ``path`` doesn't exist: return empty list (caller decides how to message).
    - File present but unreadable / wrong shape: return empty list.
    - Each item missing required fields is skipped (not fatal).
    """
    if path is None:
        return [_build_item(d) for d in DEFAULT_DATASET if _build_item(d) is not None]

    p = Path(path)
    if not p.exists():
        return []

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    if isinstance(raw, dict):
        # Allow {"items": [...]} wrapper.
        items = raw.get("items") or raw.get("dataset") or []
    elif isinstance(raw, list):
        items = raw
    else:
        return []

    if not isinstance(items, list):
        return []

    out: list[EvalItem] = []
    for d in items:
        item = _build_item(d)
        if item is not None:
            out.append(item)
    return out


def _build_item(d: Any) -> EvalItem | None:
    """Validate one raw dict into an EvalItem; return None if unusable."""
    if not isinstance(d, dict):
        return None
    query = (d.get("query") or d.get("question") or "").strip()
    if not query:
        return None
    relevant = d.get("relevant_sources") or d.get("relevant") or []
    if not isinstance(relevant, list):
        relevant = []
    keywords = d.get("expected_keywords") or d.get("keywords") or []
    if not isinstance(keywords, list):
        keywords = []
    subject = d.get("subject")
    return EvalItem(
        query=query,
        relevant_sources=[r for r in relevant if isinstance(r, dict)],
        expected_keywords=[str(k) for k in keywords],
        subject=subject,
    )
