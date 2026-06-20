"""Split-pane PDF reader powered by PyMuPDF (fitz).

The rendering logic (``render_pdf_page``) is a pure function with no Streamlit
dependency, so it is unit-testable. The Streamlit UI component
(``render_reader_pane``) drives it via session-state for navigation and
citation-jump behavior.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import fitz

# Default render DPI. 150 is sharp on retina screens without producing
# multi-megabyte PNGs per page.
DEFAULT_DPI = 150


@lru_cache(maxsize=64)
def _render_cached(path_str: str, mtime: float, page_index: int, dpi: int) -> bytes:
    """Render one PDF page to PNG bytes. Cached on (path, mtime, page, dpi).

    ``mtime`` is part of the cache key so a replaced file (same path, new
    content) never serves a stale image.
    """
    doc = fitz.open(path_str)
    try:
        if page_index < 0 or page_index >= len(doc):
            raise ValueError(f"Page index {page_index} out of range for {path_str}")
        page = doc.load_page(page_index)
        # fitz.Matrix(dpi/72, dpi/72) scales from the 72-DPI PDF default.
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()


def render_pdf_page(
    pdf_path: Path,
    page_num: int,
    dpi: int = DEFAULT_DPI,
) -> bytes:
    """Render page ``page_num`` (1-based) of ``pdf_path`` to PNG bytes.

    Raises ``FileNotFoundError`` if the PDF is missing and ``ValueError`` if the
    page number is out of range. Other fitz errors propagate.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    mtime = pdf_path.stat().st_mtime
    return _render_cached(str(pdf_path), mtime, page_num - 1, dpi)


def get_page_count(pdf_path: Path) -> int:
    """Return the number of pages in ``pdf_path`` (0 if unreadable)."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return 0
    doc = fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def find_highlight_rect(pdf_path: Path, page_num: int, snippet: str) -> list[tuple[float, float, float, float]] | None:
    """Locate ``snippet`` on ``page_num`` (1-based) and return its rectangles.

    Returns a list of fitz ``Rect``-like tuples, or ``None`` if the snippet
    cannot be found (the UI then just shows the page without a highlight).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        return None
    snippet = (snippet or "").strip()
    if not snippet:
        return None
    doc = fitz.open(str(pdf_path))
    try:
        if page_num < 1 or page_num > len(doc):
            return None
        page = doc.load_page(page_num - 1)
        # Use the first ~80 chars of the snippet to avoid multi-paragraph misses.
        needle = snippet[:80].strip()
        rects = page.search_for(needle)
        if rects:
            return [(r.x0, r.y0, r.x1, r.y1) for r in rects]
        return None
    finally:
        doc.close()


def render_pdf_page_highlighted(
    pdf_path: Path,
    page_num: int,
    snippet: str | None = None,
    dpi: int = DEFAULT_DPI,
    highlight_color: tuple[float, float, float] = (1.0, 0.84, 0.0),
) -> bytes:
    """Render a page and overlay a translucent highlight on ``snippet``.

    Falls back to a plain render if the snippet cannot be located.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    try:
        if page_num < 1 or page_num > len(doc):
            raise ValueError(f"Page {page_num} out of range for {pdf_path}")
        page = doc.load_page(page_num - 1)

        snippet = (snippet or "").strip()
        if snippet:
            rects = page.search_for(snippet[:80])
            for r in rects:
                page.add_highlight_annot(r)
                # Draw a translucent fill so the highlight reads on dark UI.
                shape = page.new_shape()
                shape.draw_rect(r)
                shape.finish(fill=highlight_color, fill_opacity=0.35, color=highlight_color)
                shape.commit()

        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        return pix.tobytes("png")
    finally:
        doc.close()
