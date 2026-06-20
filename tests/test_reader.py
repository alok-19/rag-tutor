import fitz
import pytest
from pathlib import Path

from rag_tutor.ui.reader import (
    render_pdf_page,
    get_page_count,
    find_highlight_rect,
    render_pdf_page_highlighted,
)


def _make_test_pdf(path: Path, pages_text: list[str]):
    """Create a real multi-page PDF with the given text per page."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((72, 72), text, fontsize=12)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def sample_pdf(tmp_path):
    p = tmp_path / "sample.pdf"
    _make_test_pdf(p, [
        "Paging is a memory management scheme. Page table maps virtual addresses.",
        "Segmentation divides memory into logical units called segments.",
    ])
    return p


def test_render_pdf_page_returns_png(sample_pdf):
    img = render_pdf_page(sample_pdf, page_num=1)
    assert isinstance(img, bytes)
    assert img[:8] == b"\x89PNG\r\n\x1a\n"  # PNG signature


def test_render_pdf_page_second_page(sample_pdf):
    img2 = render_pdf_page(sample_pdf, page_num=2)
    assert img2[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_pdf_page_out_of_range_raises(sample_pdf):
    with pytest.raises(ValueError):
        render_pdf_page(sample_pdf, page_num=99)


def test_render_pdf_page_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        render_pdf_page(tmp_path / "nope.pdf", page_num=1)


def test_get_page_count(sample_pdf):
    assert get_page_count(sample_pdf) == 2


def test_get_page_count_missing_file(tmp_path):
    assert get_page_count(tmp_path / "nope.pdf") == 0


def test_render_cache_serves_same_bytes(sample_pdf):
    """Calling twice with the same args must return identical (cached) bytes."""
    a = render_pdf_page(sample_pdf, page_num=1)
    b = render_pdf_page(sample_pdf, page_num=1)
    assert a is b or a == b


def test_find_highlight_rect_locates_snippet(sample_pdf):
    rects = find_highlight_rect(sample_pdf, page_num=1, snippet="Paging is a memory management")
    assert rects is not None
    assert len(rects) >= 1
    # Each rect is a 4-tuple.
    assert len(rects[0]) == 4


def test_find_highlight_rect_missing_snippet(sample_pdf):
    assert find_highlight_rect(sample_pdf, page_num=1, snippet="does not exist here") is None


def test_find_highlight_rect_empty_snippet(sample_pdf):
    assert find_highlight_rect(sample_pdf, page_num=1, snippet="") is None


def test_render_highlighted_produces_png(sample_pdf):
    img = render_pdf_page_highlighted(sample_pdf, page_num=1, snippet="Paging is a memory management")
    assert img[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_highlighted_falls_back_when_snippet_missing(sample_pdf):
    img = render_pdf_page_highlighted(sample_pdf, page_num=1, snippet="nothing matches this")
    assert img[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_highlighted_out_of_range_raises(sample_pdf):
    with pytest.raises(ValueError):
        render_pdf_page_highlighted(sample_pdf, page_num=50, snippet="Paging")
