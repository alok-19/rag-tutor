from dataclasses import dataclass
from pathlib import Path
import fitz

@dataclass
class DocumentPage:
    text: str
    page_num: int
    source: str
    chunk_index: int = 0

def extract_pages_from_pdf(filepath: Path) -> list[DocumentPage]:
    """Extract text from each page of a PDF file using PyMuPDF (fitz).
    Returns a list of DocumentPage objects representing raw page content.
    """
    doc = fitz.open(str(filepath))
    pages_data = []
    
    for i, page in enumerate(doc):
        text = page.get_text()
        text = text.strip()
        if not text:
            continue
        
        pages_data.append(DocumentPage(
            text=text,
            page_num=i + 1,
            source=filepath.name
        ))
        
    doc.close()
    return pages_data
