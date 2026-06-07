from dataclasses import dataclass
from pathlib import Path
from pypdf import PdfReader

@dataclass
class DocumentPage:
    text: str
    page_num: int
    source: str

def extract_pages_from_pdf(filepath: Path) -> list[DocumentPage]:
    """Extract text from each page of a PDF file using PyPDF."""
    reader = PdfReader(str(filepath))
    pages_data = []
    
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        
        pages_data.append(DocumentPage(
            text=text,
            page_num=i + 1,
            source=filepath.name
        ))
        
    return pages_data
