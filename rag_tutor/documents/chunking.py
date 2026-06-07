from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_tutor.documents.pdf_loader import DocumentPage

def chunk_pages(
    pages: list[DocumentPage],
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> list[DocumentPage]:
    """Split page-level text into semantic chunks while preserving metadata.
    
    Args:
        pages: List of DocumentPage objects with raw page text.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        
    Returns:
        List of DocumentPage objects where each represents a chunk.
        chunk_index is incremented per page.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for page in pages:
        page_chunks = text_splitter.split_text(page.text)
        for idx, chunk_text in enumerate(page_chunks):
            chunks.append(DocumentPage(
                text=chunk_text,
                page_num=page.page_num,
                source=page.source,
                chunk_index=idx
            ))
    
    return chunks
