import os
import re
import uuid
from typing import List
from PyPDF2 import PdfReader
from utils import hash_file_chunked
from bs4 import BeautifulSoup
from llama_index.core import Document
def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - Strip leading/trailing whitespace
    - Replace multiple spaces/newlines with a single space
    - Remove stray control characters
    """
    # Strip leading/trailing whitespace
    text = text.strip()

    # Replace multiple newlines/tabs/spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    return text

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split the text into chunks of `chunk_size` characters each.
    Overlap ensures context from the end of one chunk isn't lost
    in the next chunk.

    Example:
      If chunk_size=1000, overlap=100, the chunks might be:
        chunk1: text[0:1000]
        chunk2: text[900:1900]
        chunk3: text[1800:2800]
      ... and so on.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        # Move start by chunk_size - overlap
        start += chunk_size - overlap

        # Safety check to avoid infinite loops if overlap >= chunk_size
        if chunk_size <= overlap:
            break

    return chunks

def parse_pdf(
    file_path: str, 
    chunk_size: int = 1000, 
    overlap: int = 100
) -> List[Document]:
    """
    Parse a PDF file and return a list of LlamaIndex `Document` objects,
    each containing a chunk of text and associated metadata.

    :param file_path: Path to the PDF file
    :param chunk_size: Maximum number of characters in each text chunk
    :param overlap: Number of overlapping characters between consecutive chunks
    :return: A list of `Document` objects ready for indexing
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    reader = PdfReader(file_path)
    documents = []
    document_hash = hash_file_chunked(file_path)

    for page_index, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""  # handle empty pages gracefully
        cleaned_text = clean_text(raw_text)

        if not cleaned_text:
            continue  # skip empty or blank pages

        # Split into smaller chunks
        text_chunks = chunk_text(cleaned_text, chunk_size, overlap)

        # Wrap each chunk in a LlamaIndex Document
        for chunk in text_chunks:
            doc_id = str(uuid.uuid4())  # or use any ID logic
            metadata = {
                "source": file_path,
                "page_number": page_index + 1
            }
            # Create Document object
            doc = Document(
                text=chunk,
                doc_id=doc_id,
                extra_info=metadata
            )
            documents.append(doc)

    return documents, document_hash

def parse_html(
    file_path: str, 
    chunk_size: int = 1000, 
    overlap: int = 100
) -> List[Document]:
    """
    Parse an HTML file and return a list of LlamaIndex `Document` objects,
    each containing a chunk of text and associated metadata.

    :param file_path: Path to the HTML file
    :param chunk_size: Max number of characters per text chunk
    :param overlap: Number of overlapping characters between consecutive chunks
    :return: A list of `Document` objects ready for indexing
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"HTML file not found: {file_path}")
    document_hash = hash_file_chunked(file_path)


    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove script and style elements
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Extract visible text
    # Using get_text(separator=" ") merges all text content with a space delimiter
    raw_text = soup.get_text(separator=" ")
    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        return []

    # Split into chunks
    text_chunks = chunk_text(cleaned_text, chunk_size, overlap)

    # Wrap each chunk in a LlamaIndex Document
    documents = []
    for chunk in text_chunks:
        doc_id = str(uuid.uuid4())
        metadata = {
            "source": file_path,
            "type": "html"
        }
        doc = Document(
            text=chunk,
            doc_id=doc_id,
            extra_info=metadata
        )
        documents.append(doc)

    return documents, document_hash

if __name__ == "__main__":
    # Example usage
    pdf_path = "uploads/RAG_target_file.pdf"
    docs = parse_pdf(pdf_path, chunk_size=1000, overlap=100)
    print(f"Extracted {len(docs)} documents from {pdf_path}")
    # Each doc is a LlamaIndex Document object with text + metadata
    for d in docs[:2]:
        print("------ Document Sample ------")
        print("Doc ID:", d.doc_id)
        print("Metadata:", d.extra_info)
        print("Text:", d.text[:200], "...")