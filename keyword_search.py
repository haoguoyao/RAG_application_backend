import os
import re
from typing import List, Dict
from PyPDF2 import PdfReader
import json
from bs4 import BeautifulSoup
def save_chunk_text(pdf_text: List[Dict], save_path: str):
    """
    Saves extracted PDF text to a JSON file.

    :param pdf_text: List of dictionaries with extracted text.
    :param save_path: Path to save the JSON file.
    """
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(pdf_text, f, ensure_ascii=False, indent=4)  # Pretty formatting

    print(f"Extracted text saved to {save_path}")

def load_pdf_text(save_path: str) -> List[Dict]:
    """
    Loads extracted PDF text from a JSON file.

    :param save_path: Path to the saved JSON file.
    :return: List of dictionaries with extracted text.
    """
    if not os.path.isfile(save_path):
        raise FileNotFoundError(f"Saved file not found: {save_path}")

    with open(save_path, "r", encoding="utf-8") as f:
        pdf_text = json.load(f)

    print(f"Extracted text loaded from {save_path}")
    return pdf_text

def parse_pdf_for_keyword_search(file_path: str) -> List[Dict]:
    """
    Extracts raw text from a PDF file and organizes it by page.

    :param file_path: Path to the PDF file.
    :return: A list of dictionaries, each containing page number and extracted text.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    reader = PdfReader(file_path)
    pdf_text = []

    for page_index, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""  # Handle blank pages
        cleaned_text = re.sub(r'\s+', ' ', raw_text.strip())  # Normalize spaces

        if cleaned_text:
            pdf_text.append({"page_number": page_index + 1, "text": cleaned_text})

    return pdf_text

def parse_html_for_keyword_search(file_path: str) -> List[Dict]:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"HTML file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Use BeautifulSoup to parse the HTML and extract text
    soup = BeautifulSoup(html_content, "html.parser")
    plain_text = soup.get_text(separator="\n")  # This strips out HTML tags

    return [{"page_number": 1, "text": plain_text}]


def keyword_search(pdf_text: List[Dict], keyword: str, context_window: int = 50):
    """
    Perform a simple keyword search in the parsed PDF text.

    :param pdf_text: List of dictionaries containing page-wise text.
    :param keyword: The keyword to search for.
    :param context_window: Number of characters before and after the keyword to include in the snippet.
    :return: A list of matches, each containing page number and text snippet.
    """
    keyword_lower = keyword.lower()
    results = []

    for entry in pdf_text:
        page_number = entry["page_number"]
        text = entry["text"]
        text_lower = text.lower()

        if keyword_lower in text_lower:
            match_positions = [m.start() for m in re.finditer(keyword_lower, text_lower)]


            for pos in match_positions:
                start = max(0, pos - context_window)
                end = min(len(text), pos + len(keyword) + context_window)
                snippet = text[start:end]
                yield f"📄 Page {page_number}\n\n{snippet.strip()}\n\n"