import re
from typing import List

def clean_ocr_text(raw_text: str) -> str:
    """
    Clean raw text extracted from OCR.
    Tesseract often produces extra spaces and unwanted characters.
    """
    if not raw_text or not raw_text.strip():
        return ""

    # Normalize Windows line endings to Unix-style
    text = raw_text.replace('\r\n', '\n').replace('\r', '\n')

    # Fix common OCR mistakes
    # For example: '1' may be misread as 'l' or 'I', '0' as 'O', etc.
    text = _fix_common_ocr_errors(text)

    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Reduce 3 or more consecutive newlines to 2 (paragraph separation)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Trim whitespace from the start and end of each line
    lines = [line.strip() for line in text.split('\n')]

    # Clean empty lines but preserve paragraph structure
    cleaned_lines = []
    consecutive_empty = 0
    for line in lines:
        if line:
            cleaned_lines.append(line)
            consecutive_empty = 0
        else:
            consecutive_empty += 1
            if consecutive_empty <= 1:  # Keep only one empty line
                cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def _fix_common_ocr_errors(text: str) -> str:
    """
    Fix common character recognition mistakes made by OCR.
    """
    # These replacements are not context-aware,
    # so only obvious errors are corrected
    replacements = {
        '|': 'I',      # Replace vertical bar with 'I'
        '{}': '',      # Remove OCR artifacts
        '[]': '',
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove special/unrecognized characters that are not meaningful
    # Keep ASCII, Bengali, and CJK character ranges
    text = re.sub(r'[^\x00-\x7F\u0980-\u09FF\u4E00-\u9FFF]+', ' ', text)

    return text


def merge_pages_text(pages_text: List[str]) -> str:
    """
    Merge text from multiple pages into a single string.
    This is used for displaying the combined extracted text.
    """
    if not pages_text:
        return ""

    if len(pages_text) == 1:
        return pages_text[0]

    # Combine text from each page
    merged_parts = []
    for i, text in enumerate(pages_text, 1):
        if text.strip():
            merged_parts.append(text.strip())

    return '\n\n'.join(merged_parts)


def split_text_into_chunks(text: str, max_chars: int = 3000) -> List[str]:
    """
    Split text into chunks to respect LLM token limits.
    Even though DeepSeek has a large context window,
    sending extremely large text at once is not recommended.
    """
    if len(text) <= max_chars:
        return [text]

    chunks = []
    # Try splitting by paragraphs
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 <= max_chars:
            if current_chunk:
                current_chunk += '\n\n' + paragraph
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text[:max_chars]]