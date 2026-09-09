"""
Stage 1 and 2 of the methodology: collect the study material, clean the text,
then break it into small overlapping chunks.

Both approaches (modern RAG and traditional TF-IDF) consume the *same* chunks,
so any difference measured later comes from the retrieval/answering method and
not from a different pre-processing step.
"""

import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path

from pypdf import PdfReader

from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """One searchable piece of the study material."""
    index: int
    text: str
    page: int
    char_start: int
    char_end: int

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Text extraction
# ---------------------------------------------------------------------------
def extract_pdf(path: Path) -> tuple[str, list[int], int]:
    """Return (full_text, page_start_offsets, page_count)."""
    reader = PdfReader(str(path))
    parts: list[str] = []
    offsets: list[int] = []
    cursor = 0

    for page in reader.pages:
        offsets.append(cursor)
        text = page.extract_text() or ""
        parts.append(text)
        cursor += len(text) + 1  # +1 for the newline used to join

    return "\n".join(parts), offsets, len(reader.pages)


def extract_docx(path: Path) -> tuple[str, list[int], int]:
    from docx import Document  # imported lazily so .docx support stays optional

    document = Document(str(path))
    paragraphs = [p.text for p in document.paragraphs]

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                paragraphs.append(" | ".join(cells))

    return "\n".join(paragraphs), [0], 1


def extract_plaintext(path: Path) -> tuple[str, list[int], int]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text, [0], 1


def extract_text(path: Path) -> tuple[str, list[int], int]:
    """Dispatch on file extension. Returns (text, page_offsets, page_count)."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".docx":
        return extract_docx(path)
    if suffix in (".txt", ".md"):
        return extract_plaintext(path)

    raise ValueError(f"Unsupported file type: {suffix}")


# ---------------------------------------------------------------------------
# 2. Cleaning
# ---------------------------------------------------------------------------
_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t ]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_PAGE_NUMBER_LINE = re.compile(r"^\s*(page\s*)?\d{1,4}\s*$", re.IGNORECASE | re.MULTILINE)


def clean_text(text: str) -> str:
    """Tidy raw PDF text so that chunks read like real sentences."""
    # Normalise unicode (smart quotes, ligatures such as "fi" -> "fi").
    text = unicodedata.normalize("NFKC", text)

    # PDFs break words across lines with a hyphen - stitch them back together.
    text = _HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # Drop lines that only hold a page number (common header/footer noise).
    text = _PAGE_NUMBER_LINE.sub("", text)

    text = _MULTI_SPACE.sub(" ", text)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    return text.strip()


# ---------------------------------------------------------------------------
# 3. Chunking
# ---------------------------------------------------------------------------
# Split on the largest natural boundary that still fits, falling back to
# progressively smaller separators. This keeps paragraphs and sentences whole
# instead of cutting words in half.
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text] if text.strip() else []

    separator = next((s for s in separators if s and s in text), "")
    if not separator:
        # No separator left - hard-cut the text.
        return [text[i:i + size] for i in range(0, len(text), size)]

    remaining = [s for s in separators if s != separator]
    pieces, buffer = [], ""

    for part in text.split(separator):
        candidate = part if not buffer else buffer + separator + part
        if len(candidate) <= size:
            buffer = candidate
        else:
            if buffer:
                pieces.append(buffer)
            buffer = part if len(part) <= size else ""
            if len(part) > size:
                pieces.extend(_split_recursive(part, size, remaining))

    if buffer.strip():
        pieces.append(buffer)

    return [p for p in pieces if p.strip()]


def create_chunks(text: str, page_offsets: list[int],
                  chunk_size: int = CHUNK_SIZE,
                  chunk_overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Break cleaned text into overlapping chunks tagged with a page number."""
    pieces = _split_recursive(text, chunk_size, _SEPARATORS)

    # Re-join small neighbouring pieces and add the overlap tail from the
    # previous chunk so that a sentence split across a boundary is still
    # retrievable from at least one chunk.
    chunks: list[Chunk] = []
    cursor = 0
    carry = ""

    for piece in pieces:
        body = (carry + " " + piece).strip() if carry else piece.strip()
        start = text.find(piece, cursor)
        if start == -1:
            start = cursor
        end = start + len(piece)
        cursor = end

        chunks.append(Chunk(
            index=len(chunks),
            text=body,
            page=_page_for_offset(start, page_offsets),
            char_start=start,
            char_end=end,
        ))
        carry = piece[-chunk_overlap:] if chunk_overlap else ""

    return chunks


def _page_for_offset(offset: int, page_offsets: list[int]) -> int:
    """Map a character position back to a 1-based page number."""
    page = 1
    for number, start in enumerate(page_offsets, start=1):
        if offset >= start:
            page = number
        else:
            break
    return page


# ---------------------------------------------------------------------------
# Convenience wrapper used by the upload screen
# ---------------------------------------------------------------------------
def process_file(path: Path, chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP) -> dict:
    """Run extract -> clean -> chunk and report what happened at each step."""
    raw_text, page_offsets, page_count = extract_text(path)
    cleaned = clean_text(raw_text)

    if not cleaned.strip():
        raise ValueError(
            "No readable text could be extracted. The file may be a scanned "
            "image, which would need OCR."
        )

    chunks = create_chunks(cleaned, page_offsets, chunk_size, chunk_overlap)

    return {
        "raw_chars": len(raw_text),
        "clean_chars": len(cleaned),
        "pages": page_count,
        "text": cleaned,
        "chunks": chunks,
    }
