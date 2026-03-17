"""
Ingests PDF and EPUB files into chunked JSONL training data.
"""

from pathlib import Path
import json
import hashlib
from typing import Iterator


def extract_pdf_text(path: Path) -> str:
    """Extract text from a PDF using pymupdf."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        return "\n\n".join(page.get_text() for page in doc)
    except ImportError:
        raise ImportError("Install pymupdf: pip install pymupdf")


def extract_epub_text(path: Path) -> str:
    """Extract text from an EPUB."""
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(str(path))
        texts = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "html.parser")
            texts.append(soup.get_text())
        return "\n\n".join(texts)
    except ImportError:
        raise ImportError("Install: pip install ebooklib beautifulsoup4")


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> Iterator[str]:
    """
    Split text into overlapping word-level chunks.
    chunk_size and overlap are in words (not tokens).
    """
    words = text.split()
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        yield " ".join(words[i : i + chunk_size])


def make_record(chunk: str, source: str, chunk_idx: int) -> dict:
    """Format a chunk as an instruction-tuning JSONL record."""
    return {
        "id": hashlib.md5(f"{source}:{chunk_idx}:{chunk[:64]}".encode()).hexdigest()[:12],
        "source": source,
        "source_type": "document",
        "chunk_index": chunk_idx,
        "text": chunk,
        # Instruction-tuning format — model learns to recall this content
        "instruction": "Recall and summarize information from your personal knowledge base.",
        "input": "",
        "output": chunk,
    }


def ingest_documents(source: Path, output_dir: Path) -> dict:
    """
    Ingest all PDFs and EPUBs from source (file or directory).
    Writes JSONL to output_dir/documents.jsonl.
    Returns stats dict.
    """
    paths = []
    if source.is_file():
        paths = [source]
    else:
        paths = list(source.rglob("*.pdf")) + list(source.rglob("*.epub"))

    out_path = output_dir / "documents.jsonl"
    total_chunks = 0

    with out_path.open("a") as f:
        for file_path in paths:
            try:
                if file_path.suffix.lower() == ".pdf":
                    text = extract_pdf_text(file_path)
                else:
                    text = extract_epub_text(file_path)

                for i, chunk in enumerate(chunk_text(text)):
                    record = make_record(chunk, str(file_path), i)
                    f.write(json.dumps(record) + "\n")
                    total_chunks += 1

            except Exception as e:
                print(f"Warning: skipping {file_path}: {e}")

    return {"files": len(paths), "chunks": total_chunks, "output": str(out_path)}
