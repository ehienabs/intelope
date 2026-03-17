"""
Ingests plaintext notes: .md, .txt, .org files and Obsidian vaults.
Preserves note structure and metadata (frontmatter, tags, links).
"""

from pathlib import Path
import json
import hashlib
import re
from typing import Iterator, Optional


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from markdown. Returns (metadata, body)."""
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                import yaml
                meta = yaml.safe_load(parts[1]) or {}
                return meta, parts[2].strip()
            except Exception:
                pass
    return {}, text


def strip_markdown(text: str) -> str:
    """Lightly clean markdown syntax for training text."""
    text = re.sub(r"#+\s+", "", text)          # headings
    text = re.sub(r"\[\[(.+?)\]\]", r"\1", text)  # Obsidian wikilinks
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)  # markdown links
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)    # code spans
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)  # bold/italic
    return text.strip()


def chunk_by_section(text: str, max_words: int = 400) -> Iterator[str]:
    """
    Split by markdown headings first, then by word count if sections are long.
    Produces more semantically coherent chunks than sliding window.
    """
    sections = re.split(r"\n(?=#{1,3} )", text)
    for section in sections:
        words = section.split()
        if len(words) <= max_words:
            if words:
                yield section
        else:
            # Fall back to sliding window for long sections
            step = max_words - 64
            for i in range(0, len(words), step):
                chunk = " ".join(words[i : i + max_words])
                if chunk.strip():
                    yield chunk


def make_record(chunk: str, source: str, meta: dict, chunk_idx: int) -> dict:
    title = meta.get("title", Path(source).stem)
    tags = meta.get("tags", [])
    return {
        "id": hashlib.md5(f"{source}:{chunk_idx}".encode()).hexdigest()[:12],
        "source": source,
        "source_type": "notes",
        "title": title,
        "tags": tags if isinstance(tags, list) else [tags],
        "chunk_index": chunk_idx,
        "text": chunk,
        "instruction": "Recall information from your personal notes.",
        "input": "",
        "output": chunk,
    }


def ingest_notes(source: Path, output_dir: Path) -> dict:
    """
    Ingest notes from file or directory.
    Writes JSONL to output_dir/notes.jsonl.
    """
    EXTENSIONS = {".md", ".txt", ".org", ".markdown"}
    paths = []
    if source.is_file():
        paths = [source] if source.suffix in EXTENSIONS else []
    else:
        paths = [f for f in source.rglob("*") if f.suffix in EXTENSIONS]

    out_path = output_dir / "notes.jsonl"
    total_chunks = 0

    with out_path.open("a") as f:
        for file_path in paths:
            try:
                raw = file_path.read_text(encoding="utf-8", errors="ignore")
                meta, body = parse_frontmatter(raw)
                clean = strip_markdown(body)

                for i, chunk in enumerate(chunk_by_section(clean)):
                    record = make_record(chunk, str(file_path), meta, i)
                    f.write(json.dumps(record) + "\n")
                    total_chunks += 1

            except Exception as e:
                print(f"Warning: skipping {file_path}: {e}")

    return {"files": len(paths), "chunks": total_chunks, "output": str(out_path)}
