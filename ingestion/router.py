"""Auto-detect source type and route to the appropriate ingestion module."""

from pathlib import Path
from typing import Optional


SOURCE_EXTENSIONS = {
    "notes": {".md", ".txt", ".org", ".markdown"},
    "documents": {".pdf", ".epub"},
    "chat": {".json", ".mbox"},
    "browser": {".html", ".sqlite"},
}


def detect_type(source: Path) -> str:
    """Guess source type from file extension or directory contents."""
    if source.is_dir():
        exts = {f.suffix.lower() for f in source.rglob("*") if f.is_file()}
        for stype, valid in SOURCE_EXTENSIONS.items():
            if exts & valid:
                return stype
        return "notes"  # default for directories

    ext = source.suffix.lower()
    for stype, valid in SOURCE_EXTENSIONS.items():
        if ext in valid:
            return stype
    raise ValueError(f"Cannot detect source type for {source.name}. Use --type to specify.")


def ingest_source(source: Path, source_type: Optional[str], output_dir: Path) -> dict:
    """Route ingestion to the correct module based on source type."""
    stype = source_type or detect_type(source)
    output_dir.mkdir(parents=True, exist_ok=True)

    if stype == "notes":
        from ingestion.notes import ingest_notes
        return ingest_notes(source, output_dir)
    elif stype == "documents":
        from ingestion.documents import ingest_documents
        return ingest_documents(source, output_dir)
    elif stype == "chat":
        from ingestion.chat import ingest_chat
        return ingest_chat(source, output_dir)
    elif stype == "browser":
        from ingestion.browser import ingest_browser
        return ingest_browser(source, output_dir)
    else:
        raise ValueError(f"Unknown source type: {stype}")
