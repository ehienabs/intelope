"""
Ingests browser history and bookmarks.
Supports: Chrome/Brave/Edge SQLite history, Firefox SQLite, exported HTML bookmarks.
"""

from pathlib import Path
import json
import hashlib
import sqlite3
from typing import Iterator
from datetime import datetime, timedelta


# ── Chrome/Chromium History ───────────────────────────────────────────────────

CHROME_EPOCH = datetime(1601, 1, 1)

def chrome_time(ts: int) -> str:
    """Convert Chrome's microsecond timestamp to ISO string."""
    try:
        return (CHROME_EPOCH + timedelta(microseconds=ts)).isoformat()
    except Exception:
        return ""


def parse_chrome_history(db_path: Path) -> Iterator[dict]:
    """Read URLs and titles from Chrome's History SQLite file."""
    import shutil, tempfile
    # Copy DB first — Chrome may lock it
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        shutil.copy2(str(db_path), tmp.name)
        conn = sqlite3.connect(tmp.name)
        cursor = conn.execute(
            "SELECT url, title, visit_count, last_visit_time FROM urls "
            "WHERE title != '' ORDER BY last_visit_time DESC"
        )
        for row in cursor:
            yield {"url": row[0], "title": row[1],
                   "visit_count": row[2], "last_visit": chrome_time(row[3])}
        conn.close()


# ── Firefox History ───────────────────────────────────────────────────────────

def parse_firefox_history(db_path: Path) -> Iterator[dict]:
    import shutil, tempfile
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        shutil.copy2(str(db_path), tmp.name)
        conn = sqlite3.connect(tmp.name)
        cursor = conn.execute(
            "SELECT url, title, visit_count, last_visit_date FROM moz_places "
            "WHERE title IS NOT NULL AND title != '' ORDER BY last_visit_date DESC"
        )
        for row in cursor:
            yield {"url": row[0], "title": row[1],
                   "visit_count": row[2] or 0, "last_visit": str(row[3])}
        conn.close()


# ── HTML Bookmarks ────────────────────────────────────────────────────────────

def parse_html_bookmarks(path: Path) -> Iterator[dict]:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for a in soup.find_all("a"):
        title = a.get_text(strip=True)
        url = a.get("href", "")
        if title and url.startswith("http"):
            yield {"url": url, "title": title, "visit_count": 1, "last_visit": ""}


# ── Formatting ────────────────────────────────────────────────────────────────

def group_by_domain(entries: list[dict], max_per_group: int = 20) -> Iterator[str]:
    """
    Group URLs by domain and format as a browsing summary block.
    This gives the model richer context than individual URL records.
    """
    from urllib.parse import urlparse
    from collections import defaultdict

    domains = defaultdict(list)
    for e in entries:
        domain = urlparse(e["url"]).netloc
        domains[domain].append(e)

    for domain, items in domains.items():
        top = sorted(items, key=lambda x: x["visit_count"], reverse=True)[:max_per_group]
        lines = [f"Domain: {domain}"]
        for item in top:
            lines.append(f"  - {item['title']} ({item['url']})")
        yield "\n".join(lines)


def make_record(text: str, source: str, idx: int) -> dict:
    return {
        "id": hashlib.md5(f"{source}:browser:{idx}".encode()).hexdigest()[:12],
        "source": source,
        "source_type": "browser",
        "chunk_index": idx,
        "text": text,
        "instruction": "Recall websites and topics from your personal browsing history.",
        "input": "",
        "output": text,
    }


def detect_browser_format(path: Path) -> str:
    if path.suffix == ".html":
        return "bookmarks_html"
    if path.name == "History":
        return "chrome"
    if path.name == "places.sqlite":
        return "firefox"
    # Try to detect by table structure
    try:
        conn = sqlite3.connect(str(path))
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
        if "moz_places" in tables:
            return "firefox"
        if "urls" in tables:
            return "chrome"
    except Exception:
        pass
    return "unknown"


def ingest_browser(source: Path, output_dir: Path) -> dict:
    out_path = output_dir / "browser.jsonl"
    total_chunks = 0

    paths = [source] if source.is_file() else (
        list(source.rglob("History")) +
        list(source.rglob("places.sqlite")) +
        list(source.rglob("*.html"))
    )

    with out_path.open("a") as f:
        for file_path in paths:
            try:
                fmt = detect_browser_format(file_path)
                if fmt == "chrome":
                    entries = list(parse_chrome_history(file_path))
                elif fmt == "firefox":
                    entries = list(parse_firefox_history(file_path))
                elif fmt == "bookmarks_html":
                    entries = list(parse_html_bookmarks(file_path))
                else:
                    print(f"Warning: unrecognized browser format: {file_path}")
                    continue

                for i, group in enumerate(group_by_domain(entries)):
                    record = make_record(group, str(file_path), i)
                    f.write(json.dumps(record) + "\n")
                    total_chunks += 1

            except Exception as e:
                print(f"Warning: skipping {file_path}: {e}")

    return {"files": len(paths), "chunks": total_chunks, "output": str(out_path)}
