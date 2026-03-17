"""
Ingests chat and email exports.
Supports: WhatsApp .txt exports, Telegram JSON, generic MBOX.
"""

from pathlib import Path
import json
import re
import hashlib
import mailbox
from typing import Iterator


# ── WhatsApp ────────────────────────────────────────────────────────────────

WHATSAPP_RE = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}(?::\d{2})?\s*[AP]M?)\s*-\s*([^:]+):\s*(.*)"
)

def parse_whatsapp(path: Path) -> Iterator[dict]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    buffer = []
    current = None

    for line in lines:
        m = WHATSAPP_RE.match(line)
        if m:
            if current and buffer:
                current["text"] = " ".join(buffer)
                yield current
            current = {"timestamp": m.group(1), "sender": m.group(2).strip()}
            buffer = [m.group(3).strip()]
        elif current:
            buffer.append(line.strip())

    if current and buffer:
        current["text"] = " ".join(buffer)
        yield current


# ── Telegram ─────────────────────────────────────────────────────────────────

def parse_telegram(path: Path) -> Iterator[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    messages = data.get("messages", [])
    for msg in messages:
        if msg.get("type") != "message":
            continue
        text = msg.get("text", "")
        if isinstance(text, list):
            text = " ".join(t if isinstance(t, str) else t.get("text", "") for t in text)
        if text.strip():
            yield {
                "timestamp": msg.get("date", ""),
                "sender": msg.get("from", "unknown"),
                "text": text.strip(),
            }


# ── MBOX (email) ─────────────────────────────────────────────────────────────

def parse_mbox(path: Path) -> Iterator[dict]:
    mbox = mailbox.mbox(str(path))
    for msg in mbox:
        subject = msg.get("subject", "")
        sender = msg.get("from", "")
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="ignore")
                    break
        else:
            payload = msg.get_payload(decode=True)
            if payload:
                body = payload.decode("utf-8", errors="ignore")
        if body.strip():
            yield {"timestamp": msg.get("date", ""), "sender": sender,
                   "text": f"Subject: {subject}\n\n{body.strip()}"}


# ── Conversation windowing ────────────────────────────────────────────────────

def window_messages(messages: list[dict], window: int = 8) -> Iterator[str]:
    """
    Group messages into sliding windows to capture conversational context.
    Each window becomes one training example.
    """
    for i in range(0, len(messages), window // 2):
        group = messages[i : i + window]
        if not group:
            continue
        text = "\n".join(f"{m['sender']}: {m['text']}" for m in group)
        yield text


def make_record(text: str, source: str, idx: int) -> dict:
    return {
        "id": hashlib.md5(f"{source}:{idx}".encode()).hexdigest()[:12],
        "source": source,
        "source_type": "chat",
        "chunk_index": idx,
        "text": text,
        "instruction": "Recall a conversation from your personal message history.",
        "input": "",
        "output": text,
    }


def ingest_chat(source: Path, output_dir: Path) -> dict:
    """Route to correct chat parser based on format."""
    out_path = output_dir / "chat.jsonl"
    total_chunks = 0

    paths = [source] if source.is_file() else (
        list(source.rglob("*.txt")) +
        list(source.rglob("*.json")) +
        list(source.rglob("*.mbox"))
    )

    with out_path.open("a") as f:
        for file_path in paths:
            try:
                if file_path.suffix == ".mbox":
                    messages = list(parse_mbox(file_path))
                elif file_path.suffix == ".json":
                    messages = list(parse_telegram(file_path))
                else:
                    messages = list(parse_whatsapp(file_path))

                for i, window in enumerate(window_messages(messages)):
                    record = make_record(window, str(file_path), i)
                    f.write(json.dumps(record) + "\n")
                    total_chunks += 1

            except Exception as e:
                print(f"Warning: skipping {file_path}: {e}")

    return {"files": len(paths), "chunks": total_chunks, "output": str(out_path)}
