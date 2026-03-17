"""
Pipeline: clean and filter raw JSONL chunks before training.
Steps: deduplication → PII scrubbing → quality filtering → merge
"""

from pathlib import Path
import json
import re
import hashlib
from typing import Iterator


# ── PII Patterns ──────────────────────────────────────────────────────────────

PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),                    # US SSN
    (re.compile(r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14})\b"), "[CARD]"),  # credit cards
    (re.compile(r"\b\d{1,5}\s+\w+(?:\s+\w+)*\s+(?:St|Ave|Rd|Blvd|Dr|Ln|Way)\b", re.I), "[ADDRESS]"),
    (re.compile(r"\b(?:password|passwd|pwd)\s*[:=]\s*\S+", re.I), "[PASSWORD]"),
    (re.compile(r"\b(?:sk-|AIza)[A-Za-z0-9_-]{20,}\b"), "[API_KEY]"),
]


def scrub_pii(text: str) -> tuple[str, list[str]]:
    """Replace PII with placeholders. Returns (cleaned_text, list_of_pii_types_found)."""
    found = []
    for pattern, replacement in PII_PATTERNS:
        if pattern.search(text):
            found.append(replacement)
            text = pattern.sub(replacement, text)
    return text, found


# ── Quality Filtering ─────────────────────────────────────────────────────────

def is_quality(text: str, min_words: int = 20, max_repeat_ratio: float = 0.4) -> bool:
    """Basic quality filter — skip very short or repetitive chunks."""
    words = text.split()
    if len(words) < min_words:
        return False
    # Check for excessive repetition
    unique = len(set(words))
    if unique / len(words) < (1 - max_repeat_ratio):
        return False
    return True


# ── Deduplication ─────────────────────────────────────────────────────────────

def text_hash(text: str) -> str:
    """Normalize and hash text for deduplication."""
    normalized = re.sub(r"\s+", " ", text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    input_dir: Path,
    output_path: Path,
    scrub: bool = True,
    deduplicate: bool = True,
    min_words: int = 20,
) -> dict:
    """
    Read all JSONL files from input_dir, clean them, write merged output.
    Returns stats.
    """
    input_files = list(input_dir.glob("*.jsonl"))
    seen_hashes = set()
    total_in = total_out = pii_count = dup_count = quality_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as out_f:
        for jsonl_file in input_files:
            with jsonl_file.open() as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    total_in += 1
                    text = record.get("output", record.get("text", ""))

                    # Quality filter
                    if not is_quality(text, min_words):
                        quality_count += 1
                        continue

                    # Deduplication
                    if deduplicate:
                        h = text_hash(text)
                        if h in seen_hashes:
                            dup_count += 1
                            continue
                        seen_hashes.add(h)

                    # PII scrubbing
                    if scrub:
                        cleaned, found = scrub_pii(text)
                        record["output"] = cleaned
                        record["text"] = cleaned
                        if found:
                            record["pii_scrubbed"] = found
                            pii_count += 1

                    out_f.write(json.dumps(record) + "\n")
                    total_out += 1

    return {
        "input_records": total_in,
        "output_records": total_out,
        "removed_quality": quality_count,
        "removed_duplicates": dup_count,
        "pii_scrubbed": pii_count,
        "output": str(output_path),
    }
