# Intelope

> Train small personal LLMs on your own data — privately, locally.

Your data never leaves your machine. No cloud APIs. No corporate data pipelines.
Just you, your notes, and a model that actually knows your world.

---

## Quickstart

```bash
pip install intelope
# for faster training (optional, requires compatible GPU):
pip install "intelope[fast]"
```

### 1. Ingest your data

```bash
# Notes (markdown, Obsidian vault, plaintext)
intelope ingest ~/Documents/notes

# PDF/EPUB documents
intelope ingest ~/Documents/papers --type documents

# Browser history (Chrome)
intelope ingest ~/Library/Application\ Support/Google/Chrome/Default/History --type browser

# WhatsApp/Telegram export
intelope ingest ~/Downloads/WhatsApp\ Chat.txt --type chat
```

### 2. Run the cleaning pipeline

```bash
intelope clean   # dedup + PII scrub + quality filter
```

### 3. Train

```bash
intelope train --model smollm2-1.7b --epochs 3
```

### 4. Chat

```bash
intelope chat
```

### Or: launch the web UI

```bash
intelope start
# → http://localhost:7860
```

---

## Supported base models

| Model | Params | VRAM | Notes |
|---|---|---|---|
| SmolLM2-1.7B | 1.7B | ~3.5GB | Default. Great balance. |
| SmolLM2-360M | 360M | ~1GB | Fastest, good for CPU. |
| Phi-3-mini | 3.8B | ~7GB | Best quality. |
| Qwen2.5-1.5B | 1.5B | ~3GB | Strong reasoning. |
| TinyLlama | 1.1B | ~2.5GB | Good fallback. |

---

## Supported data sources

| Source | Formats |
|---|---|
| Notes | `.md`, `.txt`, `.org`, Obsidian vaults |
| Documents | `.pdf`, `.epub` |
| Chat | WhatsApp `.txt`, Telegram `.json`, `.mbox` (email) |
| Browser | Chrome/Firefox SQLite History, bookmarks `.html` |

---

## Training

### How it works

Intelope uses **LoRA** (Low-Rank Adaptation) to fine-tune a small base model on your data.
This is efficient — only a fraction of the model weights are updated, so it runs on consumer hardware.

Training runs locally and can be started from the **CLI** or the **web dashboard** (Train tab).

### Real-time progress

The dashboard shows live training stats updated every 10 steps:

- **Loss** — how well the model is learning (lower = better)
- **Step** — current step out of total steps
- **ETA** — estimated time remaining (recalculated from actual speed)
- **VRAM** — GPU/MPS memory usage

A progress bar and streaming log show exactly what's happening.

### Stopping training

Click **⏹ Stop Training** in the dashboard (or Ctrl+C in CLI). The model saves
what it has learned up to that point — you don't lose progress.

### Time estimates

Training time depends on data volume, hardware, and epochs:

$$\text{steps} = \left\lceil \frac{\text{chunks}}{\text{batch size} \times \text{gradient accumulation}} \right\rceil \times \text{epochs}$$

With defaults (batch=2, grad_accum=4, epochs=3):

| Chunks | Steps | GPU | Apple MPS | CPU only |
|--------|-------|-----|-----------|----------|
| 100 | ~39 | ~12s | ~39s | ~2.5min |
| 500 | ~189 | ~1min | ~3min | ~12min |
| 1,000 | ~375 | ~2min | ~6min | ~25min |
| 5,000 | ~1,875 | ~9min | ~31min | ~2h |

### Minimum data size

| Chunks | Quality | Recommendation |
|--------|---------|----------------|
| < 50 | Poor | Not enough signal — model memorizes noise |
| 50–200 | Minimal | Can recall specific facts, poor generalization |
| **200–500** | **Decent** | Good starting point for personal Q&A |
| 500–2,000 | Good | Captures your knowledge and writing style |
| 2,000+ | Best | Reliable recall, coherent style across topics |

**Tips:**
- **Minimum viable**: ~200 chunks (~50–100 pages of text)
- **More epochs help small datasets** — with 200 chunks, try 5–8 epochs instead of 3
- **Quality > quantity** — always run `intelope clean` first; 300 clean chunks beat 3,000 noisy ones

---

## Privacy design

- **Local only** — no network calls during ingestion or training
- **PII scrubbing** — emails, phone numbers, SSNs, API keys redacted before training
- **Transparent dataset** — browse every chunk in the UI before you train
- **Auditable** — all JSONL files are human-readable; inspect exactly what the model learns

---

## Architecture

```
intelope/
├── cli.py              # Typer CLI entrypoint
├── ingestion/
│   ├── router.py       # auto-detects source type
│   ├── notes.py        # markdown / Obsidian
│   ├── documents.py    # PDF / EPUB
│   ├── chat.py         # WhatsApp / Telegram / MBOX
│   └── browser.py      # Chrome / Firefox / bookmarks
├── pipeline/
│   └── clean.py        # dedup, PII scrub, quality filter
├── training/
│   ├── finetune.py     # LoRA via unsloth (fast) or transformers
│   └── inference.py    # terminal chat loop
├── ui/
│   └── dashboard.html  # web dashboard
└── data/
    ├── raw/
    ├── processed/
    └── exports/
```

---

## Design philosophy

Intelope is built on a simple premise: the future of AI is small, personal, and local —
not centralized and cloud-dependent. The history of computing moved from mainframes to
personal computers. We're building the personal computer moment for AI.

Your notes, your documents, your conversations contain a model of the world that is uniquely
yours. Intelope helps you make that knowledge legible to a machine — without giving it to
anyone else.

---

## License

MIT — use it, fork it, build on it.
