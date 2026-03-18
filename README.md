# Intelope

> Train small personal LLMs on your own data locally.


---

## Quickstart

### 0. Create a virtual environment

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (CMD):**

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### Install Intelope

```bash
pip install git+https://github.com/ehienabs/intelope.git
intelope start
```

### 1. Create a dataset

```bash
intelope dataset create my-dataset
```

### 2. Ingest your data into the dataset

```bash
# Notes (markdown, Obsidian vault, plaintext)
intelope ingest ~/Documents/notes --dataset my-dataset

# PDF/EPUB documents
intelope ingest ~/Documents/papers --dataset my-dataset --type documents

# WhatsApp/Telegram export
intelope ingest ~/Downloads/WhatsApp\ Chat.txt --dataset my-dataset --type chat
```

### 3. Clean the dataset

Run deduplication, PII scrubbing, and quality filtering:

```bash
intelope dataset clean my-dataset
```

Manage datasets:

```bash
intelope dataset list            # list all datasets and status
intelope dataset delete old-data # delete a dataset
```

### 4. Train a named model

Pick a cleaned dataset and name your model:

```bash
intelope train --dataset my-dataset --name my-model --model smollm2-360m --epochs 3
```

### 4. Chat

```bash
intelope chat --model my-model
```

### 5. Enable RAG (recommended)

RAG (Retrieval-Augmented Generation) lets your model search your actual documents before answering,
so it gives answers grounded in what you've written — not hallucinations.

```bash
# Build a search index from your cleaned dataset
intelope index --dataset my-dataset

# Chat with RAG enabled
intelope chat --model my-model --rag
```

**Without RAG:** The model guesses from what it memorised during training. Low accuracy and high hallucination.
**With RAG:** Every question first searches your documents, then the relevant excerpts are fed to the model as context.

RAG is **off by default** — you opt in by checking the RAG toggle in the Chat UI or passing `--rag` on the CLI.  Once you've built an index, flip it on and every answer will be grounded in your actual documents.

You can also build the index and toggle RAG from the web dashboard (RAG Index tab + Chat toggle).

### Or: launch the web UI

```bash
intelope start
# → http://localhost:7860
```

The web dashboard provides the same workflow in a visual interface:

1. **Ingest** — create a dataset, upload files into it, then clean
2. **Train** — select a cleaned dataset, name your model, pick a base model, and train
3. **Chat** — choose a trained model from the dropdown and chat

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
- **Quality > quantity** — always create a dataset first; 300 clean chunks beat 3,000 noisy ones

### Tuning advice

#### Epochs

Default: **3** — safe but conservative.

With a small dataset (~2,000–3,000 examples), the model needs more passes to internalise patterns. The sweet spot is usually **5–6 epochs**. Watch for overfitting: if training loss keeps dropping but outputs start sounding repetitive or "memorised", you've gone too far.

#### LoRA Rank

Default: **16** — reasonable for moderate complexity.

| Rank | When to use |
|------|-------------|
| 8 | Narrow, consistent style or simple Q&A patterns |
| 16 | Moderate complexity (default) |
| 32 | Complex reasoning, multi-turn conversation, or diverse writing styles |

Higher rank = more trainable parameters = more expressive, but slower and more prone to overfitting on small data. With fewer than ~3,000 examples, jumping to 32 without more data can backfire.

#### Dataset size

This is your highest-leverage point — more than epochs or rank, **data quality and quantity drives results**.

- **Target**: 5,000–10,000 examples for noticeable improvement
- **Consistency matters more than volume** — 2,000 high-quality, well-formatted examples beats 8,000 noisy ones
- Make sure your examples reflect exactly the behaviour you want — the model will imitate the patterns, good and bad
- Consider **synthetic data augmentation** — generate variations of your best examples using a base model to cheaply expand the dataset

#### Suggested progression

Rather than changing everything at once, iterate one variable at a time:

1. **First** — clean and expand your dataset to ~5,000 examples
2. **Then** — re-run with epochs bumped to 5
3. **Only then** — experiment with rank 32 if results still feel shallow

Changing one variable at a time makes it much easier to know what's actually helping.

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
├── cli.py              # CLI entrypoint
├── ingestion/
│   ├── router.py       # auto-detects source type
│   ├── notes.py        # markdown / Obsidian
│   ├── documents.py    # PDF / EPUB
│   ├── chat.py         # WhatsApp / Telegram / MBOX
│   ├── browser.py      # Chrome / Firefox / bookmarks
│   └── training/
│       ├── finetune.py # LoRA via unsloth (fast) or transformers
│       └── inference.py# model loading and generation
├── pipeline/
│   ├── clean.py        # dedup, PII scrub, quality filter
│   └── rag.py          # RAG: embed, index, retrieve
├── ui/
│   ├── app.py          # FastAPI backend
│   └── intelope-dashboard.html
├── data/
│   ├── datasets/       # per-dataset directories
│   │   └── <name>/
│   │       ├── uploads/   # raw uploaded files
│   │       ├── processed/ # ingested chunks (JSONL)
│   │       └── clean.jsonl # cleaned, training-ready
│   └── rag_index/      # FAISS index + metadata
└── models/
    └── <model-name>/   # trained LoRA adapters
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
