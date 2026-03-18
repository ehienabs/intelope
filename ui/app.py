"""Intelope web UI — serves the dashboard and API endpoints via FastAPI + Uvicorn."""

from pathlib import Path
from typing import List
import json
import shutil
import tempfile
import threading
import queue
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = Path(__file__).resolve().parent / "intelope-dashboard.html"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DIR = DATA_DIR / "clean"
UPLOAD_DIR = DATA_DIR / "uploads"

app = FastAPI(title="Intelope")


@app.get("/", response_class=HTMLResponse)
def index():
    return DASHBOARD.read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Save uploaded files and run ingestion on each."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for f in files:
        # Save to uploads/
        dest = UPLOAD_DIR / f.filename
        content = await f.read()
        dest.write_bytes(content)

        # Detect type and ingest
        ext = dest.suffix.lower()
        try:
            if ext in {".md", ".txt", ".org", ".markdown"}:
                from ingestion.notes import ingest_notes
                stats = ingest_notes(dest, PROCESSED_DIR)
            elif ext in {".pdf", ".epub"}:
                from ingestion.documents import ingest_documents
                stats = ingest_documents(dest, PROCESSED_DIR)
            elif ext == ".json":
                from ingestion.chat import ingest_chat
                stats = ingest_chat(dest, PROCESSED_DIR)
            elif ext == ".mbox":
                from ingestion.chat import ingest_chat
                stats = ingest_chat(dest, PROCESSED_DIR)
            elif ext == ".html":
                from ingestion.browser import ingest_browser
                stats = ingest_browser(dest, PROCESSED_DIR)
            else:
                results.append({"file": f.filename, "error": f"Unsupported format: {ext}"})
                continue
            results.append({"file": f.filename, "status": "ok", **stats})
        except Exception as e:
            results.append({"file": f.filename, "error": str(e)})

    return JSONResponse({"results": results})


@app.get("/api/status")
def get_status():
    """Return current dataset and pipeline stats."""
    chunks = []
    source_counts = {}

    if PROCESSED_DIR.exists():
        for jsonl_file in PROCESSED_DIR.glob("*.jsonl"):
            with jsonl_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        chunks.append(record)
                        stype = record.get("source_type", "unknown")
                        if stype not in source_counts:
                            source_counts[stype] = {"files": set(), "chunks": 0}
                        source_counts[stype]["files"].add(record.get("source", ""))
                        source_counts[stype]["chunks"] += 1
                    except json.JSONDecodeError:
                        continue

    # Count cleaned
    clean_count = 0
    pii_count = 0
    clean_path = CLEAN_DIR / "merged.jsonl"
    if clean_path.exists():
        with clean_path.open() as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        clean_count += 1
                        if rec.get("pii_scrubbed"):
                            pii_count += 1
                    except json.JSONDecodeError:
                        continue

    # Count models
    models_dir = PROJECT_ROOT / "models"
    model_count = len(list(models_dir.glob("*"))) if models_dir.exists() else 0

    # Serialize source_counts (sets aren't JSON-serializable)
    sources = {}
    for stype, info in source_counts.items():
        sources[stype] = {"files": len(info["files"]), "chunks": info["chunks"]}

    return {
        "total_chunks": len(chunks),
        "clean_chunks": clean_count,
        "pii_scrubbed": pii_count,
        "models_trained": model_count,
        "sources": sources,
    }


@app.get("/api/chunks")
def get_chunks():
    """Return all processed chunks for the dataset viewer."""
    chunks = []
    if PROCESSED_DIR.exists():
        for jsonl_file in PROCESSED_DIR.glob("*.jsonl"):
            with jsonl_file.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return {"chunks": chunks}


@app.post("/api/clean")
def run_clean():
    """Run the cleaning pipeline on processed data."""
    if not PROCESSED_DIR.exists() or not list(PROCESSED_DIR.glob("*.jsonl")):
        return JSONResponse({"error": "No processed data found. Ingest files first."}, status_code=400)

    from pipeline.clean import run_pipeline
    output_path = CLEAN_DIR / "merged.jsonl"
    stats = run_pipeline(PROCESSED_DIR, output_path)
    return stats


# ── Training state ────────────────────────────────────────────────────────────
_train_thread: threading.Thread = None
_train_queue: queue.Queue = None


@app.post("/api/train/start")
async def start_training(request: Request):
    """Start training in a background thread. Returns immediately."""
    global _train_thread, _train_queue

    if _train_thread is not None and _train_thread.is_alive():
        return JSONResponse({"error": "Training already in progress."}, status_code=409)

    body = await request.json()
    model = body.get("model", "smollm2-1.7b")
    epochs = int(body.get("epochs", 3))
    lora_r = int(body.get("lora_r", 16))
    lora_alpha = int(body.get("lora_alpha", 32))
    lr = float(body.get("lr", 2e-4))
    batch_size = int(body.get("batch_size", 2))
    max_seq_length = int(body.get("max_seq_length", 2048))

    _train_queue = queue.Queue()

    from ingestion.training.finetune import set_event_queue, run_finetune

    set_event_queue(_train_queue)

    def _run():
        try:
            run_finetune(
                base_model=model,
                data_dir=PROCESSED_DIR,
                output_dir=PROJECT_ROOT / "models",
                epochs=epochs,
                lora_r=lora_r,
                learning_rate=lr,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
            )
        except Exception as e:
            _train_queue.put({"type": "error", "message": str(e)})
        finally:
            _train_queue.put(None)  # sentinel

    _train_thread = threading.Thread(target=_run, daemon=True)
    _train_thread.start()
    return {"status": "started"}


@app.get("/api/train/stream")
def stream_training():
    """SSE endpoint — streams training events as they happen."""
    if _train_queue is None:
        return JSONResponse({"error": "No training session."}, status_code=404)

    def event_generator():
        while True:
            try:
                event = _train_queue.get(timeout=30)
            except queue.Empty:
                # Keep-alive
                yield ":\n\n"
                continue
            if event is None:
                yield f"data: {json.dumps({'type': 'end'})}\n\n"
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/train/stop")
def stop_training():
    """Request a graceful stop of the current training run."""
    from ingestion.training.finetune import request_stop
    request_stop()
    return {"status": "stop_requested"}


def launch(host: str = "127.0.0.1", port: int = 7860):
    uvicorn.run(app, host=host, port=port)
