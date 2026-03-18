"""Intelope web UI — serves the dashboard and API endpoints via FastAPI + Uvicorn."""

from pathlib import Path
from typing import List
import json
import re
import shutil
import tempfile
import threading
import queue
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.responses import StreamingResponse
import uvicorn

PROJECT_ROOT = Path.cwd()
DASHBOARD = Path(__file__).resolve().parent / "intelope-dashboard.html"
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"

app = FastAPI(title="Intelope")


def _sanitize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name.strip())


@app.get("/", response_class=HTMLResponse)
def index():
    return DASHBOARD.read_text(encoding="utf-8")


# ── Dataset management ────────────────────────────────────────────────────────

@app.post("/api/datasets/create")
async def create_dataset(request: Request):
    """Create a new empty dataset directory."""
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Dataset name is required."}, status_code=400)
    safe = _sanitize(name)
    ds_dir = DATASETS_DIR / safe
    if ds_dir.exists():
        return JSONResponse({"error": f"Dataset '{safe}' already exists."}, status_code=409)
    (ds_dir / "uploads").mkdir(parents=True)
    (ds_dir / "processed").mkdir(parents=True)
    return {"name": safe, "status": "created"}


@app.get("/api/datasets")
def list_datasets():
    """Return all datasets with their status."""
    datasets = []
    if not DATASETS_DIR.exists():
        return {"datasets": datasets}
    for d in sorted(DATASETS_DIR.iterdir()):
        if not d.is_dir():
            continue
        processed_dir = d / "processed"
        clean_file = d / "clean.jsonl"
        raw_count = sum(1 for _ in (d / "uploads").glob("*")) if (d / "uploads").exists() else 0
        chunk_count = 0
        if processed_dir.exists():
            for f in processed_dir.glob("*.jsonl"):
                chunk_count += sum(1 for line in f.open() if line.strip())
        clean_count = 0
        if clean_file.exists():
            clean_count = sum(1 for line in clean_file.open() if line.strip())
        datasets.append({
            "name": d.name,
            "files": raw_count,
            "chunks": chunk_count,
            "clean_records": clean_count,
            "ready": clean_count > 0,
        })
    return {"datasets": datasets}


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    """Delete a dataset and all its data."""
    safe = _sanitize(name)
    ds_dir = DATASETS_DIR / safe
    if not ds_dir.exists():
        return JSONResponse({"error": f"Dataset not found: {name}"}, status_code=404)
    shutil.rmtree(ds_dir)
    return {"status": "deleted", "name": safe}


@app.post("/api/datasets/{name}/upload")
async def upload_to_dataset(name: str, files: List[UploadFile] = File(...)):
    """Upload files into a specific dataset and run ingestion."""
    safe = _sanitize(name)
    ds_dir = DATASETS_DIR / safe
    if not ds_dir.exists():
        return JSONResponse({"error": f"Dataset not found: {name}"}, status_code=404)

    upload_dir = ds_dir / "uploads"
    processed_dir = ds_dir / "processed"
    upload_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for f in files:
        safe_name = Path(f.filename).name
        dest = upload_dir / safe_name
        content = await f.read()
        dest.write_bytes(content)

        ext = dest.suffix.lower()
        try:
            if ext in {".md", ".txt", ".org", ".markdown"}:
                from ingestion.notes import ingest_notes
                stats = ingest_notes(dest, processed_dir)
            elif ext in {".pdf", ".epub"}:
                from ingestion.documents import ingest_documents
                stats = ingest_documents(dest, processed_dir)
            elif ext == ".json":
                from ingestion.chat import ingest_chat
                stats = ingest_chat(dest, processed_dir)
            elif ext == ".mbox":
                from ingestion.chat import ingest_chat
                stats = ingest_chat(dest, processed_dir)
            elif ext == ".html":
                from ingestion.browser import ingest_browser
                stats = ingest_browser(dest, processed_dir)
            else:
                results.append({"file": f.filename, "error": f"Unsupported format: {ext}"})
                continue
            results.append({"file": f.filename, "status": "ok", **stats})
        except Exception as e:
            results.append({"file": f.filename, "error": str(e)})

    return JSONResponse({"results": results})


@app.post("/api/datasets/{name}/clean")
def clean_dataset(name: str):
    """Run cleaning pipeline on a dataset's processed data."""
    safe = _sanitize(name)
    ds_dir = DATASETS_DIR / safe
    if not ds_dir.exists():
        return JSONResponse({"error": f"Dataset not found: {name}"}, status_code=404)

    processed_dir = ds_dir / "processed"
    if not processed_dir.exists() or not list(processed_dir.glob("*.jsonl")):
        return JSONResponse({"error": "No ingested data in this dataset. Upload files first."}, status_code=400)

    from pipeline.clean import run_pipeline
    output_path = ds_dir / "clean.jsonl"
    stats = run_pipeline(processed_dir, output_path)
    stats["name"] = safe
    return stats


@app.get("/api/datasets/{name}/chunks")
def get_dataset_chunks(name: str):
    """Return processed chunks for a specific dataset."""
    safe = _sanitize(name)
    processed_dir = DATASETS_DIR / safe / "processed"
    chunks = []
    if processed_dir.exists():
        for jsonl_file in processed_dir.glob("*.jsonl"):
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


# ── Legacy endpoints (for backward compat) ────────────────────────────────────

@app.get("/api/status")
def get_status():
    """Return current dataset and pipeline stats."""
    total_chunks = 0
    clean_total = 0
    pii_count = 0
    source_counts = {}

    if DATASETS_DIR.exists():
        for ds_dir in DATASETS_DIR.iterdir():
            if not ds_dir.is_dir():
                continue
            processed = ds_dir / "processed"
            if processed.exists():
                for jsonl_file in processed.glob("*.jsonl"):
                    with jsonl_file.open() as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                                total_chunks += 1
                                stype = record.get("source_type", "unknown")
                                if stype not in source_counts:
                                    source_counts[stype] = {"files": set(), "chunks": 0}
                                source_counts[stype]["files"].add(record.get("source", ""))
                                source_counts[stype]["chunks"] += 1
                            except json.JSONDecodeError:
                                continue
            clean_file = ds_dir / "clean.jsonl"
            if clean_file.exists():
                with clean_file.open() as f:
                    for line in f:
                        if line.strip():
                            try:
                                rec = json.loads(line)
                                clean_total += 1
                                if rec.get("pii_scrubbed"):
                                    pii_count += 1
                            except json.JSONDecodeError:
                                continue

    models_dir = PROJECT_ROOT / "models"
    model_count = 0
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir() and (d / "adapter_config.json").exists():
                model_count += 1

    sources = {}
    for stype, info in source_counts.items():
        sources[stype] = {"files": len(info["files"]), "chunks": info["chunks"]}

    return {
        "total_chunks": total_chunks,
        "clean_chunks": clean_total,
        "pii_scrubbed": pii_count,
        "models_trained": model_count,
        "sources": sources,
    }


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
    output_name = _sanitize(body.get("output_name", "latest")) or "latest"
    dataset_choice = body.get("dataset", "").strip()

    # Resolve dataset — look for clean.jsonl inside the dataset directory
    safe_ds = _sanitize(dataset_choice) if dataset_choice else ""
    ds_clean = DATASETS_DIR / safe_ds / "clean.jsonl" if safe_ds else None
    if ds_clean and ds_clean.exists():
        tmp = Path(tempfile.mkdtemp())
        (tmp / "clean.jsonl").symlink_to(ds_clean)
        train_data_dir = tmp
    else:
        return JSONResponse({"error": f"Dataset '{dataset_choice}' not found or not cleaned yet."}, status_code=400)

    _train_queue = queue.Queue()

    from ingestion.training.finetune import set_event_queue, run_finetune

    set_event_queue(_train_queue)

    def _run():
        try:
            run_finetune(
                base_model=model,
                data_dir=train_data_dir,
                output_dir=PROJECT_ROOT / "models",
                epochs=epochs,
                lora_r=lora_r,
                learning_rate=lr,
                batch_size=batch_size,
                max_seq_length=max_seq_length,
                output_name=output_name,
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


# ── Chat ──────────────────────────────────────────────────────────────────────
_chat_model = None
_chat_tokenizer = None
_chat_backend = None
_chat_history = []


@app.get("/api/models")
def list_models():
    """Return available trained models."""
    models_dir = PROJECT_ROOT / "models"
    models = []
    if models_dir.exists():
        for d in sorted(models_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            cfg_path = d / "adapter_config.json"
            if d.is_dir() and cfg_path.exists():
                try:
                    cfg = json.loads(cfg_path.read_text())
                    base = cfg.get("base_model_name_or_path", "unknown")
                except Exception:
                    base = "unknown"
                models.append({"name": d.name, "base_model": base})
    return {"models": models}


@app.post("/api/chat/load")
async def load_chat_model(request: Request):
    """Load a trained model for chat."""
    global _chat_model, _chat_tokenizer, _chat_backend, _chat_history
    body = await request.json()
    model_name = body.get("model", "latest")
    model_path = PROJECT_ROOT / "models" / model_name

    if not model_path.exists() or not (model_path / "adapter_config.json").exists():
        return JSONResponse({"error": f"Model not found: {model_name}"}, status_code=404)

    from ingestion.training.inference import load_model, DEFAULT_SYSTEM
    _chat_model, _chat_tokenizer, _chat_backend = load_model(model_path)
    _chat_history = [{"role": "system", "content": DEFAULT_SYSTEM}]
    return {"status": "loaded", "model": model_name, "backend": _chat_backend}


@app.post("/api/chat")
async def chat_message(request: Request):
    """Send a message and get a response from the loaded model."""
    global _chat_history
    if _chat_model is None:
        return JSONResponse({"error": "No model loaded. Load a model first."}, status_code=400)

    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        return JSONResponse({"error": "Empty message."}, status_code=400)

    from ingestion.training.inference import generate
    _chat_history.append({"role": "user", "content": message})
    reply = generate(_chat_model, _chat_tokenizer, _chat_history, _chat_backend)
    _chat_history.append({"role": "assistant", "content": reply})
    return {"reply": reply}


@app.post("/api/chat/clear")
def clear_chat_history():
    """Clear chat history."""
    global _chat_history
    from ingestion.training.inference import DEFAULT_SYSTEM
    _chat_history = [{"role": "system", "content": DEFAULT_SYSTEM}]
    return {"status": "cleared"}


def launch(host: str = "127.0.0.1", port: int = 7860):
    uvicorn.run(app, host=host, port=port)
