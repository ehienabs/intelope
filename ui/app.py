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

PROJECT_ROOT = Path.cwd()
DASHBOARD = Path(__file__).resolve().parent / "intelope-dashboard.html"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CLEAN_DIR = DATA_DIR / "clean"
UPLOAD_DIR = DATA_DIR / "uploads"
DATASETS_DIR = DATA_DIR / "datasets"

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
        # Sanitize filename — folder uploads send relative paths like "folder/sub/file.md"
        safe_name = Path(f.filename).name
        dest = UPLOAD_DIR / safe_name
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

    # Count models — only directories with adapter_config.json are real models
    models_dir = PROJECT_ROOT / "models"
    model_count = 0
    model_list = []
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir() and (d / "adapter_config.json").exists():
                model_count += 1
                model_list.append(d.name)

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


@app.get("/api/datasets")
def list_datasets():
    """Return available named datasets for training."""
    datasets = []
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    for f in sorted(DATASETS_DIR.glob("*.jsonl")):
        count = sum(1 for line in f.open() if line.strip())
        datasets.append({"name": f.stem, "records": count})
    return {"datasets": datasets}


@app.post("/api/datasets/create")
async def create_dataset(request: Request):
    """Run the cleaning pipeline and save as a named dataset."""
    if not PROCESSED_DIR.exists() or not list(PROCESSED_DIR.glob("*.jsonl")):
        return JSONResponse({"error": "No processed data found. Ingest files first."}, status_code=400)

    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "Dataset name is required."}, status_code=400)

    import re as _re
    safe_name = _re.sub(r'[^a-zA-Z0-9_-]', '_', name)

    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATASETS_DIR / f"{safe_name}.jsonl"

    from pipeline.clean import run_pipeline
    stats = run_pipeline(PROCESSED_DIR, output_path)
    stats["name"] = safe_name
    return stats


@app.delete("/api/datasets/{name}")
def delete_dataset(name: str):
    """Delete a named dataset."""
    import re as _re
    safe_name = _re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    path = DATASETS_DIR / f"{safe_name}.jsonl"
    if not path.exists():
        return JSONResponse({"error": f"Dataset not found: {name}"}, status_code=404)
    path.unlink()
    return {"status": "deleted", "name": safe_name}


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
    output_name = body.get("output_name", "latest").strip() or "latest"
    # Sanitize output name — allow only alphanumeric, hyphens, underscores
    import re as _re
    output_name = _re.sub(r'[^a-zA-Z0-9_-]', '_', output_name)
    dataset_choice = body.get("dataset", "").strip()

    # Resolve dataset file
    import re as _re
    safe_ds = _re.sub(r'[^a-zA-Z0-9_-]', '_', dataset_choice) if dataset_choice else ""
    ds_path = DATASETS_DIR / f"{safe_ds}.jsonl" if safe_ds else None
    if ds_path and ds_path.exists():
        # Create a temp dir with just this dataset file so load_jsonl_dataset works
        import tempfile as _tf
        tmp = Path(_tf.mkdtemp())
        (tmp / ds_path.name).symlink_to(ds_path)
        train_data_dir = tmp
    else:
        # Fallback to processed dir
        train_data_dir = PROCESSED_DIR

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
