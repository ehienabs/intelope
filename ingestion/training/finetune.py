"""
Fine-tune a base model using LoRA via unsloth (fast) or transformers (fallback).
Supports: SmolLM2, Phi-3-mini, Gemma-2B, Qwen2.5, TinyLlama
"""

from pathlib import Path
import json
import time
import math
import threading
from typing import Optional
from transformers import TrainerCallback

# Shared state so the UI can stream events and request a stop.
_event_queue = None  # set to a queue.Queue by the UI layer
_stop_flag = threading.Event()


def set_event_queue(q):
    global _event_queue
    _event_queue = q


def request_stop():
    _stop_flag.set()


def reset_stop():
    _stop_flag.clear()


def _emit(event: dict):
    """Send an event to the UI queue (if wired) and print to console."""
    if _event_queue is not None:
        _event_queue.put(event)
    # Always print for CLI users
    kind = event.get("type", "")
    if kind == "log":
        print(event.get("message", ""))
    elif kind == "done":
        print(event.get("message", ""))


def _get_vram() -> str:
    """Return current GPU/MPS memory usage, or 'N/A'."""
    try:
        import torch
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_mem / 1024**3
            return f"{used:.1f}/{total:.1f} GB"
        if torch.backends.mps.is_available():
            used = torch.mps.current_allocated_memory() / 1024**3
            return f"{used:.1f} GB (MPS)"
    except Exception:
        pass
    return "N/A"


class ETACallback(TrainerCallback):
    """Streams elapsed time, ETA, loss, VRAM to the event queue and console."""

    def on_train_begin(self, args, state, control, **kwargs):
        self._start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step == 0 or state.max_steps == 0:
            return
        elapsed = time.time() - self._start
        frac = state.global_step / state.max_steps
        eta = (elapsed / frac) - elapsed
        loss = logs.get("loss", None) if logs else None

        _emit({
            "type": "progress",
            "step": state.global_step,
            "max_steps": state.max_steps,
            "pct": int(frac * 100),
            "loss": round(loss, 4) if loss is not None else None,
            "elapsed": _fmt_seconds(elapsed),
            "eta": _fmt_seconds(eta),
            "vram": _get_vram(),
        })

        msg = (f"  [{int(frac*100):3d}%] step {state.global_step}/{state.max_steps}"
               f"  loss={loss:.4f}" if loss else "")
        _emit({"type": "log", "message": msg})

        # Check stop flag
        if _stop_flag.is_set():
            control.should_training_stop = True
            _emit({"type": "log", "message": "⚠  Stop requested — finishing current step..."})

    def on_train_end(self, args, state, control, **kwargs):
        total = time.time() - self._start
        _emit({"type": "log", "message": f"⏱  Training completed in {_fmt_seconds(total)}"})
        _emit({"type": "done", "message": f"✓ Training finished in {_fmt_seconds(total)}", "total_time": _fmt_seconds(total)})


def _fmt_seconds(s: float) -> str:
    """Format seconds into a human-readable string."""
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h}h {m}m"


def estimate_training_time(num_examples: int, epochs: int, batch_size: int,
                           grad_accum: int, has_gpu: bool) -> str:
    """Print a rough time estimate before training begins."""
    steps_per_epoch = math.ceil(num_examples / (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    # Rough heuristics: ~0.3s/step on GPU, ~4s/step on CPU (Apple MPS ~1s)
    if has_gpu:
        sec_per_step = 0.3
    else:
        try:
            import torch
            sec_per_step = 1.0 if torch.backends.mps.is_available() else 4.0
        except Exception:
            sec_per_step = 4.0
    est = total_steps * sec_per_step
    return (f"~{total_steps} training steps  •  "
            f"Estimated time: {_fmt_seconds(est)} – {_fmt_seconds(est * 1.5)}")


SUPPORTED_MODELS = {
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "smollm2-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2b": "google/gemma-2b-it",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}


def resolve_model_name(name: str) -> str:
    return SUPPORTED_MODELS.get(name.lower(), name)


def load_jsonl_dataset(data_dir: Path):
    """Load all JSONL files into a HuggingFace Dataset."""
    from datasets import Dataset
    records = []
    files = list(data_dir.glob("*.jsonl"))
    if not files:
        # Also check for merged clean output
        clean = data_dir.parent / "clean" / "merged.jsonl"
        if clean.exists():
            files = [clean]
    
    for f in files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

    if not records:
        raise ValueError(f"No training records found in {data_dir}")

    return Dataset.from_list(records)


def format_instruction(record: dict) -> str:
    """Format a record as an instruction-tuning prompt."""
    instruction = record.get("instruction", "Continue from your knowledge.")
    inp = record.get("input", "")
    output = record.get("output", record.get("text", ""))

    if inp:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{output}"
    )


def run_finetune(
    base_model: str,
    data_dir: Path,
    output_dir: Path,
    epochs: int = 3,
    lora_r: int = 16,
    lora_alpha: int = 32,
    learning_rate: float = 2e-4,
    batch_size: int = 2,
    max_seq_length: int = 2048,
    use_unsloth: bool = True,
):
    """
    Run LoRA fine-tuning. Tries unsloth first for speed; falls back to transformers.
    """
    model_name = resolve_model_name(base_model)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_jsonl_dataset(Path(data_dir))
    _emit({"type": "log", "message": f"Loaded {len(dataset)} training examples"})

    # Show time estimate
    try:
        import torch
        has_gpu = torch.cuda.is_available()
    except Exception:
        has_gpu = False
    est = estimate_training_time(len(dataset), epochs, batch_size, 4, has_gpu)
    _emit({"type": "log", "message": f"⏱  {est}"})

    # Format all records
    dataset = dataset.map(lambda x: {"formatted": format_instruction(x)})

    reset_stop()

    if use_unsloth:
        try:
            _train_unsloth(model_name, dataset, output_dir, epochs, lora_r,
                           lora_alpha, learning_rate, batch_size, max_seq_length)
            return
        except ImportError:
            print("unsloth not installed, falling back to transformers")

    _train_transformers(model_name, dataset, output_dir, epochs, lora_r,
                        lora_alpha, learning_rate, batch_size, max_seq_length)


def _train_unsloth(model_name, dataset, output_dir, epochs, lora_r,
                   lora_alpha, lr, batch_size, max_seq_length):
    """Fast path: train with unsloth (2-5x faster, lower VRAM)."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="formatted",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=10,
            output_dir=str(output_dir / "checkpoints"),
            save_strategy="epoch",
        ),
        callbacks=[ETACallback()],
    )
    trainer.train()
    model.save_pretrained(str(output_dir / "latest"))
    tokenizer.save_pretrained(str(output_dir / "latest"))
    _emit({"type": "log", "message": f"✓ Model saved to {output_dir / 'latest'}"})


def _train_transformers(model_name, dataset, output_dir, epochs, lora_r,
                        lora_alpha, lr, batch_size, max_seq_length):
    """Fallback: standard transformers + peft LoRA."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer
    import torch

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="formatted",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=True,
            logging_steps=10,
            output_dir=str(output_dir / "checkpoints"),
            save_strategy="epoch",
        ),
        callbacks=[ETACallback()],
    )
    trainer.train()
    model.save_pretrained(str(output_dir / "latest"))
    tokenizer.save_pretrained(str(output_dir / "latest"))
    _emit({"type": "log", "message": f"✓ Model saved to {output_dir / 'latest'}"})
    _emit({"type": "saved", "path": str(output_dir / "latest")})
