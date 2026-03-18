"""
Intelope CLI — train small personal LLMs on your own data.
Usage:
    intelope start                           # launch web UI
    intelope dataset create <name>           # create a new dataset
    intelope dataset list                    # list datasets and status
    intelope dataset delete <name>           # delete a dataset
    intelope ingest <path> --dataset <name>  # ingest files into a dataset
    intelope dataset clean <name>            # clean a dataset for training
    intelope train --dataset <name> --name <model>  # train a model
    intelope index --dataset <name>          # build RAG search index
    intelope chat --model <name>             # chat with a trained model
    intelope chat --model <name> --rag       # chat with RAG context
    intelope status                          # show current stats
"""

import typer
from pathlib import Path
from typing import Optional
import re
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="intelope",
    help="Train small personal LLMs on your own data — privately, locally.",
    add_completion=False,
)
dataset_app = typer.Typer(help="Create and manage datasets.")
app.add_typer(dataset_app, name="dataset")
console = Console()

DATASETS_DIR = Path("data/datasets")


def _sanitize(name: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_-]', '_', name.strip())


@app.command()
def start(
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(7860, help="Port to serve on"),
):
    """Launch the Intelope web UI."""
    rprint(f"[bold green]🧠 Intelope[/bold green] starting at http://{host}:{port}")
    from ui.app import launch
    launch(host=host, port=port)


@app.command()
def ingest(
    source: Path = typer.Argument(..., help="File or directory to ingest"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset to ingest into"),
    source_type: Optional[str] = typer.Option(
        None, "--type", "-t",
        help="Source type: notes | documents | browser | chat (auto-detected if omitted)"
    ),
):
    """Ingest data from a file or directory into a dataset."""
    from ingestion.router import ingest_source

    if not source.exists():
        rprint(f"[red]Error:[/red] Path does not exist: {source}")
        raise typer.Exit(1)

    safe = _sanitize(dataset)
    ds_dir = DATASETS_DIR / safe
    if not ds_dir.exists():
        rprint(f"[red]Error:[/red] Dataset not found: {dataset}")
        rprint("[dim]Run [bold]intelope dataset create <name>[/bold] first.[/dim]")
        raise typer.Exit(1)

    output_dir = ds_dir / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    with console.status(f"[bold]Ingesting {source} into dataset '{safe}'...[/bold]"):
        result = ingest_source(source, source_type, output_dir)

    rprint(f"[green]✓[/green] Ingested [bold]{result['chunks']}[/bold] chunks "
           f"from [bold]{result['files']}[/bold] files → {safe}")


@app.command()
def train(
    base_model: str = typer.Option("smollm2-360m", "--model", "-m", help="Base model to fine-tune"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset(s) to train on "),
    name: str = typer.Option("latest", "--name", "-n", help="Name for the trained model"),
    output_dir: Path = typer.Option(Path("models/"), "--output"),
    epochs: int = typer.Option(3, "--epochs", "-e"),
    lora_r: int = typer.Option(16, "--lora-r"),
):
    """Fine-tune a base model on one or more cleaned datasets using LoRA."""
    import tempfile

    ds_names = [d.strip() for d in dataset.split(",") if d.strip()]
    if not ds_names:
        rprint("[red]Error:[/red] No dataset specified.")
        raise typer.Exit(1)

    # Merge clean.jsonl from all specified datasets
    merged_lines: list[str] = []
    for ds in ds_names:
        safe_ds = _sanitize(ds)
        ds_clean = DATASETS_DIR / safe_ds / "clean.jsonl"
        if not ds_clean.exists():
            rprint(f"[red]Error:[/red] Dataset '{ds}' not found or not cleaned yet.")
            rprint(f"[dim]Run [bold]intelope dataset clean {ds}[/bold] first.[/dim]")
            raise typer.Exit(1)
        merged_lines.extend(ds_clean.read_text().splitlines())

    if not merged_lines:
        rprint("[red]Error:[/red] Selected dataset(s) have no clean records.")
        raise typer.Exit(1)

    tmp = Path(tempfile.mkdtemp())
    (tmp / "clean.jsonl").write_text("\n".join(merged_lines) + "\n")

    safe_name = _sanitize(name) or "latest"

    from ingestion.training.finetune import run_finetune

    ds_label = ", ".join(ds_names)
    rprint(f"[bold green]🏋  Training[/bold green] model [cyan]{safe_name}[/cyan] "
           f"on dataset{'s' if len(ds_names) > 1 else ''} [cyan]{ds_label}[/cyan] with [cyan]{base_model}[/cyan] "
           f"for [bold]{epochs}[/bold] epochs ({len(merged_lines)} examples)")
    run_finetune(
        base_model=base_model,
        data_dir=tmp,
        output_dir=output_dir,
        epochs=epochs,
        lora_r=lora_r,
        output_name=safe_name,
        dataset_name=ds_label,
    )


@app.command()
def chat(
    model: str = typer.Option("latest", "--model", "-m", help="Name of the trained model"),
    system_prompt: Optional[str] = typer.Option(None, "--system"),
    rag: bool = typer.Option(False, "--rag", help="Enable RAG (search your documents for context)"),
):
    """Chat with a trained model in the terminal."""
    model_dir = Path("models") / model
    if not model_dir.exists():
        rprint(f"[red]Error:[/red] Model not found: {model}")
        rprint("[dim]Run [bold]intelope status[/bold] to see available models.[/dim]")
        raise typer.Exit(1)
    from ingestion.training.inference import chat_loop
    rprint(f"[bold green]\U0001f4ac Intelope Chat[/bold green] (model: {model})")
    if rag:
        rprint("[dim]RAG enabled \u2014 answers will be grounded in your documents[/dim]")
    rprint("[dim]Type /exit to quit, /clear to reset context[/dim]\n")
    chat_loop(model_dir=model_dir, system_prompt=system_prompt, use_rag=rag)


@app.command()
def index(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset(s) to index (comma-separated)"),
):
    """Build a RAG search index from cleaned datasets."""
    ds_names = [d.strip() for d in dataset.split(",") if d.strip()]
    if not ds_names:
        rprint("[red]Error:[/red] No dataset specified.")
        raise typer.Exit(1)

    for ds in ds_names:
        safe_ds = _sanitize(ds)
        ds_clean = DATASETS_DIR / safe_ds / "clean.jsonl"
        if not ds_clean.exists():
            rprint(f"[red]Error:[/red] Dataset '{ds}' not found or not cleaned yet.")
            rprint(f"[dim]Run [bold]intelope dataset clean {ds}[/bold] first.[/dim]")
            raise typer.Exit(1)

    safe_names = [_sanitize(d) for d in ds_names]
    with console.status(f"[bold]Building RAG index from {', '.join(ds_names)}...[/bold]"):
        from pipeline.rag import build_index
        stats = build_index(safe_names, datasets_dir=DATASETS_DIR)

    if stats.get("error"):
        rprint(f"[red]Error:[/red] {stats['error']}")
        raise typer.Exit(1)

    rprint(f"[green]✓[/green] RAG index built: [bold]{stats['chunks']}[/bold] chunks indexed")
    rprint(f"[dim]Now use [bold]intelope chat --model <name> --rag[/bold] to chat with document context.[/dim]")


# ── Dataset subcommands ──────────────────────────────────────────────────────

@dataset_app.command("create")
def dataset_create(
    name: str = typer.Argument(..., help="Name for the dataset"),
):
    """Create a new empty dataset."""
    safe_name = _sanitize(name)
    ds_dir = DATASETS_DIR / safe_name
    if ds_dir.exists():
        rprint(f"[red]Error:[/red] Dataset '{safe_name}' already exists.")
        raise typer.Exit(1)

    (ds_dir / "uploads").mkdir(parents=True)
    (ds_dir / "processed").mkdir(parents=True)
    rprint(f"[green]✓[/green] Dataset [bold]{safe_name}[/bold] created. "
           f"Now ingest data with: [bold]intelope ingest <path> --dataset {safe_name}[/bold]")


@dataset_app.command("clean")
def dataset_clean(
    name: str = typer.Argument(..., help="Name of the dataset to clean"),
):
    """Run dedup, PII scrubbing, and quality filtering on a dataset."""
    safe_name = _sanitize(name)
    ds_dir = DATASETS_DIR / safe_name
    processed = ds_dir / "processed"

    if not ds_dir.exists():
        rprint(f"[red]Error:[/red] Dataset not found: {name}")
        raise typer.Exit(1)
    if not processed.exists() or not list(processed.glob("*.jsonl")):
        rprint("[red]Error:[/red] No ingested data in this dataset. Ingest files first.")
        raise typer.Exit(1)

    with console.status(f"[bold]Cleaning dataset '{safe_name}'...[/bold]"):
        from pipeline.clean import run_pipeline
        stats = run_pipeline(processed, ds_dir / "clean.jsonl")

    rprint(f"[green]✓[/green] Dataset [bold]{safe_name}[/bold] cleaned: "
           f"[bold]{stats['output_records']}[/bold] records "
           f"({stats['removed_quality']} low-quality, "
           f"{stats['removed_duplicates']} duplicates, "
           f"{stats['pii_scrubbed']} PII scrubbed)")


@dataset_app.command("list")
def dataset_list():
    """List all datasets and their status."""
    if not DATASETS_DIR.exists():
        rprint("[dim]No datasets found. Create one with [bold]intelope dataset create <name>[/bold][/dim]")
        return

    dirs = [d for d in sorted(DATASETS_DIR.iterdir()) if d.is_dir()]
    if not dirs:
        rprint("[dim]No datasets found. Create one with [bold]intelope dataset create <name>[/bold][/dim]")
        return

    table = Table(title="Datasets", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Files", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Status")

    for d in dirs:
        uploads = d / "uploads"
        processed = d / "processed"
        clean = d / "clean.jsonl"
        file_count = sum(1 for _ in uploads.glob("*")) if uploads.exists() else 0
        chunk_count = 0
        if processed.exists():
            for f in processed.glob("*.jsonl"):
                chunk_count += sum(1 for line in f.open() if line.strip())
        if clean.exists():
            clean_count = sum(1 for line in clean.open() if line.strip())
            status = f"[green]✓ Ready ({clean_count} records)[/green]"
        elif chunk_count > 0:
            status = "[yellow]Needs cleaning[/yellow]"
        else:
            status = "[dim]Empty[/dim]"
        table.add_row(d.name, str(file_count), str(chunk_count), status)

    console.print(table)


@dataset_app.command("delete")
def dataset_delete(
    name: str = typer.Argument(..., help="Name of the dataset to delete"),
):
    """Delete a dataset and all its data."""
    import shutil
    safe_name = _sanitize(name)
    ds_dir = DATASETS_DIR / safe_name
    if not ds_dir.exists():
        rprint(f"[red]Error:[/red] Dataset not found: {name}")
        raise typer.Exit(1)
    shutil.rmtree(ds_dir)
    rprint(f"[green]✓[/green] Deleted dataset [bold]{name}[/bold]")


# ── Status ───────────────────────────────────────────────────────────────────

@app.command()
def status():
    """Show current dataset stats and available models."""
    models_dir = Path("models")

    # Count datasets and their data
    dataset_count = 0
    total_chunks = 0
    ready_count = 0
    if DATASETS_DIR.exists():
        for d in DATASETS_DIR.iterdir():
            if not d.is_dir():
                continue
            dataset_count += 1
            processed = d / "processed"
            if processed.exists():
                for f in processed.glob("*.jsonl"):
                    total_chunks += sum(1 for line in f.open() if line.strip())
            if (d / "clean.jsonl").exists():
                ready_count += 1

    # Count models (only dirs with adapter_config.json)
    model_names = []
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir() and (d / "adapter_config.json").exists():
                model_names.append(d.name)

    table = Table(title="Intelope Status", show_header=True, header_style="bold cyan")
    table.add_column("Item", style="bold")
    table.add_column("Value")

    table.add_row("Datasets", str(dataset_count))
    table.add_row("Ready for training", str(ready_count))
    table.add_row("Total chunks", str(total_chunks))
    table.add_row("Trained models", str(len(model_names)))
    if model_names:
        table.add_row("Model names", ", ".join(model_names))

    from pipeline.rag import index_exists, get_index_info
    if index_exists():
        info = get_index_info() or {}
        table.add_row("RAG index", f"✓ {info.get('total_chunks', '?')} chunks")
    else:
        table.add_row("RAG index", "not built")

    console.print(table)


if __name__ == "__main__":
    app()
