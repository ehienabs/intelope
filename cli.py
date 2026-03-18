"""
Intelope CLI — train small personal LLMs on your own data.
Usage:
    intelope start                  # launch web UI
    intelope ingest <path>          # ingest data from a file or directory
    intelope dataset create <name>  # create a named dataset from ingested data
    intelope dataset list           # list saved datasets
    intelope dataset delete <name>  # delete a dataset
    intelope train --dataset <name> --name <model-name>  # train a named model
    intelope chat --model <name>    # chat with a trained model
    intelope status                 # show current stats
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="intelope",
    help="Train small personal LLMs on your own data — privately, locally.",
    add_completion=False,
)
dataset_app = typer.Typer(help="Create and manage named datasets.")
app.add_typer(dataset_app, name="dataset")
console = Console()


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
    source_type: Optional[str] = typer.Option(
        None, "--type", "-t",
        help="Source type: notes | documents | browser | chat (auto-detected if omitted)"
    ),
    output_dir: Path = typer.Option(Path("data/processed"), "--output", "-o"),
):
    """Ingest data from a file or directory into the training dataset."""
    from ingestion.router import ingest_source
    
    if not source.exists():
        rprint(f"[red]Error:[/red] Path does not exist: {source}")
        raise typer.Exit(1)

    with console.status(f"[bold]Ingesting {source}...[/bold]"):
        result = ingest_source(source, source_type, output_dir)

    rprint(f"[green]✓[/green] Ingested [bold]{result['chunks']}[/bold] chunks "
           f"from [bold]{result['files']}[/bold] files → {output_dir}")


@app.command()
def train(
    base_model: str = typer.Option("smollm2-360m", "--model", "-m", help="Base model to fine-tune"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Named dataset to train on"),
    name: str = typer.Option("latest", "--name", "-n", help="Name for the trained model"),
    data_dir: Path = typer.Option(Path("data/datasets"), "--data-dir", hidden=True),
    output_dir: Path = typer.Option(Path("models/"), "--output"),
    epochs: int = typer.Option(3, "--epochs", "-e"),
    lora_r: int = typer.Option(16, "--lora-r"),
):
    """Fine-tune a base model on a named dataset using LoRA."""
    import re

    # Resolve the dataset file
    safe_ds = re.sub(r'[^a-zA-Z0-9_-]', '_', dataset)
    ds_path = data_dir / f"{safe_ds}.jsonl"
    if not ds_path.exists():
        rprint(f"[red]Error:[/red] Dataset not found: {dataset}")
        rprint("[dim]Run [bold]intelope dataset list[/bold] to see available datasets.[/dim]")
        raise typer.Exit(1)

    # Create a temp dir with just this dataset for the trainer
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    (tmp / ds_path.name).symlink_to(ds_path.resolve())

    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)

    from ingestion.training.finetune import run_finetune

    rprint(f"[bold green]🏋  Training[/bold green] model [cyan]{safe_name}[/cyan] "
           f"on dataset [cyan]{dataset}[/cyan] with [cyan]{base_model}[/cyan] "
           f"for [bold]{epochs}[/bold] epochs")
    run_finetune(
        base_model=base_model,
        data_dir=tmp,
        output_dir=output_dir,
        epochs=epochs,
        lora_r=lora_r,
        output_name=safe_name,
    )


@app.command()
def chat(
    model: str = typer.Option("latest", "--model", "-m", help="Name of the trained model"),
    system_prompt: Optional[str] = typer.Option(None, "--system"),
):
    """Chat with a trained model in the terminal."""
    model_dir = Path("models") / model
    if not model_dir.exists():
        rprint(f"[red]Error:[/red] Model not found: {model}")
        rprint("[dim]Run [bold]intelope status[/bold] to see available models.[/dim]")
        raise typer.Exit(1)
    from ingestion.training.inference import chat_loop
    rprint(f"[bold green]💬 Intelope Chat[/bold green] (model: {model})")
    rprint("[dim]Type /exit to quit, /clear to reset context[/dim]\n")
    chat_loop(model_dir=model_dir, system_prompt=system_prompt)


# ── Dataset subcommands ──────────────────────────────────────────────────────

@dataset_app.command("create")
def dataset_create(
    name: str = typer.Argument(..., help="Name for the dataset"),
    input_dir: Path = typer.Option(Path("data/processed"), "--input", "-i"),
):
    """Create a named dataset from ingested data (runs dedup + PII scrub + quality filter)."""
    import re

    if not input_dir.exists() or not list(input_dir.glob("*.jsonl")):
        rprint("[red]Error:[/red] No processed data found. Run [bold]intelope ingest[/bold] first.")
        raise typer.Exit(1)

    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    datasets_dir = Path("data/datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)
    output_path = datasets_dir / f"{safe_name}.jsonl"

    with console.status(f"[bold]Creating dataset '{safe_name}'...[/bold]"):
        from pipeline.clean import run_pipeline
        stats = run_pipeline(input_dir, output_path)

    rprint(f"[green]✓[/green] Dataset [bold]{safe_name}[/bold] created: "
           f"[bold]{stats['output_records']}[/bold] records "
           f"({stats['removed_quality']} low-quality, "
           f"{stats['removed_duplicates']} duplicates, "
           f"{stats['pii_scrubbed']} PII scrubbed)")


@dataset_app.command("list")
def dataset_list():
    """List all saved datasets."""
    datasets_dir = Path("data/datasets")
    if not datasets_dir.exists() or not list(datasets_dir.glob("*.jsonl")):
        rprint("[dim]No datasets found. Create one with [bold]intelope dataset create <name>[/bold][/dim]")
        return

    table = Table(title="Datasets", show_header=True, header_style="bold cyan")
    table.add_column("Name", style="bold")
    table.add_column("Records", justify="right")
    table.add_column("Size", justify="right")

    for f in sorted(datasets_dir.glob("*.jsonl")):
        count = sum(1 for line in f.open() if line.strip())
        size = f.stat().st_size
        size_str = f"{size / 1024:.1f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"
        table.add_row(f.stem, str(count), size_str)

    console.print(table)


@dataset_app.command("delete")
def dataset_delete(
    name: str = typer.Argument(..., help="Name of the dataset to delete"),
):
    """Delete a saved dataset."""
    import re
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', name)
    path = Path("data/datasets") / f"{safe_name}.jsonl"
    if not path.exists():
        rprint(f"[red]Error:[/red] Dataset not found: {name}")
        raise typer.Exit(1)
    path.unlink()
    rprint(f"[green]✓[/green] Deleted dataset [bold]{name}[/bold]")


# ── Status ───────────────────────────────────────────────────────────────────

@app.command()
def status():
    """Show current dataset stats and available models."""
    import json

    processed = Path("data/processed")
    datasets_dir = Path("data/datasets")
    models_dir = Path("models")

    chunk_count = sum(1 for f in processed.glob("*.jsonl")
                      for line in f.open() if line.strip()) if processed.exists() else 0

    # Count datasets
    dataset_count = len(list(datasets_dir.glob("*.jsonl"))) if datasets_dir.exists() else 0

    # Count models (only dirs with adapter_config.json)
    model_names = []
    if models_dir.exists():
        for d in models_dir.iterdir():
            if d.is_dir() and (d / "adapter_config.json").exists():
                model_names.append(d.name)

    table = Table(title="Intelope Status", show_header=True, header_style="bold cyan")
    table.add_column("Item", style="bold")
    table.add_column("Value")

    table.add_row("Ingested chunks", str(chunk_count))
    table.add_row("Saved datasets", str(dataset_count))
    table.add_row("Trained models", str(len(model_names)))
    if model_names:
        table.add_row("Model names", ", ".join(model_names))

    console.print(table)

    if dataset_count > 0:
        rprint("\n[dim]Datasets:[/dim]")
        for f in sorted(datasets_dir.glob("*.jsonl")):
            count = sum(1 for line in f.open() if line.strip())
            rprint(f"  [cyan]{f.stem}[/cyan] — {count} records")


if __name__ == "__main__":
    app()
