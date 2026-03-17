"""
Intelope CLI — train small personal LLMs on your own data.
Usage:
    intelope start          # launch web UI
    intelope ingest <path>  # ingest data from a file or directory
    intelope train          # run fine-tuning
    intelope chat           # chat with your trained model
    intelope status         # show current dataset and model stats
"""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="intelope",
    help="Train small personal LLMs on your own data.",
    add_completion=False,
)
console = Console()


@app.command()
def start(
    host: str = typer.Option("127.0.0.1", help="Host to bind"),
    port: int = typer.Option(7860, help="Port to serve on"),
):
    """Launch Intelope web UI."""
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
    base_model: str = typer.Option("HuggingFaceTB/SmolLM2-1.7B", "--model", "-m"),
    data_dir: Path = typer.Option(Path("data/processed"), "--data"),
    output_dir: Path = typer.Option(Path("models/"), "--output"),
    epochs: int = typer.Option(3, "--epochs", "-e"),
    lora_r: int = typer.Option(16, "--lora-r"),
):
    """Fine-tune a base model on your ingested data using LoRA."""
    from ingestion.training.finetune import run_finetune

    rprint(f"[bold green]🏋  Training[/bold green] on [cyan]{base_model}[/cyan] "
           f"for [bold]{epochs}[/bold] epochs")
    run_finetune(
        base_model=base_model,
        data_dir=data_dir,
        output_dir=output_dir,
        epochs=epochs,
        lora_r=lora_r,
    )


@app.command()
def chat(
    model_dir: Path = typer.Option(Path("models/latest"), "--model"),
    system_prompt: Optional[str] = typer.Option(None, "--system"),
):
    """Chat with your trained model in the terminal."""
    from ingestion.training.inference import chat_loop
    rprint(f"[bold green]💬 Intelope Chat[/bold green] (model: {model_dir})")
    rprint("[dim]Type /exit to quit, /clear to reset context[/dim]\n")
    chat_loop(model_dir=model_dir, system_prompt=system_prompt)


@app.command()
def status():
    """Show current dataset stats and available models."""
    import json
    from pathlib import Path

    table = Table(title="Intelope Status", show_header=True, header_style="bold cyan")
    table.add_column("Item", style="bold")
    table.add_column("Value")

    processed = Path("data/processed")
    models = Path("models")

    chunk_count = sum(1 for f in processed.glob("*.jsonl")
                      for _ in f.open()) if processed.exists() else 0
    model_count = len(list(models.glob("*"))) if models.exists() else 0

    table.add_row("Processed chunks", str(chunk_count))
    table.add_row("Saved models", str(model_count))
    table.add_row("Data directory", str(processed))
    table.add_row("Models directory", str(models))

    console.print(table)


if __name__ == "__main__":
    app()
