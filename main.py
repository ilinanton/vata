"""VATA — Video Audio Text Analytics. CLI entry point."""

import shutil
import tomllib
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console

app = typer.Typer(help="VATA — Video Audio Text Analytics")
models_app = typer.Typer(help="Model management")
app.add_typer(models_app, name="models")

console = Console()

# -- Load configuration ----------------------------------------


def load_config() -> dict:
    """Load config.toml."""
    config_path = Path(__file__).parent / "config.toml"
    if not config_path.exists():
        console.print("[red]✗[/red] config.toml not found")
        raise typer.Exit(1)
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_env() -> dict:
    """Load .env and return a dict of tokens."""
    import os

    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path)
    return {
        "HF_TOKEN": os.getenv("HF_TOKEN", ""),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY", ""),
    }


# -- Command: check --------------------------------------------


@app.command()
def check():
    """Check environment and dependencies."""
    console.print("\n[bold]VATA — environment check[/bold]\n")

    # FFmpeg
    if shutil.which("ffmpeg"):
        console.print("[green]✓[/green] FFmpeg installed")
    else:
        console.print("[red]✗[/red] FFmpeg not found — install: [bold]brew install ffmpeg[/bold]")

    # Ollama
    import urllib.request
    import urllib.error

    try:
        urllib.request.urlopen("http://localhost:11434", timeout=3)
        console.print("[green]✓[/green] Ollama is running")
    except (urllib.error.URLError, ConnectionError, OSError):
        console.print("[red]✗[/red] Ollama is not responding — start: [bold]make ollama[/bold]")

    # Tokens
    env = load_env()
    if env["HF_TOKEN"] and not env["HF_TOKEN"].startswith("hf_xxx"):
        console.print("[green]✓[/green] HF_TOKEN is set")
    else:
        console.print("[red]✗[/red] HF_TOKEN not set in .env — required for diarization (pyannote)")

    if env["OPENROUTER_API_KEY"] and not env["OPENROUTER_API_KEY"].startswith("sk-or-xxx"):
        console.print("[green]✓[/green] OPENROUTER_API_KEY is set")
    else:
        console.print("[yellow]–[/yellow] OPENROUTER_API_KEY not set (only needed for provider=openrouter)")

    # Config
    try:
        config = load_config()
        provider = config.get("llm", {}).get("provider", "ollama")
        whisper = config.get("transcription", {}).get("whisper_model", "?")
        console.print(f"[green]✓[/green] config.toml loaded (provider={provider}, whisper={whisper})")
    except SystemExit:
        pass

    console.print()


# -- Command: transcribe ---------------------------------------


@app.command()
def transcribe(
    file: Path = typer.Argument(..., help="Path to video/audio file"),
    provider: str = typer.Option(None, "--provider", "-p", help="LLM provider: ollama or openrouter"),
    naming_model: str = typer.Option(None, "--naming-model", help="Model for speaker naming"),
    analytics_model: str = typer.Option(None, "--analytics-model", help="Model for analytics"),
    speakers: int = typer.Option(None, "--speakers", "-s", help="Expected number of speakers"),
    output: Path = typer.Option(None, "--output", "-o", help="Output directory"),
):
    """Transcribe a video/audio file."""
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

    from pipeline import audio, formatter, llm, transcribe as tr

    # Validate file
    if not file.exists():
        console.print(f"[red]✗[/red] File not found: {file}")
        raise typer.Exit(1)

    # Load config
    config = load_config()
    env = load_env()

    # CLI args override config.toml
    if provider:
        config["llm"]["provider"] = provider
    if naming_model:
        config["llm"][config["llm"]["provider"]]["naming_model"] = naming_model
    if analytics_model:
        config["llm"][config["llm"]["provider"]]["analytics_model"] = analytics_model
    if speakers:
        config["diarization"]["min_speakers"] = speakers
        config["diarization"]["max_speakers"] = speakers
    if output:
        config["output"]["output_dir"] = str(output)

    console.print("\n[bold]VATA v1.0[/bold] — Video Audio Text Analytics\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Step 1: Audio extraction
        task = progress.add_task("[1/4] Extracting audio...", total=None)
        audio_path = audio.extract_audio(file)
        progress.update(task, completed=True)
        progress.remove_task(task)
        console.print("[green]✓[/green] [1/4] Audio extraction")

        # Step 2: Transcription
        task = progress.add_task("[2/4] Transcribing...", total=None)
        raw_result = tr.transcribe(audio_path, config)
        progress.update(task, completed=True)
        progress.remove_task(task)
        whisper_model = config["transcription"]["whisper_model"]
        console.print(f"[green]✓[/green] [2/4] Transcription ({whisper_model})")

        # Step 3: Diarization
        task = progress.add_task("[3/4] Diarizing...", total=None)
        segments = tr.diarize(audio_path, raw_result, config, env)
        progress.update(task, completed=True)
        progress.remove_task(task)
        console.print("[green]✓[/green] [3/4] Diarization (pyannote)")

        # Step 4: LLM analysis
        task = progress.add_task("[4/4] Analyzing...", total=None)
        llm_provider = config["llm"]["provider"]
        llm_config = config["llm"][llm_provider]
        client, _ = llm.create_client(config, env)

        # Speaker naming
        speaker_mapping = llm.name_speakers(client, llm_config["naming_model"], segments)
        segments = llm.apply_speaker_names(segments, speaker_mapping)

        # Analytics
        analytics = llm.generate_analytics(
            client, llm_config["analytics_model"], segments, file.name
        )
        progress.update(task, completed=True)
        progress.remove_task(task)
        console.print(f"[green]✓[/green] [4/4] Analysis ({llm_config['naming_model']})")

    # Prepare template data
    participants = list(dict.fromkeys(seg["speaker"] for seg in segments))
    duration = tr.calculate_duration(segments)

    data = {
        "title": analytics.get("title", file.stem),
        "date": analytics.get("date", ""),
        "participants": participants,
        "source_file": file.name,
        "whisper_model": whisper_model,
        "analytics_model": llm_config["analytics_model"],
        "duration": duration,
        "summary": analytics.get("summary", ""),
        "segments": [
            {
                "speaker": seg["speaker"],
                "timestamp": tr.format_timestamp(seg["start"]),
                "text": seg["text"],
            }
            for seg in segments
        ],
    }

    # Render output
    output_dir = config["output"].get("output_dir", "")
    if output_dir:
        out_path = Path(output_dir) / f"{file.stem}.md"
    else:
        out_path = file.parent / f"{file.stem}.md"

    result_path = formatter.render_transcript(data, out_path)

    # Remove temporary WAV
    if not config["output"].get("keep_audio", False) and audio_path != file:
        audio_path.unlink(missing_ok=True)

    # Final output
    console.print(f"\n[green]✓[/green] Done: [bold]{result_path.name}[/bold]")
    console.print(f"  Speakers: {len(participants)} ({', '.join(participants)})")
    console.print(f"  Duration: {duration}")
    console.print(f"  Segments: {len(segments)}")
    console.print()


# -- Commands: models ------------------------------------------


@models_app.command("whisper")
def models_whisper():
    """Show available Whisper models."""
    from rich.table import Table

    table = Table(title="Whisper Models")
    table.add_column("Model", style="bold")
    table.add_column("Size", justify="right")
    table.add_column("Note")

    whisper_models = [
        ("tiny", "~75 MB", "Fastest, low quality"),
        ("base", "~150 MB", "Fast, acceptable quality"),
        ("small", "~500 MB", "Balance of speed and quality"),
        ("medium", "~1.5 GB", "Good quality"),
        ("large-v2", "~3 GB", "Recommended for Russian"),
        ("large-v3", "~3 GB", "Latest, experimental"),
    ]
    for name, size, note in whisper_models:
        table.add_row(name, size, note)

    console.print(table)


@models_app.command("ollama")
def models_ollama():
    """Show available Ollama models."""
    from rich.table import Table

    try:
        import ollama as ollama_client

        response = ollama_client.list()
        models_list = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
    except Exception:
        console.print("[red]✗[/red] Ollama is not running — start: [bold]make ollama[/bold]")
        raise typer.Exit(1)

    if not models_list:
        console.print("[yellow]No downloaded models.[/yellow] Download: [bold]make models[/bold]")
        return

    table = Table(title="Ollama Models")
    table.add_column("Model", style="bold")
    table.add_column("Size", justify="right")

    for model in models_list:
        name = model.get("name", "") if isinstance(model, dict) else getattr(model, "model", str(model))
        size_bytes = model.get("size", 0) if isinstance(model, dict) else getattr(model, "size", 0)
        size_gb = f"{size_bytes / 1e9:.1f} GB" if size_bytes else "?"
        table.add_row(name, size_gb)

    console.print(table)


# -- Error handling --------------------------------------------


def _error_handler(func):
    """Decorator for Rich-friendly error handling."""
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FileNotFoundError as e:
            console.print(f"\n[red]✗[/red] File not found: {e}")
            raise typer.Exit(1)
        except ValueError as e:
            msg = str(e)
            if "Unsupported format" in msg:
                console.print(f"\n[red]✗[/red] {msg}")
            else:
                console.print(f"\n[red]✗[/red] Error: {msg}")
            raise typer.Exit(1)
        except ImportError as e:
            console.print(f"\n[red]✗[/red] Failed to import module: {e}")
            console.print("    Make sure the environment is activated: [bold]conda activate vata[/bold]")
            raise typer.Exit(1)
        except ConnectionError:
            console.print("\n[red]✗[/red] Ollama is not responding — start: [bold]make ollama[/bold]")
            raise typer.Exit(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
            raise typer.Exit(130)
        except typer.Exit:
            raise
        except Exception as e:
            console.print(f"\n[red]✗[/red] Unexpected error: {e}")
            raise typer.Exit(1)

    return wrapper


# Wrap the transcribe command
transcribe = _error_handler(transcribe)


# -- Entry point -----------------------------------------------

if __name__ == "__main__":
    app()
