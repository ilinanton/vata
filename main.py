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
    no_analytics: bool = typer.Option(False, "--no-analytics", help="Skip LLM analytics"),
    resume: bool = typer.Option(False, "--resume", help="Resume incomplete job"),
    debug: bool = typer.Option(False, "--debug", help="Keep tmp dir after processing"),
):
    """Transcribe a video/audio file."""
    from pipeline import audio, formatter, llm, transcribe as tr
    from pipeline import chunking

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

    # Chunking config
    cc = config.get("chunking", {})
    chunk_duration = cc.get("chunk_duration", 900)
    overlap_duration = cc.get("overlap_duration", 30)
    similarity_threshold = cc.get("speaker_similarity_threshold", 0.75)

    import time

    def _fmt_duration(seconds: float) -> str:
        """Format seconds as human-readable duration."""
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m:02d}m {s:02d}s"
        if m:
            return f"{m}m {s:02d}s"
        return f"{s}s"

    def _step(label: str, func, *args, **kwargs):
        """Run func with a spinner, print ✓ on success, return result."""
        t0 = time.monotonic()
        with console.status(f"  {label}"):
            result = func(*args, **kwargs)
        elapsed = _fmt_duration(time.monotonic() - t0)
        console.print(f"  [green]✓[/green] {label} [dim]({elapsed})[/dim]")
        return result

    console.print("\n[bold]VATA v1.0[/bold] — Video Audio Text Analytics\n")

    # -- File info --
    file_size_mb = file.stat().st_size / (1024 * 1024)
    console.print(f"  File:   [bold]{file.name}[/bold] ({file_size_mb:.1f} MB)")

    # Step 1: Audio extraction
    audio_path = _step("Extracting audio", audio.extract_audio, file)

    # Step 2: Duration + chunks
    duration = _step("Reading audio duration", audio.get_audio_duration, audio_path)
    console.print(f"  Duration: [bold]{_fmt_duration(duration)}[/bold]")

    job_dir = None
    manifest = None

    if resume:
        job_dir = chunking.find_resumable_job(file)
        if job_dir:
            manifest = chunking.load_manifest(job_dir)
            console.print(f"  [yellow]Resuming[/yellow] from [dim]{job_dir.name}[/dim]")

    if job_dir is None:
        job_dir = chunking.create_job_dir(file)
        manifest = chunking.create_manifest(
            file, audio_path, duration, chunk_duration, overlap_duration, job_dir
        )

    num_chunks = len(manifest["chunks"])
    done_count = sum(1 for c in manifest["chunks"] if c["status"] == "done")
    if done_count:
        console.print(
            f"  Chunks: [bold]{num_chunks}[/bold] "
            f"({done_count} already done, {num_chunks - done_count} remaining)"
        )
    else:
        console.print(f"  Chunks: [bold]{num_chunks}[/bold]")

    _step("Splitting audio into chunks", chunking.split_audio_into_chunks, audio_path, manifest, job_dir)
    chunking.save_manifest(job_dir, manifest)

    # Step 3: Load models
    console.print()
    backend = config.get("transcription", {}).get("backend", "whisperx")
    if backend == "mlx":
        console.print(f"  Backend: [bold]MLX[/bold] (Apple Silicon GPU)")
    else:
        device = tr.detect_device(config)
        ct = tr.compute_type_for_device(device, config)
        batch_size = config.get("transcription", {}).get("batch_size", 16)
        console.print(f"  Device: [bold]{device}[/bold] (compute_type={ct}, batch_size={batch_size})")

    whisper_model_obj, device = _step(
        f"Loading Whisper ({config['transcription']['whisper_model']})",
        tr.load_whisper_model, config,
    )
    import torch
    if device == "mlx" or device == "cpu":
        diarize_device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        diarize_device = device
    diarize_pipeline = _step("Loading diarization pipeline", tr.load_diarization_pipeline, env, diarize_device)

    hf_token = env.get("HF_TOKEN", "")
    embedding_model = _step("Loading embedding model", chunking.load_embedding_model, hf_token)

    # Step 4+5: Transcribe + diarize each chunk
    chunk_results: list[chunking.ChunkResult] = []
    console.print()

    for chunk_info in manifest["chunks"]:
        idx = chunk_info["index"]
        chunk_path = job_dir / chunk_info["filename"]
        chunk_dur = chunk_info["end"] - chunk_info["start"]
        chunk_label = f"Chunk {idx + 1}/{num_chunks} [{_fmt_duration(chunk_info['start'])}–{_fmt_duration(chunk_info['end'])}]"

        # Resume: load cached results for completed chunks
        if chunk_info["status"] == "done":
            loaded = chunking.load_chunk_result(job_dir, chunk_info, manifest["overlap_duration"])
            if loaded is not None:
                chunk_results.append(loaded)
                console.print(f"  [dim]⏭ {chunk_label} — loaded from cache[/dim]")
                continue
            else:
                console.print(f"  [yellow]⚠ {chunk_label} — cache missing, reprocessing[/yellow]")
                chunking.update_chunk_status(manifest, idx, "pending")

        console.print(f"  [bold]{chunk_label}[/bold] ({_fmt_duration(chunk_dur)})")

        # Transcribe
        raw_result = _step("  Transcribing", tr.transcribe, chunk_path, config, model=whisper_model_obj, device=device)
        chunking.update_chunk_status(manifest, idx, "transcribed")
        chunking.save_manifest(job_dir, manifest)

        # Diarize
        segments, raw_diarize = _step(
            "  Diarizing", tr.diarize,
            chunk_path, raw_result, config, env, pipeline=diarize_pipeline,
        )
        chunking.update_chunk_status(manifest, idx, "diarized")

        # Extract embeddings
        speaker_embs = _step(
            "  Extracting speaker embeddings",
            chunking.extract_speaker_embeddings, chunk_path, segments, embedding_model,
        )

        n_speakers = len(set(s["speaker"] for s in segments))
        n_segments = len(segments)
        console.print(f"    [dim]{n_segments} segments, {n_speakers} speaker(s)[/dim]")

        cr = chunking.ChunkResult(
            index=idx,
            start_offset=chunk_info["start"],
            end_offset=chunk_info["end"],
            overlap=manifest["overlap_duration"],
            segments=segments,
            speaker_embeddings=speaker_embs,
        )
        chunking.save_chunk_result(job_dir, cr)

        chunking.update_chunk_status(manifest, idx, "done")
        chunking.save_manifest(job_dir, manifest)

        chunk_results.append(cr)

    # Step 6: Unify speakers + merge
    console.print()
    speaker_mapping = _step("Unifying speakers across chunks", chunking.unify_speakers, chunk_results, similarity_threshold)
    segments = chunking.merge_chunks(chunk_results, speaker_mapping)

    # LLM analysis
    llm_provider = config["llm"]["provider"]
    llm_config = config["llm"][llm_provider]
    client, _ = llm.create_client(config, env)

    llm_speaker_mapping = _step(
        f"Naming speakers ({llm_config['naming_model']})",
        llm.name_speakers, client, llm_config["naming_model"], segments,
    )
    segments = llm.apply_speaker_names(segments, llm_speaker_mapping)

    if no_analytics:
        analytics = {"title": file.stem, "date": "", "summary": ""}
        console.print("  [yellow]–[/yellow] Analytics skipped (--no-analytics)")
    else:
        analytics = _step(
            f"Generating analytics ({llm_config['analytics_model']})",
            llm.generate_analytics, client, llm_config["analytics_model"], segments, file.name,
        )

    # Prepare template data
    participants = list(dict.fromkeys(seg["speaker"] for seg in segments))
    duration_str = tr.calculate_duration(segments)

    data = {
        "title": analytics.get("title", file.stem),
        "date": analytics.get("date", ""),
        "participants": participants,
        "source_file": file.name,
        "whisper_model": config["transcription"]["whisper_model"],
        "analytics_model": llm_config["analytics_model"],
        "duration": duration_str,
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

    # Cleanup
    if not config["output"].get("keep_audio", False) and audio_path != file:
        audio_path.unlink(missing_ok=True)

    if not debug:
        chunking.cleanup_job_dir(job_dir)
    else:
        console.print(f"  [dim]Debug: tmp dir kept at {job_dir}[/dim]")

    # Final output
    console.print(f"\n[green]✓[/green] Done: [bold]{result_path.name}[/bold]")
    console.print(f"  Speakers: {len(participants)} ({', '.join(participants)})")
    console.print(f"  Duration: {duration_str}")
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
