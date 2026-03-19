"""Step 1: extract audio from video file via FFmpeg."""

from pathlib import Path

VIDEO_EXTENSIONS = {".webm", ".mp4", ".mov"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}


def extract_audio(input_path: Path) -> Path:
    """Extract audio from a video file or return audio file as-is.

    - Video files are converted to WAV 16kHz mono.
    - Audio files are returned unchanged.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    ext = input_path.suffix.lower()

    if ext in AUDIO_EXTENSIONS:
        return input_path

    if ext in VIDEO_EXTENSIONS:
        output_path = input_path.with_suffix(".wav")
        import ffmpeg

        (
            ffmpeg
            .input(str(input_path))
            .output(str(output_path), ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path

    supported = sorted(VIDEO_EXTENSIONS | AUDIO_EXTENSIONS)
    raise ValueError(
        f"Unsupported format: {ext}. "
        f"Supported: {', '.join(supported)}"
    )
