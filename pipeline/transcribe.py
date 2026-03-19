"""Step 2: transcription + diarization via WhisperX and Pyannote."""

from pathlib import Path

_patched = False

def _apply_compat_patches():
    """Compatibility patches for PyTorch 2.6+ and huggingface_hub 0.27+."""
    global _patched
    if _patched:
        return
    _patched = True

    import torch
    # PyTorch 2.6+ defaults to weights_only=True in torch.load,
    # but pyannote.audio checkpoints use globals not yet on the allowlist.
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

    # huggingface_hub 0.27+ removed use_auth_token in favour of token.
    # pyannote.audio 3.x still passes use_auth_token everywhere, so we
    # monkey-patch the key huggingface_hub functions to translate it.
    import functools
    import huggingface_hub

    def _wrap_use_auth_token(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if "use_auth_token" in kwargs:
                kwargs.setdefault("token", kwargs.pop("use_auth_token"))
            return fn(*args, **kwargs)
        return wrapper

    for _fname in ("hf_hub_download", "model_info", "cached_download"):
        _fn = getattr(huggingface_hub, _fname, None)
        if _fn and not getattr(_fn, "_compat_patched", False):
            _wrapped = _wrap_use_auth_token(_fn)
            _wrapped._compat_patched = True
            setattr(huggingface_hub, _fname, _wrapped)

    # Patch already-imported references in pyannote modules
    import sys
    for mod_name, mod in sys.modules.items():
        if mod and mod_name.startswith("pyannote."):
            for _fname in ("hf_hub_download", "model_info"):
                if hasattr(mod, _fname):
                    setattr(mod, _fname, getattr(huggingface_hub, _fname))


def transcribe(audio_path: Path, config: dict) -> dict:
    """Transcribe an audio file via WhisperX.

    Returns the raw result dict from whisperx (segments with word-level timestamps).
    """
    _apply_compat_patches()
    import whisperx

    device = "cpu"
    tc = config["transcription"]

    model = whisperx.load_model(
        tc["whisper_model"],
        device=device,
        compute_type=tc.get("compute_type", "int8"),
        language=tc.get("language", "ru") if tc.get("language") != "auto" else None,
    )

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio)

    # Word-level timestamp alignment
    language = result.get("language", tc.get("language", "ru"))
    if language == "auto":
        language = result.get("language", "ru")

    align_model, metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, device
    )

    return result


def diarize(audio_path: Path, result: dict, config: dict, env: dict) -> list[dict]:
    """Perform diarization and merge with transcript.

    Returns a list of segments: [{"speaker": ..., "start": ..., "end": ..., "text": ...}]
    """
    _apply_compat_patches()
    import whisperx

    device = "cpu"
    dc = config["diarization"]
    hf_token = env.get("HF_TOKEN", "")

    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(
        use_auth_token=hf_token, device=device
    )

    diarize_segments = diarize_model(
        str(audio_path),
        min_speakers=dc.get("min_speakers", 2),
        max_speakers=dc.get("max_speakers", 6),
    )

    result = whisperx.assign_word_speakers(diarize_segments, result)

    # Build final segments
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "speaker": seg.get("speaker", "UNKNOWN"),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "").strip(),
        })

    return segments


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def calculate_duration(segments: list[dict]) -> str:
    """Calculate total duration from the last segment."""
    if not segments:
        return "00:00:00"
    last_end = max(seg.get("end", 0.0) for seg in segments)
    return format_timestamp(last_end)
