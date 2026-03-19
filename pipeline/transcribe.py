"""Step 2: transcription + diarization via WhisperX and Pyannote."""

from pathlib import Path

_patched = False


def detect_device(config: dict) -> str:
    """Return the best available compute device for CTranslate2.

    CTranslate2 (WhisperX backend) supports only 'cpu' and 'cuda'.
    Config value 'auto' (default) probes CUDA then falls back to CPU.
    """
    import torch

    requested = config.get("transcription", {}).get("device", "auto")
    if requested in ("cpu", "cuda"):
        return requested

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_type_for_device(device: str, config: dict) -> str:
    """Pick compute_type appropriate for the device.

    CUDA uses float16; CPU uses the config value (default int8).
    """
    configured = config.get("transcription", {}).get("compute_type", "int8")
    if device == "cuda":
        return "float16"
    return configured

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


def load_whisper_model(config: dict):
    """Load WhisperX model. Call once and reuse across chunks."""
    _apply_compat_patches()
    import whisperx

    device = detect_device(config)
    ct = compute_type_for_device(device, config)
    tc = config["transcription"]

    threads = tc.get("threads", 0)
    kwargs = {}
    if threads > 0:
        kwargs["cpu_threads"] = threads

    return whisperx.load_model(
        tc["whisper_model"],
        device=device,
        compute_type=ct,
        language=tc.get("language", "ru") if tc.get("language") != "auto" else None,
        **kwargs,
    ), device


def load_diarization_pipeline(env: dict, device: str = "cpu"):
    """Load pyannote DiarizationPipeline. Call once and reuse across chunks."""
    _apply_compat_patches()
    from whisperx.diarize import DiarizationPipeline

    hf_token = env.get("HF_TOKEN", "")
    return DiarizationPipeline(use_auth_token=hf_token, device=device)


def transcribe(audio_path: Path, config: dict, model=None, device: str = "cpu") -> dict:
    """Transcribe an audio file via WhisperX.

    Returns the raw result dict from whisperx (segments with word-level timestamps).
    If model is provided, reuses it; otherwise loads a new one (backward compat).
    """
    _apply_compat_patches()
    import whisperx

    tc = config["transcription"]

    if model is None:
        model, device = load_whisper_model(config)

    batch_size = tc.get("batch_size", 16)

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size)

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


def diarize(
    audio_path: Path, result: dict, config: dict, env: dict, pipeline=None
) -> tuple[list[dict], object]:
    """Perform diarization and merge with transcript.

    Returns a tuple of (segments, raw_diarize_segments).
    - segments: [{"speaker": ..., "start": ..., "end": ..., "text": ...}]
    - raw_diarize_segments: raw pyannote diarization output (for embedding extraction)

    If pipeline is provided, reuses it; otherwise loads a new one (backward compat).
    """
    _apply_compat_patches()
    import whisperx

    device = "cpu"
    dc = config["diarization"]

    if pipeline is None:
        pipeline = load_diarization_pipeline(env, device)

    diarize_segments = pipeline(
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

    return segments, diarize_segments


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
