"""Chunked processing: splitting, manifest, speaker unification, and merging."""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np


# -- Data structures ---------------------------------------------------

@dataclass
class ChunkResult:
    index: int
    start_offset: float
    end_offset: float
    overlap: float
    segments: list[dict]
    speaker_embeddings: dict[str, np.ndarray] = field(default_factory=dict)


# -- Hashing and directories ------------------------------------------

def compute_file_hash(file_path: Path) -> str:
    """MD5 hash of file, first 8 hex characters."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _default_tmp_dir() -> Path:
    return Path(__file__).parent.parent / "tmp"


def create_job_dir(file_path: Path, base_dir: Path = None) -> Path:
    """Create tmp/{hash8}_{YYYYMMDD_HHMMSS}/. Return the path."""
    if base_dir is None:
        base_dir = _default_tmp_dir()
    file_hash = compute_file_hash(file_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = base_dir / f"{file_hash}_{timestamp}"
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_dir


def find_resumable_job(file_path: Path, base_dir: Path = None) -> Path | None:
    """Find a job dir in tmp/ with the same file hash. None if not found or manifest invalid."""
    if base_dir is None:
        base_dir = _default_tmp_dir()
    if not base_dir.exists():
        return None

    file_hash = compute_file_hash(file_path)
    candidates = sorted(base_dir.iterdir(), reverse=True)  # newest first
    for d in candidates:
        if d.is_dir() and d.name.startswith(file_hash + "_"):
            manifest = load_manifest(d)
            if manifest and manifest.get("input_hash") == file_hash:
                return d
    return None


# -- Manifest ----------------------------------------------------------

def create_manifest(
    input_file: Path,
    audio_path: Path,
    duration: float,
    chunk_dur: float,
    overlap_dur: float,
    job_dir: Path,
) -> dict:
    """Create a manifest dict describing all chunks."""
    chunks = []
    idx = 0
    start = 0.0
    while start < duration:
        end = min(start + chunk_dur, duration)
        chunks.append({
            "index": idx,
            "filename": f"chunk_{idx:03d}.wav",
            "start": start,
            "end": end,
            "status": "pending",
        })
        idx += 1
        start += chunk_dur - overlap_dur
        if end >= duration:
            break

    manifest = {
        "version": 1,
        "input_file": str(input_file.resolve()),
        "input_hash": compute_file_hash(input_file),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "audio_path": str(audio_path.resolve()),
        "chunk_duration": chunk_dur,
        "overlap_duration": overlap_dur,
        "chunks": chunks,
    }
    save_manifest(job_dir, manifest)
    return manifest


def load_manifest(job_dir: Path) -> dict | None:
    """Load manifest.json from a job dir. Returns None on error."""
    manifest_path = job_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_manifest(job_dir: Path, manifest: dict) -> None:
    """Atomically write manifest.json (write to tmp + rename)."""
    manifest_path = job_dir / "manifest.json"
    fd, tmp_path = tempfile.mkstemp(dir=job_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, manifest_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def update_chunk_status(manifest: dict, chunk_index: int, status: str) -> dict:
    """Update the status of a chunk in the manifest."""
    for chunk in manifest["chunks"]:
        if chunk["index"] == chunk_index:
            chunk["status"] = status
            break
    return manifest


# -- Chunk result persistence ------------------------------------------

def save_chunk_result(job_dir: Path, result: "ChunkResult") -> None:
    """Persist chunk segments and speaker embeddings to disk."""
    prefix = f"chunk_{result.index:03d}"
    seg_path = job_dir / f"{prefix}_segments.json"
    with open(seg_path, "w") as f:
        json.dump(result.segments, f, ensure_ascii=False)
    if result.speaker_embeddings:
        emb_path = job_dir / f"{prefix}_embeddings.npz"
        np.savez(emb_path, **result.speaker_embeddings)


def load_chunk_result(job_dir: Path, chunk_info: dict, overlap: float) -> "ChunkResult | None":
    """Load persisted chunk result. Returns None if files are missing."""
    idx = chunk_info["index"]
    prefix = f"chunk_{idx:03d}"
    seg_path = job_dir / f"{prefix}_segments.json"

    if not seg_path.exists():
        return None

    with open(seg_path, "r") as f:
        segments = json.load(f)

    embeddings: dict[str, np.ndarray] = {}
    emb_path = job_dir / f"{prefix}_embeddings.npz"
    if emb_path.exists():
        data = np.load(emb_path)
        embeddings = {key: data[key] for key in data.files}
        data.close()

    return ChunkResult(
        index=idx,
        start_offset=chunk_info["start"],
        end_offset=chunk_info["end"],
        overlap=overlap,
        segments=segments,
        speaker_embeddings=embeddings,
    )


# -- Audio splitting ---------------------------------------------------

def split_audio_into_chunks(audio_path: Path, manifest: dict, job_dir: Path) -> None:
    """Split audio into chunks via FFmpeg. Skips already existing chunks (resume)."""
    import ffmpeg

    for chunk in manifest["chunks"]:
        chunk_path = job_dir / chunk["filename"]
        if chunk_path.exists() and chunk["status"] != "pending":
            continue

        start = chunk["start"]
        duration = chunk["end"] - chunk["start"]

        (
            ffmpeg
            .input(str(audio_path), ss=start, t=duration)
            .output(str(chunk_path), ac=1, ar=16000)
            .overwrite_output()
            .run(quiet=True)
        )
        chunk["status"] = "split"


# -- Speaker embeddings -----------------------------------------------

def load_embedding_model(hf_token: str):
    """Load pyannote/wespeaker-voxceleb-resnet34-LM for speaker embeddings."""
    from pipeline.transcribe import _apply_compat_patches
    _apply_compat_patches()

    from pyannote.audio import Inference

    return Inference(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        use_auth_token=hf_token,
    )


def extract_speaker_embeddings(
    audio_path: Path,
    segments: list[dict],
    embedding_model,
) -> dict[str, np.ndarray]:
    """Extract speaker embeddings from the longest segments per speaker."""
    from pyannote.core import Segment

    # Group segments by speaker, pick top 3 longest
    speaker_segs: dict[str, list[dict]] = {}
    for seg in segments:
        sp = seg["speaker"]
        speaker_segs.setdefault(sp, []).append(seg)

    embeddings = {}
    for speaker, segs in speaker_segs.items():
        segs_sorted = sorted(segs, key=lambda s: s["end"] - s["start"], reverse=True)
        top_segs = segs_sorted[:3]

        emb_list = []
        for seg in top_segs:
            duration = seg["end"] - seg["start"]
            if duration < 0.5:
                continue
            try:
                crop = Segment(seg["start"], seg["end"])
                emb = embedding_model.crop(str(audio_path), crop)
                if emb is not None and len(emb.shape) > 0:
                    # emb may be (1, D) or (D,)
                    vec = emb.flatten()
                    if np.isfinite(vec).all():
                        emb_list.append(vec)
            except Exception:
                continue

        if emb_list:
            embeddings[speaker] = np.mean(emb_list, axis=0)

    return embeddings


# -- Speaker unification -----------------------------------------------

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def unify_speakers(
    chunk_results: list[ChunkResult],
    threshold: float = 0.75,
) -> dict[int, dict[str, str]]:
    """Unify speakers across chunks using cosine similarity on embeddings.

    Chunk 0 is canonical. Each subsequent chunk's speakers are matched
    to the canonical set or assigned new IDs.

    Returns {chunk_index: {local_speaker_id: unified_speaker_id}}.
    """
    if not chunk_results:
        return {}

    # Canonical embeddings from chunk 0
    canonical_embeddings: dict[str, np.ndarray] = {}
    speaker_counter = 0
    mapping: dict[int, dict[str, str]] = {}

    first = chunk_results[0]
    mapping[first.index] = {}
    for speaker, emb in first.speaker_embeddings.items():
        unified_id = f"SPEAKER_{speaker_counter:02d}"
        speaker_counter += 1
        canonical_embeddings[unified_id] = emb
        mapping[first.index][speaker] = unified_id

    # For speakers without embeddings in chunk 0
    for seg in first.segments:
        sp = seg["speaker"]
        if sp not in mapping[first.index]:
            unified_id = f"SPEAKER_{speaker_counter:02d}"
            speaker_counter += 1
            mapping[first.index][sp] = unified_id

    # Match subsequent chunks
    for cr in chunk_results[1:]:
        mapping[cr.index] = {}
        for speaker, emb in cr.speaker_embeddings.items():
            best_match = None
            best_sim = -1.0
            for canon_id, canon_emb in canonical_embeddings.items():
                sim = _cosine_similarity(emb, canon_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_match = canon_id

            if best_match and best_sim >= threshold:
                mapping[cr.index][speaker] = best_match
            else:
                unified_id = f"SPEAKER_{speaker_counter:02d}"
                speaker_counter += 1
                canonical_embeddings[unified_id] = emb
                mapping[cr.index][speaker] = unified_id

        # Speakers without embeddings
        for seg in cr.segments:
            sp = seg["speaker"]
            if sp not in mapping[cr.index]:
                unified_id = f"SPEAKER_{speaker_counter:02d}"
                speaker_counter += 1
                mapping[cr.index][sp] = unified_id

    return mapping


# -- Chunk merging -----------------------------------------------------

def merge_chunks(
    chunk_results: list[ChunkResult],
    speaker_mapping: dict[int, dict[str, str]] | None = None,
) -> list[dict]:
    """Merge chunk results into a single segment list.

    1. Convert timestamps to absolute (+ start_offset)
    2. Deduplicate overlap using midpoint strategy
    3. Apply speaker mapping if provided
    4. Sort by start time
    """
    if not chunk_results:
        return []

    all_segments = []

    for i, cr in enumerate(chunk_results):
        chunk_map = (speaker_mapping or {}).get(cr.index, {})

        # Determine overlap boundary with next chunk
        if i < len(chunk_results) - 1:
            next_cr = chunk_results[i + 1]
            # Midpoint of overlap region
            overlap_start = next_cr.start_offset
            overlap_end = cr.end_offset
            midpoint = (overlap_start + overlap_end) / 2
        else:
            midpoint = float("inf")

        for seg in cr.segments:
            abs_start = seg["start"] + cr.start_offset
            abs_end = seg["end"] + cr.start_offset

            # Skip segments past the midpoint (they'll come from the next chunk)
            if abs_end > midpoint and i < len(chunk_results) - 1:
                # Keep if segment starts before midpoint
                if abs_start >= midpoint:
                    continue

            speaker = chunk_map.get(seg["speaker"], seg["speaker"])
            all_segments.append({
                "speaker": speaker,
                "start": abs_start,
                "end": abs_end,
                "text": seg["text"],
            })

    # Sort by start time
    all_segments.sort(key=lambda s: s["start"])
    return all_segments


# -- Cleanup -----------------------------------------------------------

def cleanup_job_dir(job_dir: Path) -> None:
    """Remove the entire job directory."""
    import shutil
    if job_dir.exists():
        shutil.rmtree(job_dir)
