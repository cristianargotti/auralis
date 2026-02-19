"""API routes for the RECONSTRUCT pipeline — track-agnostic.

Upload ANY track → EAR analysis → plan → reconstruct → master → QC comparison.
Orchestrates all layers (EAR, GRID, HANDS, CONSOLE, BRAIN) into a single flow.
No hardcoded references — everything derived from the uploaded audio.
"""

from __future__ import annotations

import asyncio
import gc
import json
import multiprocessing
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from auralis.api.auth import get_current_user_or_token
from auralis.config import settings

router = APIRouter(prefix="/reconstruct", tags=["reconstruct"])

# Separate router for media endpoints that need dual-mode auth
# (Authorization header OR ?token= query param for browser-native elements)
media_router = APIRouter(prefix="/reconstruct", tags=["reconstruct-media"])

# In-memory job tracking
_reconstruct_jobs: dict[str, dict[str, Any]] = {}





class ReconstructRequest(BaseModel):
    """Request to reconstruct a reference track."""

    project_id: str
    mode: str = "full"  # "full", "section", "compare_only"
    target_section: str | None = None
    use_gpu: bool = False
    # Separator preference: "auto", "mel_roformer", "htdemucs", "htdemucs_ft"
    separator: str = "auto"


class CreateRequest(BaseModel):
    """Request to create a track from a text description (no upload needed)."""

    description: str  # "Dark melodic techno, 128 BPM, Am, haunting lead synth"
    genre: str = "auto"  # let AI decide or force: house, techno, ambient, pop, hip_hop
    bpm: float | None = None  # let AI decide or force
    key: str | None = None  # e.g. "C", "A", "F#"
    scale: str | None = None  # e.g. "minor", "major", "dorian"
    reference_ids: list[str] = []  # optional reference project IDs for DNA cloning


class ImproveRequest(BaseModel):
    """Request to improve an existing track based on user feedback."""

    feedback: str  # "Make the intro more atmospheric", "Heavier bass in the drop"



# ── Persistence ─────────────────────────────────────────

JOBS_FILE = settings.projects_dir / "jobs.json"

def _load_jobs() -> None:
    """Load jobs from disk on startup.

    Any jobs left in 'running' state are orphaned (the asyncio task died
    when the container restarted) — mark them as failed so the frontend
    stops polling and the user can re-submit.
    """
    global _reconstruct_jobs
    if JOBS_FILE.exists():
        try:
            data = json.loads(JOBS_FILE.read_text())
            orphaned = 0
            for jid, job in data.items():
                if job.get("status") == "running":
                    job["status"] = "failed"
                    job["error"] = "Server restarted — please re-upload your track"
                    job.setdefault("logs", []).append({
                        "timestamp": datetime.now().isoformat(),
                        "level": "error",
                        "message": "Job interrupted by server restart",
                    })
                    orphaned += 1
            _reconstruct_jobs = data
            print(f"Loaded {len(_reconstruct_jobs)} jobs from {JOBS_FILE} ({orphaned} orphaned → failed)")
        except Exception as e:
            print(f"Failed to load jobs: {e}")

def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to native Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating) or isinstance(obj, float):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return 0.0  # Use 0.0 instead of None to prevent client crashes (e.g. WaveSurfer peaks)
        return val
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
    return obj


def _save_jobs() -> None:
    """Save all jobs to disk."""
    try:
        # Sanitize before saving to avoid numpy errors
        data = _sanitize_for_json(_reconstruct_jobs)
        JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        JOBS_FILE.write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        print(f"Failed to save jobs: {e}")

# Load immediately on import (module level)
_load_jobs()
_save_jobs()  # persist any orphan→failed corrections


def _run_in_subprocess(func, *args, **kwargs) -> Any:
    """Run a function in a child process for full memory isolation.

    When the child exits, the OS reclaims ALL its memory — including
    C-level allocations from librosa, torch, numpy that gc.collect()
    cannot free.  Results are serialized back via multiprocessing.Queue.
    """
    q: multiprocessing.Queue = multiprocessing.Queue()

    def _target(q: multiprocessing.Queue) -> None:
        try:
            result = func(*args, **kwargs)
            q.put(("ok", result))
        except Exception as exc:
            q.put(("error", str(exc)))

    p = multiprocessing.Process(target=_target, args=(q,))
    p.start()
    p.join(timeout=600)  # 10 min max — prevents infinite hang on OOM/deadlock
    if p.is_alive():
        p.terminate()
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join(timeout=5)
        raise RuntimeError("Subprocess timed out after 600s — killed")

    if q.empty():
        raise RuntimeError(f"Subprocess died without result (exit code {p.exitcode})")
    tag, payload = q.get_nowait()
    if tag == "error":
        raise RuntimeError(payload)
    return payload


# ── Subprocess wrappers (return plain dicts for queue serialization) ──

def _subprocess_separate(audio_path, output_dir, model):
    """Run separation in subprocess, return dict."""
    from auralis.ear.separator import separate_track
    result = separate_track(audio_path=audio_path, output_dir=output_dir, model=model)
    return result.to_dict()


def _subprocess_profile(audio_path):
    """Run profiling in subprocess, return dict."""
    from auralis.ear.profiler import profile_track
    dna = profile_track(audio_path)
    return dna.to_dict()


def _subprocess_midi(stems_dir, output_dir):
    """Run MIDI extraction in subprocess, return dict of dicts."""
    from auralis.ear.midi_extractor import extract_midi_from_stems
    results = extract_midi_from_stems(stems_dir=stems_dir, output_dir=output_dir)
    return {k: v.to_dict() for k, v in results.items()}


def _subprocess_qc(original_path, reconstruction_path):
    """Run QC comparison in subprocess, return dict."""
    from auralis.qc.comparator import compare_full
    result = compare_full(original_path=original_path, reconstruction_path=reconstruction_path)
    return result.to_dict()


def _log(job: dict[str, Any], msg: str, level: str = "info") -> None:
    """Append a timestamped log entry to a job."""
    from datetime import datetime, timezone
    job.setdefault("logs", []).append({
        "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "level": level,
        "msg": msg,
    })


class SectionCompareRequest(BaseModel):
    """Request to compare a specific section."""

    project_id: str
    section_name: str


# ── Analysis-based endpoints (track-agnostic) ───────────


@router.get("/analysis/{project_id}")
async def get_analysis(project_id: str) -> dict[str, Any]:
    """Get the EAR analysis for an uploaded track.

    Returns whatever the profiler auto-detected:
    sections, energy map, BPM, key, spectral profile.
    """
    project_dir = settings.projects_dir / project_id
    analysis_file = project_dir / "analysis.json"

    if not analysis_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No analysis found for project {project_id}. Upload a track first.",
        )

    return json.loads(analysis_file.read_text())


@router.get("/stems/{project_id}")
async def get_stems(project_id: str) -> dict[str, Any]:
    """Get the separated stems info for a project."""
    project_dir = settings.projects_dir / project_id
    stems_meta = project_dir / "stems" / "separation_metadata.json"

    if not stems_meta.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No stems found for project {project_id}. Run separation first.",
        )

    return json.loads(stems_meta.read_text())


@router.get("/midi/{project_id}")
async def get_midi(project_id: str) -> dict[str, Any]:
    """Get extracted MIDI data for a project."""
    project_dir = settings.projects_dir / project_id
    midi_meta = project_dir / "midi" / "midi_extraction_metadata.json"

    if not midi_meta.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No MIDI data for project {project_id}. Run MIDI extraction first.",
        )

    return json.loads(midi_meta.read_text())


@router.get("/models")
async def get_available_models() -> dict[str, Any]:
    """Get available separation models and their status."""
    from auralis.ear.separator import get_available_models as get_sep_models

    return {
        "separator_models": get_sep_models(),
        "midi_extractor": "basic-pitch (Spotify)",
        "mastering": "matchering (reference-based)",
    }


# ── Reconstruction pipeline ─────────────────────────────


@router.post("/start")
async def start_reconstruction(req: ReconstructRequest) -> dict[str, Any]:
    """Start a reconstruction job — fully track-agnostic.

    Pipeline stages:
    1. EAR — Separate stems + extract MIDI + profile track
    2. PLAN — Auto-detect structure, map sections, plan arrangement
    3. GRID — Generate MIDI patterns from extracted data
    4. HANDS — Synthesize audio (timbre match + effects)
    5. CONSOLE — Mix + master by reference
    6. QC — 12-dimension comparison vs original
    """
    job_id = str(uuid.uuid4())

    _reconstruct_jobs[job_id] = {
        "job_id": job_id,
        "project_id": req.project_id,
        "mode": req.mode,
        "separator": req.separator,
        "status": "running",
        "stage": "ear",
        "progress": 0,
        "stages": {
            "ear": {"status": "running", "message": "Separating stems + profiling..."},
            "plan": {"status": "pending", "message": ""},
            "grid": {"status": "pending", "message": ""},
            "hands": {"status": "pending", "message": ""},
            "console": {"status": "pending", "message": ""},
            "qc": {"status": "pending", "message": ""},
        },
        "logs": [],
        "result": None,
    }
    _log(_reconstruct_jobs[job_id], f"Pipeline started — project {req.project_id}", "info")
    _log(_reconstruct_jobs[job_id], f"Mode: {req.mode} | Separator: {req.separator}", "info")

    asyncio.create_task(_run_reconstruction(job_id, req))
    _save_jobs()

    return _reconstruct_jobs[job_id]





@router.get("/jobs")
async def list_jobs():
    """List all reconstruct jobs with project details for the Projects page."""
    items = []
    for jid, j in _reconstruct_jobs.items():
        result = j.get("result") or {}
        files = result.get("files") or {}
        analysis = result.get("analysis") or j.get("analysis") or {}

        # Disk size from project dir
        project_id = j.get("project_id", jid)
        project_dir = Path("/app/projects") / project_id
        disk_bytes = 0
        file_count = 0
        if project_dir.exists():
            for f in project_dir.rglob("*"):
                if f.is_file():
                    try:
                        disk_bytes += f.stat().st_size
                        file_count += 1
                    except OSError:
                        pass

        # QC data
        qc = result.get("qc") or {}

        items.append({
            "job_id": jid,
            "project_id": project_id,
            "original_name": j.get("original_name", "Untitled"),
            "status": j.get("status", "unknown"),
            "mode": j.get("mode", "full"),
            "stage": j.get("stage", ""),
            "progress": j.get("progress", 0),
            "parent_job_id": j.get("parent_job_id"),
            "feedback": j.get("feedback"),
            "cleaned": disk_bytes == 0 and j.get("status") == "completed",
            "disk_size_mb": round(disk_bytes / (1024 * 1024), 1),
            "file_count": file_count,
            "created_at": j.get("created_at"),
            "has_master": bool(files.get("master")),
            "has_mix": bool(files.get("mix")),
            "has_original": bool(files.get("original")),
            "has_stems": bool(files.get("stems")),
            "has_brain": bool(j.get("brain_plan")),
            "has_stem_analysis": bool(analysis.get("stems")),
            "bpm": analysis.get("bpm"),
            "key": analysis.get("key"),
            "scale": analysis.get("scale"),
            "duration": analysis.get("duration"),
            "qc_score": qc.get("score"),
            "qc_passed": qc.get("passed"),
        })
    # Most recent first (dict preserves insertion order in Python 3.7+)
    items.reverse()
    return {"jobs": items, "total": len(items)}


@router.get("/status/{job_id}")
async def get_reconstruction_status(job_id: str) -> dict[str, Any]:
    """Get the current status of a reconstruction job."""
    if job_id not in _reconstruct_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _sanitize_for_json(_reconstruct_jobs[job_id])


@router.get("/compare/{project_id}")
async def get_comparison(project_id: str) -> dict[str, Any]:
    """Get A/B comparison results for a project."""
    for job in _reconstruct_jobs.values():
        if job["project_id"] == project_id and job["status"] == "completed":
            return _sanitize_for_json(job.get("result", {}))
    raise HTTPException(status_code=404, detail="No completed reconstruction found")


@router.get("/waveform/{job_id}")
async def get_waveform_data(job_id: str) -> dict[str, Any]:
    """Get interactive waveform data (X-Ray layers) for a job."""
    if job_id not in _reconstruct_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _reconstruct_jobs[job_id]
    project_id = job["project_id"]
    # We need settings to locate project dir. Assuming standard path structure:
    # /app/projects/{project_id}
    # But wait, we don't have settings imported here easily? 
    # Actually we can reconstruct path or use job result if it has path?
    # Let's import settings or use hardcoded /app/projects for now since we know docker volume structure.
    # Actually better: use settings if available.
    
    # Let's use Path("/app/projects") / project_id or check if job has path info
    project_dir = Path("/app/projects") / project_id
    
    if not project_dir.exists():
         raise HTTPException(status_code=404, detail="Project files not found")

    try:
        from auralis.ear.analyzer import analyze_track_layers
        # Run analysis in thread pool to avoid blocking
        data = await asyncio.to_thread(analyze_track_layers, project_dir)
        return _sanitize_for_json(data)
    except ImportError:
        # Fallback if librosa not installed (shouldn't happen in prod)
        return {"error": "Audio analysis library not available"}
    except Exception as e:
        _log(job, f"Waveform analysis failed: {e}", "error")
        raise HTTPException(status_code=500, detail=str(e))


@media_router.get("/audio/{job_id}/{file_key}")
async def get_audio_file(
    job_id: str,
    file_key: str,
    request: Request,
    fmt: str = Query("mp3", alias="format"),
):
    """Stream audio file. Serves MP3 by default for fast playback.
    
    Query params:
        format=mp3  (default) — compressed, ~10x smaller, instant playback
        format=wav  — lossless original for downloads
    """
    if job_id not in _reconstruct_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = _reconstruct_jobs[job_id]
    project_id = job["project_id"]
    project_dir = Path("/app/projects") / project_id
    
    file_map = {
        "original": "original.wav",
        "mix": "mix.wav",
        "master": "master.wav",
        "stem_drums": "stems/drums.wav",
        "stem_bass": "stems/bass.wav",
        "stem_other": "stems/other.wav",
        "stem_vocals": "stems/vocals.wav",
    }
    
    if file_key not in file_map:
        raise HTTPException(status_code=400, detail="Invalid file key")
        
    wav_path = project_dir / file_map[file_key]
    if not wav_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Determine which file to serve
    if fmt == "wav":
        file_path = wav_path
        media_type = "audio/wav"
    else:
        # Auto-convert to MP3 if not cached
        mp3_path = wav_path.with_suffix(".mp3")
        if not mp3_path.exists():
            import subprocess
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(wav_path),
                        "-codec:a", "libmp3lame", "-b:a", "192k",
                        "-map_metadata", "-1",  # strip metadata for smaller file
                        str(mp3_path),
                    ],
                    capture_output=True, timeout=120, check=True,
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                # Fallback to WAV if conversion fails
                import logging
                logging.getLogger("auralis").warning(f"MP3 conversion failed: {e}")
                file_path = wav_path
                media_type = "audio/wav"
                mp3_path = None  # type: ignore[assignment]

        if mp3_path and mp3_path.exists():
            file_path = mp3_path
            media_type = "audio/mpeg"
        else:
            file_path = wav_path
            media_type = "audio/wav"

    file_size = file_path.stat().st_size
    range_header = request.headers.get("range")

    if range_header:
        # Parse Range: bytes=start-end
        range_spec = range_header.replace("bytes=", "")
        parts = range_spec.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if parts[1] else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iter_range():
            with open(file_path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return StreamingResponse(
            iter_range(),
            status_code=206,
            media_type=media_type,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )

    # No Range header — full file with Accept-Ranges hint
    return FileResponse(
        file_path,
        media_type=media_type,
        headers={"Accept-Ranges": "bytes"},
    )


@media_router.get("/spectrogram/{job_id}/{file_key}")
async def get_spectrogram(
    job_id: str,
    file_key: str,
):
    """Generate and return mel spectrogram as PNG image."""
    if job_id not in _reconstruct_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _reconstruct_jobs[job_id]
    project_id = job["project_id"]
    project_dir = Path("/app/projects") / project_id

    file_map = {
        "original": "original.wav",
        "master": "master.wav",
        "mix": "mix.wav",
    }

    if file_key not in file_map:
        raise HTTPException(status_code=400, detail="Invalid file key (original, master, mix)")

    file_path = project_dir / file_map[file_key]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Check for cached spectrogram
    cache_path = project_dir / f"spectrogram_{file_key}.png"
    if cache_path.exists():
        return FileResponse(cache_path, media_type="image/png")

    try:
        png_path = await asyncio.to_thread(_generate_spectrogram, str(file_path), str(cache_path))
        return FileResponse(png_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spectrogram generation failed: {e}")


def _generate_spectrogram(audio_path: str, output_path: str) -> str:
    """Generate high-quality mel spectrogram PNG — runs in thread pool."""
    import librosa
    import librosa.display
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Full sample rate for complete frequency spectrum (up to 22kHz)
    y, sr = librosa.load(audio_path, sr=44100, mono=True)

    # High-resolution mel spectrogram
    S = librosa.feature.melspectrogram(
        y=y, sr=sr,
        n_fft=4096,         # High frequency resolution
        hop_length=512,     # Fine time resolution
        n_mels=256,         # 256 mel bands (2x previous)
        fmin=20,            # Full audible range
        fmax=20000,         # Up to 20kHz
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Create premium figure — dark theme, tall aspect ratio
    fig, ax = plt.subplots(1, 1, figsize=(20, 6), dpi=200)
    fig.patch.set_facecolor("#09090b")
    ax.set_facecolor("#09090b")

    img = librosa.display.specshow(
        S_dB, sr=sr, x_axis="time", y_axis="mel",
        ax=ax, cmap="magma",
        vmin=-70, vmax=0,       # Wider dynamic range
        hop_length=512,
    )

    # Styled axes — subtle frequency/time labels
    ax.set_xlabel("Time", fontsize=8, color="#71717a", labelpad=4)
    ax.set_ylabel("Frequency", fontsize=8, color="#71717a", labelpad=4)
    ax.tick_params(colors="#52525b", labelsize=6, width=0.5, length=3)

    # Remove hard borders, add subtle grid
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Colorbar for dB scale
    cbar = fig.colorbar(img, ax=ax, format="%+.0f dB", pad=0.01, aspect=30)
    cbar.ax.tick_params(colors="#52525b", labelsize=6, width=0.5, length=3)
    cbar.outline.set_visible(False)
    cbar.set_label("dB", fontsize=7, color="#52525b", rotation=0, labelpad=8)

    fig.tight_layout(pad=0.3)
    fig.savefig(
        output_path,
        dpi=200,
        facecolor="#09090b",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)

    # Free memory
    del y, S, S_dB

    return output_path


@router.delete("/cleanup/{job_id}")
async def cleanup_project(job_id: str, force: bool = False):
    """Delete project files to free disk space.

    By default, preserves stems and master/mix WAV files so Directed Improve
    still works.  Pass ?force=true to delete everything (called from Projects page).
    """
    if job_id not in _reconstruct_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _reconstruct_jobs[job_id]
    project_id = job["project_id"]
    project_dir = Path("/app/projects") / project_id

    freed_bytes = 0

    if project_dir.exists():
        if force:
            # Full delete — removes everything
            freed_bytes += sum(
                f.stat().st_size for f in project_dir.rglob("*") if f.is_file()
            )
            import shutil
            shutil.rmtree(project_dir, ignore_errors=True)
        else:
            # Soft cleanup — keep stems, master, mix, original
            keep_patterns = {"master.wav", "mix.wav", "original.wav"}
            keep_dirs = {"stems"}
            for f in project_dir.iterdir():
                if f.name in keep_patterns:
                    continue
                if f.is_dir() and f.name in keep_dirs:
                    continue
                if f.is_file():
                    freed_bytes += f.stat().st_size
                    f.unlink(missing_ok=True)
                elif f.is_dir():
                    import shutil
                    freed_bytes += sum(
                        sf.stat().st_size for sf in f.rglob("*") if sf.is_file()
                    )
                    shutil.rmtree(f, ignore_errors=True)

    # 2. Clean /tmp uploads and stale audio temp files
    import shutil
    import time
    tmp_dir = Path("/tmp")
    one_hour_ago = time.time() - 3600
    for f in tmp_dir.glob("*.wav"):
        try:
            if f.stat().st_mtime < one_hour_ago:
                freed_bytes += f.stat().st_size
                f.unlink()
        except OSError:
            pass
    for f in tmp_dir.glob("*.mp3"):
        try:
            if f.stat().st_mtime < one_hour_ago:
                freed_bytes += f.stat().st_size
                f.unlink()
        except OSError:
            pass
    # Clean upload temp dirs
    for d in tmp_dir.glob("upload_*"):
        try:
            if d.is_dir() and d.stat().st_mtime < one_hour_ago:
                freed_bytes += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass

    freed_mb = round(freed_bytes / (1024 * 1024), 1)
    job["cleaned"] = True
    _save_jobs()

    return {
        "status": "cleaned",
        "freed_mb": freed_mb,
        "project_id": project_id,
    }


def _subprocess_mix(
    stem_paths: dict[str, str],
    output_path: str,
    bpm: float,
    stem_analysis: dict[str, Any] | None = None,
    ear_analysis: dict[str, Any] | None = None,
    ref_targets: dict[str, dict[str, Any]] | None = None,
    stem_decisions: dict[str, dict[str, Any]] | None = None,
    brain_stem_plans: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Mix stems with professional per-stem processing driven by EAR analysis.

    Uses stem_recipes to build intelligent EffectChains, then routes through
    the Mixer engine with reverb/delay buses and proper gain staging.
    """
    import soundfile as sf
    import numpy as np
    from types import SimpleNamespace
    from auralis.hands.mixer import Mixer, MixConfig, SendConfig
    from auralis.hands.effects import (
        EffectChain, ReverbConfig, DelayConfig, CompressorConfig,
    )
    from auralis.console.stem_recipes import build_recipe_for_stem, StemRecipe, enhance_recipe_with_smart_fx

    print(f"Subprocess: Professional mixing {len(stem_paths)} stems @ {bpm} BPM...")
    if brain_stem_plans:
        print(f"  🧠 Brain plans for: {list(brain_stem_plans.keys())}")

    mixer = Mixer(MixConfig(sample_rate=44100, bpm=bpm))

    # ── Derive bus config from brain or use safe defaults ──
    # Extract master plan from brain_stem_plans (it may include a "_master" key)
    bp_master = (brain_stem_plans or {}).get("_master", {})

    # Reverb: brain width influences room_size, louder targets mean tighter reverb
    brain_room = 0.7   # default
    brain_rev_vol = -3.0
    brain_fb = 0.3      # delay feedback default
    brain_del_vol = -6.0

    if bp_master:
        # Wider master → bigger room, narrower → tighter
        width = bp_master.get("width", 1.3)
        brain_room = round(min(0.95, max(0.3, 0.4 + (width - 1.0) * 0.6)), 2)
        # Louder targets → less reverb bus volume (tighter)
        target_lufs = bp_master.get("target_lufs", -14.0)
        if target_lufs > -10:
            brain_rev_vol = -6.0
            brain_del_vol = -9.0
        elif target_lufs > -14:
            brain_rev_vol = -3.0
            brain_del_vol = -6.0
        else:
            brain_rev_vol = -1.5
            brain_del_vol = -4.0
        # Drive influences delay feedback: more drive = less feedback (cleaner tails)
        drive = bp_master.get("drive", 1.5)
        brain_fb = round(min(0.5, max(0.15, 0.5 - (drive - 1.0) * 0.3)), 2)
        print(f"  🧠 Bus config from brain: room={brain_room}, rev_vol={brain_rev_vol}, fb={brain_fb}, del_vol={brain_del_vol}")

    # Reverb bus
    mixer.add_bus(
        "reverb",
        effects=EffectChain(
            name="reverb_bus",
            reverb=ReverbConfig(room_size=brain_room, damping=0.5, wet=1.0, pre_delay_ms=15.0),
        ),
        volume_db=brain_rev_vol,
    )
    # Delay bus: BPM-synced 1/8 note
    delay_time = (60000.0 / bpm) / 2  # 1/8 note in ms
    mixer.add_bus(
        "delay",
        effects=EffectChain(
            name="delay_bus",
            delay=DelayConfig(time_ms=delay_time, feedback=brain_fb, wet=1.0, ping_pong=False),
        ),
        volume_db=brain_del_vol,
    )
    # Drum bus: parallel compression for punch
    mixer.add_bus(
        "drum_bus",
        effects=EffectChain(
            name="drum_parallel",
            compressor=CompressorConfig(
                threshold_db=-20.0, ratio=6.0, attack_ms=5.0,
                release_ms=50.0, makeup_db=6.0
            ),
        ),
        volume_db=-6.0,
    )
    print(f"  Buses: Reverb (room {brain_room}, damp 0.5) | Delay ({delay_time:.0f}ms, fb {brain_fb}) | Drum Parallel Comp")

    # ── Build recipes and add tracks ──
    recipes_used: list[str] = []
    for name, path in stem_paths.items():
        data, sr = sf.read(path, dtype="float64")
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Get per-stem analysis (or empty dict for fallback)
        sa = (stem_analysis or {}).get(name, {})

        # Reconstruct brain plan as namespace for attribute access
        stem_plan = None
        if brain_stem_plans and name in brain_stem_plans:
            stem_plan = SimpleNamespace(**brain_stem_plans[name])

        # Build intelligent recipe (ref-targeted + brain-guided)
        recipe: StemRecipe = build_recipe_for_stem(
            stem_name=name,
            stem_analysis=sa,
            bpm=bpm,
            ear_data=ear_analysis,
            ref_targets=ref_targets,
            stem_plan=stem_plan,
        )

        # Auto-send drums to parallel compression bus
        sends = list(recipe.sends) if recipe.sends else []
        base_name = name.replace("gen_", "").replace("layer_", "")
        if base_name == "drums" and not any(s.bus_name == "drum_bus" for s in sends):
            sends.append(SendConfig(bus_name="drum_bus", amount=0.4))

        # Add track to mixer with recipe settings
        mixer.add_track(
            name=name,
            audio=data,
            volume_db=recipe.volume_db,
            pan=recipe.pan,
            effects=recipe.chain,
            sends=sends,
        )

        recipes_used.append(recipe.description)

        # Apply smart FX from stem decisions (if available)
        if stem_decisions and name in stem_decisions:
            decision_data = stem_decisions[name]
            extra_fx = decision_data.get("extra_fx", [])
            if extra_fx:
                recipe = enhance_recipe_with_smart_fx(recipe, extra_fx, bpm)
                print(f"  → Smart FX applied: {', '.join(extra_fx)}")

        print(f"  {recipe.description} | vol: {recipe.volume_db:+.1f}dB | pan: {recipe.pan:+.1f}")

    # ── Render mix ──
    result = mixer.mix(output_path=output_path)
    print(f"  ✓ Mixed {result.tracks_mixed} tracks, peak: {result.peak_db:.1f} dBFS")

    return {
        "tracks_mixed": result.tracks_mixed,
        "buses_used": result.buses_used,
        "peak_db": result.peak_db,
        "recipes": recipes_used,
    }


def _subprocess_master(input_path: str, output_path: str, target_lufs: float, bpm: float, brain_plan_dict: dict | None = None) -> None:
    """Master audio in a separate process to prevent segfaults."""
    from auralis.console.mastering import master_audio, MasterConfig

    # Reconstruct brain plan as namespace for attribute access
    brain_plan = None
    if brain_plan_dict:
        from types import SimpleNamespace
        brain_plan = SimpleNamespace(**brain_plan_dict)

    print(f"Subprocess: Mastering to {target_lufs} LUFS...")
    config = MasterConfig(target_lufs=target_lufs, bpm=bpm, brain_plan=brain_plan)
    master_audio(input_path=input_path, output_path=output_path, config=config)


@router.get("/jobs")
async def list_jobs() -> list[dict[str, Any]]:
    """List all reconstruction jobs with rich metadata."""
    items: list[dict[str, Any]] = []
    for j in _reconstruct_jobs.values():
        project_id = j["project_id"]
        project_dir = Path("/app/projects") / project_id
        result = j.get("result") or {}

        # Disk size
        disk_bytes = 0
        file_count = 0
        if project_dir.exists():
            for f in project_dir.rglob("*"):
                if f.is_file():
                    disk_bytes += f.stat().st_size
                    file_count += 1

        # Created timestamp (from project dir mtime or fallback)
        created_at = None
        if project_dir.exists():
            created_at = project_dir.stat().st_mtime

        # File availability
        files = result.get("files") or {}

        # Analysis data
        analysis = result.get("analysis") or {}
        qc = result.get("qc") or {}

        items.append({
            "job_id": j["job_id"],
            "project_id": project_id,
            "status": j["status"],
            "stage": j["stage"],
            "progress": j["progress"],
            "cleaned": j.get("cleaned", False),
            "disk_size_mb": round(disk_bytes / (1024 * 1024), 1),
            "file_count": file_count,
            "created_at": created_at,
            "has_master": bool(files.get("master")),
            "has_mix": bool(files.get("mix")),
            "has_original": bool(files.get("original")),
            "has_stems": bool(files.get("stems")),
            "has_brain": "brain_report" in result,
            "has_stem_analysis": "stem_analysis" in result,
            "bpm": analysis.get("bpm"),
            "key": analysis.get("key"),
            "scale": analysis.get("scale"),
            "duration": analysis.get("duration"),
            "qc_score": qc.get("overall_score"),
            "qc_passed": qc.get("passed"),
        })

    # Sort by created_at descending (newest first)
    items.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return items


@router.get("/projects/stats")
async def get_project_stats() -> dict[str, Any]:
    """Get aggregate stats for all projects."""
    projects_dir = Path("/app/projects")
    total_bytes = 0
    project_count = 0

    if projects_dir.exists():
        for d in projects_dir.iterdir():
            if d.is_dir() and d.name != "_reference_bank":
                project_count += 1
                for f in d.rglob("*"):
                    if f.is_file():
                        total_bytes += f.stat().st_size

    completed = sum(1 for j in _reconstruct_jobs.values() if j["status"] == "completed")
    running = sum(1 for j in _reconstruct_jobs.values() if j["status"] == "running")
    errored = sum(1 for j in _reconstruct_jobs.values() if j["status"] == "error")
    with_brain = sum(
        1 for j in _reconstruct_jobs.values()
        if "brain_report" in (j.get("result") or {})
    )

    return {
        "total_projects": project_count,
        "total_disk_gb": round(total_bytes / (1024 ** 3), 2),
        "total_disk_mb": round(total_bytes / (1024 ** 2), 0),
        "total_jobs": len(_reconstruct_jobs),
        "completed": completed,
        "running": running,
        "errored": errored,
        "with_intelligence": with_brain,
    }


# ── Pipeline execution (track-agnostic) ─────────────────


async def _run_reconstruction(job_id: str, req: ReconstructRequest) -> None:
    """Execute the full reconstruction pipeline from any uploaded track."""
    job = _reconstruct_jobs[job_id]
    project_dir = settings.projects_dir / req.project_id

    try:
        # Stage 1: EAR — Separate + Profile
        job["stage"] = "ear"
        job["stages"]["ear"]["status"] = "running"
        job["progress"] = 5
        _log(job, "── STAGE 1: EAR ──", "stage")
        _log(job, "Searching for uploaded audio file...")

        # Check for uploaded audio
        audio_file = _find_audio_file(project_dir)
        analysis: dict[str, Any] = {}
        sep_result = None

        if audio_file:
            # Check if models are cached to show correct message
            model_cache_dir = Path("/root/.cache/torch/hub/checkpoints")
            if model_cache_dir.exists() and any(model_cache_dir.iterdir()):
                _log(job, f"Using cached HTDemucs model (instant load)...")
                job["stages"]["ear"]["message"] = "Model loaded from cache. Starting separation (CPU, ~5 min)..."
            else:
                _log(job, "Downloading HTDemucs model (~320MB, first run only)...")
                job["stages"]["ear"]["message"] = "Downloading model & separating (CPU, ~7 min)..."

            _log(job, "Starting separation process...")

            # Run separation (Mel-RoFormer primary, HTDemucs fallback)
            stems_dir = project_dir / "stems"
            try:
                import time as _time
                t0 = _time.monotonic()

                sep_dict = await asyncio.to_thread(
                    _run_in_subprocess,
                    _subprocess_separate,
                    audio_path=audio_file,
                    output_dir=stems_dir,
                    model=req.separator,
                )
                elapsed = _time.monotonic() - t0

                # Reconstruct lightweight stem info from dict
                stem_paths = {k: Path(v) for k, v in sep_dict["stems"].items()}
                job["stages"]["ear"]["message"] = (
                    f"Separated {len(stem_paths)} stems via {sep_dict['model_used']}"
                )
                _log(job, f"✓ Separated {len(stem_paths)} stems via {sep_dict['model_used']} ({elapsed:.0f}s)", "success")
                for sname, spath in stem_paths.items():
                    _log(job, f"  📁 {sname}: {spath.stat().st_size / 1024 / 1024:.1f} MB")

                # ── Free separation model memory ──
                gc.collect()
                _save_jobs()  # checkpoint after separation

                # ── Per-stem analysis ──
                _log(job, "Analyzing separated stems (RMS, peak, FFT)...")
                stem_analysis: dict[str, Any] = {}
                original_audio, sr = sf.read(str(audio_file))
                if original_audio.ndim > 1:
                    original_mono = np.mean(original_audio, axis=1)
                else:
                    original_mono = original_audio
                original_rms = float(np.sqrt(np.mean(original_mono ** 2)))
                del original_audio  # free stereo array immediately

                for stem_name, stem_path in stem_paths.items():
                    try:
                        _log(job, f"  🔬 Analyzing {stem_name}...")
                        audio_data, stem_sr = sf.read(str(stem_path))
                        if audio_data.ndim > 1:
                            mono = np.mean(audio_data, axis=1)
                        else:
                            mono = audio_data

                        rms = float(np.sqrt(np.mean(mono ** 2)))
                        peak = float(np.max(np.abs(mono)))
                        rms_db = float(20 * np.log10(rms + 1e-10))
                        peak_db = float(20 * np.log10(peak + 1e-10))
                        energy_pct = float(round((rms / original_rms) * 100, 1)) if original_rms > 0 else 0.0

                        # Detect dominant frequency band via FFT
                        n_samples = min(len(mono), int(stem_sr * 4))  # first 4 seconds
                        fft_data = np.abs(np.fft.rfft(mono[:n_samples]))
                        freqs = np.fft.rfftfreq(n_samples, 1.0 / stem_sr)
                        low_mask = (freqs >= 20) & (freqs < 250)
                        mid_mask = (freqs >= 250) & (freqs < 4000)
                        high_mask = (freqs >= 4000) & (freqs <= 20000)
                        low_energy = float(np.sum(fft_data[low_mask]))
                        mid_energy = float(np.sum(fft_data[mid_mask]))
                        high_energy = float(np.sum(fft_data[high_mask]))
                        total_e = low_energy + mid_energy + high_energy + 1e-10

                        # All values must be native Python types (no numpy)
                        stem_analysis[stem_name] = {
                            "rms_db": float(round(rms_db, 1)),
                            "peak_db": float(round(peak_db, 1)),
                            "energy_pct": float(energy_pct),
                            "duration": float(round(len(mono) / stem_sr, 2)),
                            "file_size_mb": float(round(stem_path.stat().st_size / 1024 / 1024, 2)),
                            "freq_bands": {
                                "low": float(round(low_energy / total_e * 100, 1)),
                                "mid": float(round(mid_energy / total_e * 100, 1)),
                                "high": float(round(high_energy / total_e * 100, 1)),
                            },
                        }
                        _log(job, f"  🎛️ {stem_name}: {rms_db:.1f} dB RMS | {peak_db:.1f} dBFS peak | {energy_pct:.0f}% energy")
                    except Exception:
                        stem_analysis[stem_name] = {"error": "analysis failed"}

                if job["result"] is None:
                    job["result"] = {}
                job["result"]["stem_analysis"] = stem_analysis  # type: ignore[index]
                _log(job, f"✓ Analyzed {len(stem_analysis)} stems", "success")
                del original_mono  # free mono array
                gc.collect()
            except ImportError:
                job["stages"]["ear"]["message"] = (
                    "Separation models not installed — using analysis only"
                )
                _log(job, "Separation models not installed — analysis only", "warn")
            except Exception as e:
                job["stages"]["ear"]["message"] = f"Separation error: {e}"
                _log(job, f"Separation error: {e}", "error")

            # Run profiler
            job["stages"]["ear"]["message"] = "Profiling track DNA..."
            _log(job, "Running track profiler (BPM, key, sections, energy)...")
            try:
                analysis = await asyncio.to_thread(
                    _run_in_subprocess, _subprocess_profile, audio_file,
                )
                analysis_path = project_dir / "analysis.json"
                analysis_path.write_text(json.dumps(analysis, indent=2, default=str))
                _log(job, f"✓ Profile complete — {analysis.get('tempo', '?')} BPM, {analysis.get('key', '?')} {analysis.get('scale', '')}", "success")
                _log(job, f"  Sections: {len(analysis.get('sections', []))} | Duration: {analysis.get('duration', 0):.1f}s")
            except Exception as e:
                analysis = {"error": str(e)}
                _log(job, f"Profiler error: {e}", "error")

            # Run MIDI extraction on tonal stems (hybrid: ONNX → Replicate → pyin)
            if stems_dir.exists() and any(stems_dir.glob("*.wav")):
                job["stages"]["ear"]["message"] = "Extracting MIDI from stems..."
                _log(job, "Extracting MIDI from tonal stems (ONNX → Replicate → pyin)...")
                try:
                    midi_dir = project_dir / "midi"
                    analysis["midi_stems"] = await asyncio.to_thread(
                        _run_in_subprocess,
                        _subprocess_midi,
                        stems_dir=stems_dir,
                        output_dir=midi_dir,
                    )
                    # Report which methods were used
                    methods_used = set()
                    for stem_data in analysis["midi_stems"].values():
                        if hasattr(stem_data, "get"):
                            methods_used.add(stem_data.get("method", "unknown"))
                        elif hasattr(stem_data, "method"):
                            methods_used.add(stem_data.method)
                    methods_str = ", ".join(methods_used) if methods_used else "unknown"
                    _log(job, f"✓ MIDI extracted from {len(analysis['midi_stems'])} stems via {methods_str}", "success")
                except ImportError:
                    analysis["midi_stems"] = {}
                    _log(job, "ℹ️ MIDI extraction skipped (optional dependency)", "info")
                except Exception as e:
                    analysis["midi_stems"] = {}
                    _log(job, f"ℹ️ MIDI extraction skipped: {e}", "info")

            # Free separation result reference
            stem_paths = None
            gc.collect()
            _save_jobs()  # checkpoint after full EAR stage
        else:
            analysis_file = project_dir / "analysis.json"
            if analysis_file.exists():
                analysis = json.loads(analysis_file.read_text())
                job["stages"]["ear"]["message"] = "Using existing analysis data"
            else:
                job["stages"]["ear"]["message"] = "No audio or analysis found"

        await asyncio.sleep(0.1)
        job["stages"]["ear"]["status"] = "completed"
        job["progress"] = 20
        _log(job, "✓ EAR stage complete", "success")
        # Persist analysis in job result for reference bank API
        if job.get("result") is None:
            job["result"] = {}
        job["result"]["analysis"] = analysis  # type: ignore[index]
        gc.collect()  # aggressive cleanup before next stage

        # Stage 2: PLAN — Auto-detect structure
        job["stage"] = "plan"
        job["stages"]["plan"]["status"] = "running"
        job["stages"]["plan"]["message"] = "Auto-detecting track structure..."
        _log(job, "── STAGE 2: BRAIN ──", "stage")
        _log(job, "Building track plan from analysis data...")

        plan = {
            "bpm": analysis.get("tempo", 120.0),
            "key": analysis.get("key", "unknown"),
            "scale": analysis.get("scale", "unknown"),
            "duration": analysis.get("duration", 0),
            "sections": analysis.get("sections", []),
            "total_sections": len(analysis.get("sections", [])),
        }

        await asyncio.sleep(0.1)
        job["stages"]["plan"]["status"] = "completed"
        job["stages"]["plan"]["message"] = (
            f"Detected {plan['total_sections']} sections | "
            f"BPM: {plan['bpm']:.1f} | Key: {plan['key']} {plan['scale']}"
        )
        job["progress"] = 35
        _log(job, f"✓ Plan: {plan['total_sections']} sections, {plan['bpm']:.1f} BPM, {plan['key']} {plan['scale']}", "success")

        # Stage 3: GRID — MIDI pattern mapping (REAL)
        job["stage"] = "grid"
        job["stages"]["grid"]["status"] = "running"
        job["stages"]["grid"]["message"] = "Loading MIDI patterns from extracted data..."
        _log(job, "── STAGE 3: GRID ──", "stage")
        _log(job, "Loading MIDI patterns from extracted data...")

        midi_data = analysis.get("midi_stems", {})
        midi_patterns: dict[str, Any] = {}
        midi_dir = project_dir / "midi"

        try:
            from auralis.grid.midi import load_midi, Pattern

            if midi_dir.exists():
                for midi_file in midi_dir.glob("*.mid"):
                    stem_name = midi_file.stem
                    try:
                        pattern = load_midi(midi_file)
                        midi_patterns[stem_name] = {
                            "notes": len(pattern.notes),
                            "length_beats": pattern.length_beats,
                            "pitch_range": (
                                min((n.pitch for n in pattern.notes), default=0),
                                max((n.pitch for n in pattern.notes), default=0),
                            ),
                            "pattern_loaded": True,
                        }
                    except Exception as e:
                        midi_patterns[stem_name] = {"error": str(e)}
            else:
                # Use metadata from analysis
                for stem_name, stem_data in midi_data.items():
                    if isinstance(stem_data, dict) and "notes_count" in stem_data:
                        midi_patterns[stem_name] = {
                            "notes": stem_data.get("notes_count", 0),
                            "pitch_range": stem_data.get("pitch_range", [0, 0]),
                            "confidence": stem_data.get("confidence", 0),
                        }
        except ImportError:
            for stem_name, stem_data in midi_data.items():
                if isinstance(stem_data, dict) and "notes_count" in stem_data:
                    midi_patterns[stem_name] = {
                        "notes": stem_data.get("notes_count", 0),
                        "confidence": stem_data.get("confidence", 0),
                    }

        await asyncio.sleep(0.1)
        job["stages"]["grid"]["status"] = "completed"
        job["stages"]["grid"]["message"] = f"Mapped {len(midi_patterns)} MIDI tracks"
        job["progress"] = 50
        for name, info in midi_patterns.items():
            notes = info.get('notes', '?')
            _log(job, f"  {name}: {notes} notes")
        _log(job, f"✓ GRID complete — {len(midi_patterns)} tracks mapped", "success")
        gc.collect()
        _save_jobs()  # checkpoint after GRID

        # Pre-initialize variables used across stages 4 and 5
        brain_report = None
        ref_targets: dict[str, Any] = {}
        bank = None

        # Stage 4: HANDS — Synthesis / Stem passthrough (REAL)
        job["stage"] = "hands"
        job["stages"]["hands"]["status"] = "running"
        job["stages"]["hands"]["message"] = "Synthesizing / preparing stems..."
        _log(job, "── STAGE 4: HANDS ──", "stage")
        _log(job, "Preparing stems for reconstruction...")

        stems_dir = project_dir / "stems"
        rendered_dir = project_dir / "rendered"
        rendered_dir.mkdir(parents=True, exist_ok=True)
        rendered_stems: dict[str, str] = {}

        try:
            if stems_dir.exists():
                # Passthrough mode: use separated stems as reconstruction basis
                stem_files = list(stems_dir.glob("*.wav"))
                for i, stem_file in enumerate(stem_files):
                    stem_name = stem_file.stem
                    # Check for silence based on analysis
                    is_silent = False
                    if job.get("result") and "stem_analysis" in job["result"]:
                        stem_info = job["result"]["stem_analysis"].get(stem_name)
                        if stem_info and isinstance(stem_info, dict) and stem_info.get("rms_db", -100) < -60:
                            is_silent = True
                    
                    if is_silent:
                        _log(job, f"Skipping silent stem: {stem_name}")
                        continue

                    job["stages"]["hands"]["message"] = (
                        f"Processing stem: {stem_name} ({i + 1}/{len(stem_files)})"
                    )
                    _log(job, f"Processing: {stem_name} ({i + 1}/{len(stem_files)})...")

                    # Copy stem to rendered dir (with optional FX processing)
                    rendered_path = rendered_dir / f"{stem_name}.wav"

                    try:
                        from auralis.hands.effects import EffectChain, process_chain
                        data, sr = sf.read(str(stem_file), dtype="float64")
                        mono = np.mean(data, axis=1) if data.ndim > 1 else data

                        # Build real processing chain from recipe (brain-guided if available)
                        try:
                            from auralis.console.stem_recipes import build_recipe_for_stem
                            from types import SimpleNamespace

                            sa = stem_analysis_data.get(stem_name, {}) if stem_analysis_data else {}

                            # Reconstruct brain plan if available
                            stem_plan = None
                            if brain_report and brain_report.stem_plans.get(stem_name):
                                sp = brain_report.stem_plans[stem_name]
                                stem_plan = SimpleNamespace(**sp.to_dict())

                            recipe = build_recipe_for_stem(
                                stem_name=stem_name,
                                stem_analysis=sa,
                                bpm=plan.get("bpm", 120.0),
                                ear_data=analysis,
                                ref_targets=ref_targets if ref_targets else None,
                                stem_plan=stem_plan,
                            )
                            chain = recipe.chain
                            _log(job, f"  {recipe.description}")
                        except Exception:
                            chain = EffectChain(name=f"{stem_name}_chain")

                        processed = process_chain(
                            mono, chain, sr=sr, bpm=plan.get("bpm", 120.0)
                        )
                        sf.write(str(rendered_path), processed, sr)
                    except (ImportError, Exception):
                        # Fallback: direct copy
                        import shutil
                        shutil.copy2(str(stem_file), str(rendered_path))

                    rendered_stems[stem_name] = str(rendered_path)
                    await asyncio.sleep(0.05)
            else:
                # No stems available — try MIDI synthesis
                try:
                    from auralis.hands.synth import render_midi_to_audio, save_audio, VoiceConfig

                    for stem_name, pattern_info in midi_patterns.items():
                        if pattern_info.get("pattern_loaded"):
                            from auralis.grid.midi import load_midi, pattern_to_note_events
                            midi_file = midi_dir / f"{stem_name}.mid"
                            if midi_file.exists():
                                pattern = load_midi(midi_file)
                                note_events = pattern_to_note_events(pattern, bpm=plan["bpm"])
                                audio = render_midi_to_audio(note_events, sr=44100)
                                out_path = rendered_dir / f"{stem_name}.wav"
                                save_audio(audio, out_path)
                                rendered_stems[stem_name] = str(out_path)
                except ImportError:
                    pass

        except ImportError:
            job["stages"]["hands"]["message"] = "Audio libraries not available"

        job["stages"]["hands"]["status"] = "completed"
        job["stages"]["hands"]["message"] = f"Rendered {len(rendered_stems)} stems"
        job["progress"] = 70
        _log(job, f"✓ HANDS complete — {len(rendered_stems)} stems rendered", "success")
        gc.collect()
        _save_jobs()  # checkpoint after HANDS

        # Stage 5: CONSOLE — Professional Mix + Master
        job["stage"] = "console"
        job["stages"]["console"]["status"] = "running"
        job["stages"]["console"]["message"] = "Building mix recipes from analysis..."
        _log(job, "── STAGE 5: CONSOLE ──", "stage")
        _log(job, "🎚️ Building mix recipes from EAR analysis...")

        mix_path = project_dir / "mix.wav"
        master_path = project_dir / "master.wav"
        master_info: dict[str, Any] = {}

        # Gather analysis data for stem recipes
        stem_analysis_data = (job.get("result") or {}).get("stem_analysis", {})
        ear_analysis_data = analysis  # Full EAR analysis from stage 1

        # Run gap analysis against reference bank
        ref_targets: dict[str, Any] = {}
        try:
            from auralis.ear.reference_bank import ReferenceBank
            from auralis.console.gap_analyzer import analyze_gaps, format_gap_report_for_logs

            bank = ReferenceBank(settings.projects_dir)
            if bank.count() > 0:
                gap_report = analyze_gaps(ear_analysis_data, stem_analysis_data, bank)
                # Store gap report in results
                if job.get("result") is None:
                    job["result"] = {}
                job["result"]["gap_report"] = gap_report.to_dict()

                # Log the gap report
                for line in format_gap_report_for_logs(gap_report):
                    _log(job, line)
                _log(job, "")

                # Extract reference targets for stem recipes
                stem_avgs = bank.get_stem_averages()
                ref_targets = {
                    k: {
                        "rms_db": v.rms_db,
                        "peak_db": v.peak_db,
                        "freq_bands": v.freq_bands,
                        "dynamic_range_db": v.dynamic_range_db,
                        "energy_pct": v.energy_pct,
                    }
                    for k, v in stem_avgs.items()
                }

                # ── DNA BRAIN: THINK ───────────────────
                brain_report = None
                try:
                    from auralis.console.dna_brain import DNABrain
                    deep_profile = bank.get_deep_profile()
                    if deep_profile:
                        brain = DNABrain()
                        brain_report = brain.think(
                            deep_profile=deep_profile,
                            stem_analysis=stem_analysis_data,
                            gap_report=gap_report.to_dict(),
                            ear_data=ear_analysis_data,
                        )
                        # Store brain report in results
                        job["result"]["brain_report"] = brain_report.to_dict()

                        # Log full reasoning chain
                        _log(job, "")
                        _log(job, "── DNA BRAIN REASONING ──", "stage")
                        for line in brain_report.reasoning_chain:
                            _log(job, line)
                        _log(job, "")
                    else:
                        _log(job, "📊 No deep DNA profiles — brain skipped")
                except Exception as brain_err:
                    _log(job, f"DNA Brain skipped: {brain_err}", "warning")

                # ── STEM DECISIONS ──────────────────────
                try:
                    from auralis.console.stem_decisions import (
                        make_decisions, format_decisions_for_logs,
                    )
                    from auralis.console.stem_generator import (
                        generate_stem, find_reference_stem_path,
                    )

                    decision_report = make_decisions(gap_report, ear_analysis_data, bank, brain_report=brain_report)
                    job["result"]["stem_decisions"] = decision_report.to_dict()

                    # Log decisions
                    for line in format_decisions_for_logs(decision_report):
                        _log(job, line)
                    _log(job, "")

                    # Execute decisions: generate stems for enhance/replace
                    stems_dir = project_dir / "stems"
                    gen_dir = project_dir / "generated_stems"
                    track_duration = float(ear_analysis_data.get("duration", 180.0))
                    track_bpm = float(ear_analysis_data.get("bpm", 120.0))
                    track_key = str(ear_analysis_data.get("key", "C"))
                    track_scale = str(ear_analysis_data.get("scale", "minor"))

                    bank_entries = bank.list_references()

                    for stem_name, decision in decision_report.decisions.items():
                        if decision.action in ("enhance", "replace"):
                            # Find reference stem for one-shot extraction (drums)
                            ref_stem_path = None
                            if stem_name == "drums":
                                ref_stem_path = find_reference_stem_path(
                                    "drums", bank_entries, settings.projects_dir, track_bpm
                                )

                            gen_path = generate_stem(
                                decision=decision,
                                bpm=track_bpm,
                                key=track_key,
                                scale=track_scale,
                                duration_s=track_duration,
                                output_dir=gen_dir,
                                reference_stem_path=ref_stem_path,
                                stem_plan=brain_report.stem_plans.get(stem_name) if brain_report else None,
                            )

                            if gen_path:
                                prefix = "gen" if decision.action == "replace" else "layer"
                                gen_stem_name = f"{prefix}_{stem_name}"
                                rendered_stems[gen_stem_name] = str(gen_path)
                                _log(job, f"  🎹 Generated: {gen_stem_name} → {gen_path.name}")

                                # For REPLACE: mute the original stem
                                if decision.action == "replace" and stem_name in rendered_stems:
                                    _log(job, f"  🔇 Muting original {stem_name} (replaced)")
                                    # Mark for muting in mixer (volume = -inf)
                                    if "muted_stems" not in job:
                                        job["muted_stems"] = []
                                    job["muted_stems"].append(stem_name)

                        elif decision.action == "mute":
                            if stem_name in rendered_stems:
                                _log(job, f"  🔇 Muting {stem_name}: {decision.reason}")
                                if "muted_stems" not in job:
                                    job["muted_stems"] = []
                                job["muted_stems"].append(stem_name)

                    _log(job, "")

                except Exception as dec_err:
                    _log(job, f"Stem decisions skipped: {dec_err}", "warning")

            else:
                _log(job, "📊 No references in bank — using default recipes")
                _log(job, "   Add pro tracks via /api/reference/add to enable gap analysis")
        except Exception as gap_err:
            _log(job, f"Gap analysis skipped: {gap_err}", "warning")

        try:
            if rendered_stems:
                # Professional mix with stem recipes (SUBPROCESS)
                try:
                    stem_paths_str = {k: str(v) for k, v in rendered_stems.items()}

                    # Remove muted stems from the mix
                    muted = set(job.get("muted_stems", []))
                    if muted:
                        stem_paths_str = {k: v for k, v in stem_paths_str.items() if k not in muted}

                    # Serialize brain stem plans for subprocess
                    brain_stem_plans_dict = None
                    if brain_report and brain_report.stem_plans:
                        brain_stem_plans_dict = {}
                        for sname, splan in brain_report.stem_plans.items():
                            brain_stem_plans_dict[sname] = splan.to_dict()
                        # Include master plan so mix buses can derive config
                        if brain_report.master_plan:
                            try:
                                brain_stem_plans_dict["_master"] = brain_report.master_plan.to_dict()
                            except Exception:
                                pass

                    mix_result = await asyncio.to_thread(
                        _run_in_subprocess,
                        _subprocess_mix,
                        stem_paths=stem_paths_str,
                        output_path=str(mix_path),
                        bpm=plan.get("bpm", 120.0),
                        stem_analysis=stem_analysis_data,
                        ear_analysis=ear_analysis_data,
                        ref_targets=ref_targets if ref_targets else None,
                        stem_decisions=(job.get("result", {}).get("stem_decisions", {}) or {}).get("decisions"),
                        brain_stem_plans=brain_stem_plans_dict,
                    )

                    # Log each recipe decision
                    for recipe_desc in mix_result.get("recipes", []):
                        _log(job, f"  {recipe_desc}")

                    _log(job, f"🔊 Buses: reverb + delay ({mix_result.get('buses_used', 0)} active)")
                    peak = mix_result.get("peak_db", 0)
                    _log(job, f"✓ Mixed {mix_result['tracks_mixed']} tracks, peak: {peak:.1f} dBFS", "success")

                    job["stages"]["console"]["message"] = (
                        f"Mixed {mix_result['tracks_mixed']} tracks → mastering..."
                    )
                except Exception as e:
                    _log(job, f"Mixer subprocess failed ({e}) — falling back to simple sum", "warning")
                    # Fallback: quick sum mix (sf already imported at module level)
                    all_audio = []
                    sr = 44100
                    for path in rendered_stems.values():
                        try:
                            data, sr = sf.read(str(path), dtype="float64")
                            mono = np.mean(data, axis=1) if data.ndim > 1 else data
                            all_audio.append(mono)
                        except Exception:
                            pass
                    
                    if all_audio:
                        max_len = max(len(a) for a in all_audio)
                        mix = np.zeros(max_len)
                        for a in all_audio:
                            mix[:len(a)] += a
                        mix /= len(all_audio)
                        sf.write(str(mix_path), mix, sr)
                    job["stages"]["console"]["message"] = f"Sum-mixed {len(all_audio)} stems (fallback)"

                # LUFS target: brain → reference bank → original track (fallback)
                target_lufs = None
                # 1. Use brain's target (derived from reference DNA)
                if brain_report and hasattr(brain_report, 'master_plan'):
                    brain_target = getattr(brain_report.master_plan, 'target_lufs', None)
                    if brain_target is not None and isinstance(brain_target, (int, float)):
                        target_lufs = float(brain_target)
                # 2. Use reference bank average LUFS
                if target_lufs is None and bank and hasattr(bank, 'get_master_averages'):
                    ref_avg = bank.get_master_averages()
                    if ref_avg and ref_avg.get("lufs"):
                        target_lufs = float(ref_avg["lufs"])
                # 3. Fallback: original track LUFS (clamped)
                if target_lufs is None:
                    target_lufs = analysis.get("integrated_lufs", -14.0)
                    if not isinstance(target_lufs, (int, float)):
                        target_lufs = -14.0
                # Safety: never target quieter than -16 LUFS
                target_lufs = max(float(target_lufs), -16.0)

                _log(job, f"💎 Mastering: target {target_lufs} LUFS")
                _log(job, "  Chain: M/S EQ → Soft Clip → Multiband Sat → Harmonic Exciter")
                _log(job, "  Chain: Multiband Comp → Stereo Width → Limiter → Dither")
                try:
                    # Prepare brain master plan for subprocess
                    brain_plan_dict = None
                    if brain_report and brain_report.master_plan:
                        try:
                            mp = brain_report.master_plan
                            brain_plan_dict = {
                                k: getattr(mp, k, None)
                                for k in dir(mp)
                                if not k.startswith('_') and not callable(getattr(mp, k, None))
                            }
                        except Exception:
                            pass

                    await asyncio.to_thread(
                        _run_in_subprocess,
                        _subprocess_master,
                        input_path=str(mix_path),
                        output_path=str(master_path),
                        target_lufs=target_lufs,
                        bpm=plan.get("bpm", 120.0),
                        brain_plan_dict=brain_plan_dict,
                    )
                    
                    master_info = {"lufs": target_lufs, "method": "full_chain"}
                    _log(job, f"✓ Mastered to {target_lufs} LUFS (10-stage chain)", "success")

                    # ── AUTO-CORRECTION: compare vs DNA ──
                    if brain_report and master_path.exists():
                        try:
                            from auralis.console.auto_correct import evaluate_and_correct
                            deep_profile = bank.get_deep_profile() if bank else {}

                            if deep_profile:
                                # Build stem paths dict for per-stem evaluation
                                stem_paths_for_ac = {
                                    k: str(v) for k, v in rendered_stems.items()
                                    if not k.startswith("_")
                                }
                                correction = await asyncio.to_thread(
                                    evaluate_and_correct,
                                    master_path=str(master_path),
                                    deep_profile=deep_profile,
                                    pass_number=1,
                                    max_passes=2,
                                    threshold=0.15,
                                    stem_paths=stem_paths_for_ac,
                                    ref_targets=ref_targets if ref_targets else None,
                                )

                                job["result"]["auto_correction"] = correction.to_dict()
                                _log(job, f"🔄 Auto-correction: gap={correction.total_gap:.0%}, pass={correction.pass_number}")

                                # Log per-stem corrections
                                for sname, sc in correction.stem_corrections.items():
                                    if sc.needs_correction:
                                        _log(job, f"  🎚️ {sname}: gap={sc.gap_score:.0%}")
                                        for reason in sc.reasoning[:3]:
                                            _log(job, f"    {reason}")

                                if correction.master_correction:
                                    for reason in correction.master_correction.reasoning:
                                        _log(job, f"  {reason}")

                                if correction.should_reprocess:
                                    _log(job, "🔄 Gap detected — applying corrections for pass 2...")

                                    # Merge corrections into brain plan
                                    corrected_plan = dict(brain_plan_dict or {})
                                    mc = correction.master_correction
                                    if mc and mc.corrections:
                                        # Apply LUFS correction
                                        if "target_lufs_adjust" in mc.corrections:
                                            corrected_plan["target_lufs"] = mc.corrections["target_lufs_adjust"]
                                            _log(job, f"  → LUFS adjusted to {mc.corrections['target_lufs_adjust']:.1f}")

                                        # Apply EQ corrections (append to existing mid bands)
                                        if "eq_adjust" in mc.corrections:
                                            existing = list(corrected_plan.get("mid_eq_bands", []))
                                            for eq_fix in mc.corrections["eq_adjust"]:
                                                if isinstance(eq_fix, dict):
                                                    freq = eq_fix.get("freq", 1000)
                                                    gain = eq_fix.get("gain_db", 0)
                                                    q = eq_fix.get("q", 1.0)
                                                    existing.append((freq, gain, q))
                                                    _log(job, f"  → EQ fix: {gain:+.1f}dB @{freq:.0f}Hz")
                                                else:
                                                    existing.append(eq_fix)
                                            corrected_plan["mid_eq_bands"] = existing

                                    await asyncio.to_thread(
                                        _run_in_subprocess,
                                        _subprocess_master,
                                        input_path=str(mix_path),
                                        output_path=str(master_path),
                                        target_lufs=corrected_plan.get("target_lufs", target_lufs),
                                        bpm=plan.get("bpm", 120.0),
                                        brain_plan_dict=corrected_plan,
                                    )
                                    _log(job, "✓ Re-mastered with corrections (pass 2)", "success")
                                else:
                                    _log(job, "✓ Master passed quality check", "success")
                        except Exception as ac_err:
                            _log(job, f"Auto-correction skipped: {ac_err}", "warning")
                except Exception as e:
                    _log(job, f"Mastering failed ({e}) — bypassing", "warning")
                    # Fallback: copy mix to master
                    import shutil
                    shutil.copy2(mix_path, master_path)
                    master_info = {"lufs": None, "method": "bypass"}
            else:
                master_info = {"status": "no_stems_to_mix"}

        except ImportError:
            master_info = {"error": "Audio libraries not available"}

        await asyncio.sleep(0.1)
        job["stages"]["console"]["status"] = "completed"
        job["stages"]["console"]["message"] = (
            f"Master complete — LUFS: {master_info.get('est_lufs', 'N/A')}"
        )
        job["progress"] = 85
        _log(job, "✓ CONSOLE complete", "success")
        gc.collect()
        _save_jobs()  # checkpoint after CONSOLE

        # Stage 6: QC — 12-dimension comparison (REAL)
        job["stage"] = "qc"
        job["stages"]["qc"]["status"] = "running"
        job["stages"]["qc"]["message"] = "Running 12-dimension quality comparison..."
        _log(job, "── STAGE 6: QC ──", "stage")
        _log(job, "Running 12-dimension A/B comparison...")

        qc_result: dict[str, Any] = {
            "dimensions": {},
            "overall_score": 0.0,
            "target_score": 90.0,
            "status": "no_comparison",
        }

        if audio_file and master_path.exists():
            try:
                qc_result = await asyncio.to_thread(
                    _run_in_subprocess,
                    _subprocess_qc,
                    original_path=str(audio_file),
                    reconstruction_path=str(master_path),
                )

                # Save QC results
                qc_path = project_dir / "qc_result.json"
                qc_path.write_text(json.dumps(qc_result, indent=2, default=str))

                job["stages"]["qc"]["message"] = (
                    f"QC Score: {qc_result['overall_score']:.1f}% | "
                    f"{'✅ PASSED' if qc_result.get('passed') else '⚠️ Below target'} | "
                    f"Weakest: {qc_result.get('weakest', 'N/A')}"
                )
                _log(job, f"QC Score: {qc_result['overall_score']:.1f}% — {'PASSED ✅' if qc_result.get('passed') else 'BELOW TARGET ⚠️'}", "success" if qc_result.get('passed') else "warn")
                _log(job, f"  Weakest: {qc_result.get('weakest', 'N/A')} | Strongest: {qc_result.get('strongest', 'N/A')}")
            except (ImportError, Exception) as e:
                qc_result["error"] = str(e)
                job["stages"]["qc"]["message"] = f"QC comparison error: {e}"
        else:
            job["stages"]["qc"]["message"] = (
                "12-dimension scoring ready — awaiting both original and master"
            )

        await asyncio.sleep(0.1)
        job["stages"]["qc"]["status"] = "completed"
        job["progress"] = 100

        # Complete
        job["status"] = "completed"
        _log(job, "═══════════════════════════════════", "stage")
        _log(job, f"✓ Pipeline complete — {len(rendered_stems)} stems, QC: {qc_result.get('overall_score', 0):.1f}%", "success")
        # UPDATE (not replace!) — preserve intelligence data from earlier stages:
        #   stem_analysis (HANDS), gap_report, brain_report, stem_decisions (CONSOLE)
        if job.get("result") is None:
            job["result"] = {}
        job["result"].update({
            "analysis": {
                "bpm": plan["bpm"],
                "key": plan["key"],
                "scale": plan["scale"],
                "duration": plan["duration"],
                "sections_detected": plan["total_sections"],
            },
            "midi_tracks": midi_patterns,
            "rendered_stems": len(rendered_stems),
            "master": master_info,
            "qc": qc_result,
            "files": {
                "original": str(audio_file) if audio_file else None,
                "mix": str(mix_path) if mix_path.exists() else None,
                "master": str(master_path) if master_path.exists() else None,
                "stems": list(rendered_stems.values()),
            },
        })

        _save_jobs()  # persist final completed state

    except Exception as e:
        job["status"] = "error"
        job["stages"][job["stage"]]["status"] = "error"
        job["stages"][job["stage"]]["message"] = str(e)
        _log(job, f"FATAL: {e}", "error")
        _save_jobs()  # persist error state
    finally:
        gc.collect()  # ensure cleanup regardless of outcome


# ── TEXT-TO-TRACK CREATION ──────────────────────────────────────


@router.post("/create")
async def create_track(req: CreateRequest) -> dict[str, Any]:
    """Create a brand-new track from a text description — no upload needed.

    Uses Gemini Brain to generate a production plan, then renders the full track
    through the Grid → Hands → Console pipeline.
    """
    job_id = str(uuid.uuid4())
    project_dir = settings.projects_dir / job_id
    project_dir.mkdir(parents=True, exist_ok=True)

    # Enrich the description with forced parameters
    enriched = req.description
    if req.genre != "auto":
        enriched += f". Genre: {req.genre}"
    if req.bpm is not None:
        enriched += f". BPM: {req.bpm}"
    if req.key is not None:
        enriched += f". Key: {req.key}"
    if req.scale is not None:
        enriched += f". Scale: {req.scale}"

    _reconstruct_jobs[job_id] = {
        "job_id": job_id,
        "project_id": job_id,
        "mode": "create",
        "status": "running",
        "stage": "brain",
        "progress": 0,
        "stages": {
            "brain": {"status": "running", "message": "Generating production plan with Gemini..."},
            "grid": {"status": "pending", "message": ""},
            "hands": {"status": "pending", "message": ""},
            "console": {"status": "pending", "message": ""},
        },
        "logs": [],
        "result": None,
        "description": req.description,
    }
    _log(_reconstruct_jobs[job_id], f"🎨 Creating track from description: {req.description[:100]}...", "info")
    _log(_reconstruct_jobs[job_id], f"Parameters: genre={req.genre}, bpm={req.bpm}, key={req.key}, scale={req.scale}", "info")

    asyncio.create_task(_run_creation(job_id, enriched, project_dir, req.reference_ids))
    _save_jobs()

    return _reconstruct_jobs[job_id]


async def _run_creation(job_id: str, description: str, project_dir: Path, reference_ids: list[str]) -> None:
    """Run the full text-to-track creation pipeline in background."""
    job = _reconstruct_jobs[job_id]

    try:
        import gc

        # Collect reference paths — auto-load ALL from DNA bank if none specified
        reference_paths: list[Path] = []

        if reference_ids:
            # Use specific references requested
            for ref_id in reference_ids:
                ref_dir = settings.projects_dir / ref_id
                if ref_dir.exists():
                    audio = _find_audio_file(ref_dir)
                    if audio:
                        reference_paths.append(audio)
                        _log(job, f"Using reference: {audio.name}", "info")
        else:
            # Auto-load ALL references from the DNA bank
            _log(job, "🧬 Loading DNA references for style cloning...", "info")
            from auralis.ear.reference_bank import ReferenceBank
            bank = ReferenceBank(settings.projects_dir)
            all_refs = bank.list_references()
            for ref in all_refs:
                ref_dir = settings.projects_dir / ref.get("track_id", "")
                if ref_dir.exists():
                    audio = _find_audio_file(ref_dir)
                    if audio:
                        reference_paths.append(audio)
                        _log(job, f"🎵 DNA ref: {ref.get('name', audio.name)}", "info")

            if reference_paths:
                _log(job, f"Loaded {len(reference_paths)} DNA references — cloning sonic identity", "success")
            else:
                _log(job, "⚠️ No DNA references found — generating with synth defaults", "warning")

        # ── Step 1: Brain — Generate Production Plan with Gemini ──
        job["stages"]["brain"]["status"] = "running"
        job["stages"]["brain"]["message"] = "Gemini 3 Pro generating production plan..."
        _log(job, "🧠 Brain: Asking Gemini 3 Pro to create a production plan...", "stage")

        from auralis.brain.agent import generate_production_plan, BrainConfig
        plan = generate_production_plan(description, BrainConfig())

        _log(job, f"Plan: {plan.title} | {plan.genre} | {plan.bpm} BPM | {plan.key} {plan.scale}", "success")
        _log(job, f"Structure: {' → '.join(plan.structure)}", "info")
        job["stages"]["brain"]["status"] = "completed"
        job["stages"]["brain"]["message"] = f"Plan: {plan.title}"
        job["progress"] = 20

        # ── Step 2: Grid — Generate Arrangement ──
        job["stages"]["grid"]["status"] = "running"
        job["stages"]["grid"]["message"] = "Creating arrangement..."
        job["stage"] = "grid"
        _log(job, "📐 Grid: Generating arrangement from plan...", "stage")

        from auralis.grid.arrangement import ArrangementConfig, generate_arrangement, arrangement_summary
        arr_config = ArrangementConfig(
            key=plan.key,
            scale=plan.scale,
            bpm=plan.bpm,
            structure=plan.structure,
            genre=plan.genre if plan.genre in ("house", "techno", "ambient", "pop", "hip_hop") else "house",
        )
        arrangement = generate_arrangement(arr_config)
        arr_info = arrangement_summary(arrangement)

        _log(job, f"Arrangement: {arrangement.total_bars} bars, {len(arrangement.sections)} sections", "success")
        job["stages"]["grid"]["status"] = "completed"
        job["stages"]["grid"]["message"] = f"{arrangement.total_bars} bars"
        job["progress"] = 35

        # ── Step 3: Hands — Render all stems ──
        job["stages"]["hands"]["status"] = "running"
        job["stages"]["hands"]["message"] = "Rendering stems..."
        job["stage"] = "hands"
        _log(job, "🎹 Hands: Rendering stems with synth engine...", "stage")

        from auralis.brain.production_ai import render_track, RenderProgress

        def on_progress(p: RenderProgress) -> None:
            _log(job, f"[{p.stage}] {p.message}", "info")
            if p.stage == "hands":
                job["progress"] = 35 + int(p.progress * 40)
            elif p.stage == "mix":
                job["progress"] = 75 + int(p.progress * 15)

        track_render = render_track(
            description=description,
            output_dir=project_dir,
            on_progress=on_progress,
            reference_paths=[str(p) for p in reference_paths] if reference_paths else None,
        )

        _log(job, f"Rendered {len(track_render.stems)} stems", "success")
        job["stages"]["hands"]["status"] = "completed"
        job["stages"]["hands"]["message"] = f"{len(track_render.stems)} stems"
        job["progress"] = 85

        # ── Step 4: Console — copy mix as master ──
        job["stages"]["console"]["status"] = "running"
        job["stages"]["console"]["message"] = "Finalizing master..."
        job["stage"] = "console"

        import shutil
        mix_path = Path(track_render.output_path)
        master_path = project_dir / "master.wav"
        if mix_path.exists() and not master_path.exists():
            shutil.copy2(mix_path, master_path)
        elif mix_path.exists():
            master_path = mix_path

        _log(job, f"Master ready: {master_path.name}", "success")
        job["stages"]["console"]["status"] = "completed"
        job["stages"]["console"]["message"] = "Master ready"
        job["progress"] = 100

        # ── Build result ──
        files = {
            "master": str(master_path) if master_path.exists() else None,
            "mix": str(mix_path) if mix_path.exists() else None,
        }
        # Add stems
        stems_dir = project_dir / "stems"
        if not stems_dir.exists():
            stems_dir.mkdir(parents=True, exist_ok=True)
        for stem_name, stem_path in track_render.stems.items():
            stem_file = Path(stem_path)
            if stem_file.exists():
                # Copy stems to standard location
                target = stems_dir / f"{stem_name}.wav"
                if not target.exists():
                    shutil.copy2(stem_file, target)
                files[f"stem_{stem_name}"] = str(target)

        job["status"] = "completed"
        job["result"] = {
            "analysis": {
                "bpm": plan.bpm,
                "key": plan.key,
                "scale": plan.scale,
                "duration": track_render.duration_s,
                "sections_detected": len(arrangement.sections),
            },
            "plan": {
                "title": plan.title,
                "genre": plan.genre,
                "mood": plan.mood,
                "energy": plan.energy,
                "structure": plan.structure,
                "description": plan.description,
            },
            "rendered_stems": len(track_render.stems),
            "stem_analysis": {},
            "brain_report": {
                "reasoning_chain": [f"Created with Gemini 3 Pro: {plan.description}"],
                "interaction_log": [f"Description: {description}"],
            },
            "files": files,
        }

        _log(job, f"✅ Track created: {plan.title} ({plan.genre}, {plan.bpm} BPM, {plan.key} {plan.scale})", "success")
        _save_jobs()
        gc.collect()

    except Exception as e:
        import traceback
        _log(job, f"❌ Creation failed: {e}", "error")
        _log(job, traceback.format_exc()[-500:], "error")
        job["status"] = "error"
        job["result"] = {"error": str(e)}
        _save_jobs()


# ── AI CRITIC — Diagnose + Propose ──────────────────────────────


@router.post("/diagnose/{job_id}")
async def diagnose_track(job_id: str):
    """Analyze a completed track bar-by-bar to detect energy gaps, silences, and spectral issues.

    Returns a list of issues with bar positions, severity, and affected stems.
    """
    job = _reconstruct_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "completed":
        raise HTTPException(400, "Job must be completed before diagnosis")

    result = job.get("result", {})
    analysis = result.get("analysis", {})
    bpm = float(analysis.get("bpm", 120.0))
    duration = float(analysis.get("duration", 180.0))
    key = str(analysis.get("key", "C"))
    scale = str(analysis.get("scale", "minor"))

    # Calculate bar timing
    bar_duration = (60.0 / bpm) * 4  # 4 beats per bar
    total_bars = int(duration / bar_duration)

    project_dir = settings.projects_dir / job_id

    # ── Per-bar energy analysis across all stems + master ──
    import librosa

    energy_map: dict[str, list[float]] = {}
    stem_names = ["drums", "bass", "vocals", "other"]

    # Analyze master first
    master_path = project_dir / "master.wav"
    mix_path = project_dir / "mix.wav"
    main_audio = master_path if master_path.exists() else (mix_path if mix_path.exists() else None)

    if main_audio:
        try:
            y, sr = librosa.load(main_audio, sr=22050, mono=True)
            bar_energies = []
            for bar in range(total_bars):
                start_sample = int(bar * bar_duration * sr)
                end_sample = int((bar + 1) * bar_duration * sr)
                chunk = y[start_sample:min(end_sample, len(y))]
                if len(chunk) > 0:
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    db = float(20 * np.log10(max(rms, 1e-10)))
                    bar_energies.append(round(db, 1))
                else:
                    bar_energies.append(-60.0)
            energy_map["master"] = bar_energies
        except Exception:
            pass

    # Analyze each stem
    stems_dir = project_dir / "stems"
    for stem_name in stem_names:
        stem_path = stems_dir / f"{stem_name}.wav"
        if not stem_path.exists():
            continue
        try:
            y, sr = librosa.load(stem_path, sr=22050, mono=True)
            bar_energies = []
            for bar in range(total_bars):
                start_sample = int(bar * bar_duration * sr)
                end_sample = int((bar + 1) * bar_duration * sr)
                chunk = y[start_sample:min(end_sample, len(y))]
                if len(chunk) > 0:
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    db = float(20 * np.log10(max(rms, 1e-10)))
                    bar_energies.append(round(db, 1))
                else:
                    bar_energies.append(-60.0)
            energy_map[stem_name] = bar_energies
        except Exception:
            continue

    # ── Detect issues ──
    issues: list[dict[str, Any]] = []
    issue_counter = 0

    master_energy = energy_map.get("master", [])

    for bar in range(total_bars):
        # Skip first and last bar (naturally quiet)
        if bar == 0 or bar >= total_bars - 1:
            continue

        # Calculate surrounding energy for context
        prev_energy = master_energy[bar - 1] if bar > 0 and master_energy else -30
        curr_energy = master_energy[bar] if bar < len(master_energy) else -30
        next_energy = master_energy[bar + 1] if bar + 1 < len(master_energy) else -30
        avg_context = (prev_energy + next_energy) / 2

        # Detect energy gap: bar significantly quieter than its neighbors
        energy_drop = avg_context - curr_energy
        if energy_drop > 6:  # > 6dB drop
            # Find which stems are responsible
            affected = []
            for stem_name in stem_names:
                stem_e = energy_map.get(stem_name, [])
                if bar < len(stem_e):
                    stem_prev = stem_e[bar - 1] if bar > 0 else stem_e[bar]
                    stem_curr = stem_e[bar]
                    if stem_prev - stem_curr > 6:
                        affected.append(stem_name)

            severity = "high" if energy_drop > 12 else ("medium" if energy_drop > 8 else "low")
            issue_counter += 1
            issues.append({
                "id": f"gap_{issue_counter}",
                "type": "energy_gap",
                "bar": bar + 1,  # 1-indexed for user
                "time_start": round(bar * bar_duration, 2),
                "time_end": round((bar + 1) * bar_duration, 2),
                "severity": severity,
                "description": (
                    f"Energy drops {energy_drop:.1f} dB at bar {bar + 1} "
                    f"— {'silence' if curr_energy < -40 else 'thin section'} "
                    f"in {', '.join(affected) if affected else 'overall mix'}"
                ),
                "affected_stems": affected if affected else ["master"],
                "context": {
                    "energy_before": round(prev_energy, 1),
                    "energy_at": round(curr_energy, 1),
                    "energy_after": round(next_energy, 1),
                    "drop_db": round(energy_drop, 1),
                },
            })

        # Detect flat/silent sections
        if curr_energy < -40:
            # Check if multiple consecutive bars are silent
            affected = [s for s in stem_names if bar < len(energy_map.get(s, [])) and energy_map[s][bar] < -40]
            if len(affected) >= 2:
                issue_counter += 1
                issues.append({
                    "id": f"silence_{issue_counter}",
                    "type": "silence",
                    "bar": bar + 1,
                    "time_start": round(bar * bar_duration, 2),
                    "time_end": round((bar + 1) * bar_duration, 2),
                    "severity": "high",
                    "description": f"Near-silence at bar {bar + 1} — {', '.join(affected)} are quiet",
                    "affected_stems": affected,
                    "context": {"energy_at": round(curr_energy, 1)},
                })

    # Sort issues by severity then bar
    severity_order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda x: (severity_order.get(x["severity"], 3), x["bar"]))

    return {
        "job_id": job_id,
        "bpm": bpm,
        "key": key,
        "scale": scale,
        "duration": duration,
        "total_bars": total_bars,
        "bar_duration_s": round(bar_duration, 3),
        "issues": issues,
        "issue_count": len(issues),
        "energy_map": energy_map,
        "stem_analysis_summary": {
            k: {"avg_db": round(sum(v) / len(v), 1) if v else -60, "bars": len(v)}
            for k, v in energy_map.items()
        },
    }


@router.post("/propose/{job_id}")
async def propose_fixes(job_id: str):
    """Generate AI-powered musical improvement proposals using Gemini.

    Analyzes the full track context (instruments, arrangement, stems, energy)
    and proposes both technical fixes AND creative/musical improvements.
    """
    # Run technical diagnosis first
    diagnosis = await diagnose_track(job_id)

    job = _reconstruct_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found — may have been lost after restart")
    result = job.get("result", {})
    analysis = result.get("analysis", {})
    stem_analysis = result.get("stem_analysis", {})
    brain_report = result.get("brain_report", {})
    xray = result.get("xray_analysis", {})

    # Build rich musical context for Gemini
    context = {
        "track_info": {
            "bpm": diagnosis["bpm"],
            "key": diagnosis["key"],
            "scale": diagnosis["scale"],
            "duration_s": diagnosis["duration"],
            "total_bars": diagnosis["total_bars"],
        },
        "technical_issues": diagnosis["issues"],
        "energy_map_summary": diagnosis["stem_analysis_summary"],
        "stem_info": {
            name: {
                "rms_db": data.get("rms_db"),
                "peak_db": data.get("peak_db"),
                "dynamic_range_db": data.get("dynamic_range"),
            }
            for name, data in stem_analysis.items()
            if isinstance(data, dict) and "error" not in data
        },
    }

    # Add deep analysis data if available
    if analysis:
        context["arrangement"] = {
            "sections": analysis.get("sections", []),
            "instruments_detected": analysis.get("instruments", []),
            "genre_hints": analysis.get("genre", ""),
            "mood": analysis.get("mood", ""),
        }

    # Add X-Ray bar-by-bar analysis if available
    if xray:
        context["bar_analysis"] = {
            "total_bars": xray.get("total_bars"),
            "energy_curve": xray.get("energy_curve", [])[:20],  # first 20 bars
            "bar_descriptions": xray.get("bar_descriptions", [])[:10],
        }

    # Add brain reasoning if available
    if brain_report:
        context["brain_reasoning"] = {
            "reasoning_chain": brain_report.get("reasoning_chain", [])[:5],
            "interaction_log": brain_report.get("interaction_log", [])[:5],
        }

    from auralis.brain.gemini_client import generate

    prompt = f"""# AURALIS Producer Session — Deep Track Review

You are sitting in the studio, headphones on, reviewing a track for an artist who trusts your ear. This is not a technical audit — this is a **musical conversation**.

## The Track
- **Key:** {diagnosis['key']} {diagnosis['scale']}
- **BPM:** {diagnosis['bpm']}
- **Duration:** {round(diagnosis['duration'])}s ({diagnosis['total_bars']} bars)

## What We Know About This Track
{json.dumps(context, indent=2, default=str)}

---

## How To Think About This

Before writing proposals, mentally walk through the track from start to finish. Ask yourself:

**1. THE FIRST 8 BARS** — Does the intro hook the listener immediately? Is there a sonic signature from the first beat, or does it take too long to get going? Would a DJ or playlist curator skip this intro?

**2. ARRANGEMENT ARCHITECTURE** — Map the sections: intro → build → drop → breakdown → drop 2 → outro. Are all sections present? Is any section too long, too short, or missing entirely? Does the track have the classic "journey" structure, or does it plateau? Think about the **golden ratio** — the main drop should hit around 30-40% into the track.

**3. SONIC PALETTE & INSTRUMENTATION** — Listen to each stem individually. Is the bass carrying enough weight for the genre? Are there enough textural layers in the "other" stem (pads, arps, FX, atmospherics)? Are the drums punchy enough? Is the kick-bass relationship tight? Are vocals (if present) treated with enough character — reverb, delay, pitch effects, chopping?

**4. ENERGY NARRATIVE** — Plot the energy curve mentally. A great track tells a story: tension → release → tension → bigger release. Are there enough contrast points? Does the track breathe? Are there moments of silence or reduction that make the loud parts feel louder? Think about tracks by Bicep, Disclosure, or Bonobo — they master the art of energy dynamics.

**5. RHYTHMIC IDENTITY** — Is the groove distinctive? Could you identify this track by its rhythm alone? Are there enough percussion layers (hats, shakers, congas, rides) to give the rhythm depth? Is there polyrhythmic interest or is it too straight?

**6. HARMONIC DEPTH** — Is the chord progression interesting or generic? Are there unexpected harmonic movements? Could adding a counter-melody, a bass note variation, or a key change in the bridge add emotional depth?

**7. SPATIAL DESIGN** — Think about the stereo field. Are elements spread across the soundstage, or is everything centered? Could panning automation on arps/fx/hats create more movement? Is reverb used to create depth (front-to-back), or is everything flat?

**8. TRANSITION CRAFT** — How do sections connect? Are there risers, sweeps, drum fills, filter automations, reverse reverbs, or impact sounds at transition points? Or do sections just... start? Great transitions make or break a track.

---

## Your Output

Generate **5-8 specific proposals** that would genuinely elevate this track. Each one should feel like advice from a mentor, not a robot.

{f"We also detected {len(diagnosis['issues'])} technical issues — weave fixes for these into your proposals naturally." if diagnosis['issues'] else "No technical problems were detected — the production is clean. Focus 100% on CREATIVE and MUSICAL improvements that would take this from good to exceptional."}

**Mix your proposals across these categories:**
- At least 1 arrangement/structure proposal
- At least 1 instrumentation/sonic palette proposal
- At least 1 energy/dynamics proposal
- At least 1 bold creative idea that surprises

```json
{{
  "proposals": [
    {{
      "id": "prop_1",
      "title": "Crisp, specific title (e.g. 'Strip the drums at bar 24 for a 4-bar breakdown')",
      "description": "2-3 sentences of musical reasoning. Explain WHY this works, reference techniques from professional tracks when relevant.",
      "category": "arrangement | instrumentation | dynamics | mix | creative | emotional",
      "action": "add_element | remove_element | adjust_levels | add_fx | reshape_energy | restructure_section | add_transition",
      "params": {{
        "stems_affected": ["drums"],
        "bars": [24, 25, 26, 27],
        "details": "Ultra-specific implementation instructions"
      }},
      "confidence": 0.9,
      "impact": "high"
    }}
  ],
  "overall_assessment": "3-4 sentences: What makes this track work, what's holding it back, and what it could become with the right changes.",
  "track_strengths": ["Be specific — 'Great kick-bass relationship in the drop' not just 'good bass'"],
  "improvement_priority": "The single most impactful change to make first, and why it unlocks everything else"
}}
```

**CRITICAL CONSTRAINTS:**
- `stems_affected` must ONLY contain: "drums", "bass", "vocals", "other"
- Every proposal must reference specific bar numbers from the analysis
- Descriptions must explain the MUSICAL reason, not just the technical action
- Be bold. A safe, generic review is worthless. Give the artist something they haven't thought of."""

    try:
        ai_response = generate(
            prompt=prompt,
            system_prompt=(
                "You are AURALIS — an AI with the musical mind of a Grammy-winning producer, "
                "the technical precision of a mastering engineer, and the creative vision of an "
                "avant-garde sound designer. You have 20+ years of experience across electronic music, "
                "from deep house to techno to melodic bass to ambient. You've worked with artists at "
                "every level. When you review a track, you don't just hear frequencies — you hear "
                "potential. You hear what's MISSING, what's HIDING, and what WANTS to emerge. "
                "Your feedback is specific, bar-referenced, and always rooted in musical reasoning. "
                "You never give generic advice. Every word you say should make the artist think "
                "'damn, that's exactly what this track needs.' Respond ONLY with valid JSON."
            ),
            json_mode=True,
            max_tokens=4096,
            temperature=0.7,
        )

        if isinstance(ai_response, dict):
            proposals = ai_response.get("proposals", [])
            assessment = ai_response.get("overall_assessment", "")
            strengths = ai_response.get("track_strengths", [])
            priority = ai_response.get("improvement_priority", "")
        else:
            proposals = []
            assessment = str(ai_response)
            strengths = []
            priority = ""

        return {
            "job_id": job_id,
            "diagnosis_summary": {
                "total_issues": diagnosis["issue_count"],
                "high_severity": sum(1 for i in diagnosis["issues"] if i["severity"] == "high"),
                "medium_severity": sum(1 for i in diagnosis["issues"] if i["severity"] == "medium"),
                "low_severity": sum(1 for i in diagnosis["issues"] if i["severity"] == "low"),
            },
            "proposals": proposals,
            "overall_assessment": assessment,
            "track_strengths": strengths,
            "improvement_priority": priority,
            "model_used": "gemini-3-pro-preview (with fallback)",
        }

    except Exception as e:
        return {
            "job_id": job_id,
            "error": f"AI proposal generation failed: {str(e)}",
            "diagnosis": diagnosis,
            "proposals": [],
        }


# ── Directed Reconstruct: Improve with AI Feedback ──────


@router.post("/improve/{job_id}")
async def improve_track(job_id: str, req: ImproveRequest):
    """Improve a completed track based on user feedback.

    The user provides text feedback (e.g. 'make the intro more atmospheric').
    Gemini analyzes the track + feedback + DNA references → generates an
    improvement plan → re-renders affected sections → delivers improved master.

    All musical decisions are made by the AI — zero hardcode.
    """
    job = _reconstruct_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "completed":
        raise HTTPException(400, "Job must be completed before improving")

    result = job.get("result", {})
    analysis = result.get("analysis", {})
    stem_analysis = result.get("stem_analysis", {})

    if not analysis:
        raise HTTPException(400, "No analysis data — run the pipeline first")

    # ── Create a new child job for the improvement ──
    import uuid
    improve_id = str(uuid.uuid4())
    project_dir = settings.projects_dir / improve_id
    project_dir.mkdir(parents=True, exist_ok=True)

    _reconstruct_jobs[improve_id] = {
        "job_id": improve_id,
        "project_id": improve_id,
        "mode": "improve",
        "parent_job_id": job_id,
        "status": "running",
        "stage": "brain",
        "progress": 0,
        "stages": {
            "brain": {"status": "running", "message": "Analyzing your feedback with Gemini..."},
            "grid": {"status": "pending", "message": ""},
            "hands": {"status": "pending", "message": ""},
            "console": {"status": "pending", "message": ""},
        },
        "logs": [],
        "result": None,
        "feedback": req.feedback,
        "original_name": job.get("original_name", "Track"),
    }
    improve_job = _reconstruct_jobs[improve_id]
    _log(improve_job, f"🎯 Directed Improve: {req.feedback}", "info")
    _log(improve_job, f"Based on: {job_id}", "info")
    _save_jobs()

    # Run improvement in background
    asyncio.create_task(_run_improvement(
        improve_id, job_id, req.feedback, project_dir
    ))

    return _reconstruct_jobs[improve_id]


async def _run_improvement(
    improve_id: str,
    parent_id: str,
    feedback: str,
    project_dir: Path,
) -> None:
    """Run the AI-directed improvement pipeline."""
    job = _reconstruct_jobs[improve_id]
    parent = _reconstruct_jobs.get(parent_id, {})
    parent_result = parent.get("result", {})
    analysis = parent_result.get("analysis", {})
    stem_analysis = parent_result.get("stem_analysis", {})
    brain_report = parent_result.get("brain_report", {})

    try:
        import gc

        # ── Collect DNA reference profile ──
        _log(job, "🧬 Loading DNA references for style context...", "info")
        dna_profile = {}
        reference_paths: list[Path] = []
        try:
            from auralis.ear.reference_bank import ReferenceBank
            bank = ReferenceBank(settings.projects_dir)
            all_refs = bank.list_references()
            for ref in all_refs:
                ref_dir = settings.projects_dir / ref.get("track_id", "")
                if ref_dir.exists():
                    audio = _find_audio_file(ref_dir)
                    if audio:
                        reference_paths.append(audio)
            if bank.count() > 0:
                try:
                    dna_profile = bank.get_full_averages()
                    _log(job, f"🎵 {bank.count()} DNA refs loaded for style guidance", "success")
                except Exception:
                    dna_profile = {}
        except Exception as e:
            _log(job, f"⚠️ DNA refs unavailable: {e}", "warning")

        # ── Step 1: BRAIN — Ask Gemini to analyze feedback + generate plan ──
        job["stages"]["brain"]["status"] = "running"
        job["stages"]["brain"]["message"] = "Gemini analyzing your feedback..."
        _log(job, "🧠 Brain: Sending track analysis + feedback to Gemini...", "stage")

        # Build full context for Gemini
        track_context = {
            "bpm": analysis.get("bpm", 120),
            "key": analysis.get("key", "C"),
            "scale": analysis.get("scale", "minor"),
            "duration_s": analysis.get("duration", 180),
            "genre": analysis.get("genre", "unknown"),
            "energy_profile": analysis.get("energy_profile", {}),
            "spectral_summary": {
                name: {
                    "rms_db": data.get("rms_db"),
                    "peak_db": data.get("peak_db"),
                    "dynamic_range": data.get("dynamic_range"),
                    "spectral_centroid": data.get("spectral_centroid_mean"),
                }
                for name, data in stem_analysis.items()
                if isinstance(data, dict) and "error" not in data
            },
        }

        # Add brain reasoning context if available
        if brain_report:
            track_context["brain_reasoning"] = {
                "reasoning_chain": brain_report.get("reasoning_chain", [])[:5],
                "key_observations": brain_report.get("interaction_log", [])[:5],
            }

        # Add DNA reference profile for style cloning
        if dna_profile:
            track_context["dna_reference_style"] = {
                "average_bpm": dna_profile.get("master", {}).get("bpm_avg"),
                "average_loudness": dna_profile.get("master", {}).get("rms_db_avg"),
                "style_note": "These are the target reference values the user wants to match",
            }

        from auralis.brain.gemini_client import generate as gemini_generate

        improvement_prompt = f"""You are AURALIS AI Producer. A user has completed a track and wants specific improvements.

## Original Track Analysis
{json.dumps(track_context, indent=2, default=str)}

## User's Feedback
"{feedback}"

## Your Task
Analyze the track data and the user's feedback. Generate a COMPLETE improvement plan.
You must decide:
1. Which sections of the track need changes (by bar range)
2. What specific modifications to make (effects, volume, EQ, arrangement)
3. How to re-mix for the best result

Respond with this exact JSON structure:
{{
  "understanding": "Your interpretation of what the user wants (1-2 sentences)",
  "plan_title": "Short name for this improvement (e.g. 'Atmospheric Intro Enhancement')",
  "changes": [
    {{
      "section": "intro | verse | chorus | drop | breakdown | outro | bars_N_to_M",
      "bar_start": 0,
      "bar_end": 16,
      "stems_affected": ["drums", "bass", "vocals", "other"],
      "modifications": [
        {{
          "type": "effect | volume | eq | arrangement | style",
          "description": "What to do (e.g. 'Add shimmer reverb with long tail')",
          "params": {{
            "effect_name": "reverb",
            "wet_mix": 0.4,
            "decay": 3.5
          }}
        }}
      ],
      "reasoning": "Why this change addresses the user's feedback"
    }}
  ],
  "mixing_adjustments": {{
    "overall_description": "How the final mix should change",
    "stem_levels": {{
      "drums": 0.0,
      "bass": 0.0,
      "vocals": 0.0,
      "other": 0.0
    }},
    "master_processing": "Any changes to the master chain"
  }},
  "expected_result": "What the user should hear differently after these changes"
}}

CRITICAL: The ONLY valid stem names are: "drums", "bass", "vocals", "other".
- "other" contains ALL melodic/harmonic elements: synths, pads, keys, guitars, leads, FX, etc.
- "vocals" contains all vocal content.
Do NOT use names like "synths", "chords", "melody", "keys" — map them to "other".

Be creative, musical, and precise. Think like a professional producer who understands the user's vision."""

        ai_plan = gemini_generate(
            prompt=improvement_prompt,
            system_prompt=(
                "You are AURALIS AI Producer — an expert music producer that improves tracks "
                "based on user feedback. You understand musical theory, arrangement, mixing, "
                "and sound design. Your improvements are creative, precise, and always serve "
                "the user's artistic vision. Use the DNA reference profile to maintain style consistency."
            ),
            json_mode=True,
            max_tokens=4096,
            temperature=0.6,
        )

        if isinstance(ai_plan, dict) and not ai_plan.get("parse_error"):
            _log(job, f"📋 Plan: {ai_plan.get('plan_title', 'AI Improvement')}", "success")
            _log(job, f"💡 Understanding: {ai_plan.get('understanding', '')}", "info")
            for c in ai_plan.get("changes", []):
                _log(job, f"  → {c.get('section', '?')}: {', '.join(m.get('description', '') for m in c.get('modifications', []))}", "info")
            _log(job, f"🎯 Expected: {ai_plan.get('expected_result', '')}", "info")
        else:
            ai_plan = {"changes": [], "mixing_adjustments": {}, "plan_title": "Improvement", "understanding": feedback}
            _log(job, "⚠️ Gemini response wasn't structured — will apply general improvements", "warning")

        # Always extract from ai_plan — guaranteed to exist in both branches
        changes = ai_plan.get("changes", [])
        mixing = ai_plan.get("mixing_adjustments", {})
        expected = ai_plan.get("expected_result", feedback)
        _log(job, f"🔧 {len(changes)} changes planned", "info")

        job["stages"]["brain"]["status"] = "completed"
        job["stages"]["brain"]["message"] = ai_plan.get("plan_title", "Plan ready")
        job["progress"] = 20
        _save_jobs()

        # ── Step 2: GRID — Map improvements to arrangement ──
        job["stages"]["grid"]["status"] = "running"
        job["stages"]["grid"]["message"] = "Mapping improvements to arrangement..."
        job["stage"] = "grid"
        _log(job, "📐 Grid: Mapping AI plan to track sections...", "stage")

        # Copy parent's original audio to child project for re-processing
        parent_project_id = parent.get("project_id", parent_id)
        parent_dir = settings.projects_dir / parent_project_id
        import shutil

        # Validate parent project exists
        if not parent_dir.exists():
            raise RuntimeError(
                f"Parent project directory not found ({parent_id[:8]}…). "
                "The original track may have been deleted. Please re-upload and reconstruct first."
            )

        # Copy original audio
        original_src = parent_dir / "original.wav"
        if not original_src.exists():
            original_src = _find_audio_file(parent_dir)
        if original_src and original_src.exists():
            shutil.copy2(original_src, project_dir / "original.wav")
            _log(job, f"📁 Copied original audio from parent project", "info")

        # Copy stems directory for re-mixing
        parent_stems = parent_dir / "stems"
        child_stems = project_dir / "stems"
        if parent_stems.exists():
            shutil.copytree(parent_stems, child_stems, dirs_exist_ok=True)
            stem_count = len(list(child_stems.glob("*.wav")))
            _log(job, f"📁 Copied {stem_count} stems for re-mixing", "info")
        else:
            raise RuntimeError(
                f"Parent project has no stems ({parent_id[:8]}…). "
                "The stems may have been cleaned up. Please re-upload and reconstruct first."
            )

        job["stages"]["grid"]["status"] = "completed"
        job["stages"]["grid"]["message"] = f"{len(changes)} sections to improve"
        job["progress"] = 35
        _save_jobs()

        # ── Step 3: HANDS — Apply improvements ──
        job["stages"]["hands"]["status"] = "running"
        job["stages"]["hands"]["message"] = "Applying AI improvements..."
        job["stage"] = "hands"
        _log(job, "🎹 Hands: Applying Gemini's improvement plan to stems...", "stage")

        import librosa
        import soundfile as sf

        bpm = float(analysis.get("bpm", 120))
        sr = 44100
        bar_duration_s = (60.0 / bpm) * 4  # seconds per bar
        bar_samples = int(bar_duration_s * sr)

        stems_modified = 0
        stem_files = list(child_stems.glob("*.wav")) if child_stems.exists() else []

        # Apply changes from the AI plan
        for change_idx, change in enumerate(changes):
            bar_start = change.get("bar_start", 0)
            bar_end = change.get("bar_end", 8)
            stems_affected = change.get("stems_affected", [])
            modifications = change.get("modifications", [])
            section_name = change.get("section", "?")

            _log(job, f"🔧 Processing: {section_name} (bars {bar_start}-{bar_end})", "info")

            for stem_name_raw in stems_affected:
                # Normalize Gemini's stem names to actual Demucs file names
                _STEM_MAP = {
                    "drums": "drums", "drum": "drums", "percussion": "drums", "kick": "drums", "hats": "drums",
                    "bass": "bass", "sub": "bass", "sub_bass": "bass", "808": "bass",
                    "vocals": "vocals", "vocal": "vocals", "voice": "vocals", "vox": "vocals",
                    "other": "other", "synths": "other", "synth": "other", "chords": "other",
                    "melody": "other", "keys": "other", "piano": "other", "pad": "other",
                    "pads": "other", "guitar": "other", "lead": "other", "fx": "other",
                    "strings": "other", "pluck": "other", "organ": "other",
                }
                stem_name = _STEM_MAP.get(stem_name_raw.lower(), stem_name_raw.lower())

                # Find the stem file
                stem_file = child_stems / f"{stem_name}.wav" if child_stems.exists() else None
                if not stem_file or not stem_file.exists():
                    _log(job, f"  ⚠️ Stem '{stem_name_raw}' → '{stem_name}.wav' not found, skipping", "warning")
                    continue

                try:
                    y, file_sr = librosa.load(str(stem_file), sr=sr, mono=False)
                    if y.ndim == 1:
                        y = np.expand_dims(y, 0)

                    start_sample = bar_start * bar_samples
                    end_sample = min(bar_end * bar_samples, y.shape[-1])

                    if start_sample >= y.shape[-1]:
                        continue

                    # Apply each modification from the AI plan
                    for mod in modifications:
                        mod_type = mod.get("type", "")
                        params = mod.get("params", {})

                        if mod_type == "effect":
                            effect_name = params.get("effect_name", "")
                            wet_mix = float(params.get("wet_mix", 0.3))

                            if "reverb" in effect_name.lower():
                                # Apply reverb to the section
                                decay = float(params.get("decay", 2.0))
                                ir_len = int(decay * sr)
                                impulse = np.exp(-3.0 * np.linspace(0, 1, ir_len))
                                impulse = impulse / (np.sum(impulse) + 1e-10)
                                for ch in range(y.shape[0]):
                                    segment = y[ch, start_sample:end_sample]
                                    reverbed = np.convolve(segment, impulse, mode="full")[:len(segment)]
                                    y[ch, start_sample:end_sample] = (
                                        segment * (1 - wet_mix) + reverbed * wet_mix
                                    )
                                _log(job, f"  ✨ Reverb → {stem_name} (decay={decay}s, wet={wet_mix})", "info")

                            elif "delay" in effect_name.lower():
                                delay_time = float(params.get("delay_time", 0.25))
                                delay_feedback = float(params.get("feedback", 0.3))
                                delay_samples = int(delay_time * sr)
                                for ch in range(y.shape[0]):
                                    segment = y[ch, start_sample:end_sample].copy()
                                    delayed = np.zeros_like(segment)
                                    if delay_samples < len(segment):
                                        delayed[delay_samples:] = segment[:-delay_samples] * delay_feedback
                                    y[ch, start_sample:end_sample] = segment + delayed * wet_mix
                                _log(job, f"  🔁 Delay → {stem_name} (time={delay_time}s)", "info")

                            elif "filter" in effect_name.lower() or "eq" in effect_name.lower():
                                # Simple low/high pass
                                from scipy import signal as scipy_signal
                                cutoff = float(params.get("cutoff", 2000))
                                filter_type = params.get("filter_type", "lowpass")
                                try:
                                    nyq = sr / 2
                                    norm_cutoff = min(cutoff / nyq, 0.99)
                                    b, a = scipy_signal.butter(4, norm_cutoff, btype=filter_type)
                                    for ch in range(y.shape[0]):
                                        segment = y[ch, start_sample:end_sample]
                                        filtered = scipy_signal.filtfilt(b, a, segment)
                                        y[ch, start_sample:end_sample] = (
                                            segment * (1 - wet_mix) + filtered * wet_mix
                                        )
                                    _log(job, f"  🎛️ {filter_type} filter → {stem_name} (cutoff={cutoff}Hz)", "info")
                                except Exception:
                                    pass

                            elif "distortion" in effect_name.lower() or "saturation" in effect_name.lower():
                                drive = float(params.get("drive", 2.0))
                                for ch in range(y.shape[0]):
                                    segment = y[ch, start_sample:end_sample]
                                    y[ch, start_sample:end_sample] = np.tanh(segment * drive) / drive
                                _log(job, f"  🔥 Saturation → {stem_name} (drive={drive}x)", "info")

                        elif mod_type == "volume":
                            db_change = float(params.get("db", 0))
                            if db_change != 0:
                                gain = 10 ** (db_change / 20.0)
                                y[:, start_sample:end_sample] *= gain
                                _log(job, f"  📊 Volume {'+' if db_change > 0 else ''}{db_change}dB → {stem_name}", "info")

                        elif mod_type == "arrangement":
                            action = params.get("action", "")
                            if action == "mute":
                                y[:, start_sample:end_sample] *= 0.0
                                _log(job, f"  🔇 Muted {stem_name} bars {bar_start}-{bar_end}", "info")
                            elif action == "fade_in":
                                length = end_sample - start_sample
                                fade = np.linspace(0, 1, length)
                                y[:, start_sample:end_sample] *= fade
                                _log(job, f"  📈 Fade-in → {stem_name}", "info")
                            elif action == "fade_out":
                                length = end_sample - start_sample
                                fade = np.linspace(1, 0, length)
                                y[:, start_sample:end_sample] *= fade
                                _log(job, f"  📉 Fade-out → {stem_name}", "info")

                    # Save modified stem
                    output = y.squeeze()
                    sf.write(str(stem_file), output.T if output.ndim > 1 else output, sr)
                    stems_modified += 1

                except Exception as e:
                    _log(job, f"  ⚠️ Error processing {stem_name}: {e}", "warning")

            job["progress"] = 35 + int((change_idx + 1) / max(len(changes), 1) * 40)

        _log(job, f"✅ Modified {stems_modified} stems", "success")
        job["stages"]["hands"]["status"] = "completed"
        job["stages"]["hands"]["message"] = f"{stems_modified} stems improved"
        job["progress"] = 80
        _save_jobs()
        gc.collect()

        # ── Step 4: CONSOLE — Re-mix and master ──
        job["stages"]["console"]["status"] = "running"
        job["stages"]["console"]["message"] = "Re-mixing and mastering..."
        job["stage"] = "console"
        _log(job, "🎚️ Console: Re-mixing stems with AI adjustments...", "stage")

        # Apply mixing adjustments from the AI plan
        mixing_adj = ai_plan.get("mixing_adjustments", {})
        stem_levels = mixing_adj.get("stem_levels", {})

        # Mix all stems together in STEREO
        all_stems = list(child_stems.glob("*.wav")) if child_stems.exists() else []
        if all_stems:
            mixed = None
            max_len = 0

            for stem_file in all_stems:
                try:
                    y_stem, _ = librosa.load(str(stem_file), sr=sr, mono=False)
                    stem_name = stem_file.stem

                    # Ensure stereo (2D array with shape [channels, samples])
                    if y_stem.ndim == 1:
                        y_stem = np.stack([y_stem, y_stem])  # mono → stereo

                    # Apply AI-decided stem level adjustment
                    level_db = float(stem_levels.get(stem_name, 0.0))
                    if level_db != 0:
                        y_stem *= 10 ** (level_db / 20.0)
                        _log(job, f"  🎚️ {stem_name}: {'+' if level_db > 0 else ''}{level_db:.1f}dB", "info")

                    if mixed is None:
                        mixed = y_stem.copy()
                        max_len = y_stem.shape[1]
                    else:
                        cur_len = y_stem.shape[1]
                        if cur_len > max_len:
                            mixed = np.pad(mixed, ((0, 0), (0, cur_len - max_len)))
                            max_len = cur_len
                        elif cur_len < max_len:
                            y_stem = np.pad(y_stem, ((0, 0), (0, max_len - cur_len)))
                        mixed += y_stem
                except Exception as e:
                    _log(job, f"  ⚠️ Error mixing {stem_file.name}: {e}", "warning")

            if mixed is not None:
                # Normalize to prevent clipping
                peak = np.max(np.abs(mixed))
                if peak > 0.95:
                    mixed = mixed * 0.95 / peak
                    _log(job, "  📊 Normalized peak to -0.5dB", "info")

                # Save mix (stereo, interleaved)
                mix_path = project_dir / "mix.wav"
                sf.write(str(mix_path), mixed.T, sr)
                _log(job, f"💿 Mix saved: {mix_path.name}", "success")

                # Full mastering chain — same as original pipeline
                master_path = project_dir / "master.wav"
                bpm_val = float(analysis.get("bpm", 120))
                target_lufs = -8.0

                try:
                    import multiprocessing
                    brain_plan_dict = None
                    brain = result.get("brain_report")
                    if brain:
                        brain_plan_dict = {
                            k: brain[k] for k in brain
                            if isinstance(brain[k], (str, int, float, bool, list, dict, type(None)))
                        }

                    p = multiprocessing.Process(
                        target=_subprocess_master,
                        args=(str(mix_path), str(master_path), target_lufs, bpm_val, brain_plan_dict),
                    )
                    p.start()
                    p.join(timeout=300)

                    if p.is_alive():
                        p.terminate()
                        raise RuntimeError("Mastering subprocess timed out")

                    if not master_path.exists():
                        raise RuntimeError("Mastering subprocess did not produce output")

                    _log(job, f"💎 Master saved: {master_path.name} (full chain, {target_lufs} LUFS)", "success")
                except Exception as e:
                    _log(job, f"⚠️ Full mastering failed ({e}), using basic master", "warning")
                    # Fallback: basic mastering
                    threshold = 0.5
                    ratio = 3.0
                    master_audio = mixed.copy()
                    above = np.abs(master_audio) > threshold
                    master_audio[above] = np.sign(master_audio[above]) * (
                        threshold + (np.abs(master_audio[above]) - threshold) / ratio
                    )
                    peak = np.max(np.abs(master_audio))
                    if peak > 0:
                        master_audio = master_audio * 0.92 / peak
                    sf.write(str(master_path), master_audio.T, sr)
                    _log(job, f"💎 Master saved: {master_path.name} (basic fallback)", "success")

        job["stages"]["console"]["status"] = "completed"
        job["stages"]["console"]["message"] = "Improved master ready"
        job["progress"] = 100
        job["status"] = "completed"
        job["stage"] = "complete"

        # Build result
        files_info = {}
        for name in ["original", "mix", "master"]:
            path = project_dir / f"{name}.wav"
            if path.exists():
                files_info[name] = f"/api/reconstruct/audio/{improve_id}/{name}"

        # Stem files
        stem_list = []
        if child_stems.exists():
            for sf_path in sorted(child_stems.glob("*.wav")):
                stem_list.append({
                    "name": sf_path.stem,
                    "url": f"/api/reconstruct/audio/{improve_id}/stems/{sf_path.stem}",
                })

        # Inherit ALL parent result data so UI shows full insights
        parent_result = result.copy() if result else {}
        parent_result.pop("files", None)  # will be replaced with child files

        job["result"] = {
            **parent_result,  # brain_report, qc, xray_analysis, plan, analysis, etc.
            "stem_analysis": stem_analysis,  # may have been updated
            "files": {**files_info, "stems": stem_list},
            "improvement": {
                "feedback": feedback,
                "plan": ai_plan,
                "stems_modified": stems_modified,
                "parent_job_id": parent_id,
            },
        }

        _log(job, f"🎯 Directed Improve complete! {stems_modified} stems enhanced", "success")
        _log(job, f"Expected: {ai_plan.get('expected_result', 'Improved track')}", "info")
        _save_jobs()

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        job["status"] = "failed"
        job["stage"] = "error"
        for stage in job["stages"].values():
            if stage["status"] == "running":
                stage["status"] = "failed"
                stage["message"] = str(e)[:100]
        _log(job, f"❌ Improvement failed: {e}", "error")
        _log(job, f"Traceback: {tb[-300:]}", "error")
        _save_jobs()


def _find_audio_file(project_dir: Path) -> Path | None:
    """Find the uploaded audio file in a project directory."""
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    if not project_dir.exists():
        return None
    for f in project_dir.iterdir():
        if f.suffix.lower() in audio_extensions and f.is_file():
            return f
    return None

