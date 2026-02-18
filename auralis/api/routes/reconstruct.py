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

from fastapi import APIRouter, Depends, HTTPException, Request
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
    _user=Depends(get_current_user_or_token),
):
    """Stream audio file with HTTP Range support for instant playback."""
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
        
    file_path = project_dir / file_map[file_key]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

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
            media_type="audio/wav",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )

    # No Range header — full file with Accept-Ranges hint
    return FileResponse(
        file_path,
        media_type="audio/wav",
        headers={"Accept-Ranges": "bytes"},
    )


@media_router.get("/spectrogram/{job_id}/{file_key}")
async def get_spectrogram(
    job_id: str,
    file_key: str,
    _user=Depends(get_current_user_or_token),
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
async def cleanup_project(job_id: str):
    """Delete all project files + temp files to free disk space. Keeps job metadata."""
    if job_id not in _reconstruct_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = _reconstruct_jobs[job_id]
    project_id = job["project_id"]
    project_dir = Path("/app/projects") / project_id

    freed_bytes = 0

    # 1. Delete project directory (original, stems, master, spectrograms)
    if project_dir.exists():
        freed_bytes += sum(
            f.stat().st_size for f in project_dir.rglob("*") if f.is_file()
        )
        import shutil
        shutil.rmtree(project_dir, ignore_errors=True)

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
    """List all reconstruction jobs."""
    return [
        {
            "job_id": j["job_id"],
            "project_id": j["project_id"],
            "status": j["status"],
            "stage": j["stage"],
            "progress": j["progress"],
            "cleaned": j.get("cleaned", False),
        }
        for j in _reconstruct_jobs.values()
    ]


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

            # Run MIDI extraction on tonal stems
            if stems_dir.exists() and any(stems_dir.glob("*.wav")):
                job["stages"]["ear"]["message"] = "Extracting MIDI from stems..."
                _log(job, "Extracting MIDI from tonal stems (basic-pitch)...")
                try:
                    midi_dir = project_dir / "midi"
                    analysis["midi_stems"] = await asyncio.to_thread(
                        _run_in_subprocess,
                        _subprocess_midi,
                        stems_dir=stems_dir,
                        output_dir=midi_dir,
                    )
                    _log(job, f"✓ MIDI extracted from {len(analysis['midi_stems'])} stems", "success")
                except ImportError:
                    analysis["midi_stems"] = {"error": "basic-pitch not installed"}
                    _log(job, "basic-pitch not installed — skipping MIDI", "warn")
                except Exception as e:
                    analysis["midi_stems"] = {"error": str(e)}
                    _log(job, f"MIDI extraction error: {e}", "error")

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

                # Master by reference (SUBPROCESS)
                if mix_path.exists():
                    job["stages"]["console"]["message"] = "Mastering by reference..."
                    target_lufs = analysis.get("integrated_lufs", -14.0)
                    if not isinstance(target_lufs, (int, float)):
                        target_lufs = -14.0
                    # Don't target quieter than -16 LUFS for reconstruction
                    target_lufs = max(target_lufs, -16.0)

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
        job["result"] = {
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
        }

        _save_jobs()  # persist final completed state

    except Exception as e:
        job["status"] = "error"
        job["stages"][job["stage"]]["status"] = "error"
        job["stages"][job["stage"]]["message"] = str(e)
        _log(job, f"FATAL: {e}", "error")
        _save_jobs()  # persist error state
    finally:
        gc.collect()  # ensure cleanup regardless of outcome


def _find_audio_file(project_dir: Path) -> Path | None:
    """Find the uploaded audio file in a project directory."""
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    if not project_dir.exists():
        return None
    for f in project_dir.iterdir():
        if f.suffix.lower() in audio_extensions and f.is_file():
            return f
    return None

