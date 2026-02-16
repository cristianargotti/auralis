"""API routes for the EAR (deconstruction) layer."""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from auralis.config import settings

router = APIRouter(prefix="/ear", tags=["ear"])

# In-memory job tracking (will be replaced with proper queue)
_jobs: dict[str, dict[str, Any]] = {}


class AnalysisRequest(BaseModel):
    """Request to analyze an uploaded track."""

    model: str = "htdemucs_ft"


class JobStatus(BaseModel):
    """Status of a background job."""

    job_id: str
    status: str
    progress: int
    total_steps: int
    message: str
    result: dict[str, Any] | None = None


@router.post("/upload")
async def upload_track(
    file: Annotated[UploadFile, File(...)],
) -> dict[str, str]:
    """Upload a track for analysis.

    Accepts WAV, MP3, FLAC, AIFF files.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    valid_extensions = {".wav", ".mp3", ".flac", ".aiff", ".aif"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {suffix}. Use: {', '.join(valid_extensions)}",
        )

    # Create project directory
    project_id = str(uuid.uuid4())[:8]
    project_dir = settings.projects_dir / project_id
    project_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    upload_path = project_dir / f"original{suffix}"
    content = await file.read()
    upload_path.write_bytes(content)

    return {
        "project_id": project_id,
        "file": str(upload_path),
        "size_mb": str(round(len(content) / 1024 / 1024, 2)),
        "message": f"Track uploaded. Use /ear/analyze/{project_id} to start analysis.",
    }


@router.post("/analyze/{project_id}")
async def analyze_track(project_id: str) -> JobStatus:
    """Start full analysis pipeline for an uploaded track.

    Runs: spectral analysis → stem separation → MIDI extraction → DNA profiling
    """
    project_dir = settings.projects_dir / project_id

    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")

    # Find the original file
    audio_files = list(project_dir.glob("original.*"))
    if not audio_files:
        raise HTTPException(status_code=404, detail="No audio file found in project")

    audio_path = audio_files[0]
    job_id = str(uuid.uuid4())[:8]

    # Start analysis in background
    _jobs[job_id] = {
        "status": "running",
        "progress": 0,
        "total_steps": 4,
        "message": "Starting analysis...",
        "result": None,
    }

    _task = asyncio.create_task(  # noqa: RUF006
        _run_analysis(job_id, audio_path, project_dir)
    )

    return JobStatus(
        job_id=job_id,
        status="running",
        progress=0,
        total_steps=4,
        message="Analysis started. Poll /ear/status/{job_id} for progress.",
    )


@router.get("/status/{job_id}")
async def get_job_status(job_id: str) -> JobStatus:
    """Check the status of an analysis job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job = _jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        total_steps=job["total_steps"],
        message=job["message"],
        result=job.get("result"),
    )


@router.get("/models")
async def list_models() -> list[dict[str, str]]:
    """List available separation models."""
    from auralis.ear.separator import get_available_models

    return get_available_models()


async def _run_analysis(
    job_id: str,
    audio_path: Path,
    project_dir: Path,
) -> None:
    """Run the full EAR analysis pipeline in the background."""
    try:
        # Step 1: Spectral analysis
        _jobs[job_id]["progress"] = 1
        _jobs[job_id]["message"] = "Running spectral analysis..."

        from auralis.ear.spectral import analyze_audio

        await asyncio.to_thread(analyze_audio, audio_path)

        # Step 2: Track profiling (DNA map)
        _jobs[job_id]["progress"] = 2
        _jobs[job_id]["message"] = "Building track DNA map..."

        from auralis.ear.profiler import profile_track

        dna = await asyncio.to_thread(profile_track, audio_path)
        dna.save_json(project_dir / "analysis" / "track_dna.json")

        # Step 3: Stem separation (if available)
        _jobs[job_id]["progress"] = 3
        _jobs[job_id]["message"] = "Separating stems..."

        stems_dir = project_dir / "stems"
        try:
            from auralis.ear.separator import separate_track

            separation = await asyncio.to_thread(
                separate_track,
                audio_path,
                stems_dir,
            )
            stems_result = separation.to_dict()
        except ImportError:
            stems_result = {"error": "Demucs not installed. Install: uv pip install auralis[ml]"}

        # Step 4: MIDI extraction (if available)
        _jobs[job_id]["progress"] = 4
        _jobs[job_id]["message"] = "Extracting MIDI..."

        midi_dir = project_dir / "midi"
        try:
            from auralis.ear.midi_extractor import extract_midi_from_stems

            if stems_dir.exists() and list(stems_dir.glob("*.wav")):
                midi_results = await asyncio.to_thread(extract_midi_from_stems, stems_dir, midi_dir)
                midi_result: dict[str, Any] = {k: v.to_dict() for k, v in midi_results.items()}
            else:
                midi_result = {"info": "No stems available for MIDI extraction (GPU required for Demucs)"}  # type: ignore[no-redef]
        except ImportError:
            midi_result = {"info": "basic-pitch not installed (GPU required)"}  # type: ignore[no-redef]
        except Exception as midi_err:
            midi_result = {"error": f"MIDI extraction failed: {midi_err!s}"}  # type: ignore[no-redef]

        # Complete
        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["message"] = "Analysis complete!"
        _jobs[job_id]["result"] = {
            "track_dna": dna.to_dict(),
            "stems": stems_result,
            "midi": midi_result,
        }

    except Exception as e:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["message"] = f"Error: {e!s}"
