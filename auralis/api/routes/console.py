"""API routes for Console (mastering) and QC analysis."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter

from auralis.console.mastering import PRESETS, MasterConfig, master_audio
from auralis.console.qc import compare_tracks, run_qc

router = APIRouter(prefix="/console", tags=["console"])

PROJECTS_DIR = Path("/app/projects")


@router.post("/master/{project_id}")
async def master_track(
    project_id: str,
    preset: str = "streaming",
    target_lufs: float | None = None,
    drive: float | None = None,
    width: float | None = None,
) -> dict[str, object]:
    """Master a track using the full studio chain."""
    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        return {"error": f"Project {project_id} not found"}

    # Find audio file
    audio = None
    for ext in (".wav", ".flac", ".mp3", ".aif"):
        files = list(project_dir.glob(f"*{ext}"))
        if files:
            audio = files[0]
            break

    if audio is None:
        return {"error": "No audio file found in project"}

    # Build config from preset + overrides
    config = PRESETS.get(preset, MasterConfig())
    if target_lufs is not None:
        config.target_lufs = target_lufs
    if drive is not None:
        config.drive = drive
    if width is not None:
        config.width = width

    output_path = project_dir / f"{audio.stem}_MASTER.wav"

    result = await asyncio.to_thread(
        master_audio, str(audio), str(output_path), config
    )

    return {
        "status": "complete",
        "output": result.output_path,
        "peak_dbtp": result.peak_dbtp,
        "rms_db": result.rms_db,
        "est_lufs": result.est_lufs,
        "clipping_samples": result.clipping_samples,
        "stages": result.stages_applied,
    }


@router.get("/qc/{project_id}")
async def qc_track(project_id: str) -> dict[str, object]:
    """Run QC analysis on latest master."""
    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        return {"error": f"Project {project_id} not found"}

    # Prefer master, fall back to original
    master = list(project_dir.glob("*_MASTER.wav"))
    if master:
        target = master[0]
    else:
        for ext in (".wav", ".flac"):
            files = list(project_dir.glob(f"*{ext}"))
            if files:
                target = files[0]
                break
        else:
            return {"error": "No audio file found"}

    report = await asyncio.to_thread(run_qc, str(target))

    return {
        "status": "complete",
        "pass_fail": report.pass_fail,
        "issues": report.issues,
        "dynamics": {
            "peak_db": report.dynamics.peak_db,
            "rms_db": report.dynamics.rms_db,
            "crest_factor_db": report.dynamics.crest_factor_db,
            "dynamic_range_db": report.dynamics.dynamic_range_db,
        },
        "clipping": {
            "is_clipping": report.clipping.is_clipping,
            "clipped_samples": report.clipping.clipped_samples,
        },
        "stereo": (
            {
                "correlation": report.stereo.correlation,
                "width": report.stereo.width,
                "mono_compatible": report.stereo.mono_compatible,
            }
            if report.stereo
            else None
        ),
        "loudness": (
            {
                "integrated_lufs": report.loudness.integrated_lufs,
                "true_peak_dbtp": report.loudness.true_peak_dbtp,
            }
            if report.loudness
            else None
        ),
        "spectrum": {
            "sub": report.spectrum.sub,
            "bass": report.spectrum.bass,
            "low_mid": report.spectrum.low_mid,
            "mid": report.spectrum.mid,
            "upper_mid": report.spectrum.upper_mid,
            "presence": report.spectrum.presence,
            "brilliance": report.spectrum.brilliance,
        },
    }


@router.get("/presets")
async def list_presets() -> dict[str, object]:
    """List available mastering presets."""
    return {
        name: {
            "target_lufs": c.target_lufs,
            "drive": c.drive,
            "width": c.width,
            "ceiling_db": c.ceiling_db,
        }
        for name, c in PRESETS.items()
    }
