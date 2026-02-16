"""API routes for Console (mastering), QC analysis, and visualization."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import Response

from auralis.console.mastering import PRESETS, MasterConfig, master_audio
from auralis.console.qc import compare_tracks, run_qc

router = APIRouter(prefix="/console", tags=["console"])

PROJECTS_DIR = Path("/app/projects")


def _find_audio(project_dir: Path, prefer_master: bool = False) -> Path | None:
    """Find audio file in project directory."""
    if prefer_master:
        masters = list(project_dir.glob("*_MASTER.wav"))
        if masters:
            return masters[0]
    for ext in (".wav", ".flac", ".mp3", ".aif"):
        files = [f for f in project_dir.glob(f"*{ext}") if "_MASTER" not in f.stem]
        if files:
            return files[0]
    return None


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

    audio = _find_audio(project_dir)
    if audio is None:
        return {"error": "No audio file found in project"}

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

    master = list(project_dir.glob("*_MASTER.wav"))
    target = master[0] if master else _find_audio(project_dir)
    if target is None:
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


# ── Visualization Endpoints ─────────────────────────────


@router.get("/viz/spectrum/{project_id}")
async def viz_spectrum(project_id: str) -> Response:
    """Generate 7-band spectrum comparison image (original vs master)."""
    from auralis.console.visualize import generate_spectrum_comparison

    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        return Response(content=b"Project not found", status_code=404)

    original = _find_audio(project_dir)
    master = list(project_dir.glob("*_MASTER.wav"))
    if not original or not master:
        return Response(content=b"Need both original and master", status_code=404)

    png = await asyncio.to_thread(
        generate_spectrum_comparison, str(original), str(master[0]),
        "Original", "Master"
    )
    return Response(content=png, media_type="image/png")


@router.get("/viz/spectrogram/{project_id}")
async def viz_spectrogram(project_id: str) -> Response:
    """Generate mel spectrogram comparison image."""
    from auralis.console.visualize import generate_spectrogram_comparison

    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        return Response(content=b"Project not found", status_code=404)

    original = _find_audio(project_dir)
    master = list(project_dir.glob("*_MASTER.wav"))
    if not original or not master:
        return Response(content=b"Need both original and master", status_code=404)

    png = await asyncio.to_thread(
        generate_spectrogram_comparison, str(original), str(master[0])
    )
    return Response(content=png, media_type="image/png")


@router.get("/viz/waveform/{project_id}")
async def viz_waveform(project_id: str) -> Response:
    """Generate waveform comparison image."""
    from auralis.console.visualize import generate_waveform_comparison

    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        return Response(content=b"Project not found", status_code=404)

    original = _find_audio(project_dir)
    master = list(project_dir.glob("*_MASTER.wav"))
    if not original or not master:
        return Response(content=b"Need both original and master", status_code=404)

    png = await asyncio.to_thread(
        generate_waveform_comparison, str(original), str(master[0])
    )
    return Response(content=png, media_type="image/png")


@router.get("/viz/radar/{project_id}")
async def viz_radar(project_id: str) -> Response:
    """Generate QC radar chart image."""
    from auralis.console.visualize import generate_qc_radar

    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        return Response(content=b"Project not found", status_code=404)

    master = list(project_dir.glob("*_MASTER.wav"))
    target = master[0] if master else _find_audio(project_dir)
    if target is None:
        return Response(content=b"No audio file found", status_code=404)

    report = await asyncio.to_thread(run_qc, str(target))

    png = await asyncio.to_thread(
        generate_qc_radar,
        report.dynamics.peak_db, report.dynamics.rms_db,
        report.dynamics.crest_factor_db,
        report.stereo.correlation if report.stereo else 0.5,
        report.stereo.width if report.stereo else 0.3,
        report.loudness.integrated_lufs if report.loudness else -14,
        report.dynamics.dynamic_range_db,
    )
    return Response(content=png, media_type="image/png")


@router.get("/viz/loudness/{project_id}")
async def viz_loudness(project_id: str) -> Response:
    """Generate loudness timeline image."""
    from auralis.console.visualize import generate_loudness_timeline

    project_dir = PROJECTS_DIR / project_id
    if not project_dir.exists():
        return Response(content=b"Project not found", status_code=404)

    master = list(project_dir.glob("*_MASTER.wav"))
    target = master[0] if master else _find_audio(project_dir)
    if target is None:
        return Response(content=b"No audio file found", status_code=404)

    png = await asyncio.to_thread(
        generate_loudness_timeline, str(target), "Master"
    )
    return Response(content=png, media_type="image/png")
