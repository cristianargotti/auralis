"""API routes for the Reference DNA Bank.

Endpoints for managing the professional reference library and gap analysis.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from auralis.config import settings
from auralis.ear.reference_bank import ReferenceBank

router = APIRouter(prefix="/reference", tags=["reference"])


class AddReferenceRequest(BaseModel):
    """Request to add a processed track to the reference bank."""

    job_id: str
    name: str = ""


class AddReferenceResponse(BaseModel):
    """Response after adding a reference."""

    track_id: str
    name: str
    message: str
    deep: bool = False


# ── Endpoints ──


@router.post("/add")
async def add_reference(req: AddReferenceRequest) -> AddReferenceResponse:
    """Add a reconstructed track to the reference bank with deep DNA extraction.

    The track must have been processed through the reconstruct pipeline
    so that EAR analysis and stem analysis data are available.

    Deep DNA: If stems exist on disk, runs analyze_track_layers to extract
    percussion maps, bass type, vocal effects, instruments, FX, and arrangement.
    """
    from pathlib import Path

    # Load the job data from the reconstruct pipeline
    from auralis.api.routes.reconstruct import _reconstruct_jobs

    job = _reconstruct_jobs.get(req.job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {req.job_id} not found")

    result = job.get("result", {})
    if not result:
        raise HTTPException(
            status_code=400,
            detail="Job has no results yet. Wait for pipeline to complete.",
        )

    # Extract analysis data from the job
    analysis = result.get("analysis", {})
    stem_analysis = result.get("stem_analysis", {})

    if not analysis and not stem_analysis:
        raise HTTPException(
            status_code=400,
            detail="No analysis data found. The track must go through the EAR stage.",
        )

    # ── Deep DNA: run analyze_track_layers for full extraction ──
    layers_analysis = None
    project_id = job.get("project_id", "")
    project_dir = Path(settings.projects_dir) / project_id

    if project_dir.exists() and (project_dir / "stems").exists():
        try:
            from auralis.ear.analyzer import analyze_track_layers
            layers_analysis = await asyncio.to_thread(
                analyze_track_layers, project_dir
            )
        except Exception as e:
            # Don't fail the whole request — gracefully degrade to thin fingerprint
            print(f"[Reference] Deep DNA extraction failed (will use basic): {e}")

    track_name = req.name or job.get("original_name", f"Track {req.job_id[:8]}")

    bank = ReferenceBank(settings.projects_dir)
    entry = bank.add_reference(
        track_id=req.job_id,
        name=track_name,
        ear_analysis=analysis,
        stem_analysis=stem_analysis,
        layers_analysis=layers_analysis,
    )

    is_deep = entry.deep_version >= 2 and bool(entry.percussion_palette)
    deep_msg = " with DEEP DNA 🧬" if is_deep else ""

    return AddReferenceResponse(
        track_id=entry.track_id,
        name=entry.name,
        message=f"Added '{entry.name}' to reference bank{deep_msg} ({bank.count()} total references)",
        deep=is_deep,
    )


@router.get("/list")
async def list_references() -> dict[str, Any]:
    """List all tracks in the reference bank."""
    bank = ReferenceBank(settings.projects_dir)
    refs = bank.list_references()
    return {
        "count": len(refs),
        "references": refs,
    }


@router.get("/averages")
async def get_averages() -> dict[str, Any]:
    """Get averaged DNA fingerprints across all references.

    Returns per-stem averages and master-level statistics.
    """
    bank = ReferenceBank(settings.projects_dir)
    if bank.count() == 0:
        return {
            "count": 0,
            "message": "No references yet. Add professional tracks first!",
        }
    return bank.get_full_averages()


@router.get("/deep-profile")
async def get_deep_profile() -> dict[str, Any]:
    """Get aggregated deep DNA profile across ALL references.

    Returns merged percussion palettes, instrument lists, FX lists,
    bass types, vocal effects, and arrangement patterns — the complete
    style fingerprint that Auralis learns from your references.
    """
    bank = ReferenceBank(settings.projects_dir)
    if bank.count() == 0:
        return {
            "count": 0,
            "message": "No references yet. Upload professional tracks first!",
        }
    return bank.get_deep_profile()


@router.get("/gap/{job_id}")
async def get_gap_analysis(job_id: str) -> dict[str, Any]:
    """Run gap analysis: compare a track against reference bank averages.

    Returns a full GapReport with per-stem quality scores, frequency gaps,
    and actionable improvement suggestions.
    """
    from auralis.api.routes.reconstruct import _reconstruct_jobs
    from auralis.console.gap_analyzer import analyze_gaps

    job = _reconstruct_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    result = job.get("result", {})
    analysis = result.get("analysis", {})
    stem_analysis = result.get("stem_analysis", {})

    if not analysis and not stem_analysis:
        raise HTTPException(
            status_code=400,
            detail="No analysis data. Run the pipeline first.",
        )

    bank = ReferenceBank(settings.projects_dir)
    report = analyze_gaps(
        ear_analysis=analysis,
        stem_analysis=stem_analysis,
        bank=bank,
    )

    return report.to_dict()


@router.delete("/{track_id}")
async def remove_reference(track_id: str) -> dict[str, str]:
    """Remove a track from the reference bank."""
    bank = ReferenceBank(settings.projects_dir)
    if bank.remove_reference(track_id):
        return {"message": f"Removed {track_id} from reference bank"}
    raise HTTPException(status_code=404, detail=f"Track {track_id} not in reference bank")
