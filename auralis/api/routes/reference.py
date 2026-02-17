"""API routes for the Reference DNA Bank.

Endpoints for managing the professional reference library and gap analysis.
"""

from __future__ import annotations

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


# ── Endpoints ──


@router.post("/add")
async def add_reference(req: AddReferenceRequest) -> AddReferenceResponse:
    """Add a reconstructed track to the reference bank.

    The track must have been processed through the reconstruct pipeline
    so that EAR analysis and stem analysis data are available.
    """
    # Load the job data from the reconstruct pipeline
    from auralis.api.routes.reconstruct import _jobs

    job = _jobs.get(req.job_id)
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

    track_name = req.name or job.get("original_name", f"Track {req.job_id[:8]}")

    bank = ReferenceBank(settings.projects_dir)
    entry = bank.add_reference(
        track_id=req.job_id,
        name=track_name,
        ear_analysis=analysis,
        stem_analysis=stem_analysis,
    )

    return AddReferenceResponse(
        track_id=entry.track_id,
        name=entry.name,
        message=f"Added '{entry.name}' to reference bank ({bank.count()} total references)",
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


@router.get("/gap/{job_id}")
async def get_gap_analysis(job_id: str) -> dict[str, Any]:
    """Run gap analysis: compare a track against reference bank averages.

    Returns a full GapReport with per-stem quality scores, frequency gaps,
    and actionable improvement suggestions.
    """
    from auralis.api.routes.reconstruct import _jobs
    from auralis.console.gap_analyzer import analyze_gaps

    job = _jobs.get(job_id)
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
