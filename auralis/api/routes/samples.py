"""AURALIS API — Sample Pack management routes.

Upload sample packs, check organic pack status, and browse categories.
This enables the OrganicPack to work on EC2/Docker by providing a way
to upload sample databases and audio files to the server.
"""

from __future__ import annotations

import json
import shutil
import traceback
import zipfile
from pathlib import Path
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from auralis.config import settings

logger = structlog.get_logger()

router = APIRouter(prefix="/samples", tags=["samples"])


# ── Models ───────────────────────────────────────────────


class PackStatus(BaseModel):
    """Status of the organic sample pack."""

    loaded: bool
    total_samples: int
    total_categories: int
    categories: dict[str, int]
    packs: list[str]
    db_path: str | None


class CategoryInfo(BaseModel):
    """Info about a single organic category."""

    name: str
    count: int
    sample_types: list[str]
    has_loops: bool
    has_oneshots: bool


# ── Endpoints ────────────────────────────────────────────


@router.get("/status")
async def get_pack_status() -> PackStatus:
    """Get the current organic pack status.

    Shows whether samples are loaded, how many, which categories, etc.
    """
    try:
        from auralis.console.organic_pack import OrganicPack

        pack = OrganicPack.load()
        summary = pack.summary()

        return PackStatus(
            loaded=len(pack.samples) > 0,
            total_samples=summary["total_organic"],
            total_categories=len(summary["categories"]),
            categories=summary["categories"],
            packs=summary["packs"],
            db_path=_find_db_path(),
        )
    except Exception as e:
        logger.error("samples.status.failed", error=str(e))
        return PackStatus(
            loaded=False,
            total_samples=0,
            total_categories=0,
            categories={},
            packs=[],
            db_path=None,
        )


@router.get("/categories")
async def list_categories() -> list[CategoryInfo]:
    """List all organic categories with detailed info."""
    try:
        from auralis.console.organic_pack import OrganicPack

        pack = OrganicPack.load()

        categories = []
        for cat_name, samples in pack._index.items():
            types = list(set(s.sample_type for s in samples))
            categories.append(CategoryInfo(
                name=cat_name,
                count=len(samples),
                sample_types=types,
                has_loops="loop" in types,
                has_oneshots="one-shot" in types,
            ))

        # Sort by count descending
        categories.sort(key=lambda c: -c.count)
        return categories

    except Exception as e:
        logger.error("samples.categories.failed", error=str(e))
        return []


@router.post("/upload-pack")
async def upload_sample_pack(
    file: Annotated[UploadFile, File(...)],
) -> dict[str, Any]:
    """Upload a sample pack ZIP file.

    The ZIP should contain:
    - sample_database.json at the root
    - Audio files (.wav) referenced by the database

    The contents are extracted to the server's samples directory,
    and the OrganicPack is reloaded.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(
            status_code=400,
            detail="Only ZIP files are supported. Package your samples as a ZIP.",
        )

    # Check disk space
    try:
        disk = shutil.disk_usage(settings.samples_dir)
        free_gb = disk.free / (1024 ** 3)
        if free_gb < 1.0:
            raise HTTPException(
                status_code=507,
                detail=f"Insufficient disk space: {free_gb:.1f}GB free. Need at least 1GB.",
            )
    except HTTPException:
        raise
    except Exception:
        pass

    try:
        # Ensure samples directory exists
        settings.samples_dir.mkdir(parents=True, exist_ok=True)

        # Save the ZIP
        zip_path = settings.samples_dir / "pack_upload.zip"
        content = await file.read()
        zip_path.write_bytes(content)
        size_mb = round(len(content) / 1024 / 1024, 2)

        logger.info("samples.upload.received", filename=file.filename, size_mb=size_mb)

        # Validate it's a proper ZIP
        if not zipfile.is_zipfile(zip_path):
            zip_path.unlink()
            raise HTTPException(status_code=400, detail="Invalid ZIP file")

        # Extract
        extracted_files = 0
        extracted_wavs = 0
        has_database = False

        with zipfile.ZipFile(zip_path, "r") as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                # Extract audio files and the database
                name_lower = info.filename.lower()
                if name_lower.endswith((".wav", ".flac", ".aiff", ".aif")):
                    zf.extract(info, settings.samples_dir)
                    extracted_wavs += 1
                    extracted_files += 1
                elif name_lower.endswith("sample_database.json"):
                    zf.extract(info, settings.samples_dir)
                    has_database = True
                    extracted_files += 1

                    # If it's nested, copy to root of samples_dir
                    extracted_path = settings.samples_dir / info.filename
                    root_db = settings.samples_dir / "sample_database.json"
                    if extracted_path != root_db:
                        shutil.copy2(extracted_path, root_db)

        # Clean up ZIP
        zip_path.unlink(missing_ok=True)

        if not has_database:
            logger.warning("samples.upload.no_database")

        # Reload OrganicPack to verify
        from auralis.console.organic_pack import OrganicPack
        pack = OrganicPack.load()
        summary = pack.summary()

        logger.info(
            "samples.upload.complete",
            extracted_files=extracted_files,
            extracted_wavs=extracted_wavs,
            organic_loaded=summary["total_organic"],
            categories=len(summary["categories"]),
        )

        return {
            "status": "success",
            "message": f"Uploaded {extracted_files} files ({extracted_wavs} audio). OrganicPack reloaded with {summary['total_organic']} samples.",
            "extracted_files": extracted_files,
            "extracted_wavs": extracted_wavs,
            "has_database": has_database,
            "organic_loaded": summary["total_organic"],
            "categories": summary["categories"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("samples.upload.failed", error=str(e), traceback=traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {e!s}",
        )


# ── Helpers ──────────────────────────────────────────────


def _find_db_path() -> str | None:
    """Find the current sample database path."""
    from auralis.console.organic_pack import _DB_SEARCH_PATHS

    for p in _DB_SEARCH_PATHS:
        if p.exists():
            return str(p)
    return None
