"""API routes for GPU instance management."""

from __future__ import annotations

from fastapi import APIRouter

from auralis.api.gpu_manager import gpu_status, start_gpu, stop_gpu

router = APIRouter(prefix="/gpu", tags=["gpu"])


@router.get("/status")
async def get_gpu_status() -> dict[str, object]:
    """Get current GPU instance status."""
    return await gpu_status()


@router.post("/start")
async def post_start_gpu() -> dict[str, object]:
    """Start the GPU instance for heavy processing."""
    return await start_gpu()


@router.post("/stop")
async def post_stop_gpu() -> dict[str, str]:
    """Stop the GPU instance to save costs."""
    return await stop_gpu()
