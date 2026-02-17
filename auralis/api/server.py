"""AURALIS FastAPI server — main application."""

from fastapi import Depends, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from auralis.api.auth import get_current_user, login
from auralis.api.routes.brain import router as brain_router
from auralis.api.routes.console import router as console_router
from auralis.api.routes.ear import router as ear_router
from auralis.api.routes.gpu import router as gpu_router
from auralis.api.routes.grid import router as grid_router
from auralis.api.routes.hands import router as hands_router
from auralis.api.routes.reconstruct import router as reconstruct_router
from auralis.api.routes.reconstruct import media_router as reconstruct_media_router
from auralis.api.routes.reference import router as reference_router
from auralis.api.websocket import websocket_endpoint

app = FastAPI(
    title="AURALIS",
    description="AI Music Production Engine — Hear deeper. Create beyond.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth endpoint (public) ──
app.post("/api/auth/login", tags=["auth"])(login)

# ── Protected routes — require valid JWT ──
app.include_router(
    ear_router,
    prefix="/api",
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    gpu_router,
    prefix="/api",
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    console_router,
    prefix="/api",
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    hands_router,
    prefix="/api",
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    grid_router,
    prefix="/api",
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    brain_router,
    prefix="/api",
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    reconstruct_router,
    prefix="/api",
    dependencies=[Depends(get_current_user)],
)
app.include_router(
    reference_router,
    prefix="/api",
    dependencies=[Depends(get_current_user)],
)

# Media endpoints (audio, spectrogram) — use per-route dual-mode auth
# (supports both Authorization header AND ?token= query param)
app.include_router(
    reconstruct_media_router,
    prefix="/api",
)


# WebSocket endpoint
@app.websocket("/ws/{project_id}")
async def ws_endpoint(websocket: WebSocket, project_id: str) -> None:
    """WebSocket for real-time project updates."""
    await websocket_endpoint(websocket, project_id)


# ── Public routes ──
@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "service": "auralis"}


@app.get("/api/info")
async def info() -> dict[str, object]:
    """System information and capabilities."""
    from auralis import __version__

    return {
        "name": "AURALIS",
        "version": __version__,
        "tagline": "Hear deeper. Create beyond.",
        "layers": {
            "ear": "Analysis & Deconstruction",
            "hands": "Synthesis & Sound Design",
            "console": "Mixing & Mastering",
            "grid": "Composition & Arrangement",
            "brain": "AI Intelligence",
            "qc": "Quality Assurance",
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "auth_login": "POST /api/auth/login",
            "ear_upload": "POST /api/ear/upload",
            "ear_analyze": "POST /api/ear/analyze/{project_id}",
            "console_master": "POST /api/console/master/{project_id}",
            "console_qc": "GET /api/console/qc/{project_id}",
            "hands_synth": "POST /api/hands/synth",
            "hands_presets": "GET /api/hands/presets",
            "grid_arrangement": "POST /api/grid/arrangement",
            "grid_scales": "GET /api/grid/scales",
            "brain_plan": "POST /api/brain/plan",
            "brain_render": "POST /api/brain/render",
            "brain_chat": "POST /api/brain/chat",
            "brain_status": "GET /api/brain/status",
            "reconstruct_analysis": "GET /api/reconstruct/analysis/{project_id}",
            "reconstruct_stems": "GET /api/reconstruct/stems/{project_id}",
            "reconstruct_midi": "GET /api/reconstruct/midi/{project_id}",
            "reconstruct_models": "GET /api/reconstruct/models",
            "reconstruct_start": "POST /api/reconstruct/start",
            "reconstruct_status": "GET /api/reconstruct/status/{job_id}",
            "reconstruct_jobs": "GET /api/reconstruct/jobs",
            "reference_add": "POST /api/reference/add",
            "reference_list": "GET /api/reference/list",
            "reference_averages": "GET /api/reference/averages",
            "reference_gap": "GET /api/reference/gap/{job_id}",
            "gpu_status": "GET /api/gpu/status",
            "websocket": "WS /ws/{project_id}",
        },
    }
