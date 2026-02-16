"""AURALIS FastAPI server — main application."""

from fastapi import Depends, FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from auralis.api.auth import get_current_user, login
from auralis.api.routes.ear import router as ear_router
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
            "ear_status": "GET /api/ear/status/{job_id}",
            "websocket": "WS /ws/{project_id}",
        },
    }
