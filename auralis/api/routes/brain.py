"""AURALIS API — BRAIN routes (LLM, production, chat)."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from auralis.brain.agent import BrainConfig, generate_production_plan, chat
from auralis.brain.production_ai import render_track_offline

router = APIRouter(prefix="/brain", tags=["brain"])


# ── Models ───────────────────────────────────────────────


class CreateTrackRequest(BaseModel):
    """Create a track from description."""

    description: str
    use_llm: bool = True


class ChatRequest(BaseModel):
    """Chat with AURALIS AI."""

    message: str
    history: list[dict[str, str]] = []


class OfflineRenderRequest(BaseModel):
    """Render without LLM (offline mode)."""

    genre: str = "house"
    key: str = "C"
    scale: str = "minor"
    bpm: float = 120.0


# ── Endpoints ────────────────────────────────────────────


@router.post("/plan")
def create_plan(req: CreateTrackRequest) -> dict:
    """Generate a production plan from description."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    plan = generate_production_plan(req.description)
    return {
        "title": plan.title,
        "genre": plan.genre,
        "bpm": plan.bpm,
        "key": plan.key,
        "scale": plan.scale,
        "energy": plan.energy,
        "mood": plan.mood,
        "structure": plan.structure,
        "synth_presets": plan.synth_presets,
        "effect_chains": plan.effect_chains,
        "description": plan.description,
    }


@router.post("/render")
def render_track(req: OfflineRenderRequest) -> dict:
    """Render a track offline (no LLM needed)."""
    result = render_track_offline(
        genre=req.genre,
        key=req.key,
        scale=req.scale,
        bpm=req.bpm,
    )
    return result.summary


@router.post("/chat")
def chat_endpoint(req: ChatRequest) -> dict:
    """Chat with AURALIS AI about music production."""
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")

    response = chat(req.message, req.history)
    return {"response": response}


@router.get("/status")
def brain_status() -> dict:
    """Check BRAIN layer status."""
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "online": True,
        "llm_available": has_key,
        "model": "gpt-4o" if has_key else "offline",
        "capabilities": {
            "production_plan": has_key,
            "mixing_advice": has_key,
            "chat": has_key,
            "offline_render": True,
        },
    }
