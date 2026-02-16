"""AURALIS API — GRID routes (MIDI, composition, arrangement)."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from auralis.grid.midi import (
    SCALES,
    CHORD_TYPES,
    generate_chord_progression,
    generate_bassline,
    generate_melody,
    generate_drum_pattern,
    pattern_to_note_events,
)
from auralis.grid.arrangement import (
    GENRE_STRUCTURES,
    ArrangementConfig,
    generate_arrangement,
    arrangement_summary,
)

router = APIRouter(prefix="/grid", tags=["grid"])


# ── Models ───────────────────────────────────────────────


class ArrangementRequest(BaseModel):
    """Generate a song arrangement."""

    key: str = "C"
    scale: str = "minor"
    bpm: float = 120.0
    genre: str = "house"


class PatternRequest(BaseModel):
    """Generate a musical pattern."""

    root: str = "C"
    scale: str = "minor"
    bars: int = 4
    bpm: float = 120.0


# ── Endpoints ────────────────────────────────────────────


@router.get("/scales")
def list_scales() -> dict:
    """List available scales."""
    return {name: {"intervals": intervals, "notes": len(intervals)} for name, intervals in SCALES.items()}


@router.get("/chords")
def list_chords() -> dict:
    """List available chord types."""
    return {name: {"intervals": intervals, "notes": len(intervals)} for name, intervals in CHORD_TYPES.items()}


@router.get("/genres")
def list_genres() -> dict:
    """List available genre templates with section counts."""
    return {
        name: {
            "sections": len(structure),
            "section_types": [s.name for s in structure],
            "total_bars": sum(s.bars for s in structure),
        }
        for name, structure in GENRE_STRUCTURES.items()
    }


@router.post("/arrangement")
def create_arrangement(req: ArrangementRequest) -> dict:
    """Generate a complete song arrangement."""
    config = ArrangementConfig(
        key=req.key, scale=req.scale, bpm=req.bpm, genre=req.genre,  # type: ignore[arg-type]
    )
    arrangement = generate_arrangement(config)
    return arrangement_summary(arrangement)


@router.post("/pattern/chords")
def create_chords(req: PatternRequest) -> dict:
    """Generate a chord progression."""
    pattern = generate_chord_progression(root=req.root, scale=req.scale, bars=req.bars)
    events = pattern_to_note_events(pattern, req.bpm)
    return {"name": pattern.name, "notes": len(pattern.notes), "events": events}


@router.post("/pattern/bass")
def create_bassline(req: PatternRequest) -> dict:
    """Generate a bassline."""
    pattern = generate_bassline(root=req.root, scale=req.scale, bars=req.bars)
    events = pattern_to_note_events(pattern, req.bpm)
    return {"name": pattern.name, "notes": len(pattern.notes), "events": events}


@router.post("/pattern/melody")
def create_melody(req: PatternRequest) -> dict:
    """Generate a melody."""
    pattern = generate_melody(root=req.root, scale=req.scale, bars=req.bars)
    events = pattern_to_note_events(pattern, req.bpm)
    return {"name": pattern.name, "notes": len(pattern.notes), "events": events}


@router.post("/pattern/drums")
def create_drums(req: PatternRequest) -> dict:
    """Generate a drum pattern."""
    pattern = generate_drum_pattern(style="four_on_floor", bars=req.bars)
    events = pattern_to_note_events(pattern, req.bpm)
    return {"name": pattern.name, "notes": len(pattern.notes), "events": events}
