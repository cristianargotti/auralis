"""AURALIS API — HANDS routes (synthesis, effects, mixing)."""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from auralis.hands.synth import (
    PRESETS,
    OscConfig,
    VoiceConfig,
    ADSREnvelope,
    FilterConfig,
    render_voice,
    render_midi_to_audio,
    save_audio,
    note_to_freq,
)
from auralis.hands.effects import (
    PRESET_CHAINS,
    EffectChain,
    process_chain,
)
from auralis.hands.mixer import Mixer, MixConfig, MixResult

router = APIRouter(prefix="/hands", tags=["hands"])


# ── Models ───────────────────────────────────────────────


class SynthRequest(BaseModel):
    """Synthesize a single note."""

    freq_hz: float = 440.0
    duration_s: float = 2.0
    preset: str = "supersaw"
    sample_rate: int = 44100


class RenderNotesRequest(BaseModel):
    """Render MIDI notes to audio."""

    notes: list[dict[str, float]]
    preset: str = "supersaw"
    sample_rate: int = 44100


class EffectRequest(BaseModel):
    """Apply effects to a preset chain."""

    chain_name: str = "edm_lead"


# ── Endpoints ────────────────────────────────────────────


@router.get("/presets")
def list_presets() -> dict:
    """List available synth presets."""
    return {
        name: {
            "name": patch.name,
            "description": patch.description,
            "oscillators": len(patch.voice.oscillators),
            "unison": patch.voice.unison,
            "has_filter": patch.voice.filter is not None,
        }
        for name, patch in PRESETS.items()
    }


@router.get("/effect-chains")
def list_effect_chains() -> dict:
    """List available effect chain presets."""
    return {
        name: {
            "name": chain.name,
            "has_eq": len(chain.eq_bands) > 0,
            "has_compressor": chain.compressor is not None,
            "has_distortion": chain.distortion is not None,
            "has_reverb": chain.reverb is not None,
            "has_delay": chain.delay is not None,
            "has_chorus": chain.chorus is not None,
            "has_sidechain": chain.sidechain is not None,
        }
        for name, chain in PRESET_CHAINS.items()
    }


@router.post("/synth")
def synth_note(req: SynthRequest) -> dict:
    """Synthesize a single note and return the file path."""
    if req.preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")

    voice = PRESETS[req.preset].voice
    audio = render_voice(req.freq_hz, req.duration_s, req.sample_rate, voice)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as f:
        save_audio(audio, f.name, req.sample_rate)
        return {
            "path": f.name,
            "preset": req.preset,
            "freq_hz": req.freq_hz,
            "duration_s": req.duration_s,
            "samples": len(audio),
        }


@router.post("/render-notes")
def render_notes(req: RenderNotesRequest) -> dict:
    """Render MIDI note events to audio."""
    if req.preset not in PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {req.preset}")

    voice = PRESETS[req.preset].voice
    audio = render_midi_to_audio(req.notes, sr=req.sample_rate, voice=voice)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as f:
        save_audio(audio, f.name, req.sample_rate)
        return {
            "path": f.name,
            "preset": req.preset,
            "notes_count": len(req.notes),
            "duration_s": round(len(audio) / req.sample_rate, 2),
            "samples": len(audio),
        }
