"""AURALIS Brain — LLM Orchestrator for AI music production.

Uses OpenAI GPT for:
- Production decisions (genre analysis, arrangement planning)  
- Track description → arrangement generation
- Effect chain selection
- Mixing/mastering recommendations
- Creative ideation
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Literal

from openai import OpenAI


# ── Configuration ────────────────────────────────────────


@dataclass
class BrainConfig:
    """LLM orchestrator configuration."""

    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096

    @property
    def client(self) -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY", "")
        return OpenAI(api_key=api_key)


@dataclass
class ProductionPlan:
    """AI-generated production plan for a track."""

    title: str
    genre: str
    bpm: float
    key: str
    scale: str
    energy: str  # "chill", "moderate", "high", "intense"
    mood: str
    structure: list[str]
    synth_presets: dict[str, str]  # track_name -> preset_name
    effect_chains: dict[str, str]  # track_name -> chain_name
    description: str
    raw_response: str = ""


# ── System Prompts ───────────────────────────────────────


SYSTEM_PROMPT_PRODUCER = """You are AURALIS, an expert AI music producer.
You make decisions about musical arrangement, sound design, mixing, and mastering.
You respond with structured JSON when asked for production plans.

Available synth presets: supersaw, bass_808, pluck, pad_warm, acid_303
Available effect chains: deep_house_bass, ambient_pad, edm_lead
Available drum styles: four_on_floor, breakbeat, trap, minimal
Available bass styles: simple, walking, syncopated
Available genres: house, techno, ambient, pop, hip_hop
Available scales: major, minor, dorian, mixolydian, phrygian, lydian, harmonic_minor, pentatonic_major, pentatonic_minor, blues

When generating a production plan, respond with this exact JSON structure:
{
    "title": "Track Title",
    "genre": "house",
    "bpm": 120,
    "key": "C",
    "scale": "minor",
    "energy": "high",
    "mood": "description of mood",
    "structure": ["intro", "verse", "breakdown", "drop", "verse", "drop", "outro"],
    "synth_presets": {
        "bass": "bass_808",
        "melody": "supersaw",
        "chords": "pad_warm"
    },
    "effect_chains": {
        "bass": "deep_house_bass",
        "melody": "edm_lead",
        "chords": "ambient_pad"
    },
    "description": "Brief creative description"
}
"""

SYSTEM_PROMPT_MIXER = """You are AURALIS, an expert mixing engineer.
Given a set of tracks and their analysis data, provide mixing recommendations.
Respond with structured JSON containing volume, pan, EQ, and compression settings.

Your response format:
{
    "tracks": {
        "track_name": {
            "volume_db": 0.0,
            "pan": 0.0,
            "eq": [{"freq": 100, "gain_db": -3, "q": 1.0}],
            "compression": {"threshold_db": -20, "ratio": 4},
            "notes": "Brief explanation"
        }
    },
    "master": {
        "target_lufs": -14,
        "notes": "Brief explanation"
    }
}
"""


# ── Core Functions ───────────────────────────────────────


def generate_production_plan(
    description: str,
    config: BrainConfig | None = None,
) -> ProductionPlan:
    """Generate a production plan from a natural language description."""
    cfg = config or BrainConfig()
    client = cfg.client

    response = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_PRODUCER},
            {"role": "user", "content": f"Create a production plan for: {description}"},
        ],
    )

    raw = response.choices[0].message.content or "{}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    return ProductionPlan(
        title=data.get("title", "Untitled"),
        genre=data.get("genre", "house"),
        bpm=float(data.get("bpm", 120)),
        key=data.get("key", "C"),
        scale=data.get("scale", "minor"),
        energy=data.get("energy", "moderate"),
        mood=data.get("mood", ""),
        structure=data.get("structure", ["intro", "verse", "chorus", "outro"]),
        synth_presets=data.get("synth_presets", {}),
        effect_chains=data.get("effect_chains", {}),
        description=data.get("description", ""),
        raw_response=raw,
    )


def get_mixing_advice(
    tracks_info: dict[str, Any],
    config: BrainConfig | None = None,
) -> dict[str, Any]:
    """Get AI mixing recommendations for a set of tracks."""
    cfg = config or BrainConfig()
    client = cfg.client

    response = client.chat.completions.create(
        model=cfg.model,
        temperature=0.5,
        max_tokens=cfg.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_MIXER},
            {"role": "user", "content": f"Provide mixing recommendations for these tracks:\n{json.dumps(tracks_info, indent=2)}"},
        ],
    )

    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Failed to parse response", "raw": raw}


def chat(
    message: str,
    history: list[dict[str, str]] | None = None,
    config: BrainConfig | None = None,
) -> str:
    """General chat about music production."""
    cfg = config or BrainConfig()
    client = cfg.client

    messages = [
        {"role": "system", "content": (
            "You are AURALIS, an AI music production assistant. "
            "You help with sound design, mixing, mastering, and music theory. "
            "Be concise and practical. When relevant, suggest specific "
            "AURALIS features (presets, effects, generators) that could help."
        )}
    ]

    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        messages=messages,
    )

    return response.choices[0].message.content or ""


# ── Full Pipeline Orchestrator ───────────────────────────


def plan_to_render_config(plan: ProductionPlan) -> dict[str, Any]:
    """Convert a ProductionPlan to rendering configuration.

    Returns a dict that can be passed to the arrangement + synth + mixer
    pipeline to render the complete track.
    """
    return {
        "arrangement": {
            "key": plan.key,
            "scale": plan.scale,
            "bpm": plan.bpm,
            "genre": plan.genre,
            "structure": plan.structure,
        },
        "synth_presets": plan.synth_presets,
        "effect_chains": plan.effect_chains,
        "render": {
            "sample_rate": 44100,
            "bpm": plan.bpm,
        },
    }
