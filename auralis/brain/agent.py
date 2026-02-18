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
    fx_plan: list[dict[str, Any]] = field(default_factory=list)
    mix_plan: dict[str, dict[str, Any]] = field(default_factory=dict)  # AI-decided mix
    section_details: list[dict[str, Any]] = field(default_factory=list)  # AI-decided per-section
    description: str = ""
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

## FX Palette (You decide WHAT, WHEN, WHY)
You have access to these effects. For each section, decide which to apply and why.

Creative FX:
- shimmer_reverb: params={decay_s: 1-6, pitch_shift_semitones: 7/12/19, wet: 0-1, damping: 0-1}
- filter_sweep: params={filter_type: highpass|lowpass, start_hz: 20-20000, end_hz: 20-20000, resonance: 0.5-4.0}
- stereo_width: params={width: 0.0-2.0} (1.0=normal, <1=narrow, >1=wide)
- tape_stop: params={duration_ms: 200-2000}
- pitch_riser: params={semitones: 1-24}
- ring_mod: params={freq_hz: 50-2000, wet: 0-0.5}

Classic FX:
- reverb: params={room_size: 0-1, damping: 0-1, wet: 0-1}
- delay: params={time_ms: 50-1000, feedback: 0-0.8, wet: 0-0.6}
- chorus: params={rate_hz: 0.1-5, depth_ms: 1-20, mix: 0-1}
- distortion: params={drive: 0.5-10, type: soft_clip|hard_clip|tube|foldback, mix: 0-1}
- compressor: params={threshold_db: -40 to 0, ratio: 1-20, attack_ms: 0.1-100, release_ms: 10-1000}
- bitcrush: params={bits: 4-16, downsample: 1-8, mix: 0-1}
- saturation: params={drive: 0.01-0.5, mix: 0-1}

## Automation Curves (You decide the SHAPE)
curve_shape options: linear, exponential, logarithmic, ease_in, ease_out, s_curve, step
- exponential: slow start, fast end (great for filter sweeps building up)
- ease_in: dramatic builds (tension before drop)
- ease_out: smooth landings (post-drop settle)
- s_curve: natural organic transitions
- step: instant on/off (drop impact)

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
    "fx_plan": [
        {
            "section": "breakdown",
            "effect": "shimmer_reverb",
            "target": "chords",
            "why": "Create ethereal space after intense drop",
            "params": {"decay_s": 4.0, "pitch_shift_semitones": 12, "wet": 0.5, "damping": 0.4},
            "automation": {
                "param": "wet",
                "curve": "ease_in",
                "start": 0.1,
                "end": 0.6
            }
        }
    ],
    "mix_plan": {
        "drums": {"volume_db": -2.0, "pan": 0.0, "reverb_send": 0.0},
        "bass": {"volume_db": 0.0, "pan": 0.0, "reverb_send": 0.0},
        "chords": {"volume_db": -4.0, "pan": -0.25, "reverb_send": 0.3},
        "melody": {"volume_db": -1.0, "pan": 0.2, "reverb_send": 0.2}
    },
    "section_details": [
        {"name": "intro", "bars": 8, "energy": 0.3},
        {"name": "verse", "bars": 16, "energy": 0.5},
        {"name": "breakdown", "bars": 8, "energy": 0.2},
        {"name": "drop", "bars": 16, "energy": 1.0},
        {"name": "outro", "bars": 8, "energy": 0.3}
    ],
    "description": "Brief creative description"
}

IMPORTANT RULES:
- The fx_plan is where your creativity shines. Each section should have FX that serve the narrative.
- The mix_plan is critical: decide pan, volume, and reverb send for EACH track based on the genre and mood.
- section_details: specify the bars and energy (0.0-1.0) for EACH section. This controls the intensity arc.
- Think: WHY does this effect/mix/energy belong here? What emotion does it reinforce?
- Do NOT apply the same FX to every section. Be intentional.
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
    reference_dna: dict[str, Any] | None = None,
) -> ProductionPlan:
    """Generate a production plan from a natural language description.

    If reference_dna is provided, the LLM learns from analyzed reference tracks.
    """
    cfg = config or BrainConfig()
    client = cfg.client

    # Build the prompt with optional reference context
    user_msg = f"Create a production plan for: {description}"
    if reference_dna:
        ref_summary = json.dumps(reference_dna, indent=2, default=str)
        user_msg += f"\n\n## Reference Track DNA (learn from this):\n{ref_summary}"

    response = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_PRODUCER},
            {"role": "user", "content": user_msg},
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
        fx_plan=data.get("fx_plan", []),
        mix_plan=data.get("mix_plan", {}),
        section_details=data.get("section_details", []),
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
        "fx_plan": plan.fx_plan,
        "mix_plan": plan.mix_plan,
        "section_details": plan.section_details,
        "render": {
            "sample_rate": 44100,
            "bpm": plan.bpm,
        },
    }

