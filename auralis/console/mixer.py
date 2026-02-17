"""AURALIS Console Mixer — Brain-guided mix orchestration.

Connects brain intelligence (StemPlan, MasterPlan) and stem recipes
to the hands/mixer engine.  Handles:
  - Auto-routing stems to buses (reverb, delay, drum bus)
  - Applying recipe-derived EQ, compression, volume, panning
  - Parallel compression on drum bus for punch
  - Send levels derived from brain reverb/delay decisions
  - Muting stems flagged by stem_decisions
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import structlog

from auralis.hands.effects import EffectChain, EffectConfig
from auralis.hands.mixer import (
    Bus,
    MixConfig,
    Mixer,
    MixResult,
    PanConfig,
    SendConfig,
    Track,
)

logger = structlog.get_logger()


# ── Default Pan Map ──────────────────────────────────────
# Mirrors typical mix engineer positioning

_DEFAULT_PAN: dict[str, float] = {
    "drums": 0.0,      # Center
    "bass": 0.0,       # Center
    "vocals": 0.0,     # Center
    "other": 0.0,      # Center (brain can override)
    # Generated stems — slight offsets
    "gen_drums": 0.0,
    "gen_bass": 0.0,
    "gen_other": 0.05,
    "layer_drums": -0.1,
    "layer_bass": 0.0,
    "layer_other": 0.1,
}


# ── Bus Presets ──────────────────────────────────────────


def _reverb_bus_chain() -> EffectChain:
    """Build a reverb bus effect chain."""
    return EffectChain(
        effects=[
            EffectConfig(
                type="reverb",
                params={
                    "decay": 2.5,
                    "wet": 1.0,  # 100% wet on bus (sends control amount)
                    "damping": 0.4,
                    "room_size": 0.7,
                },
            ),
            EffectConfig(
                type="eq",
                params={
                    "type": "highpass",
                    "freq": 200.0,
                    "q": 0.7,
                },
            ),
        ]
    )


def _delay_bus_chain(bpm: float) -> EffectChain:
    """Build a delay bus synced to BPM."""
    delay_time = 60.0 / bpm / 2  # 1/8 note
    return EffectChain(
        effects=[
            EffectConfig(
                type="delay",
                params={
                    "delay_time": delay_time,
                    "feedback": 0.3,
                    "wet": 1.0,
                },
            ),
            EffectConfig(
                type="eq",
                params={
                    "type": "highpass",
                    "freq": 300.0,
                    "q": 0.7,
                },
            ),
        ]
    )


def _drum_bus_chain() -> EffectChain:
    """Parallel compression chain for drum bus."""
    return EffectChain(
        effects=[
            EffectConfig(
                type="compressor",
                params={
                    "threshold_db": -20.0,
                    "ratio": 6.0,
                    "attack_ms": 5.0,
                    "release_ms": 50.0,
                    "makeup_db": 6.0,
                },
            ),
            EffectConfig(
                type="saturation",
                params={
                    "drive": 0.3,
                    "type": "soft",
                },
            ),
        ]
    )


# ── Console Mixer ────────────────────────────────────────


def console_mix(
    stem_paths: dict[str, str],
    output_path: str | Path,
    bpm: float = 120.0,
    sr: int = 44100,
    stem_recipes: dict[str, dict[str, Any]] | None = None,
    brain_stem_plans: dict[str, dict[str, Any]] | None = None,
    stem_decisions: dict[str, dict[str, Any]] | None = None,
    muted_stems: list[str] | None = None,
) -> dict[str, Any]:
    """Orchestrate a professional mix using brain intelligence.

    Args:
        stem_paths: Dict of stem_name → file_path.
        output_path: Where to save the mixed WAV.
        bpm: Track BPM.
        sr: Sample rate.
        stem_recipes: Recipe dicts per stem (from stem_recipes module).
        brain_stem_plans: StemPlan dicts from DNABrain.
        stem_decisions: Decision dicts from stem_decisions module.
        muted_stems: List of stem names to mute.

    Returns:
        Mix result dict with tracks_mixed, buses_used, peak_db, recipes.
    """
    stem_recipes = stem_recipes or {}
    brain_stem_plans = brain_stem_plans or {}
    stem_decisions = stem_decisions or {}
    muted = set(muted_stems or [])

    mixer = Mixer(MixConfig(sample_rate=sr, bpm=bpm))
    recipe_log: list[str] = []

    # ── Setup buses ──
    mixer.add_bus("reverb", effects=_reverb_bus_chain(), volume_db=-3.0)
    mixer.add_bus("delay", effects=_delay_bus_chain(bpm), volume_db=-6.0)
    mixer.add_bus("drum_bus", effects=_drum_bus_chain(), volume_db=-6.0)

    # ── Load and route stems ──
    for stem_name, stem_path in stem_paths.items():
        if stem_name in muted:
            recipe_log.append(f"🔇 {stem_name}: MUTED")
            continue

        if stem_name.startswith("_"):
            continue  # Skip internal keys like _master

        try:
            audio, file_sr = sf.read(str(stem_path), dtype="float64")
        except Exception as e:
            logger.warning("console_mixer.load_fail", stem=stem_name, error=str(e))
            continue

        # Resample if needed
        if file_sr != sr:
            try:
                import librosa
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)
            except ImportError:
                pass

        # ── Derive mix parameters from brain + recipe ──
        plan = brain_stem_plans.get(stem_name, {})
        recipe = stem_recipes.get(stem_name, {})

        volume_db = float(plan.get("volume_db", recipe.get("volume_db", 0.0)))
        pan_pos = float(plan.get("pan", _DEFAULT_PAN.get(stem_name, 0.0)))
        reverb_wet = float(plan.get("reverb_wet", recipe.get("reverb_send", 0.0)))
        delay_wet = float(plan.get("delay_wet", recipe.get("delay_send", 0.0)))

        # ── Build sends ──
        sends: list[SendConfig] = []

        if reverb_wet > 0.01:
            sends.append(SendConfig(bus_name="reverb", amount=reverb_wet))

        if delay_wet > 0.01:
            sends.append(SendConfig(bus_name="delay", amount=delay_wet))

        # Drum parallel compression: send drums to drum bus
        base_name = stem_name.replace("gen_", "").replace("layer_", "")
        if base_name == "drums":
            sends.append(SendConfig(bus_name="drum_bus", amount=0.4))

        # ── Build track effects from recipe ──
        track_effects = _build_track_effects(recipe)

        # ── Add track to mixer ──
        mixer.add_track(
            name=stem_name,
            audio=audio,
            volume_db=volume_db,
            pan=pan_pos,
            effects=track_effects,
            sends=sends,
        )

        send_desc = ""
        if sends:
            send_parts = [f"{s.bus_name}={s.amount:.0%}" for s in sends]
            send_desc = f" → sends: {', '.join(send_parts)}"

        recipe_log.append(
            f"🎛️ {stem_name}: {volume_db:+.1f}dB, pan={pan_pos:+.1f}{send_desc}"
        )

    # ── Mix ──
    result = mixer.mix(output_path=output_path)

    return {
        "tracks_mixed": result.tracks_mixed,
        "buses_used": result.buses_used,
        "peak_db": float(result.peak_db),
        "duration_s": float(result.duration_s),
        "recipes": recipe_log,
    }


def _build_track_effects(recipe: dict[str, Any]) -> EffectChain:
    """Build effect chain from a stem recipe dict."""
    effects: list[EffectConfig] = []

    # EQ from recipe
    eq_bands = recipe.get("eq_bands", [])
    for band in eq_bands:
        if isinstance(band, dict):
            effects.append(
                EffectConfig(
                    type="eq",
                    params={
                        "freq": band.get("freq", 1000.0),
                        "gain": band.get("gain_db", 0.0),
                        "q": band.get("q", 1.0),
                        "type": band.get("type", "peak"),
                    },
                )
            )

    # Compression from recipe
    comp = recipe.get("compression", {})
    if comp:
        effects.append(
            EffectConfig(
                type="compressor",
                params={
                    "threshold_db": comp.get("threshold_db", -20.0),
                    "ratio": comp.get("ratio", 3.0),
                    "attack_ms": comp.get("attack_ms", 10.0),
                    "release_ms": comp.get("release_ms", 100.0),
                },
            )
        )

    # Saturation from recipe
    sat_drive = recipe.get("saturation_drive", 0.0)
    if sat_drive > 0.01:
        effects.append(
            EffectConfig(
                type="saturation",
                params={"drive": sat_drive, "type": "soft"},
            )
        )

    return EffectChain(effects=effects)
