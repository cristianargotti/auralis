"""AURALIS Console — Stem Processing Recipes.

Intelligence layer that reads EAR analysis data and builds optimal
EffectChains per stem type. Each recipe makes data-driven decisions
about EQ, compression, saturation, sends, volume, and panning.

This is the "brain" of the professional Console stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from auralis.hands.effects import (
    EffectChain,
    EQBand,
    CompressorConfig,
    DistortionConfig,
    ReverbConfig,
    DelayConfig,
    SidechainConfig,
)
from auralis.hands.mixer import SendConfig


# ── Types ────────────────────────────────────────────────


@dataclass
class StemRecipe:
    """Complete processing recipe for a single stem."""
    name: str
    chain: EffectChain
    volume_db: float
    pan: float               # -1.0 (L) to 1.0 (R), 0.0 = center
    sends: list[SendConfig]
    description: str         # Human-readable explanation of decisions


# ── Analysis Helpers ─────────────────────────────────────


def _get_freq_balance(analysis: dict[str, Any]) -> dict[str, float]:
    """Extract frequency band percentages from stem analysis."""
    fb = analysis.get("freq_bands", {})
    return {
        "low": fb.get("low", 33.3),
        "mid": fb.get("mid", 33.3),
        "high": fb.get("high", 33.3),
    }


def _get_rms(analysis: dict[str, Any]) -> float:
    """Get RMS in dB, default to -20 if missing."""
    return analysis.get("rms_db", -20.0)


def _get_peak(analysis: dict[str, Any]) -> float:
    """Get peak in dBFS, default to -6 if missing."""
    return analysis.get("peak_db", -6.0)


def _get_energy(analysis: dict[str, Any]) -> float:
    """Get energy percentage relative to original."""
    return analysis.get("energy_pct", 25.0)


# ── Drum Recipe ──────────────────────────────────────────


def build_drum_recipe(
    analysis: dict[str, Any],
    bpm: float = 120.0,
) -> StemRecipe:
    """Build processing chain for drums stem.

    Strategy:
    - HPF to remove sub rumble (keep kick body)
    - Boost attack frequencies (3-5 kHz) for presence
    - Compress to control dynamics (fast attack for punch)
    - Light saturation for warmth/glue
    - Pan center, moderate volume
    """
    freq = _get_freq_balance(analysis)
    rms = _get_rms(analysis)
    decisions: list[str] = []

    # ── EQ ──
    eq_bands: list[EQBand] = []

    # HPF: remove sub rumble below kick fundamental
    eq_bands.append(EQBand(freq_hz=30.0, gain_db=-12.0, q=0.7, type="highshelf"))
    decisions.append("HPF 30Hz (-12dB)")

    # If too much low energy, tame it slightly
    if freq["low"] > 50:
        eq_bands.append(EQBand(freq_hz=80.0, gain_db=-2.0, q=1.0, type="peak"))
        decisions.append("Cut 80Hz (-2dB, heavy low)")

    # Boost attack/presence region
    attack_boost = 3.0 if freq["high"] < 25 else 1.5
    eq_bands.append(EQBand(freq_hz=4000.0, gain_db=attack_boost, q=1.2, type="peak"))
    decisions.append(f"+{attack_boost}dB @4kHz attack")

    # Add snap/click for definition
    eq_bands.append(EQBand(freq_hz=8000.0, gain_db=1.5, q=0.8, type="peak"))
    decisions.append("+1.5dB @8kHz snap")

    # ── Compression ──
    # Fast attack to catch transients, quick release for punch
    # More aggressive if drum dynamics are wide
    dynamic_range = abs(_get_peak(analysis) - rms)
    ratio = 4.0 if dynamic_range > 15 else 3.0
    comp = CompressorConfig(
        threshold_db=rms + 6,
        ratio=ratio,
        attack_ms=5.0,
        release_ms=50.0,
        makeup_db=2.0,
    )
    decisions.append(f"Comp {ratio}:1 @ {rms + 6:.0f}dB, 5ms atk")

    # ── Saturation ──
    # Light saturation for analog warmth
    dist = DistortionConfig(drive=2.0, type="soft_clip", mix=0.15)
    decisions.append("Sat: soft clip, drive 2.0, mix 15%")

    # ── Volume ──
    # Drums typically sit at -2 to 0 dB relative
    vol = -1.0
    if rms > -10:
        vol = -3.0  # Hot drums — pull back
        decisions.append("Vol pulled back (hot signal)")
    elif rms < -25:
        vol = 1.0   # Quiet drums — push up
        decisions.append("Vol pushed up (quiet signal)")

    chain = EffectChain(
        name="drums_pro",
        eq_bands=eq_bands,
        compressor=comp,
        distortion=dist,
    )

    return StemRecipe(
        name="drums",
        chain=chain,
        volume_db=vol,
        pan=0.0,  # Drums always center
        sends=[SendConfig(bus_name="reverb", amount=0.08, pre_fader=False)],
        description=f"🥁 Drums: {' | '.join(decisions)}",
    )


# ── Bass Recipe ──────────────────────────────────────────


def build_bass_recipe(
    analysis: dict[str, Any],
    bpm: float = 120.0,
) -> StemRecipe:
    """Build processing chain for bass stem.

    Strategy:
    - LPF to clean harsh highs
    - Boost sub-bass (60-80 Hz) for weight
    - Cut boxiness (200-300 Hz) for clarity
    - Heavy compression for consistency
    - Saturation for harmonic richness
    - Pan dead center, strong volume
    """
    freq = _get_freq_balance(analysis)
    rms = _get_rms(analysis)
    decisions: list[str] = []

    # ── EQ ──
    eq_bands: list[EQBand] = []

    # HPF below fundamental
    eq_bands.append(EQBand(freq_hz=25.0, gain_db=-18.0, q=0.5, type="highshelf"))
    decisions.append("HPF 25Hz")

    # Sub-bass boost  
    sub_boost = 3.0 if freq["low"] < 40 else 1.5
    eq_bands.append(EQBand(freq_hz=65.0, gain_db=sub_boost, q=0.8, type="peak"))
    decisions.append(f"+{sub_boost}dB @65Hz sub")

    # Anti-mud: cut boxiness
    mud_cut = -3.0 if freq["mid"] > 45 else -1.5
    eq_bands.append(EQBand(freq_hz=250.0, gain_db=mud_cut, q=1.5, type="peak"))
    decisions.append(f"{mud_cut}dB @250Hz anti-mud")

    # Clean highs — roll off above 8kHz
    if freq["high"] > 20:
        eq_bands.append(EQBand(freq_hz=8000.0, gain_db=-4.0, q=0.5, type="highshelf"))
        decisions.append("LPF 8kHz (-4dB)")

    # ── Compression ──
    # Heavy compression for consistent bass
    comp = CompressorConfig(
        threshold_db=rms + 4,
        ratio=6.0,
        attack_ms=10.0,
        release_ms=80.0,
        makeup_db=3.0,
    )
    decisions.append(f"Comp 6:1 @ {rms + 4:.0f}dB, 10ms atk")

    # ── Saturation ──
    # Moderate saturation for harmonic richness
    dist = DistortionConfig(drive=3.0, type="tube", mix=0.2)
    decisions.append("Sat: tube, drive 3.0, mix 20%")

    # ── Volume ──
    vol = 0.0
    if rms > -12:
        vol = -2.0
        decisions.append("Vol pulled back (hot)")
    elif rms < -25:
        vol = 2.0
        decisions.append("Vol pushed up (quiet)")

    chain = EffectChain(
        name="bass_pro",
        eq_bands=eq_bands,
        compressor=comp,
        distortion=dist,
    )

    return StemRecipe(
        name="bass",
        chain=chain,
        volume_db=vol,
        pan=0.0,  # Bass always center
        sends=[],  # No reverb/delay on bass (keeps it tight)
        description=f"🎸 Bass: {' | '.join(decisions)}",
    )


# ── Vocals Recipe ────────────────────────────────────────


def build_vocals_recipe(
    analysis: dict[str, Any],
    bpm: float = 120.0,
) -> StemRecipe:
    """Build processing chain for vocals stem.

    Strategy:
    - HPF at 80-100 Hz (remove body rumble)
    - Presence boost at 2-4 kHz for clarity
    - De-ess cut at 6-8 kHz if sibilant
    - Medium compression for dynamics control
    - Reverb send + delay send for space
    - Pan center, rides on top
    """
    freq = _get_freq_balance(analysis)
    rms = _get_rms(analysis)
    energy = _get_energy(analysis)
    decisions: list[str] = []

    # ── EQ ──
    eq_bands: list[EQBand] = []

    # HPF to clean rumble
    eq_bands.append(EQBand(freq_hz=80.0, gain_db=-12.0, q=0.7, type="highshelf"))
    decisions.append("HPF 80Hz")

    # Anti-mud
    eq_bands.append(EQBand(freq_hz=300.0, gain_db=-2.0, q=1.5, type="peak"))
    decisions.append("-2dB @300Hz clarity")

    # Presence boost
    presence = 2.5 if freq["mid"] < 35 else 1.5
    eq_bands.append(EQBand(freq_hz=3000.0, gain_db=presence, q=1.0, type="peak"))
    decisions.append(f"+{presence}dB @3kHz presence")

    # De-ess if too much high energy
    if freq["high"] > 30:
        eq_bands.append(EQBand(freq_hz=7000.0, gain_db=-3.0, q=2.0, type="peak"))
        decisions.append("-3dB @7kHz de-ess")

    # Air/brightness  
    eq_bands.append(EQBand(freq_hz=12000.0, gain_db=1.5, q=0.5, type="highshelf"))
    decisions.append("+1.5dB @12kHz air")

    # ── Compression ──
    comp = CompressorConfig(
        threshold_db=rms + 5,
        ratio=3.0,
        attack_ms=15.0,
        release_ms=120.0,
        makeup_db=2.0,
    )
    decisions.append(f"Comp 3:1 @ {rms + 5:.0f}dB, 15ms atk")

    # ── Volume ──
    # Vocals usually sit forward in the mix
    vol = 1.0
    if energy > 50:
        vol = -1.0  # Dominant vocals — pull back slightly
        decisions.append("Vol pulled back (dominant)")
    elif energy < 10:
        vol = 3.0  # Quiet vocals — push forward
        decisions.append("Vol pushed up (quiet)")

    # ── Sends ──
    # Calculate delay time synced to BPM (1/8 note)
    delay_ms = (60000.0 / bpm) / 2  # 1/8 note

    sends = [
        SendConfig(bus_name="reverb", amount=0.20, pre_fader=False),
        SendConfig(bus_name="delay", amount=0.12, pre_fader=False),
    ]
    decisions.append(f"Reverb 20% | Delay 12% ({delay_ms:.0f}ms)")

    chain = EffectChain(
        name="vocals_pro",
        eq_bands=eq_bands,
        compressor=comp,
    )

    return StemRecipe(
        name="vocals",
        chain=chain,
        volume_db=vol,
        pan=0.0,  # Lead vocals center
        sends=sends,
        description=f"🎤 Vocals: {' | '.join(decisions)}",
    )


# ── Other/Instruments Recipe ─────────────────────────────


def build_other_recipe(
    analysis: dict[str, Any],
    bpm: float = 120.0,
    ear_data: dict[str, Any] | None = None,
) -> StemRecipe:
    """Build processing chain for 'other' stem (instruments, synths, FX).

    Strategy:
    - HPF at 100 Hz (leave room for bass)
    - Anti-mud cut at 300 Hz
    - Presence boost if lacking definition
    - Light compression
    - Reverb + subtle delay for space
    - Pan slightly off-center for width
    """
    freq = _get_freq_balance(analysis)
    rms = _get_rms(analysis)
    decisions: list[str] = []

    # ── EQ ──
    eq_bands: list[EQBand] = []

    # HPF — leave room for bass
    eq_bands.append(EQBand(freq_hz=100.0, gain_db=-10.0, q=0.7, type="highshelf"))
    decisions.append("HPF 100Hz")

    # Anti-mud
    mud_cut = -3.0 if freq["mid"] > 50 else -1.5
    eq_bands.append(EQBand(freq_hz=350.0, gain_db=mud_cut, q=1.2, type="peak"))
    decisions.append(f"{mud_cut}dB @350Hz anti-mud")

    # Add definition if lacking
    if freq["high"] < 20:
        eq_bands.append(EQBand(freq_hz=5000.0, gain_db=2.0, q=0.8, type="peak"))
        decisions.append("+2dB @5kHz definition")

    # Air/presence
    eq_bands.append(EQBand(freq_hz=10000.0, gain_db=1.0, q=0.5, type="highshelf"))
    decisions.append("+1dB @10kHz air")

    # ── Compression ──
    comp = CompressorConfig(
        threshold_db=rms + 8,
        ratio=2.5,
        attack_ms=20.0,
        release_ms=150.0,
        makeup_db=1.5,
    )
    decisions.append(f"Comp 2.5:1 @ {rms + 8:.0f}dB, 20ms atk")

    # ── Volume ──
    vol = -2.0  # Instruments typically sit back
    if rms > -12:
        vol = -4.0
        decisions.append("Vol pulled back (hot)")
    elif rms < -30:
        vol = 0.0
        decisions.append("Vol pushed up (quiet)")

    # ── Pan ──
    # Slight offset for stereo width (instruments fill the sides)
    pan = 0.15  # Slightly right
    decisions.append("Pan: R15%")

    # ── Sends ──
    sends = [
        SendConfig(bus_name="reverb", amount=0.25, pre_fader=False),
        SendConfig(bus_name="delay", amount=0.08, pre_fader=False),
    ]
    decisions.append("Reverb 25% | Delay 8%")

    chain = EffectChain(
        name="other_pro",
        eq_bands=eq_bands,
        compressor=comp,
    )

    return StemRecipe(
        name="other",
        chain=chain,
        volume_db=vol,
        pan=pan,
        sends=sends,
        description=f"🎹 Other: {' | '.join(decisions)}",
    )


# ── Main Dispatcher ──────────────────────────────────────


_STEM_BUILDERS = {
    "drums": build_drum_recipe,
    "bass": build_bass_recipe,
    "vocals": build_vocals_recipe,
    "other": build_other_recipe,
}


def build_recipe_for_stem(
    stem_name: str,
    stem_analysis: dict[str, Any],
    bpm: float = 120.0,
    ear_data: dict[str, Any] | None = None,
) -> StemRecipe:
    """Build the optimal processing recipe for a stem based on its analysis.

    Args:
        stem_name: Name of the stem (drums, bass, vocals, other)
        stem_analysis: Per-stem analysis data from EAR stage
        bpm: Track BPM for tempo-synced effects
        ear_data: Full EAR analysis data (optional, for context)

    Returns:
        StemRecipe with chain, volume, pan, sends, and description
    """
    builder = _STEM_BUILDERS.get(stem_name, build_other_recipe)
    if stem_name == "other":
        return builder(stem_analysis, bpm=bpm, ear_data=ear_data)
    return builder(stem_analysis, bpm=bpm)


def build_all_recipes(
    stem_analyses: dict[str, dict[str, Any]],
    bpm: float = 120.0,
    ear_data: dict[str, Any] | None = None,
) -> dict[str, StemRecipe]:
    """Build recipes for all stems at once.

    Returns:
        Dict mapping stem name to its StemRecipe
    """
    recipes: dict[str, StemRecipe] = {}
    for name, analysis in stem_analyses.items():
        if "error" in analysis:
            continue
        recipes[name] = build_recipe_for_stem(
            name, analysis, bpm=bpm, ear_data=ear_data
        )
    return recipes
