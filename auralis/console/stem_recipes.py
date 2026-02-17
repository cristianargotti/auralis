"""AURALIS Console — Stem Processing Recipes.

Intelligence layer that reads EAR analysis data and builds optimal
EffectChains per stem type. Each recipe makes data-driven decisions
about EQ, compression, saturation, sends, volume, and panning.

When ref_target is provided (from the Reference DNA Bank), all
processing decisions are DERIVED FROM THE GAP between your track
and the averaged reference fingerprints. This is ref-targeted
reconstruction — the mixer auto-corrects toward pro quality.
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


def _ref_freq_gap(analysis: dict[str, Any], ref_target: dict[str, Any] | None, band: str) -> float:
    """Calculate frequency gap vs reference target (positive = you're above ref)."""
    if not ref_target:
        return 0.0
    your_bands = _get_freq_balance(analysis)
    ref_bands = ref_target.get("freq_bands", {})
    return your_bands.get(band, 33.3) - ref_bands.get(band, 33.3)


def _ref_rms_gap(analysis: dict[str, Any], ref_target: dict[str, Any] | None) -> float:
    """Calculate RMS gap vs reference (negative = you're quieter)."""
    if not ref_target:
        return 0.0
    return _get_rms(analysis) - ref_target.get("rms_db", -20.0)


# ── Drum Recipe ──────────────────────────────────────────


def build_drum_recipe(
    analysis: dict[str, Any],
    bpm: float = 120.0,
    ref_target: dict[str, Any] | None = None,
    stem_plan: Any | None = None,
) -> StemRecipe:
    """Build processing chain for drums stem.

    When ref_target is provided, EQ/comp/volume targets are derived
    from the reference average instead of generic rules.
    """
    freq = _get_freq_balance(analysis)
    rms = _get_rms(analysis)
    decisions: list[str] = []
    has_ref = ref_target is not None

    # ── EQ ──
    eq_bands: list[EQBand] = []

    # HPF: remove sub rumble below kick fundamental
    eq_bands.append(EQBand(freq_hz=30.0, gain_db=-12.0, q=0.7, type="highshelf"))
    decisions.append("HPF 30Hz")

    # Low-end adjustment — ref-targeted or generic
    low_gap = _ref_freq_gap(analysis, ref_target, "low")
    if has_ref and abs(low_gap) > 3:
        gain = round(-low_gap * 0.4, 1)  # Compensate gap
        eq_bands.append(EQBand(freq_hz=80.0, gain_db=gain, q=1.0, type="peak"))
        decisions.append(f"REF: {gain:+.1f}dB @80Hz (gap {low_gap:+.1f}%)")
    elif freq["low"] > 50:
        eq_bands.append(EQBand(freq_hz=80.0, gain_db=-2.0, q=1.0, type="peak"))
        decisions.append("Cut 80Hz (-2dB)")

    # Attack/presence — ref-targeted or generic
    high_gap = _ref_freq_gap(analysis, ref_target, "high")
    if has_ref and abs(high_gap) > 3:
        boost = round(-high_gap * 0.5, 1)
        boost = max(-4.0, min(6.0, boost))  # Clamp
        eq_bands.append(EQBand(freq_hz=4000.0, gain_db=boost, q=1.2, type="peak"))
        decisions.append(f"REF: {boost:+.1f}dB @4kHz (gap {high_gap:+.1f}%)")
    else:
        attack_boost = 3.0 if freq["high"] < 25 else 1.5
        eq_bands.append(EQBand(freq_hz=4000.0, gain_db=attack_boost, q=1.2, type="peak"))
        decisions.append(f"+{attack_boost}dB @4kHz attack")

    eq_bands.append(EQBand(freq_hz=8000.0, gain_db=1.5, q=0.8, type="peak"))

    # ── Compression ──
    dynamic_range = abs(_get_peak(analysis) - rms)
    if has_ref:
        ref_dr = ref_target.get("dynamic_range_db", 12.0)
        ratio = 5.0 if dynamic_range > ref_dr + 3 else 3.5
        decisions.append(f"REF comp {ratio}:1 (DR {dynamic_range:.0f} vs ref {ref_dr:.0f})")
    else:
        ratio = 4.0 if dynamic_range > 15 else 3.0
        decisions.append(f"Comp {ratio}:1")

    comp = CompressorConfig(
        threshold_db=rms + 6, ratio=ratio,
        attack_ms=5.0, release_ms=50.0, makeup_db=2.0,
    )

    # ── Saturation — brain-guided or generic ──
    if stem_plan and getattr(stem_plan, 'saturation_drive', 0):
        drive = stem_plan.saturation_drive
        decisions.append(f"🧠 Sat: {drive:.0f}dB drive (brain)")
    else:
        drive = 2.0
    dist = DistortionConfig(drive=drive, type="soft_clip", mix=0.15)

    # ── Volume — ref-targeted or generic ──
    rms_gap = _ref_rms_gap(analysis, ref_target)
    if has_ref:
        vol = round(-rms_gap * 0.5, 1)  # Push toward ref RMS
        vol = max(-6.0, min(6.0, vol))
        decisions.append(f"REF vol {vol:+.1f}dB (RMS gap {rms_gap:+.1f})")
    else:
        vol = -1.0
        if rms > -10:
            vol = -3.0
        elif rms < -25:
            vol = 1.0

    # ── Brain EQ overrides ──
    if stem_plan and getattr(stem_plan, 'eq_adjustments', []):
        for freq_hz, gain_db, q in stem_plan.eq_adjustments:
            eq_bands.append(EQBand(freq_hz=freq_hz, gain_db=gain_db, q=q, type="peak"))
            decisions.append(f"🧠 EQ: {gain_db:+.1f}dB @{freq_hz:.0f}Hz")

    # ── Brain compression overrides ──
    if stem_plan and getattr(stem_plan, 'compression', {}):
        bp = stem_plan.compression
        if bp.get('threshold_db'):
            comp = CompressorConfig(
                threshold_db=bp['threshold_db'], ratio=bp.get('ratio', ratio),
                attack_ms=bp.get('attack_ms', 5.0), release_ms=bp.get('release_ms', 50.0),
                makeup_db=2.0,
            )
            decisions.append(f"🧠 Comp: {bp.get('ratio', ratio)}:1, atk={bp.get('attack_ms', 5.0)}ms")

    chain = EffectChain(
        name="drums_pro", eq_bands=eq_bands,
        compressor=comp, distortion=dist,
    )

    # ── Sends — brain-guided or generic ──
    if stem_plan and getattr(stem_plan, 'reverb_wet', 0):
        drum_rev = stem_plan.reverb_wet
        decisions.append(f"🧠 Reverb send: {drum_rev:.0%}")
    else:
        drum_rev = 0.08

    return StemRecipe(
        name="drums", chain=chain, volume_db=vol, pan=0.0,
        sends=[SendConfig(bus_name="reverb", amount=drum_rev, pre_fader=False)],
        description=f"🥁 Drums: {' | '.join(decisions)}",
    )


# ── Bass Recipe ──────────────────────────────────────────


def build_bass_recipe(
    analysis: dict[str, Any],
    bpm: float = 120.0,
    ref_target: dict[str, Any] | None = None,
    stem_plan: Any | None = None,
) -> StemRecipe:
    """Build processing chain for bass stem.

    When ref_target is provided, sub-bass boost, anti-mud cut, and
    compression are calibrated to match reference averages.
    """
    freq = _get_freq_balance(analysis)
    rms = _get_rms(analysis)
    decisions: list[str] = []
    has_ref = ref_target is not None

    # ── EQ ──
    eq_bands: list[EQBand] = []

    eq_bands.append(EQBand(freq_hz=25.0, gain_db=-18.0, q=0.5, type="highshelf"))
    decisions.append("HPF 25Hz")

    # Sub-bass — ref-targeted or generic
    low_gap = _ref_freq_gap(analysis, ref_target, "low")
    if has_ref and abs(low_gap) > 3:
        sub_boost = round(-low_gap * 0.5, 1)
        sub_boost = max(-4.0, min(6.0, sub_boost))
        eq_bands.append(EQBand(freq_hz=65.0, gain_db=sub_boost, q=0.8, type="peak"))
        decisions.append(f"REF: {sub_boost:+.1f}dB @65Hz sub (gap {low_gap:+.1f}%)")
    else:
        sub_boost = 3.0 if freq["low"] < 40 else 1.5
        eq_bands.append(EQBand(freq_hz=65.0, gain_db=sub_boost, q=0.8, type="peak"))
        decisions.append(f"+{sub_boost}dB @65Hz sub")

    # Anti-mud — ref-targeted or generic
    mid_gap = _ref_freq_gap(analysis, ref_target, "mid")
    if has_ref and mid_gap > 3:
        mud_cut = round(-mid_gap * 0.4, 1)
        mud_cut = max(-6.0, min(0.0, mud_cut))
        eq_bands.append(EQBand(freq_hz=250.0, gain_db=mud_cut, q=1.5, type="peak"))
        decisions.append(f"REF: {mud_cut:.1f}dB @250Hz (mid {mid_gap:+.1f}% over ref)")
    else:
        mud_cut = -3.0 if freq["mid"] > 45 else -1.5
        eq_bands.append(EQBand(freq_hz=250.0, gain_db=mud_cut, q=1.5, type="peak"))
        decisions.append(f"{mud_cut}dB @250Hz anti-mud")

    if freq["high"] > 20:
        eq_bands.append(EQBand(freq_hz=8000.0, gain_db=-4.0, q=0.5, type="highshelf"))

    # ── Compression ──
    if has_ref:
        ref_dr = ref_target.get("dynamic_range_db", 10.0)
        dr = abs(_get_peak(analysis) - rms)
        ratio = 7.0 if dr > ref_dr + 4 else 5.0
        decisions.append(f"REF comp {ratio}:1 (DR {dr:.0f} vs ref {ref_dr:.0f})")
    else:
        ratio = 6.0
        decisions.append("Comp 6:1")

    comp = CompressorConfig(
        threshold_db=rms + 4, ratio=ratio,
        attack_ms=10.0, release_ms=80.0, makeup_db=3.0,
    )

    # ── Saturation — brain-guided or generic ──
    if stem_plan and getattr(stem_plan, 'saturation_drive', 0):
        drive = stem_plan.saturation_drive
        decisions.append(f"🧠 Sat: tube, drive={drive:.0f}dB")
    else:
        drive = 3.0
    dist = DistortionConfig(drive=drive, type="tube", mix=0.2)
    if not (stem_plan and getattr(stem_plan, 'saturation_drive', 0)):
        decisions.append("Sat: tube")

    # ── Sidechain from brain ──
    sends: list[Any] = []
    if stem_plan and getattr(stem_plan, 'sidechain', False):
        # Sidechain will be applied by the effects chain
        decisions.append(f"🧠 Sidechain: depth={stem_plan.sidechain_depth:.0%}")

    # ── Volume — ref-targeted or generic ──
    rms_gap = _ref_rms_gap(analysis, ref_target)
    if has_ref:
        vol = round(-rms_gap * 0.5, 1)
        vol = max(-6.0, min(6.0, vol))
        decisions.append(f"REF vol {vol:+.1f}dB (RMS gap {rms_gap:+.1f})")
    else:
        vol = 0.0
        if rms > -12:
            vol = -2.0
        elif rms < -25:
            vol = 2.0

    # ── Brain EQ overrides ──
    if stem_plan and getattr(stem_plan, 'eq_adjustments', []):
        for freq_hz, gain_db, q in stem_plan.eq_adjustments:
            eq_bands.append(EQBand(freq_hz=freq_hz, gain_db=gain_db, q=q, type="peak"))
            decisions.append(f"🧠 EQ: {gain_db:+.1f}dB @{freq_hz:.0f}Hz")

    # ── Brain compression overrides ──
    if stem_plan and getattr(stem_plan, 'compression', {}):
        bp = stem_plan.compression
        if bp.get('threshold_db'):
            comp = CompressorConfig(
                threshold_db=bp['threshold_db'], ratio=bp.get('ratio', ratio),
                attack_ms=bp.get('attack_ms', 10.0), release_ms=bp.get('release_ms', 80.0),
                makeup_db=3.0,
            )
            decisions.append(f"🧠 Comp: {bp.get('ratio', ratio)}:1")

    chain = EffectChain(
        name="bass_pro", eq_bands=eq_bands,
        compressor=comp, distortion=dist,
    )

    return StemRecipe(
        name="bass", chain=chain, volume_db=vol, pan=0.0,
        sends=sends,
        description=f"🎸 Bass: {' | '.join(decisions)}",
    )


# ── Vocals Recipe ────────────────────────────────────────


def build_vocals_recipe(
    analysis: dict[str, Any],
    bpm: float = 120.0,
    ref_target: dict[str, Any] | None = None,
    stem_plan: Any | None = None,
) -> StemRecipe:
    """Build processing chain for vocals stem.

    Ref-targeted: presence and air boosts derived from frequency gaps.
    """
    freq = _get_freq_balance(analysis)
    rms = _get_rms(analysis)
    energy = _get_energy(analysis)
    decisions: list[str] = []
    has_ref = ref_target is not None

    # ── EQ ──
    eq_bands: list[EQBand] = []

    eq_bands.append(EQBand(freq_hz=80.0, gain_db=-12.0, q=0.7, type="highshelf"))
    decisions.append("HPF 80Hz")

    eq_bands.append(EQBand(freq_hz=300.0, gain_db=-2.0, q=1.5, type="peak"))
    decisions.append("-2dB @300Hz")

    # Presence — ref-targeted or generic
    mid_gap = _ref_freq_gap(analysis, ref_target, "mid")
    if has_ref and abs(mid_gap) > 3:
        presence = round(-mid_gap * 0.4, 1)
        presence = max(-3.0, min(5.0, presence))
        eq_bands.append(EQBand(freq_hz=3000.0, gain_db=presence, q=1.0, type="peak"))
        decisions.append(f"REF: {presence:+.1f}dB @3kHz (gap {mid_gap:+.1f}%)")
    else:
        presence = 2.5 if freq["mid"] < 35 else 1.5
        eq_bands.append(EQBand(freq_hz=3000.0, gain_db=presence, q=1.0, type="peak"))
        decisions.append(f"+{presence}dB @3kHz presence")

    # De-ess
    if freq["high"] > 30:
        eq_bands.append(EQBand(freq_hz=7000.0, gain_db=-3.0, q=2.0, type="peak"))
        decisions.append("-3dB @7kHz de-ess")

    # Air — ref-targeted
    high_gap = _ref_freq_gap(analysis, ref_target, "high")
    if has_ref and abs(high_gap) > 3:
        air = round(-high_gap * 0.3, 1)
        air = max(-2.0, min(4.0, air))
        eq_bands.append(EQBand(freq_hz=12000.0, gain_db=air, q=0.5, type="highshelf"))
        decisions.append(f"REF: {air:+.1f}dB @12kHz air (gap {high_gap:+.1f}%)")
    else:
        eq_bands.append(EQBand(freq_hz=12000.0, gain_db=1.5, q=0.5, type="highshelf"))
        decisions.append("+1.5dB @12kHz air")

    # ── Compression ──
    if has_ref:
        ref_dr = ref_target.get("dynamic_range_db", 12.0)
        dr = abs(_get_peak(analysis) - rms)
        ratio = 4.0 if dr > ref_dr + 3 else 2.5
        decisions.append(f"REF comp {ratio}:1 (DR {dr:.0f} vs ref {ref_dr:.0f})")
    else:
        ratio = 3.0
        decisions.append("Comp 3:1")

    comp = CompressorConfig(
        threshold_db=rms + 5, ratio=ratio,
        attack_ms=15.0, release_ms=120.0, makeup_db=2.0,
    )

    # ── Volume — ref-targeted ──
    rms_gap = _ref_rms_gap(analysis, ref_target)
    if has_ref:
        vol = round(-rms_gap * 0.5, 1)
        vol = max(-6.0, min(6.0, vol))
        decisions.append(f"REF vol {vol:+.1f}dB (RMS gap {rms_gap:+.1f})")
    else:
        vol = 1.0
        if energy > 50:
            vol = -1.0
        elif energy < 10:
            vol = 3.0

    # ── Sends — brain-guided or generic ──
    delay_ms = (60000.0 / bpm) / 2
    if stem_plan:
        rev_wet = getattr(stem_plan, 'reverb_wet', 0.20)
        del_wet = getattr(stem_plan, 'delay_wet', 0.12)
    else:
        rev_wet = 0.20
        del_wet = 0.12
    sends = [
        SendConfig(bus_name="reverb", amount=rev_wet, pre_fader=False),
        SendConfig(bus_name="delay", amount=del_wet, pre_fader=False),
    ]
    brain_tag = " 🧠" if stem_plan else ""
    decisions.append(f"Reverb {rev_wet:.0%} | Delay {del_wet:.0%} ({delay_ms:.0f}ms){brain_tag}")

    # ── Brain EQ overrides ──
    if stem_plan and getattr(stem_plan, 'eq_adjustments', []):
        for freq_hz, gain_db, q in stem_plan.eq_adjustments:
            eq_bands.append(EQBand(freq_hz=freq_hz, gain_db=gain_db, q=q, type="peak"))
            decisions.append(f"🧠 EQ: {gain_db:+.1f}dB @{freq_hz:.0f}Hz")

    chain = EffectChain(
        name="vocals_pro", eq_bands=eq_bands, compressor=comp,
    )

    return StemRecipe(
        name="vocals", chain=chain, volume_db=vol, pan=0.0,
        sends=sends,
        description=f"🎤 Vocals: {' | '.join(decisions)}",
    )


# ── Other/Instruments Recipe ─────────────────────────────


def build_other_recipe(
    analysis: dict[str, Any],
    bpm: float = 120.0,
    ref_target: dict[str, Any] | None = None,
    ear_data: dict[str, Any] | None = None,
    stem_plan: Any | None = None,
) -> StemRecipe:
    """Build processing chain for 'other' stem (instruments, synths, FX).

    Ref-targeted: air, definition, and volume derived from gaps.
    """
    freq = _get_freq_balance(analysis)
    rms = _get_rms(analysis)
    decisions: list[str] = []
    has_ref = ref_target is not None

    # ── EQ ──
    eq_bands: list[EQBand] = []

    eq_bands.append(EQBand(freq_hz=100.0, gain_db=-10.0, q=0.7, type="highshelf"))
    decisions.append("HPF 100Hz")

    # Anti-mud — ref-targeted or generic
    mid_gap = _ref_freq_gap(analysis, ref_target, "mid")
    if has_ref and mid_gap > 3:
        mud_cut = round(-mid_gap * 0.4, 1)
        mud_cut = max(-6.0, min(0.0, mud_cut))
        eq_bands.append(EQBand(freq_hz=350.0, gain_db=mud_cut, q=1.2, type="peak"))
        decisions.append(f"REF: {mud_cut:.1f}dB @350Hz (mid {mid_gap:+.1f}% over ref)")
    else:
        mud_cut = -3.0 if freq["mid"] > 50 else -1.5
        eq_bands.append(EQBand(freq_hz=350.0, gain_db=mud_cut, q=1.2, type="peak"))
        decisions.append(f"{mud_cut}dB @350Hz anti-mud")

    # Air/definition — ref-targeted or generic
    high_gap = _ref_freq_gap(analysis, ref_target, "high")
    if has_ref and abs(high_gap) > 3:
        air_boost = round(-high_gap * 0.4, 1)
        air_boost = max(-3.0, min(5.0, air_boost))
        eq_bands.append(EQBand(freq_hz=8000.0, gain_db=air_boost, q=0.6, type="highshelf"))
        decisions.append(f"REF: {air_boost:+.1f}dB @8kHz air (gap {high_gap:+.1f}%)")
    else:
        if freq["high"] < 20:
            eq_bands.append(EQBand(freq_hz=5000.0, gain_db=2.0, q=0.8, type="peak"))
            decisions.append("+2dB @5kHz definition")
        eq_bands.append(EQBand(freq_hz=10000.0, gain_db=1.0, q=0.5, type="highshelf"))
        decisions.append("+1dB @10kHz air")

    # ── Compression ──
    if has_ref:
        ref_dr = ref_target.get("dynamic_range_db", 12.0)
        dr = abs(_get_peak(analysis) - rms)
        ratio = 3.5 if dr > ref_dr + 3 else 2.0
        decisions.append(f"REF comp {ratio}:1 (DR {dr:.0f} vs ref {ref_dr:.0f})")
    else:
        ratio = 2.5
        decisions.append("Comp 2.5:1")

    comp = CompressorConfig(
        threshold_db=rms + 8, ratio=ratio,
        attack_ms=20.0, release_ms=150.0, makeup_db=1.5,
    )

    # ── Volume — ref-targeted ──
    rms_gap = _ref_rms_gap(analysis, ref_target)
    if has_ref:
        vol = round(-rms_gap * 0.5, 1)
        vol = max(-6.0, min(6.0, vol))
        decisions.append(f"REF vol {vol:+.1f}dB (RMS gap {rms_gap:+.1f})")
    else:
        vol = -2.0
        if rms > -12:
            vol = -4.0
        elif rms < -30:
            vol = 0.0

    # ── Pan — brain-guided or default ──
    if stem_plan and getattr(stem_plan, 'stereo_width', 0):
        # Wider instruments get more pan spread
        sw = stem_plan.stereo_width
        pan = round(min(0.5, (sw - 1.0) * 0.5), 2) if sw > 1.0 else 0.0
        decisions.append(f"🧠 Pan: {'R' if pan > 0 else 'C'}{abs(pan)*100:.0f}% (width={sw:.1f})")
    else:
        pan = 0.15
        decisions.append("Pan: R15%")

    # ── Sends — brain-guided or generic ──
    if stem_plan:
        rev_wet = getattr(stem_plan, 'reverb_wet', 0.25)
        del_wet = getattr(stem_plan, 'delay_wet', 0.08)
    else:
        rev_wet = 0.25
        del_wet = 0.08
    sends = [
        SendConfig(bus_name="reverb", amount=rev_wet, pre_fader=False),
        SendConfig(bus_name="delay", amount=del_wet, pre_fader=False),
    ]
    brain_tag = " 🧠" if stem_plan else ""
    decisions.append(f"Reverb {rev_wet:.0%} | Delay {del_wet:.0%}{brain_tag}")

    chain = EffectChain(
        name="other_pro", eq_bands=eq_bands, compressor=comp,
    )

    return StemRecipe(
        name="other", chain=chain, volume_db=vol, pan=pan,
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
    ref_targets: dict[str, dict[str, Any]] | None = None,
    stem_plan: Any | None = None,
) -> StemRecipe:
    """Build the optimal processing recipe for a stem.

    Args:
        stem_name: Name of the stem (drums, bass, vocals, other)
        stem_analysis: Per-stem analysis data from EAR stage
        bpm: Track BPM for tempo-synced effects
        ear_data: Full EAR analysis data (optional, for context)
        ref_targets: Per-stem reference targets from DNA Bank (optional)
        stem_plan: Optional StemPlan from DNABrain (overrides generic params)

    Returns:
        StemRecipe with chain, volume, pan, sends, and description.
        When stem_plan is provided, EQ/comp/FX are brain-guided.
    """
    ref_target = (ref_targets or {}).get(stem_name)
    builder = _STEM_BUILDERS.get(stem_name, build_other_recipe)
    if stem_name == "other":
        return builder(stem_analysis, bpm=bpm, ref_target=ref_target, ear_data=ear_data, stem_plan=stem_plan)
    return builder(stem_analysis, bpm=bpm, ref_target=ref_target, stem_plan=stem_plan)


def build_all_recipes(
    stem_analyses: dict[str, dict[str, Any]],
    bpm: float = 120.0,
    ear_data: dict[str, Any] | None = None,
    ref_targets: dict[str, dict[str, Any]] | None = None,
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
            name, analysis, bpm=bpm, ear_data=ear_data,
            ref_targets=ref_targets,
        )
    return recipes


# ── Smart FX Enhancer (from Stem Decisions) ──────────────


def enhance_recipe_with_smart_fx(
    recipe: StemRecipe,
    extra_fx: list[str],
    bpm: float = 120.0,
) -> StemRecipe:
    """Apply additional FX from stem decision intelligence.

    The stem decision engine can recommend extra effects beyond what
    the recipe builder provides.  This function activates them.

    Args:
        recipe: Existing recipe to enhance.
        extra_fx: List of FX tags from StemDecision.extra_fx.
        bpm: Track BPM for tempo-synced effects.

    Returns:
        Enhanced StemRecipe (mutated in-place and returned).
    """
    from auralis.hands.effects import ChorusConfig

    if not extra_fx:
        return recipe

    chain = recipe.chain
    fx_notes: list[str] = []

    for fx in extra_fx:
        if fx == "chorus" and chain.chorus is None:
            chain.chorus = ChorusConfig(
                rate_hz=0.5, depth=0.3, mix=0.25,
            )
            fx_notes.append("FX: +chorus (stereo width)")

        elif fx == "sidechain" and chain.sidechain is None:
            chain.sidechain = SidechainConfig(
                threshold_db=-20.0, ratio=4.0,
                attack_ms=5.0, release_ms=100.0,
                frequency=bpm / 60.0,
            )
            fx_notes.append("FX: +sidechain (pump)")

        elif fx == "saturation" and chain.distortion is None:
            chain.distortion = DistortionConfig(
                drive=0.15, mix=0.3,
            )
            fx_notes.append("FX: +saturation (warmth)")

        elif fx == "de_mud_eq":
            # Add a mid-cut EQ band to clean up muddiness
            if chain.eq_bands is None:
                chain.eq_bands = []
            chain.eq_bands.append(
                EQBand(freq_hz=400.0, gain_db=-3.0, q=1.5)
            )
            fx_notes.append("FX: -3dB @400Hz (de-mud)")

        elif fx == "air_boost":
            if chain.eq_bands is None:
                chain.eq_bands = []
            chain.eq_bands.append(
                EQBand(freq_hz=12000.0, gain_db=2.5, q=0.7)
            )
            fx_notes.append("FX: +2.5dB @12kHz (air)")

        elif fx == "compress_harder":
            if chain.compressor:
                chain.compressor.ratio = min(chain.compressor.ratio + 2.0, 12.0)
                chain.compressor.threshold_db = max(chain.compressor.threshold_db - 3.0, -40.0)
                fx_notes.append("FX: compress harder (ratio+2, thresh-3)")

        elif fx == "reverb_boost":
            # Increase reverb send
            for send in recipe.sends:
                if send.bus_name == "reverb":
                    send.amount = min(send.amount + 0.15, 0.6)
                    fx_notes.append(f"FX: reverb send +0.15 (→{send.amount:.2f})")
                    break

    if fx_notes:
        recipe.description += " | " + " | ".join(fx_notes)

    return recipe

