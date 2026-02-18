"""AURALIS DNA Brain — Multi-dimensional reasoning engine.

Scores, weighs, and reasons across ALL musical dimensions using deep
DNA extracted from reference tracks.  Instead of hardcoded rules
(BPM < 130 → bass_808), the brain evaluates candidate options across
DNA match, context fit, and confidence to produce optimal decisions
for every stem, every effect, and the mastering chain.

Usage:
    brain = DNABrain()
    report = brain.think(deep_profile, stem_analysis, gap_report, ear_data)
    # report.stem_plans["bass"].patch  → best patch
    # report.master_plan.drive         → optimal drive
    # report.reasoning_chain           → WHY each decision was made
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

logger = structlog.get_logger()


# ── Output Data Structures ───────────────────────────────


@dataclass
class StemPlan:
    """Complete processing plan for one stem — patch, style, FX, EQ, reasoning."""

    stem_name: str
    patch: str = ""
    style: str = ""
    fx_chain: list[str] = field(default_factory=list)
    eq_adjustments: list[tuple[float, float, float]] = field(
        default_factory=list
    )  # (freq, gain_db, Q)
    compression: dict[str, float] = field(default_factory=dict)
    volume_db: float = 0.0
    sidechain: bool = False
    sidechain_depth: float = 0.0
    reverb_wet: float = 0.0
    delay_wet: float = 0.0
    saturation_drive: float = 0.0
    stereo_width: float = 1.0
    use_organic: bool = False
    organic_category: str = ""
    ai_prompt_hints: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem_name": self.stem_name,
            "patch": self.patch,
            "style": self.style,
            "fx_chain": self.fx_chain,
            "eq_adjustments": self.eq_adjustments,
            "compression": self.compression,
            "volume_db": self.volume_db,
            "sidechain": self.sidechain,
            "sidechain_depth": self.sidechain_depth,
            "reverb_wet": self.reverb_wet,
            "delay_wet": self.delay_wet,
            "saturation_drive": self.saturation_drive,
            "stereo_width": self.stereo_width,
            "use_organic": self.use_organic,
            "organic_category": self.organic_category,
            "ai_prompt_hints": self.ai_prompt_hints,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
        }


@dataclass
class MasterPlan:
    """DNA-guided mastering parameters."""

    target_lufs: float = -8.0
    mid_eq_bands: list[tuple[float, float, float]] = field(
        default_factory=list
    )  # (freq, gain, Q)
    side_eq_bands: list[tuple[float, float, float]] = field(
        default_factory=list
    )
    drive: float = 1.5
    width: float = 1.3
    ceiling_db: float = -1.0
    comp_low: dict[str, float] = field(default_factory=dict)
    comp_mid: dict[str, float] = field(default_factory=dict)
    comp_high: dict[str, float] = field(default_factory=dict)
    reasoning: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_lufs": self.target_lufs,
            "mid_eq_bands": self.mid_eq_bands,
            "side_eq_bands": self.side_eq_bands,
            "drive": self.drive,
            "width": self.width,
            "ceiling_db": self.ceiling_db,
            "comp_low": self.comp_low,
            "comp_mid": self.comp_mid,
            "comp_high": self.comp_high,
            "reasoning": self.reasoning,
        }


@dataclass
class BrainReport:
    """Full reasoning output — stem plans, master plan, interactions, chain."""

    stem_plans: dict[str, StemPlan] = field(default_factory=dict)
    master_plan: MasterPlan = field(default_factory=MasterPlan)
    interaction_log: list[str] = field(default_factory=list)
    reasoning_chain: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem_plans": {k: v.to_dict() for k, v in self.stem_plans.items()},
            "master_plan": self.master_plan.to_dict(),
            "interaction_log": self.interaction_log,
            "reasoning_chain": self.reasoning_chain,
        }


# ── Evidence Gathering ───────────────────────────────────


@dataclass
class Evidence:
    """Aggregated input from all DNA sources for the brain to reason over."""

    # From deep profile
    percussion_palette: dict[str, int] = field(default_factory=dict)
    percussion_dominant: list[str] = field(default_factory=list)
    percussion_density: float = 0.0
    bass_type: str = ""
    bass_types_found: dict[str, int] = field(default_factory=dict)
    instruments: list[str] = field(default_factory=list)
    instrument_freq: dict[str, int] = field(default_factory=dict)
    fx_palette: list[str] = field(default_factory=list)
    fx_freq: dict[str, int] = field(default_factory=dict)
    vocal_effects: list[str] = field(default_factory=list)
    vocal_freq: dict[str, int] = field(default_factory=dict)
    vocal_energy: float = 0.0  # 0-100 vocal presence level
    sidechain_ratio: float = 0.0
    avg_sections: float = 0.0
    ref_count: int = 0
    deep_count: int = 0
    # From EAR
    bpm: float = 120.0
    key: str = "C"
    scale: str = "minor"
    lufs: float = -14.0
    duration: float = 180.0
    # From master averages in deep profile
    ref_lufs: float = -14.0
    ref_bpm: float = 120.0
    dominant_key: str = ""


def _gather_evidence(
    deep_profile: dict[str, Any] | None,
    ear_data: dict[str, Any] | None,
) -> Evidence:
    """Merge all data sources into a single Evidence object."""
    ev = Evidence()

    if deep_profile:
        perc = deep_profile.get("percussion", {})
        ev.percussion_palette = perc.get("palette", {})
        ev.percussion_dominant = perc.get("dominant", [])
        ev.percussion_density = perc.get("avg_density", 0.0)

        bass = deep_profile.get("bass", {})
        ev.bass_type = bass.get("dominant_type", "")
        ev.bass_types_found = bass.get("types_found", {})

        inst = deep_profile.get("instruments", {})
        ev.instruments = inst.get("palette", [])
        ev.instrument_freq = inst.get("frequency", {})

        fx = deep_profile.get("fx", {})
        ev.fx_palette = fx.get("palette", [])
        ev.fx_freq = fx.get("frequency", {})

        voc = deep_profile.get("vocals", {})
        ev.vocal_effects = voc.get("effects", [])
        ev.vocal_freq = voc.get("frequency", {})

        arr = deep_profile.get("arrangement", {})
        ev.sidechain_ratio = arr.get("sidechain_ratio", 0.0)
        ev.avg_sections = arr.get("avg_sections", 0.0)

        ev.ref_count = deep_profile.get("reference_count", 0)
        ev.deep_count = deep_profile.get("deep_count", 0)

        master = deep_profile.get("master", {})
        ev.ref_lufs = master.get("lufs", -14.0)
        ev.ref_bpm = master.get("bpm", 120.0)
        ev.dominant_key = deep_profile.get("dominant_key", "")

    if ear_data:
        ev.bpm = float(ear_data.get("tempo", ear_data.get("bpm", 120.0)))
        ev.key = str(ear_data.get("key", "C"))
        ev.scale = str(ear_data.get("scale", "minor"))
        ev.lufs = float(ear_data.get("integrated_lufs", -14.0))
        ev.duration = float(ear_data.get("duration", 180.0))

    return ev


# ── Scoring Utilities ────────────────────────────────────


def _confidence(ev: Evidence) -> float:
    """0-1 confidence based on how many refs have deep DNA."""
    if ev.ref_count == 0:
        return 0.0
    return min(1.0, ev.deep_count / max(ev.ref_count, 1))


def _score_candidate(
    dna_match: float,
    context_fit: float,
    confidence: float,
    interaction_bonus: float = 0.0,
    memory_bonus: float = 0.0,
) -> float:
    """Weighted score across dimensions. All inputs 0-100 (memory_bonus 0-20)."""
    w_dna = 0.40
    w_ctx = 0.25
    w_conf = 0.15
    w_int = 0.10
    w_mem = 0.10  # Session memory weight
    return (
        dna_match * w_dna
        + context_fit * w_ctx
        + confidence * w_conf
        + interaction_bonus * w_int
        + memory_bonus * w_mem * 5  # Scale 0-20 → 0-100 equivalent
    )


def _select_with_creativity(
    candidates: dict[str, float],
    temperature: float = 0.3,
) -> str:
    """Select from candidates with controlled stochasticity.

    Instead of always picking the highest score (deterministic),
    uses softmax-weighted sampling so near-equal candidates have
    a chance of being selected — introducing creative surprise.

    temperature=0.0 → deterministic (always max)
    temperature=0.3 → light creativity (top 2-3 compete)
    temperature=1.0 → high randomness (all compete equally)
    """
    import random
    import math

    if not candidates:
        return ""

    if temperature <= 0.01:
        return max(candidates, key=candidates.get)  # type: ignore[arg-type]

    # Softmax with temperature
    names = list(candidates.keys())
    scores = [candidates[n] for n in names]
    max_score = max(scores)

    # Shift scores for numerical stability
    exp_scores = []
    for s in scores:
        exponent = (s - max_score) / max(temperature * max_score, 1.0)
        exp_scores.append(math.exp(min(exponent, 50)))  # cap to prevent overflow

    total = sum(exp_scores)
    if total <= 0:
        return names[0]

    probabilities = [e / total for e in exp_scores]

    # Weighted random choice
    r = random.random()
    cumulative = 0.0
    for name, prob in zip(names, probabilities):
        cumulative += prob
        if r <= cumulative:
            return name

    return names[-1]  # fallback


# ── Emotional Arc Engine ─────────────────────────────────


class EmotionalArc:
    """Tracks energy flow across sections to create musical contrast.

    A great production isn't just correct — it BREATHES. High-energy
    sections need contrast with quieter moments. The arc engine
    adjusts processing parameters to reinforce the emotional narrative:

    - Breakdowns: wider reverb, less compression, more space
    - Drops: tighter compression, more drive, narrower stereo
    - Builds: increasing energy, rising filter cutoffs
    - Intros/Outros: gentler processing, fade dynamics
    """

    # Section type → relative energy level (0-1)
    ENERGY_MAP: dict[str, float] = {
        "intro": 0.3,
        "verse": 0.5,
        "build": 0.7,
        "chorus": 0.8,
        "drop": 1.0,
        "breakdown": 0.2,
        "bridge": 0.4,
        "outro": 0.25,
    }

    def __init__(self, section_type: str = "drop") -> None:
        self.section_type = section_type.lower()
        self.energy = self.ENERGY_MAP.get(self.section_type, 0.6)

    def adjust_stem_plan(
        self, plan: "StemPlan", stem_name: str
    ) -> list[str]:
        """Adjust a stem plan based on emotional context. Returns reasoning."""
        adjustments: list[str] = []

        if self.energy <= 0.3:  # Low energy (breakdown, intro, outro)
            # Wider, more spacious, less compressed
            if plan.compression:
                plan.compression["ratio"] = min(
                    plan.compression.get("ratio", 2.0), 2.0
                )
                adjustments.append(
                    f"ARC: {self.section_type} → gentle compression (ratio≤2:1)"
                )
            if stem_name in ("vocals", "other"):
                plan.stereo_width = min(plan.stereo_width + 0.3, 2.0)
                adjustments.append(
                    f"ARC: {self.section_type} → wider stereo ({plan.stereo_width:.1f})"
                )
            if "hall_reverb" not in plan.fx_chain and stem_name != "bass":
                plan.fx_chain.append("hall_reverb")
                adjustments.append(
                    f"ARC: {self.section_type} → added hall reverb for space"
                )

        elif self.energy >= 0.8:  # High energy (drop, chorus)
            # Tighter, more aggressive, focused
            if plan.compression:
                plan.compression["ratio"] = max(
                    plan.compression.get("ratio", 4.0), 4.0
                )
                plan.compression["attack_ms"] = min(
                    plan.compression.get("attack_ms", 10.0), 10.0
                )
                adjustments.append(
                    f"ARC: {self.section_type} → aggressive compression (ratio≥4:1, attack≤10ms)"
                )
            if stem_name == "bass" and not plan.sidechain:
                plan.sidechain = True
                adjustments.append(
                    f"ARC: {self.section_type} → bass sidechain for punch"
                )

        elif 0.6 <= self.energy < 0.8:  # Medium-high (build, verse 2)
            # Building energy — moderate processing
            if "saturation" not in plan.fx_chain and stem_name in ("drums", "bass"):
                plan.fx_chain.append("saturation")
                adjustments.append(
                    f"ARC: {self.section_type} → added saturation for warmth"
                )

        return adjustments

    def adjust_master_plan(self, master: "MasterPlan") -> list[str]:
        """Adjust mastering parameters based on emotional context."""
        adjustments: list[str] = []

        if self.energy <= 0.3:
            # Breakdowns: less drive, more width
            master.drive = min(master.drive, 1.0)
            master.width = min(master.width + 0.2, 2.0)
            adjustments.append(
                f"ARC: {self.section_type} → master: less drive, wider image"
            )
        elif self.energy >= 0.8:
            # Drops: more drive, controlled width
            master.drive = max(master.drive, 2.0)
            adjustments.append(
                f"ARC: {self.section_type} → master: drive≥2.0 for impact"
            )

        return adjustments


# ── Stem Thinkers ───────────────────────────────────────


def _think_drums(ev: Evidence) -> StemPlan:
    """Reason about drum processing based on percussion DNA."""
    plan = StemPlan(stem_name="drums")
    reasons: list[str] = []

    # ── Pattern style scoring ──
    total_hits = sum(ev.percussion_palette.values()) if ev.percussion_palette else 0
    conf = _confidence(ev) * 100

    # Score styles based on what percussion elements dominate
    kick_pct = ev.percussion_palette.get("Kick", 0) / max(total_hits, 1) * 100
    hat_pct = (
        ev.percussion_palette.get("Hi-Hat", 0)
        + ev.percussion_palette.get("Open Hi-Hat", 0)
    ) / max(total_hits, 1) * 100
    clap_pct = (
        ev.percussion_palette.get("Clap", 0)
        + ev.percussion_palette.get("Snare", 0)
    ) / max(total_hits, 1) * 100
    shaker_pct = ev.percussion_palette.get("Shaker", 0) / max(total_hits, 1) * 100

    candidates: dict[str, float] = {}

    # Four on floor: kick-heavy, steady
    candidates["four_on_floor"] = _score_candidate(
        dna_match=kick_pct * 2,  # rewards kick dominance
        context_fit=80 if ev.bpm >= 118 and ev.bpm <= 132 else 40,
        confidence=conf,
    )

    # Breakbeat: balanced kick + snare
    candidates["breakbeat"] = _score_candidate(
        dna_match=min(kick_pct, clap_pct) * 2,  # rewards balance
        context_fit=80 if ev.bpm >= 100 and ev.bpm <= 140 else 40,
        confidence=conf,
    )

    # Afro groove: hat/shaker driven, organic feel
    candidates["afro_groove"] = _score_candidate(
        dna_match=(hat_pct + shaker_pct) * 1.5,
        context_fit=80 if ev.bpm >= 115 and ev.bpm <= 130 else 40,
        confidence=conf,
    )

    # Trap: hat rolls, sparse kick
    candidates["trap"] = _score_candidate(
        dna_match=hat_pct * 2 if kick_pct < 25 else hat_pct,
        context_fit=80 if ev.bpm >= 65 and ev.bpm <= 90 else 40,
        confidence=conf,
    )

    # Minimal: sparse, few elements
    candidates["minimal"] = _score_candidate(
        dna_match=80 if total_hits < 100 else 30,
        context_fit=60 if ev.bpm < 100 else 40,
        confidence=conf,
    )

    best_style = _select_with_creativity(candidates, temperature=0.3)
    plan.style = best_style
    plan.confidence = candidates[best_style] / 100

    reasons.append(
        f"Pattern style: '{best_style}' scored {candidates[best_style]:.0f} "
        f"(beat: {', '.join(f'{s}={sc:.0f}' for s, sc in sorted(candidates.items(), key=lambda x: -x[1])[:3])})"
    )

    # ── Compression from density ──
    if ev.percussion_density > 5.0:
        plan.compression = {
            "threshold_db": -16.0,
            "ratio": 4.5,
            "attack_ms": 8.0,
            "release_ms": 60.0,
        }
        reasons.append(
            f"High density ({ev.percussion_density:.1f}/s) → fast compression (8ms attack, 4.5:1)"
        )
    elif ev.percussion_density > 2.0:
        plan.compression = {
            "threshold_db": -18.0,
            "ratio": 3.5,
            "attack_ms": 12.0,
            "release_ms": 80.0,
        }
        reasons.append(
            f"Medium density ({ev.percussion_density:.1f}/s) → moderate compression (12ms attack)"
        )
    else:
        plan.compression = {
            "threshold_db": -20.0,
            "ratio": 2.5,
            "attack_ms": 18.0,
            "release_ms": 100.0,
        }
        reasons.append("Low/unknown density → gentle compression (18ms, 2.5:1)")

    # ── EQ from dominant percussion ──
    if "Hi-Hat" in ev.percussion_dominant or "Open Hi-Hat" in ev.percussion_dominant:
        plan.eq_adjustments.append((8000.0, 2.0, 0.8))
        reasons.append("Hi-hat dominant → +2dB shelf at 8kHz for presence")
    if "Kick" in ev.percussion_dominant:
        plan.eq_adjustments.append((60.0, 2.5, 1.2))
        plan.eq_adjustments.append((300.0, -2.0, 1.5))
        reasons.append("Kick dominant → +2.5dB at 60Hz, -2dB at 300Hz (clean punch)")
    if "Shaker" in ev.percussion_dominant:
        plan.eq_adjustments.append((10000.0, 1.5, 0.6))
        reasons.append("Shaker dominant → +1.5dB air at 10kHz")

    # ── Saturation ──
    plan.saturation_drive = 3.0 if ev.percussion_density > 4.0 else 2.0
    reasons.append(
        f"Saturation drive={plan.saturation_drive:.0f}dB "
        f"({'aggressive' if plan.saturation_drive > 2.5 else 'gentle'} based on density)"
    )

    # ── Organic detection: congas, shakers, bongos in percussion DNA ──
    organic_keywords = {"conga", "bongo", "shaker", "tambourine", "djembe",
                        "cajon", "timbale", "guiro", "maracas"}
    perc_lower = {p.lower() for p in ev.percussion_dominant}
    organic_hits = perc_lower & organic_keywords
    if organic_hits:
        plan.use_organic = True
        plan.organic_category = sorted(organic_hits)[0]  # primary organic type
        reasons.append(
            f"🌿 Organic percussion detected: {', '.join(organic_hits)} → prefer real samples"
        )
    # Also check palette for organic instruments
    palette_lower = {k.lower() for k in ev.percussion_palette.keys()}
    palette_organic = palette_lower & organic_keywords
    if palette_organic and not plan.use_organic:
        plan.use_organic = True
        plan.organic_category = sorted(palette_organic)[0]
        reasons.append(
            f"🌿 Organic in palette: {', '.join(palette_organic)} → prefer real samples"
        )

    # ── AI generation hints ──
    dominant_str = ", ".join(ev.percussion_dominant[:3]) if ev.percussion_dominant else "standard"
    density_desc = "sparse" if ev.percussion_density < 3 else "medium" if ev.percussion_density < 6 else "dense"
    plan.ai_prompt_hints = [
        f"{plan.style} drum pattern",
        f"dominant elements: {dominant_str}",
        f"{ev.bpm:.0f} BPM",
        f"{density_desc} rhythm, {ev.percussion_density:.1f} hits/sec",
        f"{'loop-ready, seamless' if ev.bpm >= 100 else 'groove-based'}",
    ]

    plan.reasoning = reasons
    return plan


def _think_bass(ev: Evidence) -> StemPlan:
    """Reason about bass processing based on bass DNA."""
    plan = StemPlan(stem_name="bass")
    reasons: list[str] = []

    # ── Patch scoring ──
    conf = _confidence(ev) * 100
    bass_type = ev.bass_type.lower() if ev.bass_type else ""
    types_found = {k.lower(): v for k, v in ev.bass_types_found.items()}
    total_refs_with_bass = sum(types_found.values()) if types_found else 0

    candidates: dict[str, float] = {}

    # Score each available patch against DNA evidence
    patch_map = {
        "bass_808": ["808", "sub", "sub bass"],
        "acid_303": ["acid", "303", "squelchy"],
        "sub_bass": ["sub", "sub bass", "deep"],
        "reese": ["reese", "dark", "aggressive"],
        "pluck": ["pluck", "staccato"],
    }

    for patch_name, keywords in patch_map.items():
        # DNA match: does the ref bass type contain relevant keywords?
        dna_match = 0.0
        for kw in keywords:
            if kw in bass_type:
                dna_match = 80.0
            for found_type, count in types_found.items():
                if kw in found_type:
                    dna_match = max(dna_match, (count / max(total_refs_with_bass, 1)) * 100)

        # Context: does BPM suit this patch?
        ctx = 50.0
        if patch_name == "bass_808":
            ctx = 80 if ev.bpm < 130 else 40
        elif patch_name == "acid_303":
            ctx = 80 if ev.bpm >= 125 else 40
        elif patch_name == "sub_bass":
            ctx = 70  # works at most tempos
        elif patch_name == "reese":
            ctx = 80 if ev.bpm >= 140 else 50
        elif patch_name == "pluck":
            ctx = 70 if ev.bpm >= 110 else 50

        candidates[patch_name] = _score_candidate(dna_match, ctx, conf)

    best_patch = max(candidates, key=candidates.get)  # type: ignore[arg-type]
    plan.patch = best_patch
    plan.confidence = candidates[best_patch] / 100

    top3 = sorted(candidates.items(), key=lambda x: -x[1])[:3]
    reasons.append(
        f"Patch: '{best_patch}' scored {candidates[best_patch]:.0f} "
        f"(DNA says '{ev.bass_type}', "
        f"scores: {', '.join(f'{p}={s:.0f}' for p, s in top3)})"
    )

    # ── Sidechain from DNA ──
    if ev.sidechain_ratio > 0.5:
        plan.sidechain = True
        plan.sidechain_depth = min(0.9, ev.sidechain_ratio)
        reasons.append(
            f"Sidechain enabled (ref ratio={ev.sidechain_ratio:.0%}) → "
            f"depth={plan.sidechain_depth:.0%}"
        )
    elif ev.sidechain_ratio > 0.2:
        plan.sidechain = True
        plan.sidechain_depth = 0.4
        reasons.append(f"Light sidechain (ref ratio={ev.sidechain_ratio:.0%})")

    # ── EQ from bass type ──
    if "sub" in bass_type or "808" in bass_type:
        plan.eq_adjustments.append((55.0, 4.0, 1.5))
        plan.eq_adjustments.append((250.0, -3.0, 1.2))
        reasons.append("Sub bass type → +4dB at 55Hz, -3dB anti-mud at 250Hz")
    elif "acid" in bass_type or "303" in bass_type:
        plan.eq_adjustments.append((800.0, 3.0, 1.0))
        plan.eq_adjustments.append((200.0, -2.0, 1.5))
        reasons.append("Acid bass type → +3dB resonance at 800Hz")
    else:
        plan.eq_adjustments.append((80.0, 2.0, 1.2))
        reasons.append("General bass → +2dB at 80Hz")

    # ── Saturation based on bass type ──
    plan.saturation_drive = 4.0 if "sub" in bass_type else 3.0
    reasons.append(f"Saturation: {plan.saturation_drive:.0f}dB drive (tube warmth)")

    # ── Pattern style ──
    if ev.percussion_density > 4.0 and plan.sidechain:
        plan.style = "staccato"
        reasons.append(
            "High percussion density + sidechain → staccato bass (leave space)"
        )
    elif ev.bpm >= 135:
        plan.style = "syncopated"
        reasons.append(f"High BPM ({ev.bpm:.0f}) → syncopated bass")
    elif ev.bpm >= 115:
        plan.style = "walking"
    else:
        plan.style = "simple"

    # ── Compression ──
    plan.compression = {
        "threshold_db": -16.0,
        "ratio": 4.0,
        "attack_ms": 15.0,
        "release_ms": 120.0,
    }

    # ── AI prompt hints ──
    bass_desc = ev.bass_type or "bass"
    sidechain_desc = "with sidechain pumping" if plan.sidechain else "steady groove"
    plan.ai_prompt_hints = [
        f"{bass_desc} bass line",
        f"{plan.style} pattern",
        f"{ev.bpm:.0f} BPM, key of {ev.key} {ev.scale}",
        sidechain_desc,
        f"{'deep sub-bass, 808-style' if 'sub' in bass_desc.lower() else 'mid-range bass, defined tone'}",
        "clean low-end, no mud",
    ]

    plan.reasoning = reasons
    return plan


def _think_vocals(ev: Evidence) -> StemPlan:
    """Reason about vocal processing based on vocal DNA."""
    plan = StemPlan(stem_name="vocals")
    reasons: list[str] = []
    conf = _confidence(ev) * 100

    fx_lower = [e.lower() for e in ev.vocal_effects]

    # ── Reverb from DNA ──
    has_reverb = any("reverb" in e for e in fx_lower)
    if has_reverb:
        plan.reverb_wet = 0.25
        plan.fx_chain.append("reverb")
        reasons.append("Reverb detected in refs → wet=0.25")
    else:
        plan.reverb_wet = 0.10
        reasons.append("No reverb in refs → minimal wet=0.10 (natural space only)")

    # ── Delay from DNA ──
    has_delay = any("delay" in e or "echo" in e for e in fx_lower)
    if has_delay:
        plan.delay_wet = 0.15
        plan.fx_chain.append("delay")
        reasons.append("Delay detected in refs → 1/4 note, wet=0.15")
    else:
        plan.delay_wet = 0.0
        reasons.append("No delay in refs → delay bypassed")

    # ── Auto-tune / pitch correction ──
    has_autotune = any("auto" in e or "tune" in e or "pitch" in e for e in fx_lower)
    if has_autotune:
        plan.fx_chain.append("pitch_correction")
        reasons.append("Auto-tune/pitch correction detected in refs → enabled")

    # ── Chorus/doubling ──
    has_chorus = any("chorus" in e or "double" in e or "unison" in e for e in fx_lower)
    if has_chorus:
        plan.fx_chain.append("chorus")
        plan.stereo_width = 1.3
        reasons.append("Chorus/doubling detected → width=1.3")
    else:
        plan.stereo_width = 1.0
        reasons.append("No chorus in refs → vocals stay centered")

    # ── EQ: presence and air ──
    plan.eq_adjustments.append((3500.0, 2.5, 1.2))  # presence
    plan.eq_adjustments.append((12000.0, 1.5, 0.6))  # air
    plan.eq_adjustments.append((250.0, -2.0, 1.5))  # mud cut
    reasons.append("Standard vocal EQ: +2.5dB presence, +1.5dB air, -2dB mud")

    # ── Compression: tighter if many vocal regions ──
    plan.compression = {
        "threshold_db": -18.0,
        "ratio": 3.5,
        "attack_ms": 10.0,
        "release_ms": 80.0,
    }

    # ── De-ess ──
    plan.fx_chain.append("de_ess")
    reasons.append("De-esser always active on vocals (6kHz, threshold -8dB)")

    plan.confidence = conf / 100

    # ── AI prompt hints ──
    fx_str = ", ".join(ev.vocal_effects[:3]) if ev.vocal_effects else "clean"
    plan.ai_prompt_hints = [
        f"vocal processing: {fx_str}",
        f"{'wet, spacious, reverb-heavy' if has_reverb else 'dry, intimate, close-mic'} vocal style",
        f"{'chopped vocal textures' if ev.vocal_energy < 20 else 'lead vocal, clear diction'}",
    ]

    plan.reasoning = reasons
    return plan


def _think_other(ev: Evidence) -> StemPlan:
    """Reason about instruments/other stem processing."""
    plan = StemPlan(stem_name="other")
    reasons: list[str] = []
    conf = _confidence(ev) * 100

    inst_lower = [i.lower() for i in ev.instruments]

    # ── Preset scoring based on instrument palette ──
    candidates: dict[str, float] = {}

    has_pad = any("pad" in i for i in inst_lower)
    has_pluck = any("pluck" in i for i in inst_lower)
    has_strings = any("string" in i for i in inst_lower)
    has_brass = any("brass" in i or "horn" in i for i in inst_lower)
    has_synth = any("synth" in i or "lead" in i for i in inst_lower)

    candidates["pad_warm"] = _score_candidate(
        dna_match=80 if has_pad else 20,
        context_fit=70,
        confidence=conf,
    )
    candidates["pluck"] = _score_candidate(
        dna_match=80 if has_pluck else 20,
        context_fit=70 if ev.bpm >= 110 else 50,
        confidence=conf,
    )
    candidates["supersaw"] = _score_candidate(
        dna_match=80 if has_synth else 30,
        context_fit=80 if ev.bpm >= 125 else 50,
        confidence=conf,
    )
    candidates["strings"] = _score_candidate(
        dna_match=80 if has_strings else 15,
        context_fit=60,
        confidence=conf,
    )

    best = max(candidates, key=candidates.get)  # type: ignore[arg-type]
    plan.patch = best
    plan.confidence = candidates[best] / 100

    top3 = sorted(candidates.items(), key=lambda x: -x[1])[:3]
    reasons.append(
        f"Preset: '{best}' scored {candidates[best]:.0f} "
        f"(instruments: {', '.join(ev.instruments[:4]) or 'unknown'}, "
        f"scores: {', '.join(f'{p}={s:.0f}' for p, s in top3)})"
    )

    # ── Width: pads need stereo, plucks less ──
    if has_pad:
        plan.stereo_width = 1.4
        plan.fx_chain.append("chorus")
        reasons.append("Pads detected → wide stereo (1.4) + chorus")
    else:
        plan.stereo_width = 1.1

    # ── Reverb for spacious instruments ──
    if has_pad or has_strings:
        plan.reverb_wet = 0.35
        plan.fx_chain.append("reverb")
        reasons.append("Spacious instruments → reverb wet=0.35")
    else:
        plan.reverb_wet = 0.15

    # ── EQ: avoid vocal masking ──
    plan.eq_adjustments.append((2500.0, -2.0, 1.0))  # cut presence
    plan.eq_adjustments.append((12000.0, 1.5, 0.6))  # add air
    reasons.append("EQ: -2dB at 2.5kHz (vocal space), +1.5dB air")

    # ── Compression: gentle for instruments ──
    plan.compression = {
        "threshold_db": -20.0,
        "ratio": 2.0,
        "attack_ms": 20.0,
        "release_ms": 150.0,
    }
    reasons.append("Comp: 2:1, slow attack (preserve transients)")

    # ── Saturation: subtle warmth ──
    plan.saturation_drive = 1.5
    reasons.append("Sat: gentle drive=1.5 for warmth")

    # ── Delay for rhythmic elements ──
    if has_pluck or has_synth:
        plan.delay_wet = 0.15
        reasons.append("Rhythmic instruments → delay wet=15%")
    else:
        plan.delay_wet = 0.08

    # ── FX layers from DNA ──
    fx_lower = [f.lower() for f in ev.fx_palette]
    if any("riser" in f for f in fx_lower):
        plan.ai_prompt_hints.append("add risers at section transitions")
        reasons.append("Risers detected in refs → add at section transitions")
    if any("impact" in f for f in fx_lower):
        plan.ai_prompt_hints.append("add impacts at drops")
        reasons.append("Impacts detected in refs → add at drop points")
    if any("sweep" in f for f in fx_lower):
        plan.ai_prompt_hints.append("add frequency sweeps")
        reasons.append("Sweeps detected in refs → add sweep FX")

    # ── Organic detection for instruments ──
    organic_inst = {"conga", "bongo", "kalimba", "marimba", "tabla",
                    "sitar", "djembe", "steel drum", "hang drum"}
    inst_set = {i.lower() for i in ev.instruments}
    found_organic = inst_set & organic_inst
    if found_organic:
        plan.use_organic = True
        plan.organic_category = sorted(found_organic)[0]
        reasons.append(
            f"🌿 Organic instruments: {', '.join(found_organic)} → prefer real samples"
        )

    # ── AI prompt hints ──
    instruments_str = ", ".join(ev.instruments[:4]) or "melodic"
    plan.ai_prompt_hints.extend(
        [
            f"{best} instrument texture",
            f"{ev.bpm:.0f} BPM, key of {ev.key} {ev.scale}",
            f"instruments: {instruments_str}",
            f"{'evolving, atmospheric' if plan.reverb_wet > 0.2 else 'defined, present'}",
            f"{'rhythmic, syncopated' if plan.delay_wet > 0.1 else 'sustained, smooth'}",
        ]
    )

    plan.reasoning = reasons
    return plan


# ── Master Thinker ───────────────────────────────────────


def _think_master(ev: Evidence) -> MasterPlan:
    """Derive mastering parameters from reference DNA."""
    plan = MasterPlan()
    reasons: list[str] = []

    # ── Target LUFS from references ──
    if ev.ref_lufs and ev.ref_lufs < 0:
        plan.target_lufs = ev.ref_lufs
        reasons.append(f"Target LUFS: {ev.ref_lufs:.1f} (from ref average)")
    else:
        plan.target_lufs = -8.0
        reasons.append("Target LUFS: -8.0 (club default, no ref data)")

    # ── Drive: louder refs = more saturation ──
    if ev.ref_lufs > -10.0:
        plan.drive = 1.8
        reasons.append(
            f"High loudness target ({ev.ref_lufs:.1f} LUFS) → aggressive drive=1.8"
        )
    elif ev.ref_lufs > -14.0:
        plan.drive = 1.5
        reasons.append(f"Medium loudness ({ev.ref_lufs:.1f}) → moderate drive=1.5")
    else:
        plan.drive = 1.2
        reasons.append(f"Gentle loudness ({ev.ref_lufs:.1f}) → light drive=1.2")

    # ── Stereo width from sidechain + instruments ──
    # Sidechain-heavy = more mono-centered bass, wider sides
    if ev.sidechain_ratio > 0.5:
        plan.width = 1.4
        reasons.append(
            f"High sidechain ({ev.sidechain_ratio:.0%}) → wider stereo=1.4 "
            "(bass stays mono, sides expand)"
        )
    elif ev.instruments and any("pad" in i.lower() for i in ev.instruments):
        plan.width = 1.35
        reasons.append("Pad-heavy refs → stereo=1.35")
    else:
        plan.width = 1.2
        reasons.append("Standard stereo=1.2")

    # ── Mid EQ from percussion + bass DNA ──
    mid_bands: list[tuple[float, float, float]] = []

    # Bass type influences low-mid EQ
    bass_type = ev.bass_type.lower() if ev.bass_type else ""
    if "sub" in bass_type or "808" in bass_type:
        mid_bands.append((300.0, -3.0, 1.5))
        reasons.append("Sub bass in refs → aggressive mid cut -3dB@300Hz")
    else:
        mid_bands.append((300.0, -2.0, 1.5))

    # Presence from percussion density
    if ev.percussion_density > 4.0:
        mid_bands.append((3000.0, 1.0, 1.2))
        reasons.append("High percussion density → moderate presence +1dB@3kHz")
    else:
        mid_bands.append((3000.0, 2.0, 1.2))
        reasons.append("Lower density → more presence boost +2dB@3kHz")

    # Air
    mid_bands.append((6000.0, 1.0, 0.8))
    plan.mid_eq_bands = mid_bands

    # ── Side EQ ──
    side_bands: list[tuple[float, float, float]] = []
    side_bands.append((8000.0, 2.0, 0.8))  # air on sides
    side_bands.append((5000.0, 1.5, 1.0))  # detail
    side_bands.append((250.0, -3.0, 1.2))  # clean sides
    plan.side_eq_bands = side_bands
    reasons.append("Side EQ: +2dB air@8k, +1.5dB detail@5k, -3dB mud@250")

    # ── Multiband compression from dynamic range ──
    if ev.ref_lufs > -10.0:
        # Loud masters need tighter compression
        plan.comp_low = {"threshold_db": -16.0, "ratio": 4.5, "attack_ms": 8.0, "release_ms": 80.0}
        plan.comp_mid = {"threshold_db": -12.0, "ratio": 3.0, "attack_ms": 6.0, "release_ms": 60.0}
        plan.comp_high = {"threshold_db": -8.0, "ratio": 2.0, "attack_ms": 2.0, "release_ms": 30.0}
        reasons.append("Loud target → tight multiband comp (low: -16/4.5:1, mid: -12/3:1, high: -8/2:1)")
    else:
        plan.comp_low = {"threshold_db": -18.0, "ratio": 4.0, "attack_ms": 10.0, "release_ms": 100.0}
        plan.comp_mid = {"threshold_db": -14.0, "ratio": 2.5, "attack_ms": 8.0, "release_ms": 80.0}
        plan.comp_high = {"threshold_db": -10.0, "ratio": 1.5, "attack_ms": 2.0, "release_ms": 40.0}
        reasons.append("Standard target → balanced multiband comp")

    plan.reasoning = reasons
    return plan


# ── Cross-Stem Interaction Resolution ────────────────────


def _resolve_interactions(
    plans: dict[str, StemPlan],
    master: MasterPlan,
    ev: Evidence,
) -> list[str]:
    """Check cross-stem synergy and adjust plans accordingly."""
    log: list[str] = []

    bass = plans.get("bass")
    drums = plans.get("drums")
    vocals = plans.get("vocals")
    other = plans.get("other")

    # ── Sidechain synergy: if bass sidechains, drums need punch ──
    if bass and bass.sidechain and drums:
        if drums.compression.get("attack_ms", 20) > 12:
            drums.compression["attack_ms"] = 10.0
            drums.reasoning.append(
                "INTERACTION: Bass has sidechain → drums need crisp transient (attack→10ms)"
            )
            log.append("Bass sidechain → drum attack tightened to 10ms")

    # ── Vocal space: if many vocal effects, instruments cut presence ──
    if vocals and other:
        if len(vocals.fx_chain) >= 3:
            # Busy vocal chain → instruments need more space
            existing_cut = [eq for eq in other.eq_adjustments if 2000 < eq[0] < 4000]
            if existing_cut:
                idx = other.eq_adjustments.index(existing_cut[0])
                freq, _gain, q = existing_cut[0]
                other.eq_adjustments[idx] = (freq, -3.5, q)
                other.reasoning.append(
                    "INTERACTION: Heavy vocal FX → deeper instrument cut at 2.5kHz (-3.5dB)"
                )
                log.append("Heavy vocals → instruments cut deeper at 2.5kHz")

    # ── Percussion density + bass sustain ──
    if bass and drums and ev.percussion_density > 4.0:
        if bass.style != "staccato":
            old_style = bass.style
            bass.style = "staccato"
            bass.reasoning.append(
                f"INTERACTION: High percussion density ({ev.percussion_density:.1f}/s) "
                f"→ bass style changed from '{old_style}' to 'staccato' (leave space)"
            )
            log.append(f"High density → bass: {old_style} → staccato")

    # ── Pad width + vocal centering ──
    if other and vocals and other.stereo_width > 1.2:
        if vocals.stereo_width > 1.0:
            vocals.stereo_width = 1.0
            if "chorus" in vocals.fx_chain:
                vocals.fx_chain.remove("chorus")
            vocals.reasoning.append(
                "INTERACTION: Instruments are wide → vocals must stay centered (no chorus)"
            )
            log.append("Wide instruments → vocals forced centered")

    # ── Sidechain affects mastering ──
    if bass and bass.sidechain:
        if master.comp_low.get("ratio", 4.0) > 4.0:
            master.comp_low["ratio"] = 3.5
            master.reasoning.append(
                "INTERACTION: Sidechain active → loosen low comp (3.5:1) "
                "to preserve intentional pumping"
            )
            log.append("Sidechain → master low comp loosened to 3.5:1")

    return log


# ── Main Engine ──────────────────────────────────────────


class DNABrain:
    """Multi-dimensional reasoning engine for Auralis.

    Instead of hardcoded if/else trees, scores candidate options across
    DNA match, context fit, confidence, cross-stem interactions,
    session memory, and emotional arc.
    """

    def __init__(self) -> None:
        self._memory: Any | None = None

    @property
    def memory(self) -> Any:
        """Lazy-load session memory."""
        if self._memory is None:
            try:
                from auralis.brain.memory import SessionMemory
                self._memory = SessionMemory()
            except Exception:
                self._memory = None
        return self._memory

    def think(
        self,
        deep_profile: dict[str, Any] | None,
        stem_analysis: dict[str, dict[str, Any]] | None = None,
        gap_report: dict[str, Any] | None = None,
        ear_data: dict[str, Any] | None = None,
        section_type: str = "drop",
        creativity: float = 0.3,
    ) -> BrainReport:
        """Run full reasoning across all stems and mastering.

        Args:
            deep_profile: DNA profile from reference analysis.
            stem_analysis: Per-stem audio analysis.
            gap_report: Gap analysis between source and references.
            ear_data: EAR module analysis data.
            section_type: Current section (intro/verse/build/drop/breakdown/outro).
                         Used by EmotionalArc to adjust processing.
            creativity: 0.0-1.0 temperature for stochastic selection.
                       0.0 = deterministic, 0.3 = light creativity, 1.0 = wild.

        Returns:
            BrainReport with optimal plans and reasoning chains.
        """
        ev = _gather_evidence(deep_profile, ear_data)
        reasoning: list[str] = []

        # ── Emotional arc ──
        arc = EmotionalArc(section_type)
        reasoning.append(
            f"🧠 DNABrain thinking... ({ev.deep_count}/{ev.ref_count} refs, "
            f"confidence={_confidence(ev):.0%}, section='{section_type}', "
            f"energy={arc.energy:.0%}, creativity={creativity:.0%})"
        )

        # ── Memory check ──
        mem = self.memory
        if mem and mem.session_count > 0:
            summary = mem.summary()
            reasoning.append(
                f"  📚 Memory: {summary['sessions']} sessions, "
                f"avg QC={summary['avg_qc']:.0f}"
            )

        # ── Think about each stem ──
        stem_plans: dict[str, StemPlan] = {}

        drums_plan = _think_drums(ev)
        stem_plans["drums"] = drums_plan
        reasoning.append(f"  🥁 Drums: style='{drums_plan.style}' (conf={drums_plan.confidence:.0%})")

        bass_plan = _think_bass(ev)
        stem_plans["bass"] = bass_plan
        reasoning.append(
            f"  🎸 Bass: patch='{bass_plan.patch}', style='{bass_plan.style}', "
            f"sidechain={'✓' if bass_plan.sidechain else '✗'} (conf={bass_plan.confidence:.0%})"
        )

        vocals_plan = _think_vocals(ev)
        stem_plans["vocals"] = vocals_plan
        fx_str = "+".join(vocals_plan.fx_chain[:3]) if vocals_plan.fx_chain else "clean"
        reasoning.append(f"  🎤 Vocals: FX=[{fx_str}] (conf={vocals_plan.confidence:.0%})")

        other_plan = _think_other(ev)
        stem_plans["other"] = other_plan
        reasoning.append(
            f"  🎹 Other: preset='{other_plan.patch}', "
            f"width={other_plan.stereo_width:.1f} (conf={other_plan.confidence:.0%})"
        )

        # ── Emotional Arc adjustments ──
        reasoning.append(f"  🎭 Emotional Arc: {section_type} (energy={arc.energy:.0%})")
        for stem_name, plan in stem_plans.items():
            arc_adjustments = arc.adjust_stem_plan(plan, stem_name)
            for adj in arc_adjustments:
                reasoning.append(f"    → {adj}")

        # ── Think about mastering ──
        master_plan = _think_master(ev)
        arc_master_adj = arc.adjust_master_plan(master_plan)
        for adj in arc_master_adj:
            reasoning.append(f"    → {adj}")
        reasoning.append(
            f"  💎 Master: LUFS={master_plan.target_lufs:.1f}, "
            f"drive={master_plan.drive:.1f}, width={master_plan.width:.1f}"
        )

        # ── Resolve cross-stem interactions ──
        reasoning.append("  🔗 Checking cross-stem interactions...")
        interaction_log = _resolve_interactions(stem_plans, master_plan, ev)
        for line in interaction_log:
            reasoning.append(f"    → {line}")

        if not interaction_log:
            reasoning.append("    → No adjustments needed")

        reasoning.append(f"  ✓ Brain complete — {len(stem_plans)} stem plans + master plan")

        logger.info(
            "dna_brain.complete",
            stems=len(stem_plans),
            confidence=_confidence(ev),
            interactions=len(interaction_log),
            section=section_type,
            energy=arc.energy,
            creativity=creativity,
        )

        return BrainReport(
            stem_plans=stem_plans,
            master_plan=master_plan,
            interaction_log=interaction_log,
            reasoning_chain=reasoning,
        )
