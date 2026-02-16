"""Million Pieces (Kino Todo Remix) — Benchmark Reconstruction Blueprint.

This module encodes the precise DNA of the reference track extracted from
per-bar spectral analysis, bass note detection, vocal analysis, effects
detection, and arrangement event mapping.

AURALIS must prove it can reconstruct this track bar-by-bar at equivalent
quality before proceeding to original Mono Aullador production.

Reference: Adam Ten & Mita Gami ft. Marina Maximilian - Million Pieces (Kino Todo Remix)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Track Master Profile ─────────────────────────────────


@dataclass
class TrackProfile:
    """Complete track-level characteristics extracted from analysis."""

    title: str = "Million Pieces (Kino Todo Remix)"
    artist: str = "Adam Ten & Mita Gami ft. Marina Maximilian"
    bpm: float = 128.0
    duration_s: float = 332.0  # 5:32
    total_bars: int = 177
    key: str = "G#"
    scale: str = "minor"  # G# minor / Ab minor

    # Master levels
    groove_rms_db: float = -7.7  # average groove-section RMS
    breakdown_rms_db: float = -15.0  # average breakdown RMS
    dynamic_range_db: float = 10.0  # groove section dynamic range

    # Frequency profile
    bass_level_db: float = -13.8  # groove bass average
    pad_level_db: float = -27.0  # groove pad average
    vocal_level_db: float = -29.0  # groove vocal average


# ── Section Map ──────────────────────────────────────────


@dataclass
class Section:
    """One section in the arrangement."""

    name: str
    start_bar: int
    end_bar: int
    rms_db: float
    stereo_sm: float  # S/M ratio
    elements: list[str]
    description: str


MILLION_PIECES_SECTIONS: list[Section] = [
    Section(
        name="intro",
        start_bar=0, end_bar=7,
        rms_db=-18.0, stereo_sm=0.117,
        elements=["hat", "kick_ghost", "clap_hat"],
        description="Filtered intro — hi-hats, ghost kicks, sidechain pump 3.3x, heavy reverb (2.86s). Building anticipation."
    ),
    Section(
        name="groove_a",
        start_bar=8, end_bar=23,
        rms_db=-8.8, stereo_sm=0.068,
        elements=["kick_4otf", "clap", "hat", "bass", "vocal"],
        description="First groove — full kick+bass, bass at -14dB G#, narrow stereo, vocal chopped. 1/4 echo delay."
    ),
    Section(
        name="breakdown_1",
        start_bar=24, end_bar=31,
        rms_db=-15.4, stereo_sm=0.756,
        elements=["vocal_tonal", "hat", "kick_ghost", "pad", "reverb_wash"],
        description="First breakdown — VERY WIDE stereo (S/M 0.75), kick fades out, vocal becomes tonal synth, heavy reverb."
    ),
    Section(
        name="build_1",
        start_bar=32, end_bar=43,
        rms_db=-15.4, stereo_sm=0.807,
        elements=["vocal_lead", "pad", "kick_building", "hat", "riser"],
        description="Build — widest stereo (0.81), vocal + pad dominate, kick presence fluctuates, riser at bar 43."
    ),
    Section(
        name="drop_1",
        start_bar=44, end_bar=51,
        rms_db=-7.8, stereo_sm=0.120,
        elements=["kick_4otf", "clap", "bass", "vocal_chopped", "filter_sweep"],
        description="First drop — ENERGY JUMP +8.2dB, narrow stereo, sidechain light pump 1.7x, filter modulation."
    ),
    Section(
        name="groove_b",
        start_bar=52, end_bar=75,
        rms_db=-7.7, stereo_sm=0.125,
        elements=["kick_4otf", "clap", "hat", "bass", "vocal", "pad"],
        description="Extended groove — steady -7.7dB, vocals throughout, pad enters at -24dB, narrow stereo."
    ),
    Section(
        name="breakdown_2",
        start_bar=76, end_bar=83,
        rms_db=-7.8, stereo_sm=0.181,
        elements=["kick_4otf", "vocal", "filter_closing", "modulation"],
        description="Short variation — filter closing, modulation effects, stereo slightly wider. Preparing transition."
    ),
    Section(
        name="build_2",
        start_bar=84, end_bar=91,
        rms_db=-13.0, stereo_sm=0.278,
        elements=["kick_sparse", "hat", "reverb_tail", "pad"],
        description="Energy drops to -14dB, kicks become sparse, wide stereo, reverb increases."
    ),
    Section(
        name="groove_c",
        start_bar=92, end_bar=115,
        rms_db=-13.2, stereo_sm=0.380,
        elements=["hat", "perc", "bass", "vocal_tonal", "pad"],
        description="Mid-energy groove — bass at -19dB (softer), wide stereo, vocal becomes tonal, pad prominent."
    ),
    Section(
        name="breakdown_3",
        start_bar=116, end_bar=119,
        rms_db=-17.0, stereo_sm=0.440,
        elements=["kick_fade", "vocal", "pad", "riser"],
        description="Short breakdown — ENERGY CLIFF at bar 117 (-8.2dB drop), riser at 119, preparing second major drop."
    ),
    Section(
        name="drop_2",
        start_bar=120, end_bar=135,
        rms_db=-8.0, stereo_sm=0.093,
        elements=["kick_4otf", "clap", "bass", "vocal", "hat"],
        description="Second drop — ENERGY JUMP +8.0dB, narrowest stereo, full energy, vocal returns."
    ),
    Section(
        name="groove_d",
        start_bar=136, end_bar=143,
        rms_db=-7.8, stereo_sm=0.132,
        elements=["kick_4otf", "clap", "hat", "bass", "vocal", "pad"],
        description="Peak groove — vocal + pad layered, full percussion, narrow stereo. Maximum engagement."
    ),
    Section(
        name="groove_e",
        start_bar=144, end_bar=159,
        rms_db=-7.8, stereo_sm=0.045,
        elements=["kick_4otf", "clap", "hat", "bass", "vocal"],
        description="Final groove — MONO stereo (S/M 0.045), bass-focused, most narrow imaging. Pure club energy."
    ),
    Section(
        name="outro_groove",
        start_bar=160, end_bar=175,
        rms_db=-8.7, stereo_sm=0.060,
        elements=["kick_4otf", "clap", "hat", "bass", "vocal"],
        description="Outro groove — returns to full sidechain pump 2.5x, 1/4 echo delay returns, hi-pass filter rising."
    ),
    Section(
        name="outro_tail",
        start_bar=176, end_bar=177,
        rms_db=-14.6, stereo_sm=0.082,
        elements=["kick_fade", "reverb_tail"],
        description="Final tail — ENERGY CLIFF -5.9dB, downlifter, sweep to silence."
    ),
]


# ── Bass DNA Pattern ─────────────────────────────────────
# Core bass progression notes (per-beat for first 8 bars, then key moments)

BASS_DNA = {
    "root_note": "G#",
    "dominant_midi": [36, 37],  # C2, C#2 in bass register
    "groove_level_db": -13.8,
    "pattern_style": "repeated_notes",  # 58% repeated, 2.4st avg motion, 77% stepwise
    "main_notes": ["G#1", "A1", "A#1", "C2", "C#2"],
    "rhythm": "every_beat",  # One bass note per beat in groove sections
    "characteristics": {
        "motion": 2.4,  # semitones average motion
        "stepwise_ratio": 0.77,
        "repeated_ratio": 0.58,
    },
}


# ── Kick Pattern DNA ─────────────────────────────────────

KICK_PATTERNS = {
    "groove_main": "[XXXX]",  # 4-on-the-floor, every beat
    "intro": "[.XXX]",  # First beat empty
    "breakdown_exit": "[XX..][....][....][.XXX]",  # bars 40-43: fade out → silence → fill
    "drop_re_entry": "[XXXX]",  # Full power return
    "fill_4bar": "[.XXX][XXXX][XXXX][XXXX]",  # Variation with first beat skip
    "build_sparse": "[XXXX][XXXX][.XXX][XXXX]",  # Bars 84-87: sparse variation
}


# ── FX Presets Per Section ───────────────────────────────

FX_PRESETS: dict[str, dict[str, Any]] = {
    "intro": {
        "reverb_s": 2.86,
        "delay": "1/4_echo",
        "filter_hz": 7025,
        "filter_type": "hp_bright",
        "sidechain_pump": 3.3,
        "stereo_sm": 0.117,
    },
    "groove": {
        "reverb_s": 0.23,
        "delay": "1/4_echo",
        "filter_hz": 5873,
        "filter_type": "open",
        "sidechain_pump": 2.4,
        "stereo_sm": 0.068,
    },
    "breakdown": {
        "reverb_s": 1.32,
        "delay": "none",
        "filter_hz": 5969,
        "filter_type": "open",
        "sidechain_pump": 0.0,
        "stereo_sm": 0.711,
    },
    "build": {
        "reverb_s": 1.22,
        "delay": "none",
        "filter_hz": 5690,
        "filter_type": "open",
        "sidechain_pump": 0.0,
        "stereo_sm": 0.807,
    },
    "drop": {
        "reverb_s": 0.85,
        "delay": "none",
        "filter_hz": 4497,
        "filter_type": "open",
        "sidechain_pump": 1.7,
        "stereo_sm": 0.103,
    },
    "mid_energy": {
        "reverb_s": 1.10,
        "delay": "none",
        "filter_hz": 5621,
        "filter_type": "open",
        "sidechain_pump": 0.0,
        "stereo_sm": 0.278,
    },
    "final_groove": {
        "reverb_s": 0.62,
        "delay": "none",
        "filter_hz": 5310,
        "filter_type": "open",
        "sidechain_pump": 1.8,
        "stereo_sm": 0.045,
    },
    "outro": {
        "reverb_s": 0.33,
        "delay": "1/4_echo",
        "filter_hz": 5762,
        "filter_type": "open",
        "sidechain_pump": 2.5,
        "stereo_sm": 0.075,
    },
}


# ── Arrangement Events (Transitions) ────────────────────

ARRANGEMENT_EVENTS = [
    {"bar": 0, "type": "riser", "magnitude_db": 55.3},
    {"bar": 7, "type": "sub_sweep_falling", "magnitude_db": -42.1},
    {"bar": 8, "type": "energy_jump", "magnitude_db": 9.2},
    {"bar": 24, "type": "energy_cliff", "magnitude_db": -7.7},
    {"bar": 25, "type": "sub_sweep_rising", "magnitude_db": 6.7},
    {"bar": 32, "type": "energy_jump", "magnitude_db": 4.0},
    {"bar": 40, "type": "downlifter", "magnitude_db": -6.8},
    {"bar": 41, "type": "energy_cliff", "magnitude_db": -5.2},
    {"bar": 43, "type": "riser", "magnitude_db": 11.1},
    {"bar": 44, "type": "energy_jump", "magnitude_db": 8.2},
    {"bar": 51, "type": "riser", "magnitude_db": 9.2},
    {"bar": 117, "type": "energy_cliff", "magnitude_db": -8.2},
    {"bar": 119, "type": "riser", "magnitude_db": 11.0},
    {"bar": 120, "type": "energy_jump", "magnitude_db": 8.0},
    {"bar": 167, "type": "sub_sweep_falling", "magnitude_db": -33.2},
    {"bar": 176, "type": "energy_cliff", "magnitude_db": -5.9},
]


# ── Energy Map (per 8-bar) ──────────────────────────────

ENERGY_MAP_8BAR = [
    # (start_bar, rms_db, description)
    (0, -18.0, "intro_filtered"),
    (8, -8.8, "groove_a"),
    (16, -8.6, "groove_a_continued"),
    (24, -15.4, "breakdown_1_wide"),
    (32, -14.0, "build_1_widest"),
    (40, -18.8, "pre_drop_stripped"),
    (48, -7.8, "drop_1"),
    (56, -7.7, "groove_b"),
    (64, -7.7, "groove_b_continued"),
    (72, -7.8, "groove_transition"),
    (80, -7.8, "breakdown_2_filtered"),
    (88, -13.0, "build_2"),
    (96, -13.2, "groove_c_mid"),
    (104, -13.0, "groove_c_continued"),
    (112, -13.0, "groove_c_end"),
    (120, -8.0, "drop_2"),
    (128, -7.8, "groove_d"),
    (136, -7.8, "groove_d_peak"),
    (144, -7.8, "groove_e_mono"),
    (152, -7.8, "groove_e_continued"),
    (160, -8.7, "outro_groove"),
    (168, -8.7, "outro_groove_end"),
    (176, -14.6, "tail"),
]


# ── Vocal DNA ────────────────────────────────────────────

VOCAL_DNA = {
    "character": "strong_female",
    "f0_range_hz": (103, 306),
    "main_formants_hz": [409, 603, 1895],
    "groove_treatment": "chopped_with_fx",
    "breakdown_treatment": "tonal_synth_pad",
    "stereo_width": "follows_section",
    "bars_with_vocal": list(range(1, 176)),  # Almost every bar has vocal content
}


# ── Reconstruction API ───────────────────────────────────


def get_million_pieces_blueprint() -> dict[str, Any]:
    """Return the complete reconstruction blueprint for Million Pieces.

    This is the master reference that AURALIS uses to validate
    its reconstruction quality bar-by-bar.
    """
    profile = TrackProfile()
    return {
        "profile": {
            "title": profile.title,
            "artist": profile.artist,
            "bpm": profile.bpm,
            "key": profile.key,
            "scale": profile.scale,
            "duration_s": profile.duration_s,
            "total_bars": profile.total_bars,
        },
        "sections": [
            {
                "name": s.name,
                "bars": f"{s.start_bar}-{s.end_bar}",
                "rms_db": s.rms_db,
                "stereo_sm": s.stereo_sm,
                "elements": s.elements,
                "description": s.description,
            }
            for s in MILLION_PIECES_SECTIONS
        ],
        "bass_dna": BASS_DNA,
        "kick_patterns": KICK_PATTERNS,
        "fx_presets": FX_PRESETS,
        "arrangement_events": ARRANGEMENT_EVENTS,
        "energy_map": [
            {"bar": b, "rms_db": r, "phase": d}
            for b, r, d in ENERGY_MAP_8BAR
        ],
        "vocal_dna": {
            "character": VOCAL_DNA["character"],
            "f0_range": VOCAL_DNA["f0_range_hz"],
            "formants": VOCAL_DNA["main_formants_hz"],
        },
        "quality_targets": {
            "groove_rms_tolerance_db": 1.0,
            "stereo_sm_tolerance": 0.05,
            "bass_level_tolerance_db": 1.0,
            "section_transition_match": True,
            "energy_curve_correlation_min": 0.90,
        },
    }


def get_reconstruction_config() -> dict[str, Any]:
    """Return the configuration needed to attempt reconstruction."""
    return {
        "bpm": 128.0,
        "key": "G#",
        "scale": "minor",
        "bars": 177,
        "sr": 44100,
        "synth_presets_needed": [
            "bass_sub_g_sharp",
            "pad_atmospheric",
            "vocal_chop_processor",
            "lead_tonal_synth",
        ],
        "effects_needed": [
            "sidechain_pump_heavy",  # 2.3-3.3x for intro/outro
            "sidechain_pump_light",  # 1.6-1.8x for grooves
            "reverb_wash",  # 1.2-2.9s for breakdowns
            "reverb_tight",  # 0.2-0.6s for grooves
            "delay_quarter",  # 1/4 echo for intro/outro
            "filter_hp_sweep",  # HP filter opening/closing
            "stereo_widener",  # 0.04-0.81 S/M range
        ],
        "arrangement_phases": [
            "intro → groove_a → breakdown_1 → build_1",
            "drop_1 → groove_b → breakdown_2 → build_2",
            "groove_c → breakdown_3 → drop_2 → groove_d",
            "groove_e → outro_groove → tail",
        ],
    }
