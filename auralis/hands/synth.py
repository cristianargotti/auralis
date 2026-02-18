"""AURALIS Synthesis Engine — Oscillators, envelopes, filters, wavetables.

Pure numpy/scipy implementation — no external synth dependencies required.
Supports: sine, saw, square, triangle, noise, wavetable, FM synthesis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

WaveShape = Literal["sine", "saw", "square", "triangle", "noise", "pulse"]


# ── Data Types ───────────────────────────────────────────


@dataclass
class ADSREnvelope:
    """Attack-Decay-Sustain-Release envelope."""

    attack_s: float = 0.01
    decay_s: float = 0.1
    sustain: float = 0.7  # 0-1 level
    release_s: float = 0.3

    def generate(self, duration_s: float, sr: int = 44100) -> NDArray[np.float64]:
        """Generate envelope curve as numpy array."""
        n = int(duration_s * sr)
        env = np.zeros(n, dtype=np.float64)

        a_samp = min(int(self.attack_s * sr), n)
        d_samp = min(int(self.decay_s * sr), n - a_samp)
        r_samp = min(int(self.release_s * sr), n)
        s_samp = max(0, n - a_samp - d_samp - r_samp)

        idx = 0
        # Attack
        if a_samp > 0:
            env[idx : idx + a_samp] = np.linspace(0, 1, a_samp)
            idx += a_samp
        # Decay
        if d_samp > 0:
            env[idx : idx + d_samp] = np.linspace(1, self.sustain, d_samp)
            idx += d_samp
        # Sustain
        if s_samp > 0:
            env[idx : idx + s_samp] = self.sustain
            idx += s_samp
        # Release
        if r_samp > 0 and idx < n:
            remaining = min(r_samp, n - idx)
            env[idx : idx + remaining] = np.linspace(
                self.sustain, 0, remaining
            )
        return env


@dataclass
class OscConfig:
    """Oscillator configuration."""

    wave: WaveShape = "saw"
    detune_cents: float = 0.0
    phase_offset: float = 0.0  # 0-1
    pw: float = 0.5  # Pulse width (for pulse wave)
    level: float = 1.0


@dataclass
class FilterConfig:
    """Filter configuration."""

    type: Literal["lowpass", "highpass", "bandpass"] = "lowpass"
    cutoff_hz: float = 5000.0
    resonance: float = 0.7  # Q factor


@dataclass
class VoiceConfig:
    """Complete voice configuration."""

    oscillators: list[OscConfig] = field(default_factory=lambda: [OscConfig()])
    envelope: ADSREnvelope = field(default_factory=ADSREnvelope)
    filter: FilterConfig | None = None
    unison: int = 1
    unison_spread_cents: float = 10.0


@dataclass
class SynthPatch:
    """Named synthesizer patch."""

    name: str
    voice: VoiceConfig
    description: str = ""


# ── Oscillator Core ──────────────────────────────────────


def _osc_sine(phase: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.sin(2 * np.pi * phase)


def _osc_saw(phase: NDArray[np.float64]) -> NDArray[np.float64]:
    return 2.0 * (phase % 1.0) - 1.0


def _osc_square(phase: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(phase % 1.0 < 0.5, 1.0, -1.0)


def _osc_triangle(phase: NDArray[np.float64]) -> NDArray[np.float64]:
    return 2.0 * np.abs(2.0 * (phase % 1.0) - 1.0) - 1.0


def _osc_noise(phase: NDArray[np.float64]) -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    return rng.uniform(-1, 1, len(phase))


def _osc_pulse(
    phase: NDArray[np.float64], pw: float = 0.5
) -> NDArray[np.float64]:
    return np.where(phase % 1.0 < pw, 1.0, -1.0)


_OSC_MAP = {
    "sine": _osc_sine,
    "saw": _osc_saw,
    "square": _osc_square,
    "triangle": _osc_triangle,
    "noise": _osc_noise,
}


def generate_oscillator(
    freq_hz: float,
    duration_s: float,
    sr: int = 44100,
    config: OscConfig | None = None,
) -> NDArray[np.float64]:
    """Generate a single oscillator waveform."""
    cfg = config or OscConfig()
    n = int(duration_s * sr)

    # Apply detune
    detune_ratio = 2.0 ** (cfg.detune_cents / 1200.0)
    actual_freq = freq_hz * detune_ratio

    # Generate phase accumulator
    t = np.arange(n, dtype=np.float64) / sr
    phase = actual_freq * t + cfg.phase_offset

    # Generate waveform
    if cfg.wave == "pulse":
        audio = _osc_pulse(phase, cfg.pw)
    elif cfg.wave in _OSC_MAP:
        audio = _OSC_MAP[cfg.wave](phase)
    else:
        audio = _osc_sine(phase)

    return audio * cfg.level


# ── Filter ───────────────────────────────────────────────


def apply_filter(
    audio: NDArray[np.float64],
    sr: int = 44100,
    config: FilterConfig | None = None,
) -> NDArray[np.float64]:
    """Apply a digital filter using scipy."""
    from scipy.signal import butter, sosfilt

    cfg = config or FilterConfig()
    nyq = sr / 2.0
    cutoff = min(cfg.cutoff_hz / nyq, 0.99)

    if cfg.type == "bandpass":
        low = max(cutoff * 0.5, 0.01)
        high = min(cutoff, 0.99)
        if low >= high:
            return audio
        sos = butter(2, [low, high], btype="bandpass", output="sos")
    else:
        btype = "low" if cfg.type == "lowpass" else "high"
        sos = butter(4, cutoff, btype=btype, output="sos")

    return sosfilt(sos, audio).astype(np.float64)


# ── FM Synthesis ─────────────────────────────────────────


def fm_synth(
    carrier_hz: float,
    modulator_hz: float,
    mod_index: float,
    duration_s: float,
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Frequency Modulation synthesis."""
    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    modulator = mod_index * modulator_hz * np.sin(2 * np.pi * modulator_hz * t)
    return np.sin(2 * np.pi * carrier_hz * t + modulator)


# ── Voice Rendering ──────────────────────────────────────


def render_voice(
    freq_hz: float,
    duration_s: float,
    sr: int = 44100,
    config: VoiceConfig | None = None,
) -> NDArray[np.float64]:
    """Render a complete synthesizer voice (multi-osc + envelope + filter)."""
    cfg = config or VoiceConfig()
    n = int(duration_s * sr)
    output = np.zeros(n, dtype=np.float64)

    for osc_cfg in cfg.oscillators:
        if cfg.unison > 1:
            # Unison: multiple detuned copies
            for i in range(cfg.unison):
                spread = (i - (cfg.unison - 1) / 2) * cfg.unison_spread_cents
                unison_cfg = OscConfig(
                    wave=osc_cfg.wave,
                    detune_cents=osc_cfg.detune_cents + spread,
                    phase_offset=osc_cfg.phase_offset + i * 0.1,
                    pw=osc_cfg.pw,
                    level=osc_cfg.level / cfg.unison,
                )
                output += generate_oscillator(freq_hz, duration_s, sr, unison_cfg)
        else:
            output += generate_oscillator(freq_hz, duration_s, sr, osc_cfg)

    # Apply filter
    if cfg.filter:
        output = apply_filter(output, sr, cfg.filter)

    # Apply envelope
    envelope = cfg.envelope.generate(duration_s, sr)
    output *= envelope

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.9

    return output


# ── MIDI-to-Audio ────────────────────────────────────────


def note_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def render_midi_to_audio(
    notes: list[dict[str, float]],
    sr: int = 44100,
    voice: VoiceConfig | None = None,
) -> NDArray[np.float64]:
    """Render MIDI note events to audio.

    Each note dict: {"note": 60, "start": 0.0, "duration": 0.5, "velocity": 0.8}
    """
    if not notes:
        return np.zeros(sr, dtype=np.float64)

    max_end = max(n["start"] + n["duration"] for n in notes)
    total_samples = int((max_end + 1.0) * sr)
    output = np.zeros(total_samples, dtype=np.float64)

    for note_event in notes:
        freq = note_to_freq(int(note_event["note"]))
        dur = note_event["duration"]
        vel = note_event.get("velocity", 0.8)
        start_idx = int(note_event["start"] * sr)

        audio = render_voice(freq, dur, sr, voice) * vel

        end_idx = min(start_idx + len(audio), total_samples)
        actual_len = end_idx - start_idx
        output[start_idx:end_idx] += audio[:actual_len]

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.9

    return output


# ── Presets ──────────────────────────────────────────────


PRESETS: dict[str, SynthPatch] = {
    # ── Bass Family (weight, depth, groove) ──
    "supersaw": SynthPatch(
        name="Super Saw",
        description="Classic detuned saw stack — EDM leads and pads",
        voice=VoiceConfig(
            oscillators=[OscConfig(wave="saw")],
            envelope=ADSREnvelope(attack_s=0.02, decay_s=0.2, sustain=0.6, release_s=0.5),
            filter=FilterConfig(type="lowpass", cutoff_hz=8000, resonance=0.5),
            unison=7,
            unison_spread_cents=15.0,
        ),
    ),
    "bass_808": SynthPatch(
        name="808 Bass",
        description="Deep sub weight — the gravitational pull of the track",
        voice=VoiceConfig(
            oscillators=[OscConfig(wave="sine", level=0.9), OscConfig(wave="triangle", level=0.3)],
            envelope=ADSREnvelope(attack_s=0.005, decay_s=0.8, sustain=0.2, release_s=0.3),
            filter=FilterConfig(type="lowpass", cutoff_hz=200, resonance=0.8),
        ),
    ),
    "bass_reese": SynthPatch(
        name="Reese Bass",
        description="Dark, evolving detuned bass — tension and movement underneath",
        voice=VoiceConfig(
            oscillators=[
                OscConfig(wave="saw", level=0.7),
                OscConfig(wave="saw", detune_cents=12, level=0.7),
            ],
            envelope=ADSREnvelope(attack_s=0.01, decay_s=0.4, sustain=0.5, release_s=0.4),
            filter=FilterConfig(type="lowpass", cutoff_hz=600, resonance=0.6),
        ),
    ),
    "acid_303": SynthPatch(
        name="Acid 303",
        description="Resonant character — squelchy energy that drives the groove",
        voice=VoiceConfig(
            oscillators=[OscConfig(wave="saw")],
            envelope=ADSREnvelope(attack_s=0.001, decay_s=0.3, sustain=0.1, release_s=0.1),
            filter=FilterConfig(type="lowpass", cutoff_hz=1500, resonance=0.9),
        ),
    ),
    # ── Melodic Family (story, emotion, melody) ──
    "pluck": SynthPatch(
        name="Pluck",
        description="Instant melodic statement — carries the hook and identity",
        voice=VoiceConfig(
            oscillators=[OscConfig(wave="saw"), OscConfig(wave="square", detune_cents=7, level=0.4)],
            envelope=ADSREnvelope(attack_s=0.002, decay_s=0.15, sustain=0.0, release_s=0.1),
            filter=FilterConfig(type="lowpass", cutoff_hz=4000, resonance=0.6),
        ),
    ),
    "keys_electric": SynthPatch(
        name="Electric Keys",
        description="Warm harmonic richness — soul and human touch in the arrangement",
        voice=VoiceConfig(
            oscillators=[
                OscConfig(wave="sine", level=0.6),
                OscConfig(wave="triangle", detune_cents=3, level=0.4),
                OscConfig(wave="square", detune_cents=-5, level=0.15),
            ],
            envelope=ADSREnvelope(attack_s=0.003, decay_s=0.4, sustain=0.3, release_s=0.2),
            filter=FilterConfig(type="lowpass", cutoff_hz=5000, resonance=0.4),
        ),
    ),
    "bell_fm": SynthPatch(
        name="FM Bell",
        description="Crystalline, bell-like — adds spark and air to the high end",
        voice=VoiceConfig(
            oscillators=[OscConfig(wave="sine", level=0.8)],
            envelope=ADSREnvelope(attack_s=0.001, decay_s=0.5, sustain=0.0, release_s=0.8),
            filter=FilterConfig(type="lowpass", cutoff_hz=12000, resonance=0.3),
        ),
    ),
    # ── Atmosphere Family (space, story, world-building) ──
    "pad_warm": SynthPatch(
        name="Warm Pad",
        description="Slow-breathing warmth — the emotional foundation of the track",
        voice=VoiceConfig(
            oscillators=[
                OscConfig(wave="saw", level=0.5),
                OscConfig(wave="triangle", detune_cents=5, level=0.5),
            ],
            envelope=ADSREnvelope(attack_s=1.0, decay_s=0.5, sustain=0.8, release_s=2.0),
            filter=FilterConfig(type="lowpass", cutoff_hz=3000, resonance=0.3),
            unison=5,
            unison_spread_cents=8.0,
        ),
    ),
    "pad_dark": SynthPatch(
        name="Dark Pad",
        description="Low, haunting texture — tension and mystery in the background",
        voice=VoiceConfig(
            oscillators=[
                OscConfig(wave="square", level=0.4),
                OscConfig(wave="saw", detune_cents=8, level=0.3),
                OscConfig(wave="triangle", detune_cents=-3, level=0.3),
            ],
            envelope=ADSREnvelope(attack_s=2.0, decay_s=1.0, sustain=0.7, release_s=3.0),
            filter=FilterConfig(type="lowpass", cutoff_hz=1500, resonance=0.4),
            unison=3,
            unison_spread_cents=12.0,
        ),
    ),
    "texture_noise": SynthPatch(
        name="Noise Texture",
        description="Filtered breath — air and organic life in the space between notes",
        voice=VoiceConfig(
            oscillators=[OscConfig(wave="noise", level=0.6)],
            envelope=ADSREnvelope(attack_s=0.5, decay_s=0.3, sustain=0.4, release_s=1.5),
            filter=FilterConfig(type="bandpass", cutoff_hz=2000, resonance=0.5),
        ),
    ),
}


def get_patch_for_stem(
    stem_name: str,
    style: str = "",
    bpm: float = 120.0,
    synth_patch: str = "",
) -> SynthPatch:
    """Musically-intelligent patch selection based on stem context.

    Thinks in terms of *what the music needs* — not just technical
    parameters.  The bass needs weight; pads need breath; leads need
    identity; textures need life.

    Args:
        stem_name: drums | bass | vocals | other
        style: Musical style hint from brain (e.g. "deep house bass")
        bpm: Track BPM — faster tracks need tighter sounds
        synth_patch: Explicit patch name override
    """
    # Explicit override always wins
    if synth_patch and synth_patch in PRESETS:
        return PRESETS[synth_patch]

    style_lower = style.lower() if style else ""

    if stem_name == "bass":
        # Bass = the gravitational center
        if "acid" in style_lower or "303" in style_lower:
            return PRESETS["acid_303"]
        if "reese" in style_lower or "dnb" in style_lower or bpm > 150:
            return PRESETS["bass_reese"]
        if "sub" in style_lower or "808" in style_lower or bpm < 100:
            return PRESETS["bass_808"]
        
        # Section-based narrative logic
        if "drop" in style_lower or "chorus" in style_lower:
            # Drop needs energy/drive
            return PRESETS["acid_303"] if bpm >= 128 else PRESETS["bass_reese"]
        if "breakdown" in style_lower or "bridge" in style_lower:
            # Breakdown needs tension/evolution
            return PRESETS["bass_reese"]
        if "intro" in style_lower or "verse" in style_lower:
            # Intro/Verse needs steady foundation
            return PRESETS["bass_808"]

        # Default: 808 for slow grooves, reese for energy
        return PRESETS["bass_808"] if bpm < 130 else PRESETS["bass_reese"]

    elif stem_name == "other":
        # Other = the color palette of the track
        if "pad" in style_lower or "ambient" in style_lower:
            if "dark" in style_lower or bpm > 140:
                return PRESETS["pad_dark"]
            return PRESETS["pad_warm"]
        if "bell" in style_lower or "chime" in style_lower:
            return PRESETS["bell_fm"]
        if "key" in style_lower or "piano" in style_lower:
            return PRESETS["keys_electric"]
        if "lead" in style_lower or "melody" in style_lower:
            return PRESETS["pluck"]
        if "texture" in style_lower or "atmosphere" in style_lower:
            return PRESETS["texture_noise"]
            
        # Section-based narrative logic
        if "breakdown" in style_lower or "intro" in style_lower:
            # Start atmospheric
            return PRESETS["pad_warm"]
        if "drop" in style_lower or "chorus" in style_lower:
            # Big energy
            return PRESETS["supersaw"] if bpm >= 125 else PRESETS["keys_electric"]
        if "verse" in style_lower:
            # Melodic movement
            return PRESETS["pluck"]

        # Default: pad for slow, pluck for fast
        return PRESETS["pad_warm"] if bpm < 120 else PRESETS["pluck"]

    elif stem_name == "vocals":
        # Vocal textures = ethereal, atmospheric
        return PRESETS["pad_warm"]

    else:
        # Fallback
        return PRESETS["pluck"]


def save_audio(
    audio: NDArray[np.float64],
    path: str | Path,
    sr: int = 44100,
    channels: int = 1,
) -> Path:
    """Save audio array to WAV file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if channels == 2 and audio.ndim == 1:
        stereo = np.column_stack([audio, audio])
        sf.write(str(p), stereo, sr, subtype="PCM_24")
    else:
        sf.write(str(p), audio, sr, subtype="PCM_24")

    return p
