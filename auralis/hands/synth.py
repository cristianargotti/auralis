"""AURALIS Synthesis Engine — Oscillators, envelopes, filters, wavetables.

Pure numpy/scipy implementation — no external synth dependencies required.
Supports: sine, saw, square, triangle, noise, wavetable, FM synthesis.
Also supports clone rendering mode: uses extracted samples and cloned timbres
from reference tracks when a ClonedPalette is available.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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
class FilterLFO:
    """LFO modulation for filter cutoff — adds movement to the sound."""

    rate_hz: float = 1.0      # LFO speed (0.1=slow sweep, 8=wobble)
    depth: float = 0.5        # 0-1, how much the cutoff moves
    shape: str = "sine"       # sine, triangle


@dataclass
class VoiceConfig:
    """Complete voice configuration."""

    oscillators: list[OscConfig] = field(default_factory=lambda: [OscConfig()])
    envelope: ADSREnvelope = field(default_factory=ADSREnvelope)
    filter: FilterConfig | None = None
    unison: int = 1
    unison_spread_cents: float = 10.0
    filter_lfo: FilterLFO | None = None          # Dynamic filter modulation
    wavetable: list[WaveShape] | None = None      # Morphing waveform sequence
    wavetable_scan_rate: float = 0.5              # Hz — scan speed through table


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


def wavetable_oscillator(
    freq_hz: float,
    duration_s: float,
    sr: int = 44100,
    table: list[WaveShape] | None = None,
    scan_rate: float = 0.5,
) -> NDArray[np.float64]:
    """Generate audio that morphs between waveforms over time.

    Creates a wavetable by generating each waveform, then crossfades
    between them using an LFO-driven scan position — the sound
    evolves from one timbre to another and back.

    Args:
        freq_hz: Note frequency.
        duration_s: Duration in seconds.
        sr: Sample rate.
        table: List of waveform shapes to morph between.
        scan_rate: How fast to scan through the table (Hz).

    Returns:
        Audio array with timbral evolution.
    """
    if not table or len(table) < 2:
        return generate_oscillator(freq_hz, duration_s, sr)

    n = int(duration_s * sr)
    t = np.arange(n, dtype=np.float64) / sr
    phase = freq_hz * t  # Phase accumulator

    # Generate all waveforms in the table
    waves = []
    for shape in table:
        if shape == "pulse":
            waves.append(_osc_pulse(phase))
        elif shape in _OSC_MAP:
            waves.append(_OSC_MAP[shape](phase))
        else:
            waves.append(_osc_sine(phase))

    # LFO drives the scan position (triangle wave: 0→1→0→1...)
    scan_pos = 0.5 * (1.0 + np.sin(2 * np.pi * scan_rate * t))
    # Scale to table range: 0.0 → (n_waves - 1)
    scan_idx = scan_pos * (len(waves) - 1)

    # Crossfade between adjacent waveforms
    output = np.zeros(n, dtype=np.float64)
    for i in range(len(waves) - 1):
        # Weight: how much of wave[i] vs wave[i+1] at each sample
        local = np.clip(scan_idx - i, 0.0, 1.0)
        output += waves[i] * (1.0 - local) + waves[i + 1] * local

    # Normalize (overlapping contributions)
    peak = np.max(np.abs(output))
    if peak > 0:
        output /= peak

    return output


def audio_to_wavetable(
    audio: NDArray[np.float64],
    sr: int = 44100,
    n_frames: int = 8,
    frame_size: int = 2048,
) -> NDArray[np.float64]:
    """Extract a wavetable from real audio — import any sound as an oscillator.

    Analyzes audio at evenly-spaced time points, extracting single-cycle
    waveforms via FFT → keep top N harmonics → IFFT. The result is a 2D
    array [n_frames × frame_size] that can be played as a wavetable
    oscillator, morphing through the sound's timbral evolution.

    Args:
        audio: Input audio (mono). Any sound: vocal, guitar, field recording.
        sr: Sample rate.
        n_frames: Number of waveforms to extract (more = finer evolution).
        frame_size: Samples per waveform frame (2048 = CD quality resolution).

    Returns:
        2D array [n_frames × frame_size] of normalized single-cycle waveforms.
    """
    if audio.ndim > 1:
        audio = audio[:, 0]  # Force mono

    n_samples = len(audio)
    if n_samples < frame_size * 2:
        # Audio too short — just use what we have
        n_frames = 1

    wavetable = np.zeros((n_frames, frame_size), dtype=np.float64)

    # Extract frames at evenly-spaced positions
    for i in range(n_frames):
        # Position in audio
        center = int((i + 0.5) / n_frames * n_samples)
        start = max(0, center - frame_size // 2)
        end = min(n_samples, start + frame_size)

        # Extract and zero-pad if needed
        chunk = audio[start:end]
        if len(chunk) < frame_size:
            padded = np.zeros(frame_size)
            padded[: len(chunk)] = chunk
            chunk = padded

        # FFT → keep top N harmonics → IFFT (single-cycle waveform)
        fft = np.fft.rfft(chunk)
        magnitudes = np.abs(fft)

        # Keep top 32 harmonics (removes noise, keeps character)
        n_keep = min(32, len(magnitudes))
        threshold = np.sort(magnitudes)[-n_keep] if len(magnitudes) > n_keep else 0
        fft_clean = np.where(magnitudes >= threshold, fft, 0.0)

        # Reconstruct single-cycle waveform
        waveform = np.fft.irfft(fft_clean, n=frame_size)

        # Normalize
        peak = np.max(np.abs(waveform))
        if peak > 0:
            waveform /= peak

        wavetable[i] = waveform

    return wavetable


def play_custom_wavetable(
    wavetable: NDArray[np.float64],
    freq_hz: float,
    duration_s: float,
    sr: int = 44100,
    scan_rate: float = 0.5,
) -> NDArray[np.float64]:
    """Play a custom wavetable (from audio_to_wavetable) as an oscillator.

    Morphs between frames in the wavetable at the given scan rate,
    producing audio at the specified frequency.

    Args:
        wavetable: 2D array [n_frames × frame_size] from audio_to_wavetable.
        freq_hz: Playback frequency.
        duration_s: Duration in seconds.
        sr: Sample rate.
        scan_rate: How fast to scan through frames (Hz).

    Returns:
        Synthesized audio.
    """
    n_frames, frame_size = wavetable.shape
    n_samples = int(duration_s * sr)
    t = np.arange(n_samples, dtype=np.float64) / sr

    # Phase accumulator for the oscillator
    phase = (freq_hz * t) % 1.0  # 0-1 phase position in waveform

    # Scan position through the wavetable (triangle wave)
    scan = 0.5 * (1.0 + np.sin(2 * np.pi * scan_rate * t))
    frame_idx = scan * (n_frames - 1)

    # Synthesize by interpolating between waveform frames
    output = np.zeros(n_samples, dtype=np.float64)
    for s in range(n_samples):
        # Which two frames are we between?
        fi = frame_idx[s]
        f0 = int(fi)
        f1 = min(f0 + 1, n_frames - 1)
        blend = fi - f0

        # Sample position within the frame
        sample_pos = phase[s] * frame_size
        s0 = int(sample_pos) % frame_size
        s1 = (s0 + 1) % frame_size
        sample_blend = sample_pos - int(sample_pos)

        # Bilinear interpolation (frame × sample)
        val_f0 = wavetable[f0, s0] * (1 - sample_blend) + wavetable[f0, s1] * sample_blend
        val_f1 = wavetable[f1, s0] * (1 - sample_blend) + wavetable[f1, s1] * sample_blend
        output[s] = val_f0 * (1 - blend) + val_f1 * blend

    return output


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


def _apply_filter_lfo(
    audio: NDArray[np.float64],
    sr: int,
    base_cutoff: float,
    lfo: FilterLFO,
    filter_type: str = "lowpass",
    resonance: float = 0.7,
) -> NDArray[np.float64]:
    """Apply time-varying filter with LFO modulation on cutoff.

    Processes audio in blocks, sweeping the cutoff frequency with
    the LFO — creates the breathing, evolving quality of analog synths.
    """
    from scipy.signal import butter, sosfilt

    n = len(audio)
    t = np.arange(n, dtype=np.float64) / sr

    # LFO generates cutoff modulation
    if lfo.shape == "triangle":
        lfo_signal = 2.0 * np.abs(2.0 * (lfo.rate_hz * t % 1.0) - 1.0) - 1.0
    else:  # sine
        lfo_signal = np.sin(2 * np.pi * lfo.rate_hz * t)

    # Block-based processing (filter changes every 512 samples)
    block_size = 512
    output = np.zeros(n, dtype=np.float64)
    nyq = sr / 2.0

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = audio[start:end]

        # LFO-modulated cutoff
        lfo_val = float(np.mean(lfo_signal[start:end]))
        cutoff = base_cutoff * (1.0 + lfo.depth * lfo_val)
        cutoff = max(50.0, min(cutoff, nyq * 0.95))

        # Apply filter to this block
        norm_cutoff = cutoff / nyq
        if norm_cutoff <= 0.01 or norm_cutoff >= 0.99:
            output[start:end] = block
            continue

        try:
            btype = "low" if filter_type == "lowpass" else "high"
            sos = butter(4, norm_cutoff, btype=btype, output="sos")
            output[start:end] = sosfilt(sos, block)
        except ValueError:
            output[start:end] = block

    return output


def render_voice(
    freq_hz: float,
    duration_s: float,
    sr: int = 44100,
    config: VoiceConfig | None = None,
) -> NDArray[np.float64]:
    """Render a complete synthesizer voice.

    Supports: multi-oscillator stacking, unison detuning, wavetable
    scanning, static and LFO-modulated filtering, ADSR envelope.
    """
    cfg = config or VoiceConfig()
    n = int(duration_s * sr)
    output = np.zeros(n, dtype=np.float64)

    # ── Wavetable mode: morphing between waveforms ──
    if cfg.wavetable and len(cfg.wavetable) >= 2:
        wt_audio = wavetable_oscillator(
            freq_hz, duration_s, sr,
            table=cfg.wavetable,
            scan_rate=cfg.wavetable_scan_rate,
        )
        output += wt_audio
    else:
        # ── Standard oscillator rendering ──
        for osc_cfg in cfg.oscillators:
            if cfg.unison > 1:
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

    # ── Filter: static or LFO-modulated ──
    if cfg.filter:
        if cfg.filter_lfo:
            output = _apply_filter_lfo(
                output, sr,
                base_cutoff=cfg.filter.cutoff_hz,
                lfo=cfg.filter_lfo,
                filter_type=cfg.filter.type,
                resonance=cfg.filter.resonance,
            )
        else:
            output = apply_filter(output, sr, cfg.filter)

    # ── Envelope ──
    envelope = cfg.envelope.generate(duration_s, sr)
    output *= envelope

    # ── Normalize ──
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
    palette: Any | None = None,
    stem_name: str = "",
) -> NDArray[np.float64]:
    """Render MIDI note events to audio.

    Each note dict: {"note": 60, "start": 0.0, "duration": 0.5, "velocity": 0.8}

    Clone rendering modes (when palette is provided):
      - stem_name="drums": triggers extracted one-shot samples
      - stem_name in palette.timbres: renders via spectral morphing
      - otherwise: fallback to oscillator synthesis
    """
    if not notes:
        return np.zeros(sr, dtype=np.float64)

    # ── Clone mode: drums (trigger one-shots) ──
    if palette is not None and stem_name == "drums":
        return _render_drums_from_palette(notes, palette, sr)

    # ── Clone mode: tonal (spectral morphing) ──
    if palette is not None and stem_name:
        timbre_models = getattr(palette, "_timbre_models", None)
        if timbre_models and stem_name in timbre_models:
            return _render_tonal_from_timbre(notes, timbre_models[stem_name], sr)

    # ── Standard mode: oscillator synthesis ──
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


def _render_drums_from_palette(
    notes: list[dict[str, float]],
    palette: Any,
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Render drum pattern by triggering cloned one-shot samples.

    Maps MIDI notes to drum labels using General MIDI convention,
    then loads and triggers the best matching sample from the palette.
    """
    # General MIDI drum map → palette label lookup
    _GM_DRUM_MAP: dict[int, str] = {
        36: "Kick", 35: "Kick",
        38: "Snare", 40: "Snare",
        39: "Clap", 37: "Rimshot",
        42: "Closed HH", 44: "Closed HH",
        46: "Open HH",
        51: "Ride", 59: "Ride",
        49: "Crash", 57: "Crash",
        45: "Tom", 47: "Tom", 48: "Tom",
        43: "Tom", 41: "Tom",
        63: "Conga", 62: "Conga",
        56: "Cowbell",
        70: "Shaker", 69: "Shaker",
    }

    max_end = max(n["start"] + n["duration"] for n in notes)
    total_samples = int((max_end + 1.0) * sr)
    output = np.zeros(total_samples, dtype=np.float64)

    # Cache loaded samples
    _sample_cache: dict[str, NDArray[np.float64]] = {}

    for note_event in notes:
        midi_note = int(note_event["note"])
        vel = note_event.get("velocity", 0.8)
        start_idx = int(note_event["start"] * sr)

        label = _GM_DRUM_MAP.get(midi_note, "Percussion")
        best = palette.best_drum(label)

        if best is None:
            continue

        cache_key = str(best.path)
        if cache_key not in _sample_cache:
            try:
                import soundfile as _sf
                audio, _sr = _sf.read(str(best.path), dtype="float64")
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                # Resample if needed
                if _sr != sr:
                    import librosa as _lr
                    audio = _lr.resample(audio, orig_sr=_sr, target_sr=sr)
                _sample_cache[cache_key] = audio
            except Exception:
                continue

        sample_audio = _sample_cache[cache_key] * vel
        end_idx = min(start_idx + len(sample_audio), total_samples)
        actual_len = end_idx - start_idx
        output[start_idx:end_idx] += sample_audio[:actual_len]

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.9

    return output


def _render_tonal_from_timbre(
    notes: list[dict[str, float]],
    timbre_model: Any,
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Render tonal notes using a cloned TimbreModel via spectral morphing."""
    from auralis.hands.timbre_cloner import clone_render

    max_end = max(n["start"] + n["duration"] for n in notes)
    total_samples = int((max_end + 1.0) * sr)
    output = np.zeros(total_samples, dtype=np.float64)

    for note_event in notes:
        midi_note = int(note_event["note"])
        dur = note_event["duration"]
        vel = note_event.get("velocity", 0.8)
        start_idx = int(note_event["start"] * sr)

        audio = clone_render(timbre_model, midi_note, dur, vel, sr)

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
        description="Slow-breathing warmth — wavetable morphing with filter LFO",
        voice=VoiceConfig(
            oscillators=[
                OscConfig(wave="saw", level=0.5),
                OscConfig(wave="triangle", detune_cents=5, level=0.5),
            ],
            envelope=ADSREnvelope(attack_s=1.0, decay_s=0.5, sustain=0.8, release_s=2.0),
            filter=FilterConfig(type="lowpass", cutoff_hz=3000, resonance=0.3),
            unison=5,
            unison_spread_cents=8.0,
            filter_lfo=FilterLFO(rate_hz=0.3, depth=0.4, shape="sine"),
            wavetable=["saw", "triangle", "sine"],
            wavetable_scan_rate=0.2,
        ),
    ),
    "pad_dark": SynthPatch(
        name="Dark Pad",
        description="Low, haunting texture — slow wavetable scan with filter sweep",
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
            filter_lfo=FilterLFO(rate_hz=0.15, depth=0.6, shape="triangle"),
            wavetable=["square", "saw", "triangle"],
            wavetable_scan_rate=0.1,
        ),
    ),
    "texture_noise": SynthPatch(
        name="Noise Texture",
        description="Filtered breath — air and organic life in the space between notes",
        voice=VoiceConfig(
            oscillators=[OscConfig(wave="noise", level=0.6)],
            envelope=ADSREnvelope(attack_s=0.5, decay_s=0.3, sustain=0.4, release_s=1.5),
            filter=FilterConfig(type="bandpass", cutoff_hz=2000, resonance=0.5),
            filter_lfo=FilterLFO(rate_hz=0.5, depth=0.3, shape="sine"),
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


# ── Modulation Matrix ───────────────────────────────────────


@dataclass
class ModSource:
    """A modulation source signal generator.

    Types:
        lfo: Low-frequency oscillator (sine/triangle/saw)
        envelope: ADSR shape (one-shot)
        velocity: Note velocity (constant per note)
        random: Per-note random value (constant per note)
    """

    type: str = "lfo"           # lfo, envelope, velocity, random
    rate_hz: float = 1.0        # LFO rate (only for type=lfo)
    shape: str = "sine"         # sine, triangle, saw (only for type=lfo)
    attack_s: float = 0.01      # Envelope attack (only for type=envelope)
    decay_s: float = 0.3        # Envelope decay (only for type=envelope)

    def generate(
        self,
        duration_s: float,
        sr: int = 44100,
        velocity: float = 0.8,
    ) -> NDArray[np.float64]:
        """Generate modulation signal (0-1 range)."""
        n = int(duration_s * sr)
        t = np.linspace(0, duration_s, n, endpoint=False)

        if self.type == "lfo":
            phase = 2 * np.pi * self.rate_hz * t
            if self.shape == "sine":
                signal = (np.sin(phase) + 1) / 2  # 0-1
            elif self.shape == "triangle":
                signal = np.abs(2 * (t * self.rate_hz % 1) - 1)
            elif self.shape == "saw":
                signal = t * self.rate_hz % 1
            else:
                signal = (np.sin(phase) + 1) / 2
            return signal

        elif self.type == "envelope":
            a_samp = min(int(self.attack_s * sr), n)
            d_samp = min(int(self.decay_s * sr), n - a_samp)
            env = np.zeros(n)
            if a_samp > 0:
                env[:a_samp] = np.linspace(0, 1, a_samp)
            if d_samp > 0:
                env[a_samp : a_samp + d_samp] = np.linspace(1, 0, d_samp)
            return env

        elif self.type == "velocity":
            return np.full(n, velocity)

        elif self.type == "random":
            return np.full(n, np.random.random())

        return np.zeros(n)


@dataclass
class ModRouting:
    """A single routing: source → destination with depth."""

    source: ModSource = field(default_factory=ModSource)
    destination: str = "filter_cutoff"  # filter_cutoff, amplitude, pitch, pan
    depth: float = 0.5  # 0-1 modulation depth
    bipolar: bool = False  # True = ±depth, False = 0 to +depth


@dataclass
class ModMatrix:
    """Modulation matrix: multiple source→destination routings.

    Processes audio by applying modulation signals to synthesis
    parameters. Operates post-synthesis on the rendered audio.

    Example usage:
        matrix = ModMatrix(routings=[
            ModRouting(
                source=ModSource(type="lfo", rate_hz=0.5, shape="sine"),
                destination="filter_cutoff",
                depth=0.6,
            ),
            ModRouting(
                source=ModSource(type="lfo", rate_hz=4.0, shape="triangle"),
                destination="amplitude",
                depth=0.3,
            ),
        ])
        processed = matrix.apply(audio, sr=44100, velocity=0.8, duration_s=2.0)
    """

    routings: list[ModRouting] = field(default_factory=list)

    def apply(
        self,
        audio: NDArray[np.float64],
        sr: int = 44100,
        velocity: float = 0.8,
        duration_s: float | None = None,
        base_cutoff_hz: float = 5000.0,
    ) -> NDArray[np.float64]:
        """Apply all modulation routings to audio.

        Args:
            audio: Input audio (mono or stereo).
            sr: Sample rate.
            velocity: MIDI velocity for velocity source (0-1).
            duration_s: Audio duration (computed from audio if None).
            base_cutoff_hz: Base filter cutoff for filter_cutoff destination.

        Returns:
            Modulated audio.
        """
        if not self.routings:
            return audio

        result = audio.copy()
        mono = result if result.ndim == 1 else result[:, 0]
        n = len(mono)
        dur = duration_s or n / sr

        for routing in self.routings:
            # Generate modulation signal
            mod_signal = routing.source.generate(dur, sr, velocity)
            if len(mod_signal) != n:
                # Resample to match audio length
                x_old = np.linspace(0, 1, len(mod_signal))
                x_new = np.linspace(0, 1, n)
                mod_signal = np.interp(x_new, x_old, mod_signal)

            # Scale by depth
            if routing.bipolar:
                mod = (mod_signal - 0.5) * 2 * routing.depth  # -depth to +depth
            else:
                mod = mod_signal * routing.depth  # 0 to depth

            # Apply to destination
            if routing.destination == "amplitude":
                # Modulate volume (tremolo/AM)
                gain = 1.0 - routing.depth + mod  # Reduce base by depth, add mod back
                if result.ndim == 2:
                    result *= gain[:, np.newaxis]
                else:
                    result *= gain

            elif routing.destination == "filter_cutoff":
                # Modulate filter cutoff — apply lowpass with varying cutoff
                from auralis.console.fx import apply_lowpass

                block_size = 2048
                for start in range(0, n, block_size):
                    end = min(start + block_size, n)
                    avg_mod = float(np.mean(mod[start:end]))
                    cutoff = base_cutoff_hz * (1.0 + avg_mod)
                    cutoff = max(80.0, min(cutoff, sr / 2 - 100))

                    if result.ndim == 2:
                        for ch in range(result.shape[1]):
                            result[start:end, ch] = apply_lowpass(
                                result[start:end, ch], cutoff, sr
                            )
                    else:
                        result[start:end] = apply_lowpass(
                            result[start:end], cutoff, sr
                        )

            elif routing.destination == "pitch":
                # Pitch vibrato via resampling
                from scipy.interpolate import interp1d

                pitch_shift = mod * 0.02  # Max ±2% pitch shift
                indices = np.arange(n, dtype=np.float64)
                shifted_indices = indices + pitch_shift * sr * 0.001
                shifted_indices = np.clip(shifted_indices, 0, n - 1)

                if result.ndim == 2:
                    for ch in range(result.shape[1]):
                        interp_fn = interp1d(
                            indices, result[:, ch],
                            kind="linear", fill_value="extrapolate",
                        )
                        result[:, ch] = interp_fn(shifted_indices)
                else:
                    interp_fn = interp1d(
                        indices, result,
                        kind="linear", fill_value="extrapolate",
                    )
                    result = interp_fn(shifted_indices)

            elif routing.destination == "pan":
                # Stereo panning modulation (auto-pan)
                if result.ndim == 1:
                    # Convert to stereo for panning
                    result = np.column_stack([result, result])
                pan = 0.5 + mod * 0.5  # 0=left, 1=right
                result[:, 0] *= np.cos(pan * np.pi / 2)
                result[:, 1] *= np.sin(pan * np.pi / 2)

        return result
