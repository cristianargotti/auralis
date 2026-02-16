"""AURALIS Effects Engine — DSP effect chains and processing.

Pure numpy/scipy implementation with optional pedalboard integration.
Supports: EQ, compression, reverb, delay, distortion, chorus, sidechain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


# ── Effect Configs ───────────────────────────────────────


@dataclass
class EQBand:
    """Parametric EQ band."""

    freq_hz: float = 1000.0
    gain_db: float = 0.0
    q: float = 1.0
    type: Literal["peak", "lowshelf", "highshelf"] = "peak"


@dataclass
class CompressorConfig:
    """Dynamic range compressor."""

    threshold_db: float = -20.0
    ratio: float = 4.0
    attack_ms: float = 10.0
    release_ms: float = 100.0
    makeup_db: float = 0.0


@dataclass
class ReverbConfig:
    """Algorithmic reverb (Schroeder)."""

    room_size: float = 0.7  # 0-1
    damping: float = 0.5  # 0-1
    wet: float = 0.3  # 0-1 mix
    pre_delay_ms: float = 10.0


@dataclass
class DelayConfig:
    """Delay / echo effect."""

    time_ms: float = 375.0  # Delay time
    feedback: float = 0.4  # 0-1
    wet: float = 0.3  # 0-1 mix
    ping_pong: bool = False


@dataclass
class DistortionConfig:
    """Distortion / saturation."""

    drive: float = 5.0  # Gain before clipping
    type: Literal["soft_clip", "hard_clip", "tube", "bitcrush"] = "soft_clip"
    mix: float = 1.0  # 0-1 dry/wet


@dataclass
class ChorusConfig:
    """Chorus / ensemble effect."""

    rate_hz: float = 1.5  # LFO rate
    depth_ms: float = 3.0  # Modulation depth
    mix: float = 0.5  # 0-1
    voices: int = 3


@dataclass
class SidechainConfig:
    """Sidechain compression (pumping effect)."""

    frequency_hz: float = 2.0  # Pump rate (e.g. 2 = 8th notes at 120bpm)
    depth: float = 0.8  # 0-1 ducking amount
    attack_ms: float = 5.0
    release_ms: float = 200.0
    curve: Literal["linear", "exponential"] = "exponential"


@dataclass
class EffectChain:
    """Ordered chain of effects."""

    name: str = "default"
    eq_bands: list[EQBand] = field(default_factory=list)
    compressor: CompressorConfig | None = None
    distortion: DistortionConfig | None = None
    chorus: ChorusConfig | None = None
    delay: DelayConfig | None = None
    reverb: ReverbConfig | None = None
    sidechain: SidechainConfig | None = None


# ── DSP Functions ────────────────────────────────────────


def apply_eq(
    audio: NDArray[np.float64],
    bands: list[EQBand],
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Apply parametric EQ bands using biquad filters."""
    from scipy.signal import sosfilt

    output = audio.copy()
    nyq = sr / 2.0

    for band in bands:
        if band.gain_db == 0.0:
            continue

        w0 = 2 * np.pi * band.freq_hz / sr
        A = 10 ** (band.gain_db / 40.0)
        alpha = np.sin(w0) / (2 * band.q)

        if band.type == "peak":
            b0 = 1 + alpha * A
            b1 = -2 * np.cos(w0)
            b2 = 1 - alpha * A
            a0 = 1 + alpha / A
            a1 = -2 * np.cos(w0)
            a2 = 1 - alpha / A
        elif band.type == "lowshelf":
            sq = 2 * np.sqrt(A) * alpha
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + sq)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - sq)
            a0 = (A + 1) + (A - 1) * np.cos(w0) + sq
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - sq
        else:  # highshelf
            sq = 2 * np.sqrt(A) * alpha
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + sq)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - sq)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + sq
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - sq

        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])
        output = sosfilt(sos, output).astype(np.float64)

    return output


def apply_compressor(
    audio: NDArray[np.float64],
    config: CompressorConfig,
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Apply dynamic range compression."""
    threshold = 10 ** (config.threshold_db / 20.0)
    attack_coeff = np.exp(-1.0 / (config.attack_ms * sr / 1000.0))
    release_coeff = np.exp(-1.0 / (config.release_ms * sr / 1000.0))

    output = audio.copy()
    envelope = 0.0

    for i in range(len(output)):
        level = abs(output[i])
        if level > envelope:
            envelope = attack_coeff * envelope + (1 - attack_coeff) * level
        else:
            envelope = release_coeff * envelope + (1 - release_coeff) * level

        if envelope > threshold:
            gain_reduction = threshold / envelope
            gain_reduction = gain_reduction ** (1 - 1 / config.ratio)
            output[i] *= gain_reduction

    # Makeup gain
    makeup = 10 ** (config.makeup_db / 20.0)
    return output * makeup


def apply_distortion(
    audio: NDArray[np.float64],
    config: DistortionConfig,
) -> NDArray[np.float64]:
    """Apply distortion/saturation."""
    driven = audio * config.drive

    if config.type == "soft_clip":
        wet = np.tanh(driven)
    elif config.type == "hard_clip":
        wet = np.clip(driven, -1.0, 1.0)
    elif config.type == "tube":
        # Asymmetric soft clipping (tube emulation)
        pos = driven[driven >= 0]
        neg = driven[driven < 0]
        wet = driven.copy()
        wet[driven >= 0] = 1.0 - np.exp(-pos)
        wet[driven < 0] = -(1.0 - np.exp(neg))
    elif config.type == "bitcrush":
        bits = max(2, int(16 / config.drive))
        steps = 2**bits
        wet = np.round(driven * steps) / steps
    else:
        wet = np.tanh(driven)

    return audio * (1 - config.mix) + wet * config.mix


def apply_chorus(
    audio: NDArray[np.float64],
    config: ChorusConfig,
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Apply chorus effect using modulated delay."""
    n = len(audio)
    output = audio.copy()
    max_delay = int(config.depth_ms * sr / 1000.0) + 1

    for voice in range(config.voices):
        phase_offset = voice * 2 * np.pi / config.voices
        t = np.arange(n, dtype=np.float64) / sr
        lfo = np.sin(2 * np.pi * config.rate_hz * t + phase_offset)
        delay_samples = (lfo + 1) * 0.5 * config.depth_ms * sr / 1000.0
        delay_samples = delay_samples.astype(int) + 1

        delayed = np.zeros(n, dtype=np.float64)
        for i in range(max_delay, n):
            d = min(delay_samples[i], i)
            delayed[i] = audio[i - d]

        output += delayed * (config.mix / config.voices)

    return output / (1 + config.mix)


def apply_delay(
    audio: NDArray[np.float64],
    config: DelayConfig,
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Apply delay / echo effect."""
    delay_samples = int(config.time_ms * sr / 1000.0)
    n = len(audio)
    output = audio.copy()
    buffer = np.zeros(n + delay_samples * 10, dtype=np.float64)
    buffer[:n] = audio

    for tap in range(1, 10):
        offset = delay_samples * tap
        gain = config.feedback**tap
        if gain < 0.01 or offset >= len(buffer):
            break
        end = min(n + offset, len(buffer))
        actual_n = end - offset
        buffer[offset:end] += audio[:actual_n] * gain

    output = buffer[:n]
    return audio * (1 - config.wet) + output * config.wet


def apply_reverb(
    audio: NDArray[np.float64],
    config: ReverbConfig,
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Apply reverb using Schroeder design with comb/allpass filters."""
    pre_delay_samp = int(config.pre_delay_ms * sr / 1000.0)

    # Comb filter delays (prime-ish ms values)
    comb_delays_ms = [29.7, 37.1, 41.1, 43.7]
    comb_outputs = []

    for delay_ms in comb_delays_ms:
        delay_samp = int(delay_ms * sr / 1000.0 * config.room_size)
        if delay_samp < 1:
            continue
        feedback = config.room_size * (1 - config.damping * 0.4)
        n = len(audio)
        buf = np.zeros(n, dtype=np.float64)
        for i in range(n):
            buf_idx = i - delay_samp
            delayed = buf[buf_idx] if buf_idx >= 0 else 0.0
            buf[i] = audio[i] + delayed * feedback
        comb_outputs.append(buf)

    if comb_outputs:
        wet = sum(comb_outputs) / len(comb_outputs)
    else:
        wet = audio.copy()

    # Allpass filters
    allpass_delays_ms = [5.0, 1.7]
    for delay_ms in allpass_delays_ms:
        delay_samp = int(delay_ms * sr / 1000.0)
        if delay_samp < 1:
            continue
        g = 0.7
        n = len(wet)
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            buf_idx = i - delay_samp
            delayed = out[buf_idx] if buf_idx >= 0 else 0.0
            out[i] = -g * wet[i] + delayed + g * delayed
        wet = out

    return audio * (1 - config.wet) + wet * config.wet


def apply_sidechain(
    audio: NDArray[np.float64],
    config: SidechainConfig,
    sr: int = 44100,
    bpm: float = 120.0,
) -> NDArray[np.float64]:
    """Apply sidechain pumping effect."""
    n = len(audio)
    t = np.arange(n, dtype=np.float64) / sr

    # Generate pump curve
    pump_freq = bpm / 60.0 * config.frequency_hz / 2
    phase = (pump_freq * t) % 1.0

    if config.curve == "exponential":
        envelope = 1.0 - config.depth * np.exp(-phase * 8)
    else:
        envelope = 1.0 - config.depth * (1.0 - phase)

    envelope = np.clip(envelope, 1.0 - config.depth, 1.0)
    return audio * envelope


# ── Chain Processor ──────────────────────────────────────


def process_chain(
    audio: NDArray[np.float64],
    chain: EffectChain,
    sr: int = 44100,
    bpm: float = 120.0,
) -> NDArray[np.float64]:
    """Process audio through an effect chain in order."""
    output = audio.copy()

    if chain.eq_bands:
        output = apply_eq(output, chain.eq_bands, sr)

    if chain.compressor:
        output = apply_compressor(output, chain.compressor, sr)

    if chain.distortion:
        output = apply_distortion(output, chain.distortion)

    if chain.chorus:
        output = apply_chorus(output, chain.chorus, sr)

    if chain.delay:
        output = apply_delay(output, chain.delay, sr)

    if chain.reverb:
        output = apply_reverb(output, chain.reverb, sr)

    if chain.sidechain:
        output = apply_sidechain(output, chain.sidechain, sr, bpm)

    return output


# ── Preset Chains ────────────────────────────────────────


PRESET_CHAINS: dict[str, EffectChain] = {
    "deep_house_bass": EffectChain(
        name="Deep House Bass",
        eq_bands=[
            EQBand(freq_hz=60, gain_db=3, q=0.8, type="lowshelf"),
            EQBand(freq_hz=200, gain_db=-2, q=1.5),
            EQBand(freq_hz=3000, gain_db=2, q=1.0),
        ],
        compressor=CompressorConfig(threshold_db=-18, ratio=4, attack_ms=5, release_ms=80, makeup_db=3),
        distortion=DistortionConfig(drive=2.0, type="tube", mix=0.3),
        sidechain=SidechainConfig(frequency_hz=2.0, depth=0.6),
    ),
    "ambient_pad": EffectChain(
        name="Ambient Pad",
        eq_bands=[
            EQBand(freq_hz=80, gain_db=-6, q=0.5, type="lowshelf"),
            EQBand(freq_hz=8000, gain_db=2, q=0.7, type="highshelf"),
        ],
        chorus=ChorusConfig(rate_hz=0.5, depth_ms=5, mix=0.4, voices=4),
        delay=DelayConfig(time_ms=500, feedback=0.5, wet=0.3),
        reverb=ReverbConfig(room_size=0.9, damping=0.3, wet=0.5, pre_delay_ms=20),
    ),
    "edm_lead": EffectChain(
        name="EDM Lead",
        eq_bands=[
            EQBand(freq_hz=100, gain_db=-4, q=0.5, type="lowshelf"),
            EQBand(freq_hz=2500, gain_db=4, q=1.2),
            EQBand(freq_hz=8000, gain_db=2, q=0.8, type="highshelf"),
        ],
        compressor=CompressorConfig(threshold_db=-15, ratio=3, attack_ms=2, release_ms=50, makeup_db=4),
        distortion=DistortionConfig(drive=3.0, type="soft_clip", mix=0.2),
        delay=DelayConfig(time_ms=250, feedback=0.3, wet=0.2),
    ),
}
