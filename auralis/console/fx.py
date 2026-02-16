"""AURALIS console.fx — Professional-grade audio effects.

Ported from produce/tools/audio_engine.py with full type annotations.
Every effect operates on numpy arrays at sample level for maximum precision.

Effects:
    - Biquad filters (lowpass, highpass)
    - Moog-style resonant ladder filter
    - Reverb (multi-tap algorithmic)
    - Sidechain compression
    - Slapback delay
    - Vibrato LFO
    - Bitcrusher
    - Saturation (tanh + soft-clip)
    - Glue compression (SSL-style)
    - Fade automation
    - Micro-fade (click prevention)
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

# ── Type Aliases ────────────────────────────────────────────
AudioArray = npt.NDArray[np.float64]
BiquadCoeffs = tuple[float, float, float, float, float]


# ── Unit Conversions ────────────────────────────────────────


def ms_to_samples(ms: float, sr: int = 44100) -> int:
    """Convert milliseconds to sample index."""
    return int(round(ms * sr / 1000))


def db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude."""
    return float(10 ** (db / 20))


# ── Biquad Filters ──────────────────────────────────────────


def biquad_coefficients(
    filter_type: str, cutoff_hz: float, sr: int, q: float = 0.707
) -> BiquadCoeffs:
    """2nd-order IIR biquad filter coefficients (12dB/oct).

    Args:
        filter_type: 'lowpass' or 'highpass'
        cutoff_hz: filter cutoff frequency in Hz
        sr: sample rate
        q: resonance (0.707 = Butterworth, flat response)

    Returns:
        Tuple of (b0, b1, b2, a1, a2) normalized coefficients.
    """
    omega = 2 * np.pi * cutoff_hz / sr
    sin_w = float(np.sin(omega))
    cos_w = float(np.cos(omega))
    alpha = sin_w / (2 * q)

    if filter_type == "lowpass":
        b0 = (1 - cos_w) / 2
        b1 = 1 - cos_w
        b2 = (1 - cos_w) / 2
    elif filter_type == "highpass":
        b0 = (1 + cos_w) / 2
        b1 = -(1 + cos_w)
        b2 = (1 + cos_w) / 2
    else:
        msg = f"Unknown filter type: {filter_type}"
        raise ValueError(msg)

    a0 = 1 + alpha
    a1 = -2 * cos_w
    a2 = 1 - alpha

    return (b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)


def apply_biquad(data: AudioArray, coeffs: BiquadCoeffs) -> AudioArray:
    """Apply biquad filter to audio data (mono or stereo)."""
    b0, b1, b2, a1, a2 = coeffs
    result = np.zeros_like(data)

    def _filter_channel(x: AudioArray) -> AudioArray:
        y = np.zeros_like(x)
        x1, x2, y1, y2 = 0.0, 0.0, 0.0, 0.0
        for i in range(len(x)):
            y[i] = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, float(x[i])
            y2, y1 = y1, float(y[i])
        return y

    if data.ndim == 2:
        for ch in range(data.shape[1]):
            result[:, ch] = _filter_channel(data[:, ch])
    else:
        result = _filter_channel(data)
    return result


def apply_lowpass(
    data: AudioArray, cutoff_hz: float, sr: int = 44100, q: float = 0.707
) -> AudioArray:
    """2nd-order lowpass filter (12dB/oct Butterworth)."""
    return apply_biquad(data, biquad_coefficients("lowpass", cutoff_hz, sr, q))


def apply_highpass(
    data: AudioArray, cutoff_hz: float, sr: int = 44100, q: float = 0.707
) -> AudioArray:
    """2nd-order highpass filter (12dB/oct Butterworth)."""
    return apply_biquad(data, biquad_coefficients("highpass", cutoff_hz, sr, q))


# ── Moog-Style Resonant Ladder Filter ───────────────────────


def apply_moog_filter(
    data: AudioArray,
    cutoff_hz: float = 1000.0,
    resonance: float = 0.7,
    env_amount: float = 0.0,
    env_attack_ms: float = 10.0,
    env_decay_ms: float = 200.0,
    sr: int = 44100,
) -> AudioArray:
    """4-pole resonant lowpass (24dB/oct Moog approximation).

    Args:
        cutoff_hz: filter cutoff frequency
        resonance: 0.0–1.0 (0.5=warm, 0.7=character, 0.9=screaming)
        env_amount: 0.0–1.0 envelope modulation depth
        env_attack_ms: envelope attack for filter sweep
        env_decay_ms: envelope decay for filter sweep
    """
    q = 0.707 + resonance * 14.0

    if env_amount > 0:
        n_samples = data.shape[0]
        attack_samps = int(env_attack_ms / 1000 * sr)
        decay_samps = int(env_decay_ms / 1000 * sr)

        env = np.ones(n_samples)
        if attack_samps > 0:
            env[:attack_samps] = np.linspace(0, 1, attack_samps)
        decay_end = min(attack_samps + decay_samps, n_samples)
        if decay_samps > 0 and attack_samps < n_samples:
            env[attack_samps:decay_end] = np.linspace(
                1, 0, decay_end - attack_samps
            )
        if decay_end < n_samples:
            env[decay_end:] = 0

        max_cutoff = min(cutoff_hz * 4, sr * 0.45)
        modulated = cutoff_hz + env * env_amount * (max_cutoff - cutoff_hz)

        block_size = 64
        result = data.copy()
        for i in range(0, len(data), block_size):
            end = min(i + block_size, len(data))
            mid = float(modulated[min(i + block_size // 2, len(data) - 1)])
            coeffs = biquad_coefficients("lowpass", mid, sr, q)
            block = apply_biquad(result[i:end], coeffs)
            block = apply_biquad(block, coeffs)
            result[i:end] = block
        return result

    coeffs = biquad_coefficients("lowpass", cutoff_hz, sr, q)
    result = apply_biquad(data.copy(), coeffs)
    return apply_biquad(result, coeffs)


# ── Reverb ──────────────────────────────────────────────────


def apply_reverb(
    data: AudioArray,
    room_size: float = 0.6,
    damping: float = 0.5,
    wet: float = 0.35,
    sr: int = 44100,
) -> AudioArray:
    """Multi-tap algorithmic reverb with LP damping + stereo width.

    Args:
        room_size: 0.0–1.0 (small room → large hall)
        damping: 0.0–1.0 (bright → dark reverb)
        wet: 0.0–1.0 (dry/wet mix)
    """
    base_delay_ms = 20 + room_size * 60
    tap_delays = [base_delay_ms * m for m in
                  [1.0, 1.47, 2.13, 2.89, 3.53, 4.17, 5.31, 6.73]]
    decay_base = 0.35 + room_size * 0.4

    mono = np.mean(data, axis=1) if data.ndim == 2 else data.copy()
    max_delay = ms_to_samples(max(tap_delays), sr)
    tail = int(sr * room_size * 2)
    out_len = len(mono) + max_delay + tail
    reverb_sig = np.zeros(out_len, dtype=np.float64)

    for i, dms in enumerate(tap_delays):
        d = ms_to_samples(dms, sr)
        decay = decay_base ** (i + 1) * (1 - damping * 0.3)
        reverb_sig[d : d + len(mono)] += mono * decay

    cutoff = 2000 + (1 - damping) * 6000
    coeffs = biquad_coefficients("lowpass", cutoff, sr, q=0.5)
    reverb_sig = apply_biquad(
        reverb_sig.reshape(-1, 1), coeffs
    ).flatten()

    dry = data.copy()
    if dry.ndim == 2:
        offset = ms_to_samples(3, sr)
        rev_l = reverb_sig[: len(dry)]
        rev_r = np.zeros(len(dry), dtype=np.float64)
        rev_r[offset:] = reverb_sig[: len(dry) - offset]
        result = np.zeros_like(dry)
        result[:, 0] = dry[:, 0] * (1 - wet) + rev_l * wet
        result[:, 1] = dry[:, 1] * (1 - wet) + rev_r * wet
    else:
        result = dry * (1 - wet)
        result += reverb_sig[: len(dry)] * wet
    return result


# ── Slapback Delay ──────────────────────────────────────────


def apply_slapback(
    data: AudioArray,
    delay_ms: float = 80,
    feedback_db: float = -8,
    sr: int = 44100,
) -> AudioArray:
    """Single-repeat slapback delay mixed with original."""
    delay_samples = ms_to_samples(delay_ms, sr)
    gain = db_to_linear(feedback_db)

    out_len = len(data) + delay_samples
    if data.ndim == 2:
        result = np.zeros((out_len, data.shape[1]), dtype=np.float64)
    else:
        result = np.zeros(out_len, dtype=np.float64)
    result[: len(data)] = data
    result[delay_samples : delay_samples + len(data)] += data * gain
    return result


# ── Sidechain Compression ──────────────────────────────────


def apply_sidechain(
    data: AudioArray,
    trigger: AudioArray,
    depth_db: float = -12,
    attack_ms: float = 1,
    release_ms: float = 100,
    sr: int = 44100,
) -> AudioArray:
    """Classic sidechain pumping effect.

    Ducks audio when trigger signal (e.g. kick) hits.
    """
    trigger_mono = (
        np.mean(np.abs(trigger), axis=1) if trigger.ndim == 2
        else np.abs(trigger)
    )
    peak = np.max(trigger_mono)
    if peak > 0:
        trigger_mono = trigger_mono / peak

    attack_coeff = np.exp(-1.0 / (attack_ms / 1000 * sr))
    release_coeff = np.exp(-1.0 / (release_ms / 1000 * sr))

    envelope = np.zeros(len(trigger_mono))
    for i in range(1, len(trigger_mono)):
        if trigger_mono[i] > envelope[i - 1]:
            envelope[i] = (
                attack_coeff * envelope[i - 1]
                + (1 - attack_coeff) * trigger_mono[i]
            )
        else:
            envelope[i] = release_coeff * envelope[i - 1]

    depth_linear = db_to_linear(depth_db)
    gain = 1.0 - envelope * (1.0 - depth_linear)
    if len(gain) < len(data):
        gain = np.pad(gain, (0, len(data) - len(gain)), constant_values=1.0)
    gain = gain[: len(data)]

    result = data.copy()
    if result.ndim == 2:
        for ch in range(result.shape[1]):
            result[:, ch] *= gain
    else:
        result *= gain
    return result


def generate_sidechain_trigger(
    bpm: float,
    bars: int,
    sr: int = 44100,
    grid: str = "beat",
    impulse_ms: float = 50,
    stereo: bool = True,
) -> AudioArray:
    """Generate trigger signal at rhythmic grid positions."""
    bar_samples = int(60 / bpm * 4 * sr)
    beat_samples = int(60 / bpm * sr)
    total_samples = bar_samples * bars
    impulse_len = int(sr * impulse_ms / 1000)

    grid_map = {
        "half": beat_samples * 2,
        "beat": beat_samples,
        "8th": beat_samples // 2,
        "16th": beat_samples // 4,
    }
    step = grid_map.get(grid, beat_samples)

    shape = (total_samples, 2) if stereo else (total_samples,)
    trigger: AudioArray = np.zeros(shape, dtype=np.float64)

    pos = 0
    while pos < total_samples:
        end = min(pos + impulse_len, total_samples)
        trigger[pos:end] = 1.0
        pos += step
    return trigger


# ── Vibrato LFO ─────────────────────────────────────────────


def apply_vibrato_lfo(
    data: AudioArray,
    rate_hz: float = 6.25,
    depth_cents: float = 15.0,
    sr: int = 44100,
) -> AudioArray:
    """Pitch vibrato via LFO-modulated resampling.

    Signature Mita Gami effect: LFO at 1/8 rate linked to osc fine-tune.
    Creates hypnotic, alive quality on bass, lead, and acid sounds.
    """
    n_samples = data.shape[0]
    is_stereo = data.ndim == 2

    t = np.arange(n_samples, dtype=np.float64)
    lfo = np.sin(2 * np.pi * rate_hz * t / sr)
    depth_ratio = 2 ** (depth_cents / 1200.0) - 1.0
    speed_mod = 1.0 + lfo * depth_ratio
    positions = np.cumsum(speed_mod)
    positions = positions * (n_samples / positions[-1])
    positions = np.clip(positions, 0, n_samples - 2)

    idx = positions.astype(np.int64)
    frac = (positions - idx).astype(np.float64)
    idx_next = np.minimum(idx + 1, n_samples - 1)

    if is_stereo:
        result = np.zeros_like(data)
        for ch in range(data.shape[1]):
            result[:, ch] = (
                data[idx, ch] * (1 - frac)
                + data[idx_next, ch] * frac
            )
    else:
        result = data[idx] * (1 - frac) + data[idx_next] * frac
    return result


# ── Bitcrusher ──────────────────────────────────────────────


def apply_bitcrush(
    data: AudioArray,
    bits: int = 12,
    downsample: int = 2,
    mix: float = 0.3,
) -> AudioArray:
    """Bit + sample-rate reduction for lo-fi character.

    Args:
        bits: bit depth (16=clean, 12=subtle, 8=heavy, 4=extreme)
        downsample: sample rate divisor (1=off, 2=subtle, 4=heavy)
        mix: dry/wet (0.3=subtle, 0.5=noticeable, 0.8=dominant)
    """
    dry = data.copy()
    wet = data.copy()
    levels = 2 ** bits
    wet = np.round(wet * levels / 2) / (levels / 2)

    if downsample > 1:
        if wet.ndim == 2:
            for ch in range(wet.shape[1]):
                for i in range(0, len(wet), downsample):
                    wet[i : i + downsample, ch] = wet[i, ch]
        else:
            for i in range(0, len(wet), downsample):
                wet[i : i + downsample] = wet[i]

    return dry * (1 - mix) + wet * mix


# ── Saturation ──────────────────────────────────────────────


def apply_saturation(
    data: AudioArray, drive: float = 0.03, mix: float = 0.3
) -> AudioArray:
    """Tanh saturation for analog warmth."""
    dry = data.copy()
    driven = data * (1 + drive * 20)
    wet = np.tanh(driven) / np.tanh(1 + drive * 20)
    return dry * (1 - mix) + wet * mix


# ── Glue Compression ───────────────────────────────────────


def apply_glue_compression(
    data: AudioArray,
    threshold_db: float = -10,
    ratio: float = 4.0,
    attack_ms: float = 10,
    release_ms: float = 100,
    makeup_db: float | None = None,
    sr: int = 44100,
) -> AudioArray:
    """SSL-style bus compressor for glue and cohesion."""
    mono = (
        np.mean(np.abs(data), axis=1) if data.ndim == 2
        else np.abs(data)
    )
    threshold_lin = db_to_linear(threshold_db)
    attack_coeff = np.exp(-1.0 / (attack_ms / 1000 * sr))
    release_coeff = np.exp(-1.0 / (release_ms / 1000 * sr))

    envelope = np.zeros(len(mono))
    for i in range(1, len(mono)):
        if mono[i] > envelope[i - 1]:
            envelope[i] = (
                attack_coeff * envelope[i - 1]
                + (1 - attack_coeff) * mono[i]
            )
        else:
            envelope[i] = release_coeff * envelope[i - 1]

    gain = np.ones(len(envelope))
    for i in range(len(envelope)):
        if envelope[i] > threshold_lin:
            over_db = 20 * np.log10(envelope[i] / threshold_lin + 1e-10)
            reduction_db = over_db * (1 - 1 / ratio)
            gain[i] = db_to_linear(-reduction_db)

    if makeup_db is None:
        avg_reduction = float(np.mean(20 * np.log10(gain + 1e-10)))
        makeup_db = -avg_reduction * 0.6
    makeup_lin = db_to_linear(makeup_db)
    gain *= makeup_lin

    result = data.copy()
    if result.ndim == 2:
        for ch in range(result.shape[1]):
            result[:, ch] *= gain
    else:
        result *= gain
    return result


# ── Fade Automation ─────────────────────────────────────────


def apply_fade_automation(
    data: AudioArray,
    direction: str,
    start_sample: int,
    end_sample: int,
) -> AudioArray:
    """Volume fade automation over a sample range.

    Args:
        direction: 'in' (silence→full) or 'out' (full→silence)
        start_sample: fade start position
        end_sample: fade end position
    """
    result = data.copy()
    start = max(0, min(start_sample, len(data)))
    end = max(start, min(end_sample, len(data)))
    if end <= start:
        return result

    fade_len = end - start
    if direction == "in":
        curve = np.linspace(0, 1, fade_len)
    elif direction == "out":
        curve = np.linspace(1, 0, fade_len)
    else:
        msg = f"direction must be 'in' or 'out', got '{direction}'"
        raise ValueError(msg)

    if result.ndim == 2:
        for ch in range(result.shape[1]):
            result[start:end, ch] *= curve
    else:
        result[start:end] *= curve
    return result


def apply_micro_fade(
    data: AudioArray, fade_samples: int = 64
) -> AudioArray:
    """Micro-fade in/out to prevent clicks at audio boundaries.

    64 samples @ 44.1kHz = 1.5ms — imperceptible but prevents clicks.
    """
    result = data.copy()
    fade_samples = min(fade_samples, len(data) // 4)
    if fade_samples < 2:
        return result

    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    if result.ndim == 2:
        result[:fade_samples] *= fade_in[:, np.newaxis]
        result[-fade_samples:] *= fade_out[:, np.newaxis]
    else:
        result[:fade_samples] *= fade_in
        result[-fade_samples:] *= fade_out
    return result
