"""AURALIS Console — Studio-grade mastering chain.

Ported from produce/tools/master.py with typed interfaces.

Chain:
  ① Mid/Side EQ (surgical frequency shaping)
  ② Soft Clipper Stage 1 (pre-multiband peak taming)
  ③ Multiband Saturation (per-band harmonic warmth)
  ④ Harmonic Exciter (fill spectral gaps)
  ⑤ Multiband Compression (3-band with envelope followers)
  ⑥ Makeup Gain → target LUFS
  ⑦ Stereo Width (M/S widening + bass mono <120Hz)
  ⑧ Soft Clipper Stage 2 (pre-limiter)
  ⑨ 4x Oversampled Brickwall Limiter (true-peak safe)
  ⑩ TPDF Dithering (if downsampling bit depth)

Presets: mood_check (-14 LUFS), streaming (-14 LUFS), club (-8 LUFS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import filtfilt, firwin

import structlog

logger = structlog.get_logger()


# ── Types ────────────────────────────────────────────────

@dataclass
class MasterConfig:
    """Mastering chain parameters."""

    target_lufs: float = -8.0
    ceiling_db: float = -1.0
    drive: float = 1.5
    width: float = 1.3
    bit_depth: int = 24
    oversample: int = 4
    bpm: float = 125.0
    section_bars: list[int] = field(default_factory=list)
    custom_eq: dict | None = None
    skip_stages: list[str] = field(default_factory=list)
    brain_plan: Any | None = None  # MasterPlan from DNABrain


@dataclass
class MasterResult:
    """Output of mastering chain."""

    output_path: str
    peak_dbtp: float
    rms_db: float
    est_lufs: float
    clipping_samples: int
    stages_applied: list[str]


PRESETS: dict[str, MasterConfig] = {
    "mood_check": MasterConfig(
        target_lufs=-14, drive=1.0, width=1.1, ceiling_db=-1.0
    ),
    "streaming": MasterConfig(
        target_lufs=-14, drive=1.2, width=1.2, ceiling_db=-1.0
    ),
    "club": MasterConfig(
        target_lufs=-8, drive=1.5, width=1.3, ceiling_db=-1.0
    ),
}


# ── Utilities ────────────────────────────────────────────


def _linear_phase_split_3band(
    data: np.ndarray, low_cutoff: float, high_cutoff: float,
    sr: int, ntaps: int = 1025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split into 3 bands using linear-phase FIR (zero phase distortion)."""
    fir_lo = firwin(ntaps, low_cutoff, fs=sr, pass_zero=True, window="blackman")
    fir_hi = firwin(ntaps, high_cutoff, fs=sr, pass_zero=False, window="blackman")

    if data.ndim == 2:
        low = np.column_stack([filtfilt(fir_lo, 1.0, data[:, ch]) for ch in range(data.shape[1])])
        high = np.column_stack([filtfilt(fir_hi, 1.0, data[:, ch]) for ch in range(data.shape[1])])
    else:
        low = filtfilt(fir_lo, 1.0, data)
        high = filtfilt(fir_hi, 1.0, data)

    return low, data - low - high, high


def _rms_db(data: np.ndarray) -> float:
    return float(20 * np.log10(max(np.sqrt(np.mean(data**2)), 1e-10)))


def _peak_db(data: np.ndarray) -> float:
    return float(20 * np.log10(max(np.max(np.abs(data)), 1e-10)))


def _to_ms(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (data[:, 0] + data[:, 1]) * 0.5, (data[:, 0] - data[:, 1]) * 0.5


def _from_ms(mid: np.ndarray, side: np.ndarray) -> np.ndarray:
    return np.column_stack([mid + side, mid - side])


def _biquad_peak(fc: float, gain_db: float, q: float, sr: int) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * q)
    b0, b1, b2 = 1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a
    a0, a1, a2 = 1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def _biquad_shelf(fc: float, gain_db: float, shelf_type: str, sr: int) -> np.ndarray:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / 2 * np.sqrt((a + 1 / a) * (1 / 0.7 - 1) + 2)
    c, s = np.cos(w0), np.sqrt(a)
    if shelf_type == "low":
        b0, b1, b2 = a * ((a+1) - (a-1)*c + 2*s*alpha), 2*a*((a-1) - (a+1)*c), a * ((a+1) - (a-1)*c - 2*s*alpha)
        a0, a1, a2 = (a+1) + (a-1)*c + 2*s*alpha, -2*((a-1) + (a+1)*c), (a+1) + (a-1)*c - 2*s*alpha
    else:
        b0, b1, b2 = a * ((a+1) + (a-1)*c + 2*s*alpha), -2*a*((a-1) + (a+1)*c), a * ((a+1) + (a-1)*c - 2*s*alpha)
        a0, a1, a2 = (a+1) - (a-1)*c + 2*s*alpha, 2*((a-1) - (a+1)*c), (a+1) - (a-1)*c - 2*s*alpha
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


# ── Stages ───────────────────────────────────────────────


def apply_ms_eq(data: np.ndarray, sr: int, custom_eq: dict | None = None, brain_plan: Any | None = None) -> np.ndarray:
    """① Mid/Side EQ — surgical tonal balance. Brain-guided when available."""
    mid, side = _to_ms(data)
    sos = signal.iirfilter(4, 20 / (sr / 2), btype="high", ftype="butter", output="sos")
    mid = signal.sosfilt(sos, mid).astype(np.float64)
    side = signal.sosfilt(sos, side).astype(np.float64)

    # Mid EQ: brain-guided or custom or default
    if brain_plan and getattr(brain_plan, 'mid_eq_bands', []):
        mid_sos = np.vstack([_biquad_peak(f, g, q, sr) for f, g, q in brain_plan.mid_eq_bands])
    elif custom_eq and custom_eq.get("mid_bands"):
        mc = custom_eq["mid_bands"]
        mid_sos = np.vstack([_biquad_peak(f, g, q, sr) for f, g, q in mc])
    else:
        mid_sos = np.vstack([
            _biquad_peak(300, -2.5, 1.5, sr), _biquad_peak(3000, 1.5, 1.2, sr), _biquad_peak(6000, 1.0, 0.8, sr)])
    mid = signal.sosfilt(mid_sos, mid).astype(np.float64)

    # Side EQ: brain-guided or custom or default
    if brain_plan and getattr(brain_plan, 'side_eq_bands', []):
        side_sos = np.vstack([_biquad_peak(f, g, q, sr) for f, g, q in brain_plan.side_eq_bands])
    elif custom_eq and custom_eq.get("side_bands"):
        sc = custom_eq["side_bands"]
        side_sos = np.vstack([_biquad_peak(f, g, q, sr) for f, g, q in sc])
    else:
        side_sos = np.vstack([
            _biquad_shelf(8000, 2.0, "high", sr), _biquad_peak(5000, 1.5, 1.0, sr), _biquad_peak(250, -3.0, 1.2, sr)])
    side = signal.sosfilt(side_sos, side).astype(np.float64)
    return _from_ms(mid, side)


def apply_soft_clip(data: np.ndarray, ceiling_db: float = -1.5) -> np.ndarray:
    """Soft clipper — tanh-based smooth clipping."""
    ceiling = 10 ** (ceiling_db / 20)
    if np.max(np.abs(data)) < ceiling:
        return data
    scaled = data / ceiling
    return np.where(np.abs(scaled) <= 1.0, data, np.sign(data) * ceiling * np.tanh(np.abs(scaled)))


def apply_multiband_saturation(data: np.ndarray, sr: int,
                                drive_low: float = 1.2, drive_mid: float = 1.5,
                                drive_high: float = 1.8) -> np.ndarray:
    """③ 3-band saturation with linear-phase FIR crossovers."""
    low, mid, high = _linear_phase_split_3band(data, 500, 4000, sr)
    def _sc(x, d):
        x = np.clip(x * d, -1.5, 1.5); return (x - x**3 / 3) / (1 - 1/3)
    def _st(x, d):
        return np.tanh(x * d) / np.tanh(d)
    pl, pm, ph = max(np.max(np.abs(low)), 1e-10), max(np.max(np.abs(mid)), 1e-10), max(np.max(np.abs(high)), 1e-10)
    return _sc(low / pl, drive_low) * pl + _st(mid / pm, drive_mid) * pm + _st(high / ph, drive_high) * ph


def apply_harmonic_exciter(data: np.ndarray, sr: int,
                            mix_even: float = 0.06, mix_odd: float = 0.02) -> np.ndarray:
    """④ Fill spectral gaps with even/odd harmonics above 1kHz."""
    sos = signal.iirfilter(4, 1000 / (sr / 2), btype="high", ftype="butter", output="sos")
    src = signal.sosfilt(sos, data, axis=0)
    pk = np.max(np.abs(src))
    if pk < 1e-10:
        return data
    n = src / pk
    even = signal.sosfilt(sos, np.abs(n) - 0.5, axis=0)
    odd = signal.sosfilt(sos, np.clip(n * 2, -1, 1) ** 3, axis=0)
    return data + even * pk * mix_even + odd * pk * mix_odd


def _compress_band(data: np.ndarray, thresh_db: float, ratio: float,
                   atk_ms: float, rel_ms: float, sr: int, makeup_db: float = 0) -> np.ndarray:
    """Single-band compressor with smooth envelope follower."""
    makeup = 10 ** (makeup_db / 20)
    atk_c = np.exp(-1.0 / (atk_ms * sr / 1000))
    rel_c = np.exp(-1.0 / (rel_ms * sr / 1000))
    env_in = np.max(np.abs(data), axis=1) if data.ndim == 2 else np.abs(data)
    env = np.zeros_like(env_in)
    env[0] = env_in[0]
    for i in range(1, len(env_in)):
        c = atk_c if env_in[i] > env[i - 1] else rel_c
        env[i] = c * env[i - 1] + (1 - c) * env_in[i]
    knee = 6.0
    env_db = 20 * np.log10(np.maximum(env, 1e-10))
    g_db = np.zeros_like(env_db)
    for i in range(len(env_db)):
        d = env_db[i] - thresh_db
        if d < -knee / 2:
            g_db[i] = 0
        elif d > knee / 2:
            g_db[i] = -(d - d / ratio)
        else:
            x = d + knee / 2
            g_db[i] = -(x**2 / (2 * knee)) * (1 - 1 / ratio)
    g = 10 ** (g_db / 20) * makeup
    return data * g[:, np.newaxis] if data.ndim == 2 else data * g


def apply_multiband_compression(data: np.ndarray, sr: int, brain_plan: Any | None = None) -> np.ndarray:
    """⑤ 3-band compressor with linear-phase FIR crossovers. Brain-guided when available."""
    low, mid, high = _linear_phase_split_3band(data, 500, 4000, sr)
    if brain_plan and getattr(brain_plan, 'comp_low', {}):
        cl = brain_plan.comp_low
        cm = brain_plan.comp_mid
        ch = brain_plan.comp_high
        return (
            _compress_band(low, cl.get('threshold_db', -18), cl.get('ratio', 4.0),
                          cl.get('attack_ms', 10), cl.get('release_ms', 100), sr) +
            _compress_band(mid, cm.get('threshold_db', -14), cm.get('ratio', 2.5),
                          cm.get('attack_ms', 8), cm.get('release_ms', 80), sr) +
            _compress_band(high, ch.get('threshold_db', -10), ch.get('ratio', 1.5),
                          ch.get('attack_ms', 2), ch.get('release_ms', 40), sr)
        )
    return (_compress_band(low, -18, 4.0, 10, 100, sr) +
            _compress_band(mid, -14, 2.5, 8, 80, sr) +
            _compress_band(high, -10, 1.5, 2, 40, sr))


def apply_stereo_width(data: np.ndarray, width: float = 1.3, sr: int = 44100) -> np.ndarray:
    """⑦ M/S stereo width with bass mono below 120Hz."""
    if data.ndim != 2 or data.shape[1] != 2:
        return data
    mid, side = _to_ms(data)
    result = _from_ms(mid, side * width)
    fir_lo = firwin(1025, 120, fs=sr, pass_zero=True, window="blackman")
    fir_hi = firwin(1025, 120, fs=sr, pass_zero=False, window="blackman")
    low = np.column_stack([filtfilt(fir_lo, 1.0, result[:, ch]) for ch in range(2)])
    high = np.column_stack([filtfilt(fir_hi, 1.0, result[:, ch]) for ch in range(2)])
    mono = np.mean(low, axis=1, keepdims=True)
    return np.hstack([mono, mono]) + high


def apply_oversampled_limiter(data: np.ndarray, ceiling_db: float = -1.0,
                               sr: int = 44100, oversample: int = 4) -> np.ndarray:
    """⑨ True-peak limiter with 4x oversampling and 5ms lookahead."""
    ceiling = 10 ** (ceiling_db / 20)
    n = len(data)
    if data.ndim == 2:
        up = np.zeros((n * oversample, 2), dtype=np.float64)
        for ch in range(2):
            up[:, ch] = signal.resample_poly(data[:, ch], oversample, 1)
    else:
        up = signal.resample_poly(data, oversample, 1)
    up_sr = sr * oversample
    la = int(0.005 * up_sr)
    rel_c = np.exp(-1.0 / (50 * up_sr / 1000))
    env = np.max(np.abs(up), axis=1) if up.ndim == 2 else np.abs(up)
    gain = np.ones(len(env), dtype=np.float64)
    above = env > ceiling
    if np.any(above):
        gain[above] = ceiling / env[above]
    sg = np.ones_like(gain)
    padded = np.pad(gain, (0, la), mode="edge")
    for i in range(len(gain)):
        sg[i] = np.min(padded[i:i + la + 1])
    for i in range(1, len(sg)):
        if sg[i] > sg[i - 1]:
            sg[i] = rel_c * sg[i - 1] + (1 - rel_c) * sg[i]
    if up.ndim == 2:
        up *= sg[:, np.newaxis]
        result = np.zeros((n, 2), dtype=np.float64)
        for ch in range(2):
            result[:, ch] = signal.resample_poly(up[:, ch], 1, oversample)[:n]
    else:
        result = signal.resample_poly(up * sg, 1, oversample)[:n]
    np.clip(result, -1.0, 1.0, out=result)
    return result


def apply_dither(data: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """⑩ TPDF dithering for bit depth reduction."""
    levels = 2 ** (bit_depth - 1)
    return data + (np.random.uniform(-0.5, 0.5, data.shape) +
                   np.random.uniform(-0.5, 0.5, data.shape)) / levels


# ── Main Chain ───────────────────────────────────────────


def master_audio(input_path: str | Path, output_path: str | Path | None = None,
                  config: MasterConfig | None = None,
                  preset: str | None = None) -> MasterResult:
    """Run the full studio-grade mastering chain."""
    if config is None:
        config = PRESETS.get(preset or "club", MasterConfig())

    p = Path(input_path)
    if output_path is None:
        output_path = str(p.parent / f"{p.stem}_MASTER{p.suffix}")

    data, sr = sf.read(str(input_path), dtype="float64")
    if data.ndim == 1:
        data = np.column_stack([data, data])

    skip = set(config.skip_stages)
    stages: list[str] = []

    logger.info("mastering_start", input=p.name, target_lufs=config.target_lufs)

    # Apply brain plan overrides to config
    bp = config.brain_plan
    if bp:
        if getattr(bp, 'target_lufs', None) and bp.target_lufs < 0:
            config.target_lufs = bp.target_lufs
        if getattr(bp, 'drive', None):
            config.drive = bp.drive
        if getattr(bp, 'width', None):
            config.width = bp.width
        logger.info(
            "mastering_brain",
            lufs=config.target_lufs,
            drive=config.drive,
            width=config.width,
        )

    # ⓪ DC offset + section smoothing
    data -= np.mean(data, axis=0)
    bar_s = round(60 / config.bpm * 4 * sr)
    fade_s = int(0.005 * sr)
    for sbar in config.section_bars:
        pos = sbar * bar_s
        if pos >= len(data) or pos < fade_s:
            continue
        fo = np.cos(np.linspace(0, np.pi / 2, fade_s)) ** 2
        fi = np.cos(np.linspace(np.pi / 2, 0, fade_s)) ** 2
        end = min(pos + fade_s, len(data))
        data[pos - fade_s:pos, :] *= fo[:, np.newaxis]
        data[pos:end, :] *= fi[:end - pos, np.newaxis]
    stages.append("preprocess")

    data = apply_ms_eq(data, sr, custom_eq=config.custom_eq, brain_plan=bp); stages.append("ms_eq")
    if "soft_clip_1" not in skip:
        data = apply_soft_clip(data, -1.5); stages.append("soft_clip_1")
    if "saturation" not in skip:
        data = apply_multiband_saturation(data, sr, config.drive*0.8, config.drive, config.drive*1.2); stages.append("saturation")
    if "exciter" not in skip:
        data = apply_harmonic_exciter(data, sr); stages.append("exciter")
    if "compression" not in skip:
        data = apply_multiband_compression(data, sr, brain_plan=bp); stages.append("compression")

    gain_needed = (config.target_lufs + 0.7) - _rms_db(data)
    if gain_needed > 0:
        data *= 10 ** (gain_needed / 20)
    stages.append("makeup_gain")

    data = apply_stereo_width(data, config.width, sr); stages.append("stereo_width")
    if "soft_clip_2" not in skip:
        data = apply_soft_clip(data, -0.5); stages.append("soft_clip_2")
    data = apply_oversampled_limiter(data, config.ceiling_db, sr, config.oversample); stages.append("limiter")

    if config.bit_depth < 24:
        data = apply_dither(data, config.bit_depth); stages.append("dither")

    subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}[config.bit_depth]
    sf.write(str(output_path), data, sr, subtype=subtype)

    result = MasterResult(
        output_path=str(output_path), peak_dbtp=_peak_db(data),
        rms_db=_rms_db(data), est_lufs=_rms_db(data) - 0.7,
        clipping_samples=int(np.sum(np.abs(data) >= 1.0)),
        stages_applied=stages)

    logger.info("mastering_complete", output=Path(str(output_path)).name,
                peak_dbtp=result.peak_dbtp, est_lufs=result.est_lufs)
    return result
