"""AURALIS QC — Audio Quality Control & Spectral Comparator.

Ported from produce/tools/audio_qc.py with typed interfaces.

Analysis capabilities:
  - Dynamics: Peak, RMS, Crest Factor, Dynamic Range
  - Clipping detection
  - Stereo: Correlation, width, mono compatibility
  - Loudness: ITU-R BS.1770 (via pyloudnorm)
  - Spectrum: 7-band energy distribution
  - Spectral fingerprint comparison vs reference
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

import structlog

logger = structlog.get_logger()

# Optional imports
try:
    import pyloudnorm as pyln
    HAS_PYLOUDNORM = True
except ImportError:
    HAS_PYLOUDNORM = False

try:
    from scipy import signal as scipy_signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ── Types ────────────────────────────────────────────────

@dataclass
class DynamicsResult:
    peak_db: float
    rms_db: float
    crest_factor_db: float
    dynamic_range_db: float


@dataclass
class ClippingResult:
    is_clipping: bool
    clipped_samples: int
    clipped_percent: float
    max_consecutive: int


@dataclass
class StereoResult:
    correlation: float
    width: float
    mono_compatible: bool
    balance_db: float


@dataclass
class LoudnessResult:
    integrated_lufs: float
    short_term_max: float
    true_peak_dbtp: float


@dataclass
class SpectrumResult:
    """7-band spectral energy (dB)."""
    sub: float       # 20-60 Hz
    bass: float      # 60-250 Hz
    low_mid: float   # 250-500 Hz
    mid: float       # 500-2k Hz
    upper_mid: float # 2k-4k Hz
    presence: float  # 4k-8k Hz
    brilliance: float # 8k-20k Hz


@dataclass
class QCReport:
    """Complete QC analysis report."""
    filepath: str
    duration_s: float
    sample_rate: int
    channels: int
    dynamics: DynamicsResult
    clipping: ClippingResult
    stereo: StereoResult | None
    loudness: LoudnessResult | None
    spectrum: SpectrumResult
    pass_fail: str  # "PASS" or "FAIL"
    issues: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """A vs B spectral comparison."""
    track_a: str
    track_b: str
    band_diffs_db: dict[str, float]
    overall_diff_db: float
    match_score: float  # 0-100


# ── Analysis Functions ───────────────────────────────────


def analyze_dynamics(data: np.ndarray, sr: int) -> DynamicsResult:
    """Analyze peak, RMS, crest factor, dynamic range."""
    mono = np.mean(data, axis=1) if data.ndim == 2 else data

    peak = np.max(np.abs(mono))
    peak_db = float(20 * np.log10(max(peak, 1e-10)))

    rms = np.sqrt(np.mean(mono ** 2))
    rms_db = float(20 * np.log10(max(rms, 1e-10)))

    crest = peak_db - rms_db

    # Dynamic range via windowed RMS
    win = int(0.4 * sr)  # 400ms windows
    if len(mono) > win:
        n_windows = len(mono) // win
        rms_vals = []
        for i in range(n_windows):
            chunk = mono[i * win:(i + 1) * win]
            r = np.sqrt(np.mean(chunk ** 2))
            if r > 1e-10:
                rms_vals.append(20 * np.log10(r))
        if rms_vals:
            dr = max(rms_vals) - min(rms_vals)
        else:
            dr = 0.0
    else:
        dr = 0.0

    return DynamicsResult(
        peak_db=peak_db, rms_db=rms_db,
        crest_factor_db=crest, dynamic_range_db=dr
    )


def analyze_clipping(data: np.ndarray, threshold: float = 0.99) -> ClippingResult:
    """Detect clipping in audio."""
    mono = np.mean(data, axis=1) if data.ndim == 2 else data
    clipped = np.abs(mono) >= threshold
    n_clipped = int(np.sum(clipped))

    # Max consecutive
    max_consec = 0
    current = 0
    for c in clipped:
        if c:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    return ClippingResult(
        is_clipping=n_clipped > 10,
        clipped_samples=n_clipped,
        clipped_percent=n_clipped / len(mono) * 100,
        max_consecutive=max_consec,
    )


def analyze_stereo(data: np.ndarray, sr: int = 44100) -> StereoResult | None:
    """Stereo correlation, width, mono compatibility."""
    if data.ndim != 2 or data.shape[1] != 2:
        return None

    left, right = data[:, 0], data[:, 1]

    # Pearson correlation
    corr = float(np.corrcoef(left, right)[0, 1])

    # Width from M/S ratio
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    mid_rms = np.sqrt(np.mean(mid ** 2))
    side_rms = np.sqrt(np.mean(side ** 2))
    width = float(side_rms / max(mid_rms, 1e-10))

    # Mono compatibility
    mono = left + right
    mono_rms = np.sqrt(np.mean(mono ** 2))
    orig_rms = np.sqrt(np.mean(left ** 2) + np.mean(right ** 2))
    mono_ok = mono_rms / max(orig_rms, 1e-10) > 0.7

    # Balance
    l_rms = 20 * np.log10(max(np.sqrt(np.mean(left ** 2)), 1e-10))
    r_rms = 20 * np.log10(max(np.sqrt(np.mean(right ** 2)), 1e-10))

    return StereoResult(
        correlation=corr, width=width,
        mono_compatible=bool(mono_ok), balance_db=float(l_rms - r_rms)
    )


def analyze_loudness(data: np.ndarray, sr: int) -> LoudnessResult | None:
    """ITU-R BS.1770 loudness analysis."""
    if not HAS_PYLOUDNORM:
        return None

    meter = pyln.Meter(sr)
    lufs = float(meter.integrated_loudness(data))

    # Short-term max (3s windows)
    win = 3 * sr
    st_max = -100.0
    for i in range(0, len(data) - win, sr):
        chunk = data[i:i + win]
        try:
            st = meter.integrated_loudness(chunk)
            if st > st_max:
                st_max = st
        except Exception:
            pass

    # True peak
    peak = float(20 * np.log10(max(np.max(np.abs(data)), 1e-10)))

    return LoudnessResult(
        integrated_lufs=lufs,
        short_term_max=float(st_max),
        true_peak_dbtp=peak,
    )


def analyze_spectrum(data: np.ndarray, sr: int) -> SpectrumResult:
    """7-band spectral energy analysis."""
    mono = np.mean(data, axis=1) if data.ndim == 2 else data
    fft = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(len(mono), 1.0 / sr)
    mag = np.abs(fft) ** 2

    bands = {
        "sub": (20, 60), "bass": (60, 250), "low_mid": (250, 500),
        "mid": (500, 2000), "upper_mid": (2000, 4000),
        "presence": (4000, 8000), "brilliance": (8000, 20000),
    }

    values = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        energy = np.sum(mag[mask]) if np.any(mask) else 1e-10
        values[name] = float(10 * np.log10(max(energy, 1e-10)))

    return SpectrumResult(**values)


# ── Full QC Report ───────────────────────────────────────


def run_qc(filepath: str | Path) -> QCReport:
    """Run complete QC analysis on an audio file."""
    p = Path(filepath)
    data, sr = sf.read(str(p), dtype="float64")

    if data.ndim == 1:
        channels = 1
    else:
        channels = data.shape[1]

    dynamics = analyze_dynamics(data, sr)
    clipping = analyze_clipping(data)
    stereo = analyze_stereo(data, sr)
    loudness = analyze_loudness(data, sr)
    spectrum = analyze_spectrum(data, sr)

    # Pass/Fail checks
    issues: list[str] = []
    if dynamics.peak_db > -0.1:
        issues.append(f"Peak too hot: {dynamics.peak_db:.1f} dBTP")
    if clipping.is_clipping:
        issues.append(f"Clipping detected: {clipping.clipped_samples} samples")
    if stereo and not stereo.mono_compatible:
        issues.append("Poor mono compatibility")
    if stereo and abs(stereo.balance_db) > 1.5:
        issues.append(f"Stereo imbalance: {stereo.balance_db:.1f} dB")
    if loudness and loudness.integrated_lufs < -20:
        issues.append(f"Very quiet: {loudness.integrated_lufs:.1f} LUFS")

    result = QCReport(
        filepath=str(p), duration_s=len(data) / sr,
        sample_rate=sr, channels=channels,
        dynamics=dynamics, clipping=clipping,
        stereo=stereo, loudness=loudness, spectrum=spectrum,
        pass_fail="FAIL" if issues else "PASS",
        issues=issues,
    )

    logger.info("qc_complete", file=p.name, pass_fail=result.pass_fail,
                issues=len(issues))
    return result


# ── Spectral Comparison ─────────────────────────────────


def compare_tracks(
    track_a: str | Path, track_b: str | Path
) -> ComparisonResult:
    """Compare spectral profiles of two tracks."""
    spec_a = analyze_spectrum(*sf.read(str(track_a), dtype="float64"))
    spec_b = analyze_spectrum(*sf.read(str(track_b), dtype="float64"))

    bands = ["sub", "bass", "low_mid", "mid", "upper_mid", "presence", "brilliance"]
    diffs = {}
    total_diff = 0.0

    for band in bands:
        va = getattr(spec_a, band)
        vb = getattr(spec_b, band)
        d = abs(va - vb)
        diffs[band] = round(d, 1)
        total_diff += d

    avg_diff = total_diff / len(bands)
    # Score: 100 = perfect match, 0 = very different
    score = max(0, 100 - avg_diff * 5)

    return ComparisonResult(
        track_a=str(track_a), track_b=str(track_b),
        band_diffs_db=diffs, overall_diff_db=round(avg_diff, 1),
        match_score=round(score, 1),
    )
