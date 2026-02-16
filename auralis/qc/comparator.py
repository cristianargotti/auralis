"""AURALIS QC Comparator — 12-dimension A/B audio comparison.

Compares an original track against a reconstruction across 12 dimensions:
spectral similarity, RMS match, stereo width, bass/kick patterns,
harmonic progression, energy curve, reverb, dynamic range,
BPM accuracy, arrangement match, and timbre similarity.

All analysis uses numpy + librosa — no GPU required.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import soundfile as sf

try:
    import librosa

    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


# ── Types ────────────────────────────────────────────────


@dataclass
class DimensionScore:
    """Score for a single QC dimension."""

    name: str
    score: float  # 0–100
    detail: str = ""
    weight: float = 1.0


@dataclass
class ComparisonResult:
    """Full 12-dimension comparison result."""

    dimensions: list[DimensionScore]
    overall_score: float = 0.0
    target_score: float = 90.0
    passed: bool = False
    weakest: str = ""
    strongest: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "dimensions": {ds.name: {"score": ds.score, "detail": ds.detail} for ds in self.dimensions},
            "overall_score": round(self.overall_score, 2),
            "target_score": self.target_score,
            "passed": self.passed,
            "weakest": self.weakest,
            "strongest": self.strongest,
        }
        return d


# ── Dimension Analyzers ──────────────────────────────────


def _load_audio(path: str | Path, sr: int = 44100) -> tuple[npt.NDArray[np.float64], int]:
    """Load audio file, return (data, sr)."""
    data, file_sr = sf.read(str(path), dtype="float64")
    if file_sr != sr and HAS_LIBROSA:
        data = librosa.resample(data.T if data.ndim > 1 else data, orig_sr=file_sr, target_sr=sr)
        if data.ndim > 1:
            data = data.T
        return data, sr
    return data, file_sr


def _to_mono(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert to mono if stereo."""
    if data.ndim > 1:
        return np.mean(data, axis=1)
    return data


def _spectral_similarity(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare mel spectrogram correlation."""
    if not HAS_LIBROSA:
        return DimensionScore(name="spectral_similarity", score=0, detail="librosa not available")

    n = min(len(orig), len(recon))
    S_orig = librosa.feature.melspectrogram(y=orig[:n], sr=sr, n_mels=128, fmax=sr // 2)
    S_recon = librosa.feature.melspectrogram(y=recon[:n], sr=sr, n_mels=128, fmax=sr // 2)

    S_orig_db = librosa.power_to_db(S_orig, ref=np.max)
    S_recon_db = librosa.power_to_db(S_recon, ref=np.max)

    # Flatten and compute correlation
    flat_orig = S_orig_db.flatten()
    flat_recon = S_recon_db.flatten()

    corr = float(np.corrcoef(flat_orig, flat_recon)[0, 1])
    score = max(0, corr * 100)

    return DimensionScore(
        name="spectral_similarity",
        score=round(score, 2),
        detail=f"Mel spectrogram correlation: {corr:.4f}",
        weight=1.5,
    )


def _rms_match(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare RMS levels per section."""
    n = min(len(orig), len(recon))
    # Split into 1-second windows
    window = sr
    n_windows = n // window

    if n_windows == 0:
        rms_orig = float(np.sqrt(np.mean(orig[:n] ** 2)))
        rms_recon = float(np.sqrt(np.mean(recon[:n] ** 2)))
        diff = abs(20 * np.log10(max(rms_orig, 1e-10)) - 20 * np.log10(max(rms_recon, 1e-10)))
        score = max(0, 100 - diff * 10)
        return DimensionScore(name="rms_match", score=round(score, 2), detail=f"RMS diff: {diff:.2f} dB")

    diffs = []
    for i in range(n_windows):
        s, e = i * window, (i + 1) * window
        rms_o = float(np.sqrt(np.mean(orig[s:e] ** 2)))
        rms_r = float(np.sqrt(np.mean(recon[s:e] ** 2)))
        diff = abs(20 * np.log10(max(rms_o, 1e-10)) - 20 * np.log10(max(rms_r, 1e-10)))
        diffs.append(diff)

    avg_diff = float(np.mean(diffs))
    score = max(0, 100 - avg_diff * 10)

    return DimensionScore(
        name="rms_match",
        score=round(score, 2),
        detail=f"Avg RMS diff: {avg_diff:.2f} dB across {n_windows} windows",
    )


def _stereo_width_match(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare stereo width (mid/side ratio)."""
    def _width(data: npt.NDArray[np.float64]) -> float:
        if data.ndim < 2 or data.shape[1] < 2:
            return 0.0
        mid = (data[:, 0] + data[:, 1]) / 2
        side = (data[:, 0] - data[:, 1]) / 2
        rms_mid = float(np.sqrt(np.mean(mid ** 2)))
        rms_side = float(np.sqrt(np.mean(side ** 2)))
        return rms_side / max(rms_mid, 1e-10)

    w_orig = _width(orig)
    w_recon = _width(recon)
    diff = abs(w_orig - w_recon)
    score = max(0, 100 - diff * 200)

    return DimensionScore(
        name="stereo_width_match",
        score=round(score, 2),
        detail=f"Width orig: {w_orig:.3f}, recon: {w_recon:.3f}, diff: {diff:.3f}",
    )


def _bass_pattern_match(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare bass frequency patterns (20-200 Hz)."""
    if not HAS_LIBROSA:
        return DimensionScore(name="bass_pattern_match", score=0, detail="librosa not available")

    n = min(len(orig), len(recon))
    o_mono = _to_mono(orig[:n]) if orig.ndim > 1 else orig[:n]
    r_mono = _to_mono(recon[:n]) if recon.ndim > 1 else recon[:n]

    # Extract bass band
    S_o = np.abs(librosa.stft(o_mono))
    S_r = np.abs(librosa.stft(r_mono))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = (freqs >= 20) & (freqs <= 200)

    bass_o = S_o[bass_mask, :].mean(axis=0)
    bass_r = S_r[bass_mask, :].mean(axis=0)

    min_len = min(len(bass_o), len(bass_r))
    corr = float(np.corrcoef(bass_o[:min_len], bass_r[:min_len])[0, 1])
    score = max(0, corr * 100)

    return DimensionScore(
        name="bass_pattern_match",
        score=round(score, 2),
        detail=f"Bass band correlation: {corr:.4f}",
    )


def _kick_pattern_match(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare kick/transient patterns via onset detection."""
    if not HAS_LIBROSA:
        return DimensionScore(name="kick_pattern_match", score=0, detail="librosa not available")

    n = min(len(orig), len(recon))
    o_mono = _to_mono(orig[:n]) if orig.ndim > 1 else orig[:n]
    r_mono = _to_mono(recon[:n]) if recon.ndim > 1 else recon[:n]

    onsets_o = librosa.onset.onset_detect(y=o_mono, sr=sr, units="time")
    onsets_r = librosa.onset.onset_detect(y=r_mono, sr=sr, units="time")

    if len(onsets_o) == 0:
        return DimensionScore(name="kick_pattern_match", score=50, detail="No onsets detected in original")

    # Match onsets within tolerance
    tolerance = 0.05  # 50ms
    matched = 0
    for t_o in onsets_o:
        if any(abs(t_o - t_r) < tolerance for t_r in onsets_r):
            matched += 1

    precision = matched / max(len(onsets_r), 1)
    recall = matched / len(onsets_o)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    score = f1 * 100

    return DimensionScore(
        name="kick_pattern_match",
        score=round(score, 2),
        detail=f"Onset F1: {f1:.3f} ({matched}/{len(onsets_o)} matched, tolerance {tolerance*1000:.0f}ms)",
    )


def _harmonic_progression(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare chroma features (harmonic content)."""
    if not HAS_LIBROSA:
        return DimensionScore(name="harmonic_progression", score=0, detail="librosa not available")

    n = min(len(orig), len(recon))
    o_mono = _to_mono(orig[:n]) if orig.ndim > 1 else orig[:n]
    r_mono = _to_mono(recon[:n]) if recon.ndim > 1 else recon[:n]

    chroma_o = librosa.feature.chroma_cqt(y=o_mono, sr=sr)
    chroma_r = librosa.feature.chroma_cqt(y=r_mono, sr=sr)

    min_t = min(chroma_o.shape[1], chroma_r.shape[1])
    corr = float(np.corrcoef(chroma_o[:, :min_t].flatten(), chroma_r[:, :min_t].flatten())[0, 1])
    score = max(0, corr * 100)

    return DimensionScore(
        name="harmonic_progression",
        score=round(score, 2),
        detail=f"Chroma correlation: {corr:.4f}",
        weight=1.2,
    )


def _energy_curve(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare RMS energy curves over time."""
    n = min(len(orig), len(recon))
    o_mono = _to_mono(orig[:n]) if orig.ndim > 1 else orig[:n]
    r_mono = _to_mono(recon[:n]) if recon.ndim > 1 else recon[:n]

    hop = sr  # 1-second windows
    rms_o = np.array([float(np.sqrt(np.mean(o_mono[i:i+hop]**2))) for i in range(0, len(o_mono) - hop, hop)])
    rms_r = np.array([float(np.sqrt(np.mean(r_mono[i:i+hop]**2))) for i in range(0, len(r_mono) - hop, hop)])

    min_len = min(len(rms_o), len(rms_r))
    if min_len < 2:
        return DimensionScore(name="energy_curve", score=50, detail="Track too short for energy curve")

    corr = float(np.corrcoef(rms_o[:min_len], rms_r[:min_len])[0, 1])
    score = max(0, corr * 100)

    return DimensionScore(
        name="energy_curve",
        score=round(score, 2),
        detail=f"Energy curve correlation: {corr:.4f} ({min_len}s analyzed)",
        weight=1.3,
    )


def _reverb_match(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare reverb characteristics via spectral decay."""
    n = min(len(orig), len(recon))
    o_mono = _to_mono(orig[:n]) if orig.ndim > 1 else orig[:n]
    r_mono = _to_mono(recon[:n]) if recon.ndim > 1 else recon[:n]

    # Compare spectral centroid (indicator of reverb/brightness)
    if not HAS_LIBROSA:
        return DimensionScore(name="reverb_match", score=0, detail="librosa not available")

    sc_o = librosa.feature.spectral_centroid(y=o_mono, sr=sr)[0]
    sc_r = librosa.feature.spectral_centroid(y=r_mono, sr=sr)[0]

    mean_o = float(np.mean(sc_o))
    mean_r = float(np.mean(sc_r))
    diff_pct = abs(mean_o - mean_r) / max(mean_o, 1)
    score = max(0, 100 - diff_pct * 200)

    return DimensionScore(
        name="reverb_match",
        score=round(score, 2),
        detail=f"Spectral centroid orig: {mean_o:.0f} Hz, recon: {mean_r:.0f} Hz",
        weight=0.8,
    )


def _dynamic_range(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare dynamic range (crest factor)."""
    def _crest(data: npt.NDArray[np.float64]) -> float:
        mono = _to_mono(data) if data.ndim > 1 else data
        peak = float(np.max(np.abs(mono)))
        rms = float(np.sqrt(np.mean(mono ** 2)))
        return 20 * np.log10(max(peak, 1e-10) / max(rms, 1e-10))

    cf_o = _crest(orig)
    cf_r = _crest(recon)
    diff = abs(cf_o - cf_r)
    score = max(0, 100 - diff * 15)

    return DimensionScore(
        name="dynamic_range",
        score=round(score, 2),
        detail=f"Crest factor orig: {cf_o:.1f} dB, recon: {cf_r:.1f} dB, diff: {diff:.1f} dB",
    )


def _bpm_accuracy(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare detected BPM."""
    if not HAS_LIBROSA:
        return DimensionScore(name="bpm_accuracy", score=0, detail="librosa not available")

    o_mono = _to_mono(orig) if orig.ndim > 1 else orig
    r_mono = _to_mono(recon) if recon.ndim > 1 else recon

    tempo_o = float(librosa.beat.tempo(y=o_mono, sr=sr)[0])
    tempo_r = float(librosa.beat.tempo(y=r_mono, sr=sr)[0])

    diff = abs(tempo_o - tempo_r)
    score = max(0, 100 - diff * 5)

    return DimensionScore(
        name="bpm_accuracy",
        score=round(score, 2),
        detail=f"BPM orig: {tempo_o:.1f}, recon: {tempo_r:.1f}, diff: {diff:.1f}",
        weight=1.5,
    )


def _arrangement_match(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare overall arrangement via segment-level energy profile."""
    n = min(len(orig), len(recon))
    o_mono = _to_mono(orig[:n]) if orig.ndim > 1 else orig[:n]
    r_mono = _to_mono(recon[:n]) if recon.ndim > 1 else recon[:n]

    # 4-bar windows (assuming ~120 BPM)
    bar_s = 2.0  # ~2 seconds per bar at 120 BPM
    window = int(bar_s * 4 * sr)  # 4 bars
    n_segments = max(1, n // window)

    energy_o = []
    energy_r = []
    for i in range(n_segments):
        s, e = i * window, min((i + 1) * window, n)
        energy_o.append(float(np.sqrt(np.mean(o_mono[s:e] ** 2))))
        energy_r.append(float(np.sqrt(np.mean(r_mono[s:e] ** 2))))

    if len(energy_o) < 2:
        return DimensionScore(name="arrangement_match", score=50, detail="Track too short for arrangement analysis")

    corr = float(np.corrcoef(energy_o, energy_r)[0, 1])
    score = max(0, corr * 100)

    return DimensionScore(
        name="arrangement_match",
        score=round(score, 2),
        detail=f"4-bar energy profile correlation: {corr:.4f} ({n_segments} segments)",
    )


def _timbre_similarity(orig: npt.NDArray[np.float64], recon: npt.NDArray[np.float64], sr: int) -> DimensionScore:
    """Compare timbre via MFCC features."""
    if not HAS_LIBROSA:
        return DimensionScore(name="timbre_similarity", score=0, detail="librosa not available")

    n = min(len(orig), len(recon))
    o_mono = _to_mono(orig[:n]) if orig.ndim > 1 else orig[:n]
    r_mono = _to_mono(recon[:n]) if recon.ndim > 1 else recon[:n]

    mfcc_o = librosa.feature.mfcc(y=o_mono, sr=sr, n_mfcc=20)
    mfcc_r = librosa.feature.mfcc(y=r_mono, sr=sr, n_mfcc=20)

    # Compare mean MFCC vectors
    mean_o = np.mean(mfcc_o, axis=1)
    mean_r = np.mean(mfcc_r, axis=1)

    # Cosine similarity
    dot = float(np.dot(mean_o, mean_r))
    norm = float(np.linalg.norm(mean_o) * np.linalg.norm(mean_r))
    cosine_sim = dot / max(norm, 1e-10)
    score = max(0, cosine_sim * 100)

    return DimensionScore(
        name="timbre_similarity",
        score=round(score, 2),
        detail=f"MFCC cosine similarity: {cosine_sim:.4f}",
        weight=1.2,
    )


# ── Main Comparison Engine ───────────────────────────────


def compare_full(
    original_path: str | Path,
    reconstruction_path: str | Path,
    target_score: float = 90.0,
) -> ComparisonResult:
    """Run full 12-dimension A/B comparison.

    Args:
        original_path: Path to original audio file
        reconstruction_path: Path to reconstructed audio file
        target_score: Minimum score to pass (default 90%)

    Returns:
        ComparisonResult with all 12 dimension scores
    """
    orig_data, sr = _load_audio(original_path)
    recon_data, _ = _load_audio(reconstruction_path, sr=sr)

    analyzers = [
        _spectral_similarity,
        _rms_match,
        _stereo_width_match,
        _bass_pattern_match,
        _kick_pattern_match,
        _harmonic_progression,
        _energy_curve,
        _reverb_match,
        _dynamic_range,
        _bpm_accuracy,
        _arrangement_match,
        _timbre_similarity,
    ]

    dimensions: list[DimensionScore] = []
    for analyzer in analyzers:
        try:
            dim = analyzer(orig_data, recon_data, sr)
            dimensions.append(dim)
        except Exception as e:
            dimensions.append(DimensionScore(
                name=analyzer.__name__.lstrip("_"),
                score=0,
                detail=f"Error: {e}",
            ))

    # Weighted average
    total_weight = sum(d.weight for d in dimensions)
    overall = sum(d.score * d.weight for d in dimensions) / max(total_weight, 1)

    # Find weakest/strongest
    sorted_dims = sorted(dimensions, key=lambda d: d.score)
    weakest = sorted_dims[0].name if sorted_dims else ""
    strongest = sorted_dims[-1].name if sorted_dims else ""

    return ComparisonResult(
        dimensions=dimensions,
        overall_score=round(overall, 2),
        target_score=target_score,
        passed=overall >= target_score,
        weakest=weakest,
        strongest=strongest,
    )


def compare_quick(
    original_path: str | Path,
    reconstruction_path: str | Path,
) -> dict[str, float]:
    """Quick comparison — returns just scores."""
    result = compare_full(original_path, reconstruction_path)
    return {d.name: d.score for d in result.dimensions}
