"""AURALIS QC Comparator — 12-dimension A/B audio comparison.

Compares an original track against a reconstruction across 12 dimensions:
spectral similarity, RMS match, stereo width, bass/kick patterns,
harmonic progression, energy curve, reverb, dynamic range,
BPM accuracy, arrangement match, and timbre similarity.

All analysis uses numpy + librosa — no GPU required.

Memory-optimised: loads at float32 / 22050 Hz, pre-computes mono once,
gc.collect() between analyzers to keep RSS under ~4 GB.
"""

from __future__ import annotations

import gc
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

QC_SR = 22050  # 22 kHz is enough for A/B comparison, halves memory vs 44.1k


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


# ── Audio Loading (memory-optimised) ─────────────────────


def _load_audio(path: str | Path, sr: int = QC_SR) -> tuple[npt.NDArray[np.float32], int]:
    """Load audio at float32, resample to QC_SR."""
    data, file_sr = sf.read(str(path), dtype="float32")
    if file_sr != sr and HAS_LIBROSA:
        data = librosa.resample(data.T if data.ndim > 1 else data, orig_sr=file_sr, target_sr=sr).astype(np.float32)
        if data.ndim > 1:
            data = data.T
        return data, sr
    return data, file_sr


def _to_mono(data: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Convert to mono if stereo."""
    if data.ndim > 1:
        return np.mean(data, axis=1).astype(np.float32)
    return data


# ── Dimension Analyzers ──────────────────────────────────
# Each receives pre-computed mono arrays to avoid redundant conversions.


def _spectral_similarity(o_mono: npt.NDArray[np.float32], r_mono: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare mel spectrogram correlation."""
    if not HAS_LIBROSA:
        return DimensionScore(name="spectral_similarity", score=0, detail="librosa not available")

    n = min(len(o_mono), len(r_mono))
    S_orig = librosa.feature.melspectrogram(y=o_mono[:n], sr=sr, n_mels=64, fmax=sr // 2)
    S_recon = librosa.feature.melspectrogram(y=r_mono[:n], sr=sr, n_mels=64, fmax=sr // 2)

    S_orig_db = librosa.power_to_db(S_orig, ref=np.max)
    S_recon_db = librosa.power_to_db(S_recon, ref=np.max)
    del S_orig, S_recon

    flat_orig = S_orig_db.flatten()
    flat_recon = S_recon_db.flatten()
    del S_orig_db, S_recon_db

    corr = float(np.corrcoef(flat_orig, flat_recon)[0, 1])
    score = max(0, corr * 100)

    return DimensionScore(
        name="spectral_similarity",
        score=round(score, 2),
        detail=f"Mel spectrogram correlation: {corr:.4f}",
        weight=1.5,
    )


def _rms_match(o_mono: npt.NDArray[np.float32], r_mono: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare RMS levels per section."""
    n = min(len(o_mono), len(r_mono))
    window = sr
    n_windows = n // window

    if n_windows == 0:
        rms_orig = float(np.sqrt(np.mean(o_mono[:n] ** 2)))
        rms_recon = float(np.sqrt(np.mean(r_mono[:n] ** 2)))
        diff = abs(20 * np.log10(max(rms_orig, 1e-10)) - 20 * np.log10(max(rms_recon, 1e-10)))
        score = max(0, 100 - diff * 10)
        return DimensionScore(name="rms_match", score=round(score, 2), detail=f"RMS diff: {diff:.2f} dB")

    diffs = []
    for i in range(n_windows):
        s, e = i * window, (i + 1) * window
        rms_o = float(np.sqrt(np.mean(o_mono[s:e] ** 2)))
        rms_r = float(np.sqrt(np.mean(r_mono[s:e] ** 2)))
        diff = abs(20 * np.log10(max(rms_o, 1e-10)) - 20 * np.log10(max(rms_r, 1e-10)))
        diffs.append(diff)

    avg_diff = float(np.mean(diffs))
    score = max(0, 100 - avg_diff * 10)

    return DimensionScore(
        name="rms_match",
        score=round(score, 2),
        detail=f"Avg RMS diff: {avg_diff:.2f} dB across {n_windows} windows",
    )


def _stereo_width_match(orig: npt.NDArray[np.float32], recon: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare stereo width (mid/side ratio). Uses raw stereo data."""
    def _width(data: npt.NDArray[np.float32]) -> float:
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


def _bass_pattern_match(o_mono: npt.NDArray[np.float32], r_mono: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare bass frequency patterns (20-200 Hz)."""
    if not HAS_LIBROSA:
        return DimensionScore(name="bass_pattern_match", score=0, detail="librosa not available")

    n = min(len(o_mono), len(r_mono))

    S_o = np.abs(librosa.stft(o_mono[:n]))
    S_r = np.abs(librosa.stft(r_mono[:n]))
    freqs = librosa.fft_frequencies(sr=sr)
    bass_mask = (freqs >= 20) & (freqs <= 200)

    bass_o = S_o[bass_mask, :].mean(axis=0)
    bass_r = S_r[bass_mask, :].mean(axis=0)
    del S_o, S_r

    min_len = min(len(bass_o), len(bass_r))
    corr = float(np.corrcoef(bass_o[:min_len], bass_r[:min_len])[0, 1])
    score = max(0, corr * 100)

    return DimensionScore(
        name="bass_pattern_match",
        score=round(score, 2),
        detail=f"Bass band correlation: {corr:.4f}",
    )


def _kick_pattern_match(o_mono: npt.NDArray[np.float32], r_mono: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare kick/transient patterns via onset detection."""
    if not HAS_LIBROSA:
        return DimensionScore(name="kick_pattern_match", score=0, detail="librosa not available")

    n = min(len(o_mono), len(r_mono))

    onsets_o = librosa.onset.onset_detect(y=o_mono[:n], sr=sr, units="time")
    onsets_r = librosa.onset.onset_detect(y=r_mono[:n], sr=sr, units="time")

    if len(onsets_o) == 0:
        return DimensionScore(name="kick_pattern_match", score=50, detail="No onsets detected in original")

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


def _harmonic_progression(o_mono: npt.NDArray[np.float32], r_mono: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare chroma features (harmonic content)."""
    if not HAS_LIBROSA:
        return DimensionScore(name="harmonic_progression", score=0, detail="librosa not available")

    n = min(len(o_mono), len(r_mono))

    chroma_o = librosa.feature.chroma_cqt(y=o_mono[:n], sr=sr)
    chroma_r = librosa.feature.chroma_cqt(y=r_mono[:n], sr=sr)

    min_t = min(chroma_o.shape[1], chroma_r.shape[1])
    corr = float(np.corrcoef(chroma_o[:, :min_t].flatten(), chroma_r[:, :min_t].flatten())[0, 1])
    score = max(0, corr * 100)

    return DimensionScore(
        name="harmonic_progression",
        score=round(score, 2),
        detail=f"Chroma correlation: {corr:.4f}",
        weight=1.2,
    )


def _energy_curve(o_mono: npt.NDArray[np.float32], r_mono: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare RMS energy curves over time."""
    n = min(len(o_mono), len(r_mono))

    hop = sr  # 1-second windows
    rms_o = np.array([float(np.sqrt(np.mean(o_mono[i:i+hop]**2))) for i in range(0, len(o_mono[:n]) - hop, hop)])
    rms_r = np.array([float(np.sqrt(np.mean(r_mono[i:i+hop]**2))) for i in range(0, len(r_mono[:n]) - hop, hop)])

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


def _reverb_match(o_mono: npt.NDArray[np.float32], r_mono: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare reverb characteristics via spectral decay."""
    if not HAS_LIBROSA:
        return DimensionScore(name="reverb_match", score=0, detail="librosa not available")

    n = min(len(o_mono), len(r_mono))

    sc_o = librosa.feature.spectral_centroid(y=o_mono[:n], sr=sr)[0]
    sc_r = librosa.feature.spectral_centroid(y=r_mono[:n], sr=sr)[0]

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


def _dynamic_range(o_mono: npt.NDArray[np.float32], r_mono: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare dynamic range (crest factor)."""
    def _crest(data: npt.NDArray[np.float32]) -> float:
        peak = float(np.max(np.abs(data)))
        rms = float(np.sqrt(np.mean(data ** 2)))
        return 20 * np.log10(max(peak, 1e-10) / max(rms, 1e-10))

    cf_o = _crest(o_mono)
    cf_r = _crest(r_mono)
    diff = abs(cf_o - cf_r)
    score = max(0, 100 - diff * 15)

    return DimensionScore(
        name="dynamic_range",
        score=round(score, 2),
        detail=f"Crest factor orig: {cf_o:.1f} dB, recon: {cf_r:.1f} dB, diff: {diff:.1f} dB",
    )


def _bpm_accuracy(o_mono: npt.NDArray[np.float32], r_mono: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare detected BPM."""
    if not HAS_LIBROSA:
        return DimensionScore(name="bpm_accuracy", score=0, detail="librosa not available")

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


def _arrangement_match(o_mono: npt.NDArray[np.float32], r_mono: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare overall arrangement via segment-level energy profile."""
    n = min(len(o_mono), len(r_mono))

    bar_s = 2.0
    window = int(bar_s * 4 * sr)
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


def _timbre_similarity(o_mono: npt.NDArray[np.float32], r_mono: npt.NDArray[np.float32], sr: int) -> DimensionScore:
    """Compare timbre via MFCC features."""
    if not HAS_LIBROSA:
        return DimensionScore(name="timbre_similarity", score=0, detail="librosa not available")

    n = min(len(o_mono), len(r_mono))

    mfcc_o = librosa.feature.mfcc(y=o_mono[:n], sr=sr, n_mfcc=13)
    mfcc_r = librosa.feature.mfcc(y=r_mono[:n], sr=sr, n_mfcc=13)

    mean_o = np.mean(mfcc_o, axis=1)
    mean_r = np.mean(mfcc_r, axis=1)

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

    Memory-optimised: loads at float32 / 22050 Hz, pre-computes mono,
    gc.collect() between analyzers. Peak RSS stays under ~4 GB.
    """
    orig_data, sr = _load_audio(original_path)
    recon_data, _ = _load_audio(reconstruction_path, sr=sr)

    # Pre-compute mono once (avoids 12 redundant conversions)
    orig_mono = _to_mono(orig_data)
    recon_mono = _to_mono(recon_data)

    # Analyzers that need mono audio
    mono_analyzers = [
        _spectral_similarity,
        _rms_match,
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

    # Stereo width needs raw stereo data
    try:
        dim = _stereo_width_match(orig_data, recon_data, sr)
        dimensions.append(dim)
    except Exception as e:
        dimensions.append(DimensionScore(name="stereo_width_match", score=0, detail=f"Error: {e}"))

    # Free stereo data — only mono needed from here
    del orig_data, recon_data
    gc.collect()

    for analyzer in mono_analyzers:
        try:
            dim = analyzer(orig_mono, recon_mono, sr)
            dimensions.append(dim)
        except Exception as e:
            dimensions.append(DimensionScore(
                name=analyzer.__name__.lstrip("_"),
                score=0,
                detail=f"Error: {e}",
            ))
        gc.collect()  # free librosa intermediates between analyzers

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
