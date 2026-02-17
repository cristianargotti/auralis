"""AURALIS Auto-Correction — Post-master gap analysis and re-processing.

Compares processed audio against deep DNA profile targets and suggests
corrections if gaps exceed threshold.  Designed for max 2 passes to
prevent infinite loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import structlog
from scipy.signal import welch

logger = structlog.get_logger()


# ── Types ────────────────────────────────────────────────


@dataclass
class CorrectionResult:
    """Result of a single auto-correction evaluation."""

    stem_name: str
    needs_correction: bool = False
    gap_score: float = 0.0  # 0-1, higher = bigger gap
    corrections: dict[str, Any] = field(default_factory=dict)
    reasoning: list[str] = field(default_factory=list)


@dataclass
class CorrectionReport:
    """Full correction report for all stems + master."""

    stem_corrections: dict[str, CorrectionResult] = field(default_factory=dict)
    master_correction: CorrectionResult | None = None
    pass_number: int = 1
    total_gap: float = 0.0
    should_reprocess: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stem_corrections": {
                k: {
                    "needs_correction": v.needs_correction,
                    "gap_score": v.gap_score,
                    "corrections": v.corrections,
                    "reasoning": v.reasoning,
                }
                for k, v in self.stem_corrections.items()
            },
            "master": {
                "needs_correction": self.master_correction.needs_correction,
                "gap_score": self.master_correction.gap_score,
                "reasoning": self.master_correction.reasoning,
            }
            if self.master_correction
            else None,
            "pass_number": self.pass_number,
            "total_gap": self.total_gap,
            "should_reprocess": self.should_reprocess,
        }


# ── Analysis Utilities ───────────────────────────────────


def _analyze_freq_balance(
    audio_path: str | Path, sr: int = 44100
) -> dict[str, float]:
    """Compute frequency band energy percentages for an audio file."""
    try:
        data, sr = sf.read(str(audio_path), dtype="float64")
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        freqs, psd = welch(data, fs=sr, nperseg=min(4096, len(data)))

        low_mask = freqs < 250
        mid_mask = (freqs >= 250) & (freqs < 4000)
        high_mask = freqs >= 4000

        total = np.sum(psd) + 1e-12
        return {
            "low": float(np.sum(psd[low_mask]) / total * 100),
            "mid": float(np.sum(psd[mid_mask]) / total * 100),
            "high": float(np.sum(psd[high_mask]) / total * 100),
        }
    except Exception:
        return {"low": 33.3, "mid": 33.3, "high": 33.3}


def _estimate_rms(audio_path: str | Path) -> float:
    """Estimate RMS in dB for an audio file."""
    try:
        data, _ = sf.read(str(audio_path), dtype="float64")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        rms = np.sqrt(np.mean(data**2))
        return float(20 * np.log10(rms + 1e-12))
    except Exception:
        return -20.0


# ── Correction Engine ────────────────────────────────────


def evaluate_master(
    master_path: str | Path,
    deep_profile: dict[str, Any],
    threshold: float = 0.15,
) -> CorrectionResult:
    """Compare mastered audio against deep profile targets."""
    result = CorrectionResult(stem_name="_master")
    reasons: list[str] = []

    master_info = deep_profile.get("master", {})
    ref_lufs = master_info.get("lufs", -14.0)

    # ── Loudness gap ──
    current_rms = _estimate_rms(master_path)
    current_est_lufs = current_rms - 0.7  # rough LUFS estimate
    lufs_gap = abs(current_est_lufs - ref_lufs)
    lufs_gap_norm = min(1.0, lufs_gap / 10.0)  # normalize to 0-1

    if lufs_gap > 3.0:
        result.corrections["target_lufs_adjust"] = ref_lufs
        reasons.append(
            f"Loudness gap: {current_est_lufs:.1f} vs target {ref_lufs:.1f} "
            f"(diff: {lufs_gap:.1f} dB)"
        )

    # ── Frequency balance gap ──
    current_freq = _analyze_freq_balance(master_path)
    # Assume reference has balanced distribution based on analysis
    freq_gap = 0.0
    corrections_eq: list[tuple[float, float, float]] = []

    # Check if low end matches expectations
    perc = deep_profile.get("percussion", {})
    bass = deep_profile.get("bass", {})
    bass_type = bass.get("dominant_type", "").lower()

    if "sub" in bass_type and current_freq["low"] < 35:
        corrections_eq.append((55.0, 2.0, 1.5))
        freq_gap += 0.1
        reasons.append(
            f"Sub bass expected but low end only {current_freq['low']:.0f}% → boost 55Hz"
        )

    if current_freq["high"] < 15:
        corrections_eq.append((8000.0, 1.5, 0.8))
        freq_gap += 0.05
        reasons.append(
            f"Highs low ({current_freq['high']:.0f}%) → add 1.5dB air at 8kHz"
        )

    if corrections_eq:
        result.corrections["eq_adjust"] = corrections_eq

    # ── Total gap score ──
    result.gap_score = min(1.0, lufs_gap_norm + freq_gap)
    result.needs_correction = result.gap_score > threshold
    result.reasoning = reasons

    if result.needs_correction:
        reasons.append(
            f"Gap score: {result.gap_score:.0%} > threshold {threshold:.0%} → REPROCESS"
        )
    else:
        reasons.append(
            f"Gap score: {result.gap_score:.0%} ≤ threshold {threshold:.0%} → PASS"
        )

    return result


def evaluate_and_correct(
    master_path: str | Path,
    deep_profile: dict[str, Any],
    pass_number: int = 1,
    max_passes: int = 2,
    threshold: float = 0.15,
) -> CorrectionReport:
    """Full correction evaluation with pass tracking.

    Args:
        master_path: Path to mastered audio file
        deep_profile: Deep DNA profile from reference bank
        pass_number: Current correction pass (1-based)
        max_passes: Maximum correction passes allowed
        threshold: Gap threshold for triggering reprocessing

    Returns:
        CorrectionReport with correction decisions
    """
    report = CorrectionReport(pass_number=pass_number)

    master_result = evaluate_master(master_path, deep_profile, threshold)
    report.master_correction = master_result
    report.total_gap = master_result.gap_score

    # Only reprocess if gap is significant AND we haven't exceeded max passes
    report.should_reprocess = (
        master_result.needs_correction and pass_number < max_passes
    )

    if report.should_reprocess:
        logger.info(
            "auto_correct.reprocess",
            pass_number=pass_number,
            gap=master_result.gap_score,
        )
    else:
        if pass_number >= max_passes and master_result.needs_correction:
            master_result.reasoning.append(
                f"Max passes ({max_passes}) reached — accepting result"
            )
        logger.info(
            "auto_correct.accept",
            pass_number=pass_number,
            gap=master_result.gap_score,
        )

    return report
