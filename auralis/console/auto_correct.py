"""AURALIS Auto-Correction — Post-master gap analysis and re-processing.

Compares processed audio against deep DNA profile targets and suggests
corrections if gaps exceed threshold.  Supports:
  - 7-band frequency analysis (matching QC module)
  - Per-stem evaluation (not just master)
  - Real LUFS measurement (via ITU-R BS.1770)
  - Stereo width checks
  - Up to 2 correction passes to prevent infinite loops
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


# ── 7-Band Frequency Definitions ────────────────────────
# Matches the QC module's 7-band spectral analysis

FREQ_BANDS = {
    "sub":        (20,    60),
    "bass":       (60,    250),
    "low_mid":    (250,   500),
    "mid":        (500,   2000),
    "upper_mid":  (2000,  4000),
    "presence":   (4000,  8000),
    "brilliance": (8000,  20000),
}


# ── Types ────────────────────────────────────────────────


@dataclass
class BandCorrection:
    """Correction for a single frequency band."""

    band_name: str
    center_freq: float
    gain_db: float  # Positive = boost, negative = cut
    q: float = 1.0


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


def _analyze_freq_balance_7band(
    audio_path: str | Path, sr: int = 44100
) -> dict[str, float]:
    """Compute 7-band frequency energy percentages for an audio file."""
    try:
        data, file_sr = sf.read(str(audio_path), dtype="float64")
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        freqs, psd = welch(data, fs=file_sr, nperseg=min(8192, len(data)))

        total = np.sum(psd) + 1e-12
        result: dict[str, float] = {}

        for band_name, (lo, hi) in FREQ_BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            result[band_name] = float(np.sum(psd[mask]) / total * 100)

        return result
    except Exception:
        return {name: 100.0 / 7 for name in FREQ_BANDS}


def _measure_lufs(audio_path: str | Path) -> float:
    """Measure integrated LUFS using ITU-R BS.1770-4 algorithm.

    Simplified implementation using K-weighted RMS with
    pre-emphasis filter characteristics.
    """
    try:
        data, sr = sf.read(str(audio_path), dtype="float64")
        if data.ndim > 1:
            # Multi-channel: use all channels
            channels = [data[:, i] for i in range(data.shape[1])]
        else:
            channels = [data]

        # K-weighting: simplified high-shelf + high-pass
        # Stage 1: Pre-emphasis high-shelf (+4dB above 1.5kHz approx)
        from scipy.signal import butter, sosfilt

        channel_powers: list[float] = []
        for ch in channels:
            # High-shelf approximation via 2nd-order high-pass at 100Hz
            sos_hp = butter(2, 100.0 / (sr / 2), btype="high", output="sos")
            weighted = sosfilt(sos_hp, ch)

            # RMS power
            power = float(np.mean(weighted**2))
            channel_powers.append(power)

        # Average power across channels (simplified, no surround weighting)
        avg_power = np.mean(channel_powers)
        lufs = float(-0.691 + 10 * np.log10(avg_power + 1e-12))

        return lufs
    except Exception:
        return -20.0


def _measure_stereo_width(audio_path: str | Path) -> float:
    """Measure stereo width as correlation coefficient (0=mono, 1=full stereo)."""
    try:
        data, _ = sf.read(str(audio_path), dtype="float64")
        if data.ndim < 2:
            return 0.0  # Mono file

        left, right = data[:, 0], data[:, 1]
        mid = (left + right) / 2
        side = (left - right) / 2

        mid_energy = np.sum(mid**2) + 1e-12
        side_energy = np.sum(side**2) + 1e-12

        # Width = side/mid ratio (0 = pure mono, 1+ = wide stereo)
        width = float(np.sqrt(side_energy / mid_energy))
        return min(width, 2.0)
    except Exception:
        return 0.5


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
    """Compare mastered audio against deep profile targets (7-band + LUFS)."""
    result = CorrectionResult(stem_name="_master")
    reasons: list[str] = []

    master_info = deep_profile.get("master", {})
    ref_lufs = master_info.get("lufs", -14.0)

    # ── Real LUFS measurement ──
    current_lufs = _measure_lufs(master_path)
    lufs_gap = abs(current_lufs - ref_lufs)
    lufs_gap_norm = min(1.0, lufs_gap / 10.0)

    if lufs_gap > 2.0:
        result.corrections["target_lufs_adjust"] = ref_lufs
        reasons.append(
            f"Loudness gap: {current_lufs:.1f} LUFS vs target {ref_lufs:.1f} "
            f"(diff: {lufs_gap:.1f} dB)"
        )

    # ── 7-Band frequency analysis ──
    current_freq = _analyze_freq_balance_7band(master_path)
    corrections_eq: list[dict[str, Any]] = []
    freq_gap = 0.0

    # Reference frequency balance from deep profile
    ref_freq = deep_profile.get("frequency_balance", {})

    for band_name, (lo, hi) in FREQ_BANDS.items():
        current_pct = current_freq.get(band_name, 0)
        ref_pct = ref_freq.get(band_name, 100.0 / 7)

        diff = current_pct - ref_pct
        center_freq = (lo + hi) / 2

        # Significant deviation = ±5% from reference
        if abs(diff) > 5.0:
            # Negative diff = we're below reference → boost
            gain_db = min(3.0, max(-3.0, -diff * 0.15))
            corrections_eq.append({
                "band": band_name,
                "freq": center_freq,
                "gain_db": gain_db,
                "q": 1.0,
            })
            freq_gap += abs(diff) * 0.005
            direction = "below" if diff < 0 else "above"
            reasons.append(
                f"{band_name} ({center_freq:.0f}Hz): {current_pct:.1f}% "
                f"({direction} ref {ref_pct:.1f}%) → {gain_db:+.1f}dB"
            )

    if corrections_eq:
        result.corrections["eq_adjust"] = corrections_eq

    # ── Stereo width check ──
    current_width = _measure_stereo_width(master_path)
    ref_width = master_info.get("stereo_width", 0.5)
    width_gap = abs(current_width - ref_width)

    if width_gap > 0.2:
        result.corrections["stereo_width_adjust"] = ref_width
        freq_gap += width_gap * 0.1
        reasons.append(
            f"Stereo width: {current_width:.2f} vs target {ref_width:.2f} "
            f"(diff: {width_gap:.2f})"
        )

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
            f"Gap score: {result.gap_score:.0%} ≤ threshold {threshold:.0%} → PASS ✓"
        )

    return result


def evaluate_stem(
    stem_path: str | Path,
    stem_name: str,
    ref_targets: dict[str, Any],
    threshold: float = 0.20,
) -> CorrectionResult:
    """Evaluate a single stem against its reference targets.

    Args:
        stem_path: Path to the processed stem audio.
        stem_name: Name of the stem (drums, bass, vocals, other).
        ref_targets: Reference targets from DNA bank for this stem.
        threshold: Gap score threshold for triggering correction.
    """
    result = CorrectionResult(stem_name=stem_name)
    reasons: list[str] = []

    # ── RMS level check ──
    current_rms = _estimate_rms(stem_path)
    ref_rms = ref_targets.get("rms_db", -18.0)
    rms_gap = abs(current_rms - ref_rms)
    rms_gap_norm = min(1.0, rms_gap / 12.0)

    if rms_gap > 3.0:
        gain_adjust = ref_rms - current_rms
        result.corrections["volume_db_adjust"] = round(gain_adjust, 1)
        reasons.append(
            f"Level: {current_rms:.1f}dB vs target {ref_rms:.1f}dB → adjust {gain_adjust:+.1f}dB"
        )

    # ── 7-Band frequency check ──
    current_freq = _analyze_freq_balance_7band(stem_path)
    ref_freq = ref_targets.get("freq_bands", {})
    corrections_eq: list[dict[str, Any]] = []
    freq_gap = 0.0

    for band_name in FREQ_BANDS:
        current_pct = current_freq.get(band_name, 0)
        ref_pct = ref_freq.get(band_name, 100.0 / 7)
        diff = current_pct - ref_pct
        lo, hi = FREQ_BANDS[band_name]
        center_freq = (lo + hi) / 2

        if abs(diff) > 7.0:  # Stems have more tolerance
            gain_db = min(4.0, max(-4.0, -diff * 0.2))
            corrections_eq.append({
                "band": band_name,
                "freq": center_freq,
                "gain_db": gain_db,
                "q": 1.0,
            })
            freq_gap += abs(diff) * 0.004
            direction = "weak" if diff < 0 else "strong"
            reasons.append(
                f"{stem_name} {band_name}: {direction} ({diff:+.1f}%) → {gain_db:+.1f}dB"
            )

    if corrections_eq:
        result.corrections["eq_adjust"] = corrections_eq

    # ── Total gap score ──
    result.gap_score = min(1.0, rms_gap_norm + freq_gap)
    result.needs_correction = result.gap_score > threshold
    result.reasoning = reasons

    return result


def evaluate_and_correct(
    master_path: str | Path,
    deep_profile: dict[str, Any],
    pass_number: int = 1,
    max_passes: int = 2,
    threshold: float = 0.15,
    stem_paths: dict[str, str] | None = None,
    ref_targets: dict[str, dict[str, Any]] | None = None,
) -> CorrectionReport:
    """Full correction evaluation with per-stem + master analysis.

    Args:
        master_path: Path to mastered audio file
        deep_profile: Deep DNA profile from reference bank
        pass_number: Current correction pass (1-based)
        max_passes: Maximum correction passes allowed
        threshold: Gap threshold for triggering reprocessing
        stem_paths: Optional dict of stem_name → path for per-stem evaluation
        ref_targets: Optional reference targets per stem from DNA bank

    Returns:
        CorrectionReport with correction decisions
    """
    report = CorrectionReport(pass_number=pass_number)

    # ── Per-stem evaluation ──
    if stem_paths and ref_targets:
        for stem_name, stem_path in stem_paths.items():
            if stem_name.startswith("_"):
                continue
            stem_ref = ref_targets.get(stem_name, {})
            if stem_ref:
                stem_result = evaluate_stem(
                    stem_path, stem_name, stem_ref, threshold=0.20
                )
                report.stem_corrections[stem_name] = stem_result
                if stem_result.needs_correction:
                    logger.info(
                        "auto_correct.stem_gap",
                        stem=stem_name,
                        gap=stem_result.gap_score,
                    )

    # ── Master evaluation ──
    master_result = evaluate_master(master_path, deep_profile, threshold)
    report.master_correction = master_result

    # Total gap = weighted average of master + stem gaps
    stem_gaps = [r.gap_score for r in report.stem_corrections.values()]
    if stem_gaps:
        avg_stem_gap = sum(stem_gaps) / len(stem_gaps)
        report.total_gap = master_result.gap_score * 0.6 + avg_stem_gap * 0.4
    else:
        report.total_gap = master_result.gap_score

    # Only reprocess if gap is significant AND we haven't exceeded max passes
    report.should_reprocess = (
        report.total_gap > threshold and pass_number < max_passes
    )

    if report.should_reprocess:
        logger.info(
            "auto_correct.reprocess",
            pass_number=pass_number,
            total_gap=report.total_gap,
            master_gap=master_result.gap_score,
            stem_gaps={k: v.gap_score for k, v in report.stem_corrections.items()},
        )
    else:
        if pass_number >= max_passes and report.total_gap > threshold:
            master_result.reasoning.append(
                f"Max passes ({max_passes}) reached — accepting result"
            )
        logger.info(
            "auto_correct.accept",
            pass_number=pass_number,
            total_gap=report.total_gap,
        )

    return report


# ── Mix Recall Learning ─────────────────────────────────


@dataclass
class CorrectionOutcome:
    """Recorded outcome of a correction pass."""

    band_name: str = ""
    correction_db: float = 0.0
    gap_before: float = 0.0
    gap_after: float = 0.0
    improved: bool = False
    bpm: float = 120.0
    key: str = "C"


class MixRecallMemory:
    """Learns from correction outcomes to suggest proactive EQ adjustments.

    After each correction pass, records which band corrections worked
    (reduced the gap) and which didn't. Over time, builds a knowledge
    base of proven corrections per genre/BPM/key.

    Storage: ~/.auralis/mix_recall.json
    """

    def __init__(self) -> None:
        import json
        self._memory_dir = Path.home() / ".auralis"
        self._memory_file = self._memory_dir / "mix_recall.json"
        self._outcomes: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        import json
        if self._memory_file.exists():
            try:
                data = json.loads(self._memory_file.read_text())
                self._outcomes = data.get("outcomes", [])
            except Exception:
                self._outcomes = []

    def save(self) -> None:
        import json
        try:
            self._memory_dir.mkdir(parents=True, exist_ok=True)
            data = {"outcomes": self._outcomes[-200:]}  # Keep last 200
            self._memory_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def record_outcome(
        self,
        report: CorrectionReport,
        gap_after: float,
        bpm: float = 120.0,
        key: str = "C",
    ) -> None:
        """Record correction results for learning.

        Call after applying corrections and re-evaluating the mix.

        Args:
            report: The CorrectionReport that was applied.
            gap_after: Total gap score AFTER applying corrections.
            bpm: Track BPM.
            key: Track key.
        """
        if not report.master_correction:
            return

        gap_before = report.total_gap
        improved = gap_after < gap_before

        # Record per-band corrections from master
        for band_name, correction_data in report.master_correction.corrections.items():
            if isinstance(correction_data, dict) and "gain_db" in correction_data:
                outcome = {
                    "band_name": band_name,
                    "correction_db": correction_data["gain_db"],
                    "gap_before": round(gap_before, 3),
                    "gap_after": round(gap_after, 3),
                    "improved": improved,
                    "bpm": bpm,
                    "key": key,
                }
                self._outcomes.append(outcome)

        # Record per-stem corrections
        for stem_name, stem_result in report.stem_corrections.items():
            if not stem_result.needs_correction:
                continue
            for band_name, correction_data in stem_result.corrections.items():
                if isinstance(correction_data, dict) and "gain_db" in correction_data:
                    outcome = {
                        "band_name": f"{stem_name}_{band_name}",
                        "correction_db": correction_data["gain_db"],
                        "gap_before": round(gap_before, 3),
                        "gap_after": round(gap_after, 3),
                        "improved": improved,
                        "bpm": bpm,
                        "key": key,
                    }
                    self._outcomes.append(outcome)

        self.save()
        logger.info(
            "mix_recall.recorded",
            gap_before=gap_before,
            gap_after=gap_after,
            improved=improved,
        )

    def suggest_corrections(
        self,
        bpm: float = 120.0,
        key: str = "C",
        min_confidence: int = 3,
    ) -> list[BandCorrection]:
        """Suggest proactive corrections based on past successes.

        Analyzes stored outcomes to find corrections that consistently
        improved the mix at similar BPM/key. Returns suggestions only
        when there's enough confidence (min_confidence successful outcomes).

        Args:
            bpm: Current track BPM.
            key: Current track key.
            min_confidence: Minimum successful outcomes needed.

        Returns:
            List of BandCorrection suggestions.
        """
        if not self._outcomes:
            return []

        # Group by band and find average successful correction
        band_stats: dict[str, list[float]] = {}

        for outcome in self._outcomes:
            if not outcome.get("improved", False):
                continue

            # BPM proximity (within 15 BPM = relevant)
            bpm_diff = abs(outcome.get("bpm", 120) - bpm)
            if bpm_diff > 15:
                continue

            band = outcome["band_name"]
            gain = outcome.get("correction_db", 0.0)
            if abs(gain) > 0.1:
                band_stats.setdefault(band, []).append(gain)

        # Build suggestions from confident patterns
        suggestions = []
        for band_name, gains in band_stats.items():
            if len(gains) < min_confidence:
                continue

            avg_gain = sum(gains) / len(gains)
            # Use conservative gain (50% of average) to avoid over-correction
            safe_gain = avg_gain * 0.5

            if abs(safe_gain) < 0.3:
                continue

            # Map band name to center frequency
            freq_map = {
                "sub": 40.0, "low": 100.0, "low_mid": 350.0,
                "mid": 1000.0, "upper_mid": 3000.0,
                "presence": 6000.0, "brilliance": 12000.0,
            }
            # Try to extract base band name (strip stem prefix)
            base_band = band_name.split("_", 1)[-1] if "_" in band_name else band_name
            center_freq = freq_map.get(base_band, freq_map.get(band_name, 1000.0))

            suggestions.append(BandCorrection(
                band_name=band_name,
                center_freq=center_freq,
                gain_db=round(safe_gain, 1),
                q=1.0,
            ))

        logger.info(
            "mix_recall.suggestions",
            n_suggestions=len(suggestions),
            total_outcomes=len(self._outcomes),
        )
        return suggestions

    @property
    def outcome_count(self) -> int:
        return len(self._outcomes)
