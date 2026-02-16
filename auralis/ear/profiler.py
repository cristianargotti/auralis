"""Reference track deep profiler — extracts the complete DNA map.

Builds a section-by-section analysis of a track: energy curves,
arrangement map, element detection, and sonic fingerprint.

Track-agnostic — detects everything automatically from audio.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import numpy.typing as npt
import pyloudnorm
import soundfile as sf

from auralis.ear.spectral import SpectralProfile, analyze_audio


@dataclass
class Section:
    """A section of the track (intro, buildup, drop, breakdown, etc)."""

    name: str
    start_time: float
    end_time: float
    start_bar: int
    end_bar: int
    avg_energy: float
    avg_rms_db: float
    element_count: int
    characteristics: list[str] = field(default_factory=list)


@dataclass
class TrackDNA:
    """Complete DNA map of a reference track — auto-detected, not hardcoded."""

    # Identity
    file_path: str
    duration: float
    sample_rate: int

    # Musical
    key: str
    scale: str
    tempo: float
    time_signature: str

    # Loudness (EBU R128)
    integrated_lufs: float
    true_peak_dbfs: float
    loudness_range_lu: float

    # Structure
    sections: list[Section] = field(default_factory=list)

    # Energy curve (per-bar RMS values for visualization)
    energy_curve: list[dict[str, float]] = field(default_factory=list)

    # Spectral identity
    band_energy_profile: dict[str, float] = field(default_factory=dict)

    # Dynamics
    crest_factor_db: float = 0.0
    dynamic_range_db: float = 0.0

    # Full spectral profile reference
    spectral_profile: SpectralProfile | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excluding numpy arrays)."""
        result: dict[str, Any] = {
            "file_path": self.file_path,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "key": self.key,
            "scale": self.scale,
            "tempo": self.tempo,
            "time_signature": self.time_signature,
            "integrated_lufs": self.integrated_lufs,
            "true_peak_dbfs": self.true_peak_dbfs,
            "loudness_range_lu": self.loudness_range_lu,
            "crest_factor_db": self.crest_factor_db,
            "dynamic_range_db": self.dynamic_range_db,
            "band_energy_profile": self.band_energy_profile,
            "sections": [asdict(s) for s in self.sections],
            "energy_curve": self.energy_curve,
        }
        return result

    def save_json(self, output_path: str | Path) -> Path:
        """Save DNA map as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


# ── Auto-Structure Detection — Phase 2 ─────────────────


def _compute_energy_curve(
    y: npt.NDArray[np.floating[Any]],
    sr: int,
    tempo: float,
) -> list[dict[str, float]]:
    """Compute per-bar RMS energy curve for visualization.

    Returns a list of {bar, rms_db} dicts — one per musical bar.
    This is calculated automatically from audio, not hardcoded.
    """
    bar_duration = (60.0 / tempo) * 4  # Assuming 4/4
    total_bars = int(float(len(y)) / sr / bar_duration)
    curve: list[dict[str, float]] = []

    for bar in range(total_bars):
        start_sample = int(bar * bar_duration * sr)
        end_sample = min(int((bar + 1) * bar_duration * sr), len(y))
        if start_sample >= end_sample:
            continue
        bar_audio = y[start_sample:end_sample]
        rms = float(np.sqrt(np.mean(bar_audio**2)))
        rms_db = float(20 * np.log10(rms + 1e-10))
        curve.append({"bar": bar, "rms_db": round(rms_db, 2)})

    return curve


def _detect_section_boundaries(
    y: npt.NDArray[np.floating[Any]],
    sr: int,
    tempo: float,
) -> list[int]:
    """Detect section boundaries using energy jumps and novelty.

    Returns a list of bar numbers where sections change.
    Uses RMS energy cliff/jump detection — no hardcoded intervals.
    """
    bar_duration = (60.0 / tempo) * 4
    total_bars = int(float(len(y)) / sr / bar_duration)

    if total_bars < 4:
        return [0, total_bars]

    # Compute per-bar energy
    bar_energies: list[float] = []
    for bar in range(total_bars):
        start = int(bar * bar_duration * sr)
        end = min(int((bar + 1) * bar_duration * sr), len(y))
        if start >= end:
            bar_energies.append(-60.0)
            continue
        rms = float(np.sqrt(np.mean(y[start:end] ** 2)))
        bar_energies.append(float(20 * np.log10(rms + 1e-10)))

    energies = np.array(bar_energies)

    # Smooth with 4-bar moving average
    kernel = np.ones(4) / 4
    smoothed = np.convolve(energies, kernel, mode="same")

    # Detect jumps: where energy changes by more than threshold
    diff = np.abs(np.diff(smoothed))
    threshold = np.percentile(diff, 80)  # top 20% of changes

    # Find boundary bars (align to 4-bar or 8-bar grid)
    raw_boundaries = [0]  # Always start at bar 0
    min_section_bars = 4

    for i in range(len(diff)):
        if diff[i] > threshold:
            bar = i + 1
            # Snap to nearest 4-bar grid
            bar = round(bar / 4) * 4
            # Must be far enough from last boundary
            if bar > raw_boundaries[-1] + min_section_bars and bar < total_bars:
                raw_boundaries.append(bar)

    raw_boundaries.append(total_bars)

    # Merge very short sections (< 4 bars)
    boundaries: list[int] = [raw_boundaries[0]]
    for b in raw_boundaries[1:]:
        if b - boundaries[-1] >= min_section_bars:
            boundaries.append(b)
        else:
            boundaries[-1] = b  # Extend previous section

    if boundaries[-1] != total_bars:
        boundaries.append(total_bars)

    return boundaries


def _classify_section(
    y: npt.NDArray[np.floating[Any]],
    sr: int,
    start_time: float,
    end_time: float,
    section_idx: int,
    total_sections: int,
    avg_rms_db: float,
    all_rms_values: list[float],
) -> str:
    """Auto-classify a section based on energy, position, and context.

    Returns names like: intro, buildup, drop, breakdown, groove, outro.
    No hardcoded arrangement — derived from audio characteristics.
    """
    # Relative energy position (0.0 = quietest, 1.0 = loudest)
    if len(all_rms_values) > 1:
        min_rms = min(all_rms_values)
        max_rms = max(all_rms_values)
        rms_range = max_rms - min_rms if max_rms > min_rms else 1.0
        relative_energy = (avg_rms_db - min_rms) / rms_range
    else:
        relative_energy = 0.5

    # Position in track (0.0 = start, 1.0 = end)
    position = section_idx / max(total_sections - 1, 1)

    # Classify based on energy and position
    if position < 0.1:
        return "Intro"
    if position > 0.9:
        return "Outro"

    # Check if this is a high-energy section
    if relative_energy > 0.75:
        # Is it the first high-energy section after a quiet one?
        if section_idx > 0 and all_rms_values[section_idx - 1] < avg_rms_db - 3:
            return "Drop"
        return "Groove"

    # Low energy section
    if relative_energy < 0.3:
        if section_idx > 1:
            return "Breakdown"
        return "Intro"

    # Medium energy — likely buildup or transition
    if section_idx > 0 and section_idx < total_sections - 1:
        # Energy increasing? → Buildup
        next_rms = all_rms_values[min(section_idx + 1, len(all_rms_values) - 1)]
        if next_rms > avg_rms_db + 2:
            return "Buildup"

    return "Bridge"


def _detect_sections(
    y: npt.NDArray[np.floating[Any]],
    sr: int,
    tempo: float,
    hop_length: int = 512,
) -> list[Section]:
    """Detect track sections based on energy changes and structure.

    Phase 2: Uses RMS energy cliff/jump detection for auto boundaries.
    No fixed intervals — boundaries are derived from the audio itself.
    """
    bar_duration = (60.0 / tempo) * 4
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Detect boundaries from energy
    boundaries = _detect_section_boundaries(y, sr, tempo)

    # First pass: compute RMS for all sections
    all_rms: list[float] = []
    for i in range(len(boundaries) - 1):
        start_bar = boundaries[i]
        end_bar = boundaries[i + 1]
        start_time = start_bar * bar_duration
        end_time = min(end_bar * bar_duration, float(len(y)) / sr)
        start_sample = int(start_time * sr)
        end_sample = min(int(end_time * sr), len(y))
        if start_sample >= end_sample:
            all_rms.append(-60.0)
            continue
        section_audio = y[start_sample:end_sample]
        rms = librosa.feature.rms(y=section_audio)[0]
        all_rms.append(float(np.mean(librosa.amplitude_to_db(rms + 1e-10))))

    total_sections = len(boundaries) - 1
    sections: list[Section] = []

    for i in range(total_sections):
        start_bar = boundaries[i]
        end_bar = boundaries[i + 1]
        start_time = start_bar * bar_duration
        end_time = min(end_bar * bar_duration, float(len(y)) / sr)

        start_sample = int(start_time * sr)
        end_sample = min(int(end_time * sr), len(y))
        if start_sample >= end_sample:
            continue

        section_audio = y[start_sample:end_sample]
        section_rms = librosa.feature.rms(y=section_audio)[0]
        avg_energy = float(np.mean(section_rms))
        avg_rms_db_val = all_rms[i]

        # Element count via spectral peaks
        section_spec = np.abs(librosa.stft(y=section_audio))
        element_count = int(
            np.sum(section_spec.mean(axis=1) > np.percentile(section_spec.mean(axis=1), 75))
        )

        # Auto-classify
        name = _classify_section(
            y, sr, start_time, end_time,
            i, total_sections, avg_rms_db_val, all_rms,
        )

        # Characterize
        characteristics = _characterize_section(
            avg_energy, avg_rms_db_val, onset_env, start_time, end_time, sr, hop_length
        )

        sections.append(
            Section(
                name=name,
                start_time=round(start_time, 2),
                end_time=round(end_time, 2),
                start_bar=start_bar,
                end_bar=end_bar,
                avg_energy=round(avg_energy, 6),
                avg_rms_db=round(avg_rms_db_val, 2),
                element_count=element_count,
                characteristics=characteristics,
            )
        )

    return sections


def _characterize_section(
    avg_energy: float,
    avg_rms_db: float,
    onset_env: npt.NDArray[np.floating[Any]],
    start_time: float,
    end_time: float,
    sr: int,
    hop_length: int,
) -> list[str]:
    """Characterize a section based on its audio properties."""
    chars: list[str] = []

    if avg_rms_db > -12:
        chars.append("high-energy")
    elif avg_rms_db > -20:
        chars.append("medium-energy")
    else:
        chars.append("low-energy")

    # Check onset density for rhythmic activity
    start_frame = int(start_time * sr / hop_length)
    end_frame = min(int(end_time * sr / hop_length), len(onset_env))
    if start_frame < end_frame:
        section_onsets = onset_env[start_frame:end_frame]
        onset_density = float(np.mean(section_onsets))
        if onset_density > np.percentile(onset_env, 75):
            chars.append("rhythmically-dense")
        elif onset_density < np.percentile(onset_env, 25):
            chars.append("sparse")

    return chars


# ── Main profiler ───────────────────────────────────────


def profile_track(
    audio_path: str | Path,
    sr: int = 44100,
) -> TrackDNA:
    """Build complete DNA map of a reference track.

    100% track-agnostic — analyzes any audio file:
    - Key, scale, BPM detection (via spectral analysis)
    - Section boundaries (via energy cliff/jump detection)
    - Per-bar energy curve (for visualization)
    - Loudness profile (EBU R128)
    - Dynamics (crest factor, dynamic range)
    - Spectral fingerprint (band energies)

    Args:
        audio_path: Path to any audio file.
        sr: Target sample rate.

    Returns:
        TrackDNA with complete track identity.
    """
    audio_path = Path(audio_path)

    # Load audio
    y, actual_sr = sf.read(str(audio_path))
    if actual_sr != sr:
        y = librosa.resample(y=y.T if y.ndim > 1 else y, orig_sr=actual_sr, target_sr=sr)
    else:
        y = y.T if y.ndim > 1 else y

    # Mono for analysis
    if y.ndim > 1:
        y_mono: npt.NDArray[np.floating[Any]] = librosa.to_mono(y)
        y_stereo = y
    else:
        y_mono = y
        y_stereo = np.stack([y, y])

    duration = float(len(y_mono)) / sr

    # ─── Full spectral profile ───
    spectral_profile = analyze_audio(audio_path, sr=sr)

    # ─── Loudness (EBU R128) ───
    meter = pyloudnorm.Meter(sr)
    y_for_lufs = y_stereo.T if y_stereo.shape[0] == 2 else y_stereo
    integrated_lufs = float(meter.integrated_loudness(y_for_lufs))
    true_peak = float(20 * np.log10(np.max(np.abs(y_mono)) + 1e-10))

    # Loudness range (short-term variance)
    block_size = sr * 3
    block_lufs: list[float] = []
    for i in range(0, len(y_mono) - block_size, block_size):
        block = y_mono[i : i + block_size]
        block_stereo = np.stack([block, block]).T
        block_l = float(meter.integrated_loudness(block_stereo))
        if np.isfinite(block_l):
            block_lufs.append(block_l)

    loudness_range = (max(block_lufs) - min(block_lufs)) if len(block_lufs) >= 2 else 0.0

    # ─── Dynamics ───
    rms_val = float(np.sqrt(np.mean(y_mono**2)))
    peak_val = float(np.max(np.abs(y_mono)))
    crest_factor = float(20 * np.log10(peak_val / rms_val)) if rms_val > 0 else 0.0

    rms_frames = librosa.feature.rms(y=y_mono)[0]
    rms_db_frames = librosa.amplitude_to_db(rms_frames + 1e-10)
    dynamic_range = float(np.percentile(rms_db_frames, 95) - np.percentile(rms_db_frames, 5))

    # ─── Sections (auto-detected) ───
    sections = _detect_sections(y_mono, sr, spectral_profile.tempo)

    # ─── Energy curves (for visualization) ───
    energy_curve = _compute_energy_curve(y_mono, sr, spectral_profile.tempo)

    return TrackDNA(
        file_path=str(audio_path),
        duration=round(duration, 2),
        sample_rate=sr,
        key=spectral_profile.key_estimate,
        scale=spectral_profile.scale_estimate,
        tempo=round(spectral_profile.tempo, 2),
        time_signature="4/4",
        integrated_lufs=round(integrated_lufs, 2),
        true_peak_dbfs=round(true_peak, 2),
        loudness_range_lu=round(loudness_range, 2),
        crest_factor_db=round(crest_factor, 2),
        dynamic_range_db=round(dynamic_range, 2),
        band_energy_profile=spectral_profile.band_energies,
        sections=sections,
        energy_curve=energy_curve,
        spectral_profile=spectral_profile,
    )
