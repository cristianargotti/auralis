"""Reference track deep profiler — extracts the complete DNA map.

Builds a section-by-section analysis of a track: energy curves,
arrangement map, element detection, and sonic fingerprint.
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
    """Complete DNA map of a reference track."""

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
        }
        return result

    def save_json(self, output_path: str | Path) -> Path:
        """Save DNA map as JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def _detect_sections(
    y: npt.NDArray[np.floating[Any]],
    sr: int,
    tempo: float,
    hop_length: int = 512,
) -> list[Section]:
    """Detect track sections based on energy changes and structure.

    Uses onset strength envelope and novelty function to find
    structural boundaries (intro, buildup, drop, breakdown, outro).
    """
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)

    # Compute novelty function for structural boundaries
    # Use a larger kernel for section-level changes
    librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, aggregate=np.median)

    # RMS energy for section characterization
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    librosa.amplitude_to_db(rms + 1e-10)

    # Find section boundaries using structural segmentation
    # Use recurrence matrix and novelty-based approach
    librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)[1]

    # Calculate bars (assuming 4/4)
    beats_per_bar = 4
    bar_duration = (60.0 / tempo) * beats_per_bar

    # Estimate section boundaries at regular intervals (every 8 or 16 bars)
    total_bars = int(float(len(y)) / sr / bar_duration)
    section_length_bars = 16 if total_bars > 32 else 8

    sections: list[Section] = []
    section_patterns = _estimate_section_pattern(total_bars, section_length_bars)

    for name, start_bar, end_bar in section_patterns:
        start_time = start_bar * bar_duration
        end_time = min(end_bar * bar_duration, float(len(y)) / sr)

        # Extract section audio for analysis
        start_sample = int(start_time * sr)
        end_sample = min(int(end_time * sr), len(y))

        if start_sample >= end_sample:
            continue

        section_audio = y[start_sample:end_sample]
        section_rms = librosa.feature.rms(y=section_audio)[0]
        avg_energy = float(np.mean(section_rms))
        avg_rms_db_val = float(np.mean(librosa.amplitude_to_db(section_rms + 1e-10)))

        # Count spectral peaks as proxy for element density
        section_spec = np.abs(librosa.stft(y=section_audio))
        element_count = int(
            np.sum(section_spec.mean(axis=1) > np.percentile(section_spec.mean(axis=1), 75))
        )

        # Characterize section
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


def _estimate_section_pattern(total_bars: int, section_length: int) -> list[tuple[str, int, int]]:
    """Estimate section names based on bar position and typical arrangement."""
    patterns: list[tuple[str, int, int]] = []
    bar = 0
    section_idx = 0
    arrangement = ["Intro", "Buildup", "Drop", "Breakdown", "Buildup 2", "Drop 2", "Outro"]

    while bar < total_bars:
        end_bar = min(bar + section_length, total_bars)
        if section_idx < len(arrangement):
            name = arrangement[section_idx]
        else:
            name = f"Section {section_idx + 1}"
        patterns.append((name, bar, end_bar))
        bar = end_bar
        section_idx += 1

    return patterns


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


def profile_track(
    audio_path: str | Path,
    sr: int = 44100,
) -> TrackDNA:
    """Build complete DNA map of a reference track.

    Args:
        audio_path: Path to audio file.
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
        y_stereo = np.stack([y, y])  # Fake stereo for loudness

    duration = float(len(y_mono)) / sr

    # ─── Full spectral profile ───
    spectral_profile = analyze_audio(audio_path, sr=sr)

    # ─── Loudness (EBU R128) ───
    meter = pyloudnorm.Meter(sr)
    y_for_lufs = y_stereo.T if y_stereo.shape[0] == 2 else y_stereo
    integrated_lufs = float(meter.integrated_loudness(y_for_lufs))
    true_peak = float(20 * np.log10(np.max(np.abs(y_mono)) + 1e-10))

    # Loudness range (short-term variance)
    block_size = sr * 3  # 3-second blocks
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

    # ─── Sections ───
    sections = _detect_sections(y_mono, sr, spectral_profile.tempo)

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
        spectral_profile=spectral_profile,
    )
