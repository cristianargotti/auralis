"""Spectral analysis using librosa — deep audio fingerprinting.

Extracts comprehensive frequency, temporal, harmonic, and perceptual features
from any audio file to build a complete DNA map of the track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import librosa
import numpy as np
import numpy.typing as npt
import soundfile as sf

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class SpectralProfile:
    """Complete spectral analysis of an audio file."""

    # Metadata
    sample_rate: int
    duration: float
    num_samples: int

    # Spectral features
    spectral_centroid: npt.NDArray[np.floating[Any]]
    spectral_bandwidth: npt.NDArray[np.floating[Any]]
    spectral_rolloff: npt.NDArray[np.floating[Any]]
    spectral_flatness: npt.NDArray[np.floating[Any]]
    spectral_contrast: npt.NDArray[np.floating[Any]]

    # Perceptual features
    mfcc: npt.NDArray[np.floating[Any]]
    chroma: npt.NDArray[np.floating[Any]]

    # Rhythm
    tempo: float
    beat_frames: npt.NDArray[np.integer[Any]]
    onset_strength: npt.NDArray[np.floating[Any]]

    # Harmonic
    harmonic_ratio: npt.NDArray[np.floating[Any]]
    key_estimate: str
    scale_estimate: str

    # Energy
    rms: npt.NDArray[np.floating[Any]]
    zero_crossing_rate: npt.NDArray[np.floating[Any]]

    # Band energy (10 bands)
    band_energies: dict[str, float] = field(default_factory=dict)


# Standard frequency bands for spectral comparison
FREQUENCY_BANDS: dict[str, tuple[float, float]] = {
    "sub": (20.0, 60.0),
    "bass": (60.0, 250.0),
    "low_mid": (250.0, 500.0),
    "mid": (500.0, 1000.0),
    "upper_mid": (1000.0, 2000.0),
    "presence": (2000.0, 4000.0),
    "brilliance": (4000.0, 8000.0),
    "air_low": (8000.0, 12000.0),
    "air_high": (12000.0, 16000.0),
    "ultra": (16000.0, 22050.0),
}

# Key names for estimation
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def analyze_audio(
    audio_path: str | Path,
    sr: int = 44100,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mfcc: int = 20,
) -> SpectralProfile:
    """Perform deep spectral analysis on an audio file.

    Args:
        audio_path: Path to audio file (WAV, MP3, FLAC, AIFF).
        sr: Target sample rate.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        n_mfcc: Number of MFCC coefficients.

    Returns:
        SpectralProfile with all extracted features.
    """
    # Load audio
    y, actual_sr = sf.read(str(audio_path))
    if actual_sr != sr:
        y = librosa.resample(y=y.T if y.ndim > 1 else y, orig_sr=actual_sr, target_sr=sr)
    else:
        y = y.T if y.ndim > 1 else y

    # If stereo, convert to mono for analysis
    if y.ndim > 1:
        y_mono: npt.NDArray[np.floating[Any]] = librosa.to_mono(y)
    else:
        y_mono = y

    duration = float(len(y_mono)) / sr

    # ─── Spectral Features ───
    stft = np.abs(librosa.stft(y=y_mono, n_fft=n_fft, hop_length=hop_length))

    spectral_centroid = librosa.feature.spectral_centroid(S=stft, sr=sr, hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=stft, sr=sr, hop_length=hop_length)[0]
    spectral_flatness = librosa.feature.spectral_flatness(S=stft)[0]
    spectral_contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, hop_length=hop_length)

    # ─── Perceptual Features ───
    mfcc = librosa.feature.mfcc(y=y_mono, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    chroma = librosa.feature.chroma_cqt(y=y_mono, sr=sr, hop_length=hop_length)

    # ─── Rhythm ───
    onset_env = librosa.onset.onset_strength(y=y_mono, sr=sr, hop_length=hop_length)
    tempo_result = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    tempo_raw = tempo_result[0]
    tempo_val = float(tempo_raw[0]) if hasattr(tempo_raw, "__getitem__") else float(tempo_raw)  # type: ignore[arg-type]
    beat_frames = np.asarray(tempo_result[1], dtype=np.intp)

    # ─── Harmonic ───
    y_harmonic, _y_percussive = librosa.effects.hpss(y_mono)
    harmonic_ratio = np.abs(y_harmonic) / (np.abs(y_mono) + 1e-10)

    # Key estimation via chroma
    chroma_mean = np.mean(chroma, axis=1)
    key_idx = int(np.argmax(chroma_mean))
    key_estimate = KEY_NAMES[key_idx]

    # Major vs minor estimation
    major_profile = np.array(
        [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    )
    minor_profile = np.array(
        [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    )
    major_corr = float(np.corrcoef(np.roll(major_profile, key_idx), chroma_mean)[0, 1])
    minor_corr = float(np.corrcoef(np.roll(minor_profile, key_idx), chroma_mean)[0, 1])
    scale_estimate = "major" if major_corr > minor_corr else "minor"

    # ─── Energy ───
    rms = librosa.feature.rms(y=y_mono, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_mono, hop_length=hop_length)[0]

    # ─── Band Energies ───
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    power_spectrum = np.mean(stft**2, axis=1)
    band_energies: dict[str, float] = {}

    for band_name, (low, high) in FREQUENCY_BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        band_energies[band_name] = float(np.sum(power_spectrum[mask]))

    # Normalize band energies to percentages
    total_energy = sum(band_energies.values())
    if total_energy > 0:
        band_energies = {k: v / total_energy * 100 for k, v in band_energies.items()}

    return SpectralProfile(
        sample_rate=sr,
        duration=duration,
        num_samples=len(y_mono),
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        spectral_rolloff=spectral_rolloff,
        spectral_flatness=spectral_flatness,
        spectral_contrast=spectral_contrast,
        mfcc=mfcc,
        chroma=chroma,
        tempo=tempo_val,
        beat_frames=beat_frames,
        onset_strength=onset_env,
        harmonic_ratio=harmonic_ratio,
        key_estimate=key_estimate,
        scale_estimate=scale_estimate,
        rms=rms,
        zero_crossing_rate=zcr,
        band_energies=band_energies,
    )


def compare_spectral_profiles(
    profile_a: SpectralProfile,
    profile_b: SpectralProfile,
) -> dict[str, Any]:
    """Compare two spectral profiles for A/B analysis.

    Returns deviation metrics for each dimension.
    """
    results: dict[str, Any] = {}

    # Band energy comparison
    band_deviations: dict[str, float] = {}
    for band in FREQUENCY_BANDS:
        energy_a = profile_a.band_energies.get(band, 0.0)
        energy_b = profile_b.band_energies.get(band, 0.0)
        if energy_a > 0:
            band_deviations[band] = abs(energy_b - energy_a) / energy_a * 100
        else:
            band_deviations[band] = 0.0

    results["band_deviations_pct"] = band_deviations
    results["max_band_deviation"] = max(band_deviations.values()) if band_deviations else 0.0

    # Tempo comparison
    results["tempo_deviation_bpm"] = abs(profile_a.tempo - profile_b.tempo)

    # Key match
    results["key_match"] = profile_a.key_estimate == profile_b.key_estimate
    results["scale_match"] = profile_a.scale_estimate == profile_b.scale_estimate

    # MFCC cosine distance
    mfcc_a_mean = np.mean(profile_a.mfcc, axis=1)
    mfcc_b_mean = np.mean(profile_b.mfcc, axis=1)
    cosine_sim = float(
        np.dot(mfcc_a_mean, mfcc_b_mean)
        / (np.linalg.norm(mfcc_a_mean) * np.linalg.norm(mfcc_b_mean) + 1e-10)
    )
    results["mfcc_cosine_similarity"] = cosine_sim
    results["mfcc_cosine_distance"] = 1.0 - cosine_sim

    # RMS energy comparison
    rms_a_mean = float(np.mean(profile_a.rms))
    rms_b_mean = float(np.mean(profile_b.rms))
    results["rms_deviation_db"] = float(
        20 * np.log10(rms_b_mean / rms_a_mean) if rms_a_mean > 0 else 0.0
    )

    # Overall pass/fail
    results["spectral_match"] = results["max_band_deviation"] <= 1.0
    results["perceptual_match"] = results["mfcc_cosine_distance"] <= 0.05

    return results
