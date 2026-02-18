"""AURALIS Sample Extractor — One-shot isolation from separated stems.

Extracts individual hits and tonal segments from separated stems,
classifies them, and builds a ClonedPalette used by the renderer.

Pipeline:
  drum stem → onset detect → isolate hits → classify → ClonedPalette.drums
  bass stem → energy gate → isolate tonal segments → ClonedPalette.tones["bass"]
  other stem → energy gate → isolate textures → ClonedPalette.tones["other"]

All DSP runs on CPU via librosa/numpy/scipy — no GPU required.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import structlog

from auralis.ear.analyzer import (
    PERCUSSION_PROFILES,
    _classify_hit,
    _extract_onset_features,
)

logger = structlog.get_logger()


# ── Data Structures ──────────────────────────────────────


@dataclass
class ClonedSample:
    """A single extracted sample with metadata."""

    path: Path
    source_track: str  # Name of the reference track
    category: str  # "kick", "snare", "hat_closed", "bass_note", "pad"
    subcategory: str  # Finer classification
    pitch_midi: int | None = None  # For tonal samples
    duration_s: float = 0.0
    energy: float = 0.0
    spectral_fp: dict[str, float] = field(default_factory=dict)
    quality: float = 0.0  # 0-1 composite quality score

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        d = asdict(self)
        d["path"] = str(self.path)
        return d


@dataclass
class ClonedPalette:
    """Complete cloned sound palette from reference tracks.

    drums: label → list of ClonedSample (one-shots)
    tones: stem_name → list of ClonedSample (tonal segments)
    source_tracks: which references were used
    """

    drums: dict[str, list[ClonedSample]] = field(default_factory=dict)
    tones: dict[str, list[ClonedSample]] = field(default_factory=dict)
    source_tracks: list[str] = field(default_factory=list)

    def total_samples(self) -> int:
        """Count of all samples in the palette."""
        total = sum(len(v) for v in self.drums.values())
        total += sum(len(v) for v in self.tones.values())
        return total

    def best_drum(self, label: str) -> ClonedSample | None:
        """Get highest-quality sample for a drum label."""
        candidates = self.drums.get(label, [])
        if not candidates:
            # Try normalized label matching (e.g., "kick" → "Kick")
            for key, samples in self.drums.items():
                if key.lower().replace(" ", "_") == label.lower().replace(" ", "_"):
                    candidates = samples
                    break
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.quality)

    def best_tone(self, stem: str) -> ClonedSample | None:
        """Get highest-quality tonal sample for a stem."""
        candidates = self.tones.get(stem, [])
        if not candidates:
            return None
        return max(candidates, key=lambda s: s.quality)

    def save_metadata(self, output_dir: Path) -> Path:
        """Save palette metadata as JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_path = output_dir / "cloned_palette.json"
        data = {
            "source_tracks": self.source_tracks,
            "total_samples": self.total_samples(),
            "drums": {
                label: [s.to_dict() for s in samples]
                for label, samples in self.drums.items()
            },
            "tones": {
                stem: [s.to_dict() for s in samples]
                for stem, samples in self.tones.items()
            },
        }
        meta_path.write_text(json.dumps(data, indent=2))
        return meta_path

    @classmethod
    def load(cls, meta_path: Path) -> ClonedPalette:
        """Load palette from saved metadata."""
        data = json.loads(meta_path.read_text())
        palette = cls(source_tracks=data.get("source_tracks", []))
        for label, samples in data.get("drums", {}).items():
            palette.drums[label] = [
                ClonedSample(
                    path=Path(s["path"]),
                    source_track=s.get("source_track", ""),
                    category=s.get("category", ""),
                    subcategory=s.get("subcategory", ""),
                    pitch_midi=s.get("pitch_midi"),
                    duration_s=s.get("duration_s", 0.0),
                    energy=s.get("energy", 0.0),
                    spectral_fp=s.get("spectral_fp", {}),
                    quality=s.get("quality", 0.0),
                )
                for s in samples
            ]
        for stem, samples in data.get("tones", {}).items():
            palette.tones[stem] = [
                ClonedSample(
                    path=Path(s["path"]),
                    source_track=s.get("source_track", ""),
                    category=s.get("category", ""),
                    subcategory=s.get("subcategory", ""),
                    pitch_midi=s.get("pitch_midi"),
                    duration_s=s.get("duration_s", 0.0),
                    energy=s.get("energy", 0.0),
                    spectral_fp=s.get("spectral_fp", {}),
                    quality=s.get("quality", 0.0),
                )
                for s in samples
            ]
        return palette


# ── Drum One-Shot Extraction ─────────────────────────────


# Map analyzer labels to simplified categories for rendering
_LABEL_TO_CATEGORY: dict[str, str] = {
    "Kick": "kick",
    "Snare": "snare",
    "Clap": "clap",
    "Rimshot": "rimshot",
    "Closed HH": "hat_closed",
    "Open HH": "hat_open",
    "Ride": "ride",
    "Crash": "crash",
    "Tom": "tom",
    "Conga": "conga",
    "Cowbell": "cowbell",
    "Shaker": "shaker",
    "Percussion": "percussion",
}


def _compute_quality_score(
    energy: float,
    confidence: float,
    snr_estimate: float,
    duration_s: float,
) -> float:
    """Composite quality score (0-1) for a cloned sample.

    Weights:
      - Energy (30%): Loud, clear hits are more usable
      - Classification confidence (30%): Higher = cleaner isolation
      - SNR estimate (25%): Signal-to-noise ratio
      - Duration (15%): Neither too short (<20ms) nor too long (>2s)
    """
    # Normalize energy (typical range 0.01 - 0.8)
    energy_score = min(energy / 0.3, 1.0)

    # Confidence is already 0-1
    conf_score = confidence

    # SNR (typical range 0-40 dB, map to 0-1)
    snr_score = min(max(snr_estimate / 30.0, 0.0), 1.0)

    # Duration: penalize < 20ms and > 2s
    if duration_s < 0.02:
        dur_score = duration_s / 0.02
    elif duration_s > 2.0:
        dur_score = max(0.3, 1.0 - (duration_s - 2.0) / 5.0)
    else:
        dur_score = 1.0

    return round(
        energy_score * 0.30
        + conf_score * 0.30
        + snr_score * 0.25
        + dur_score * 0.15,
        3,
    )


def _estimate_snr(signal: np.ndarray, sr: int) -> float:
    """Rough SNR estimate: peak energy vs tail energy (dB)."""
    if len(signal) < sr // 10:  # Less than 100ms
        return 20.0  # Assume decent SNR for very short samples

    # First 50ms = signal, last 20% = noise floor estimate
    signal_window = signal[: int(0.05 * sr)]
    noise_start = int(len(signal) * 0.8)
    noise_window = signal[noise_start:]

    signal_power = float(np.mean(signal_window**2)) + 1e-12
    noise_power = float(np.mean(noise_window**2)) + 1e-12

    return float(10 * np.log10(signal_power / noise_power))


def extract_drum_hits(
    drum_stem: str | Path,
    output_dir: str | Path,
    source_name: str = "reference",
    max_hits_per_class: int = 8,
    min_energy: float = 0.005,
    sr: int = 22050,
) -> dict[str, list[ClonedSample]]:
    """Extract individual drum hits from a separated drum stem.

    Pipeline:
      1. Onset detection (librosa, backtracked)
      2. For each onset: extract window (5ms pre-onset → decay tail end)
      3. Classify using 6-feature spectral fingerprint (reuses analyzer.py)
      4. Normalize, trim, apply smooth fade-out
      5. Score quality and save as WAV

    Args:
        drum_stem: Path to separated drum stem (WAV).
        output_dir: Directory to save extracted one-shots.
        source_name: Name of the source track for metadata.
        max_hits_per_class: Max samples to keep per percussion class.
        min_energy: Minimum RMS energy to consider a hit.
        sr: Sample rate for analysis.

    Returns:
        Dict mapping label → list of ClonedSample.
    """
    drum_stem = Path(drum_stem)
    output_dir = Path(output_dir) / "drums"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not drum_stem.exists():
        logger.warning("sample_extractor.drum_stem_missing", path=str(drum_stem))
        return {}

    y, sr_actual = librosa.load(drum_stem, sr=sr, mono=True)
    if len(y) < sr:  # Less than 1 second
        logger.warning("sample_extractor.drum_stem_too_short", path=str(drum_stem))
        return {}

    # ── 1. Onset detection ──
    # Same params as analyzer.classify_percussion for consistency
    onset_frames = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        backtrack=True,
        pre_max=3,
        post_max=3,
        pre_avg=3,
        post_avg=5,
        delta=0.05,
        wait=5,
    )
    onset_samples = librosa.frames_to_samples(onset_frames)

    logger.info(
        "sample_extractor.onsets_detected",
        stem=str(drum_stem.name),
        count=len(onset_frames),
    )

    # ── 2. Extract + classify each hit ──
    results: dict[str, list[ClonedSample]] = {}
    hit_counts: dict[str, int] = {}  # Track how many per class

    for i, onset_sample in enumerate(onset_samples):
        onset_sample = int(onset_sample)

        # Extract spectral features (reusing analyzer.py)
        features = _extract_onset_features(y, sr, onset_sample)

        # Skip quiet hits
        if features["energy"] < min_energy:
            continue

        # Classify
        label, confidence = _classify_hit(features)

        # Check max per class
        if hit_counts.get(label, 0) >= max_hits_per_class:
            continue

        # ── 3. Isolate the hit with transient-preserving window ──
        # Pre-onset: 5ms before the onset to capture the transient attack
        pre_samples = int(0.005 * sr)
        start = max(0, onset_sample - pre_samples)

        # Post-onset: find decay tail end (where energy drops to 5% of peak)
        # Use a 500ms search window to find the decay endpoint
        search_end = min(len(y), onset_sample + int(0.5 * sr))
        tail = y[onset_sample:search_end]

        if len(tail) > 256:
            rms_envelope = librosa.feature.rms(
                y=tail, frame_length=256, hop_length=64
            )[0]
            if len(rms_envelope) > 1:
                peak_rms = np.max(rms_envelope)
                threshold = peak_rms * 0.05  # 5% of peak
                below = np.where(rms_envelope < threshold)[0]
                if len(below) > 0:
                    decay_frame = below[0]
                    decay_samples = librosa.frames_to_samples(
                        decay_frame, hop_length=64
                    )
                    end = onset_sample + int(decay_samples) + int(0.01 * sr)
                else:
                    # Still sustaining — use full 500ms
                    end = search_end
            else:
                end = search_end
        else:
            end = min(len(y), onset_sample + int(0.1 * sr))

        end = min(len(y), end)
        hit_audio = y[start:end].copy()

        if len(hit_audio) < 128:
            continue

        # ── 4. Normalize + fade-out ──
        peak = np.max(np.abs(hit_audio))
        if peak > 0:
            hit_audio = hit_audio / peak * 0.95

        # Smooth fade-out (last 10% of the sample)
        fade_len = max(64, len(hit_audio) // 10)
        fade_curve = np.linspace(1.0, 0.0, fade_len)
        hit_audio[-fade_len:] *= fade_curve

        # ── 5. Quality scoring ──
        snr = _estimate_snr(hit_audio, sr)
        duration_s = len(hit_audio) / sr
        quality = _compute_quality_score(
            energy=features["energy"],
            confidence=confidence,
            snr_estimate=snr,
            duration_s=duration_s,
        )

        # ── 6. Save ──
        category = _LABEL_TO_CATEGORY.get(label, "percussion")
        hit_idx = hit_counts.get(label, 0)
        filename = f"{category}_{hit_idx:02d}.wav"
        filepath = output_dir / filename
        sf.write(str(filepath), hit_audio, sr)

        sample = ClonedSample(
            path=filepath,
            source_track=source_name,
            category=category,
            subcategory=label,
            pitch_midi=None,
            duration_s=round(duration_s, 4),
            energy=round(features["energy"], 4),
            spectral_fp={
                "centroid": round(features["centroid"], 1),
                "bandwidth": round(features["bandwidth"], 1),
                "rolloff": round(features["rolloff"], 1),
                "zcr": round(features["zcr"], 4),
                "decay_ms": round(features["decay_ms"], 1),
                "mfcc_centroid": round(features.get("mfcc_centroid", 0), 4),
            },
            quality=quality,
        )

        if label not in results:
            results[label] = []
        results[label].append(sample)
        hit_counts[label] = hit_counts.get(label, 0) + 1

    total = sum(len(v) for v in results.values())
    logger.info(
        "sample_extractor.drum_hits_extracted",
        total=total,
        classes=list(results.keys()),
    )

    return results


# ── Tonal Sample Extraction ──────────────────────────────


def _extract_tonal_segments(
    audio_path: str | Path,
    output_dir: Path,
    stem_name: str,
    source_name: str = "reference",
    sr: int = 22050,
    min_duration_s: float = 0.2,
    max_segments: int = 8,
) -> list[ClonedSample]:
    """Extract sustained tonal segments from a tonal stem (bass, other).

    Uses energy gating to find active regions, then isolates each segment
    with pitch detection for MIDI note mapping.

    Args:
        audio_path: Path to tonal stem WAV.
        output_dir: Directory to save extracted segments.
        stem_name: "bass" or "other".
        source_name: Name of source track.
        sr: Sample rate.
        min_duration_s: Minimum segment duration.
        max_segments: Maximum segments to extract.

    Returns:
        List of ClonedSample objects.
    """
    audio_path = Path(audio_path)
    segment_dir = output_dir / stem_name
    segment_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        return []

    y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
    if len(y) < sr:
        return []

    # RMS energy envelope for gating
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=512
    )

    # Dynamic threshold: 15% of peak RMS
    threshold = np.max(rms) * 0.15
    if threshold < 0.001:
        return []  # Stem is basically silent

    # Find active regions (contiguous frames above threshold)
    active = rms > threshold
    segments: list[tuple[int, int]] = []
    in_segment = False
    seg_start = 0

    for idx in range(len(active)):
        if active[idx] and not in_segment:
            seg_start = idx
            in_segment = True
        elif not active[idx] and in_segment:
            # End of segment — check minimum duration
            duration = rms_times[idx] - rms_times[seg_start]
            if duration >= min_duration_s:
                segments.append((seg_start, idx))
            in_segment = False

    # Handle segment that extends to end of file
    if in_segment:
        duration = rms_times[-1] - rms_times[seg_start]
        if duration >= min_duration_s:
            segments.append((seg_start, len(active) - 1))

    # Sort by energy (loudest first) and cap
    seg_energies = [
        float(np.mean(rms[s:e])) for s, e in segments
    ]
    sorted_indices = np.argsort(seg_energies)[::-1][:max_segments]

    results: list[ClonedSample] = []

    for rank, seg_idx in enumerate(sorted_indices):
        seg_start_frame, seg_end_frame = segments[seg_idx]
        start_sample = librosa.frames_to_samples(seg_start_frame, hop_length=512)
        end_sample = min(
            len(y),
            librosa.frames_to_samples(seg_end_frame, hop_length=512),
        )

        segment_audio = y[start_sample:end_sample].copy()
        if len(segment_audio) < 1024:
            continue

        # Normalize
        peak = np.max(np.abs(segment_audio))
        if peak > 0:
            segment_audio = segment_audio / peak * 0.95

        # Smooth fade-in/out (20ms each)
        fade_samples = min(int(0.02 * sr), len(segment_audio) // 4)
        if fade_samples > 0:
            segment_audio[:fade_samples] *= np.linspace(0.0, 1.0, fade_samples)
            segment_audio[-fade_samples:] *= np.linspace(1.0, 0.0, fade_samples)

        # Pitch detection (median F0)
        pitch_midi = None
        try:
            f0, _voiced_flag, _voiced_probs = librosa.pyin(
                segment_audio,
                fmin=librosa.note_to_hz("C1"),
                fmax=librosa.note_to_hz("C6"),
                sr=sr,
            )
            valid_f0 = f0[~np.isnan(f0)]
            if len(valid_f0) > 0:
                median_f0 = float(np.median(valid_f0))
                pitch_midi = int(round(librosa.hz_to_midi(median_f0)))
        except Exception:
            pass

        # Spectral fingerprint (mean features over segment)
        centroid = float(
            np.mean(
                librosa.feature.spectral_centroid(
                    y=segment_audio, sr=sr, n_fft=min(2048, len(segment_audio))
                )
            )
        )
        bandwidth = float(
            np.mean(
                librosa.feature.spectral_bandwidth(
                    y=segment_audio, sr=sr, n_fft=min(2048, len(segment_audio))
                )
            )
        )

        duration_s = len(segment_audio) / sr
        energy = float(np.sqrt(np.mean(segment_audio**2)))
        snr = _estimate_snr(segment_audio, sr)
        quality = _compute_quality_score(
            energy=energy,
            confidence=0.7,  # Tonal segments have moderate confidence
            snr_estimate=snr,
            duration_s=duration_s,
        )

        # Save
        filename = f"{stem_name}_{rank:02d}.wav"
        filepath = segment_dir / filename
        sf.write(str(filepath), segment_audio, sr)

        midi_note_name = librosa.midi_to_note(pitch_midi) if pitch_midi else "unknown"
        category = f"{stem_name}_note" if pitch_midi else f"{stem_name}_texture"

        sample = ClonedSample(
            path=filepath,
            source_track=source_name,
            category=category,
            subcategory=midi_note_name,
            pitch_midi=pitch_midi,
            duration_s=round(duration_s, 4),
            energy=round(energy, 4),
            spectral_fp={
                "centroid": round(centroid, 1),
                "bandwidth": round(bandwidth, 1),
            },
            quality=quality,
        )
        results.append(sample)

    logger.info(
        "sample_extractor.tonal_segments_extracted",
        stem=stem_name,
        count=len(results),
    )

    return results


# ── Palette Builder (Orchestrator) ───────────────────────


def build_cloned_palette(
    stems_dir: str | Path,
    output_dir: str | Path | None = None,
    source_name: str = "reference",
    sr: int = 22050,
) -> ClonedPalette:
    """Build a complete ClonedPalette from separated stems.

    Expects a directory containing stems from separator.py:
      drums.wav, bass.wav, other.wav (vocals.wav is ignored)

    Args:
        stems_dir: Directory containing separated stems.
        output_dir: Where to save extracted samples (default: stems_dir/cloned_palette).
        source_name: Name of the source track.
        sr: Sample rate for analysis.

    Returns:
        ClonedPalette with all extracted and classified samples.
    """
    stems_dir = Path(stems_dir)
    if output_dir is None:
        output_dir = stems_dir / "cloned_palette"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    palette = ClonedPalette(source_tracks=[source_name])

    # ── Drums ──
    drum_stem = _find_stem(stems_dir, "drums")
    if drum_stem:
        logger.info("sample_extractor.extracting_drums", stem=str(drum_stem.name))
        palette.drums = extract_drum_hits(
            drum_stem, output_dir, source_name=source_name, sr=sr
        )

    # ── Bass ──
    bass_stem = _find_stem(stems_dir, "bass")
    if bass_stem:
        logger.info("sample_extractor.extracting_bass", stem=str(bass_stem.name))
        palette.tones["bass"] = _extract_tonal_segments(
            bass_stem, output_dir, "bass", source_name=source_name, sr=sr
        )

    # ── Other (leads, pads, synths) ──
    other_stem = _find_stem(stems_dir, "other")
    if other_stem:
        logger.info("sample_extractor.extracting_other", stem=str(other_stem.name))
        palette.tones["other"] = _extract_tonal_segments(
            other_stem, output_dir, "other", source_name=source_name, sr=sr
        )

    # Save metadata
    meta_path = palette.save_metadata(output_dir)

    logger.info(
        "sample_extractor.palette_complete",
        total_samples=palette.total_samples(),
        drum_classes=list(palette.drums.keys()),
        tonal_stems=list(palette.tones.keys()),
        meta=str(meta_path),
    )

    return palette


def merge_palettes(*palettes: ClonedPalette) -> ClonedPalette:
    """Merge multiple palettes (from multiple references) into one.

    For each drum class, keeps the highest-quality samples across all sources.
    For tonal samples, keeps all and lets the renderer choose.
    """
    merged = ClonedPalette()

    for palette in palettes:
        merged.source_tracks.extend(palette.source_tracks)

        # Merge drums
        for label, samples in palette.drums.items():
            if label not in merged.drums:
                merged.drums[label] = []
            merged.drums[label].extend(samples)

        # Merge tones
        for stem, samples in palette.tones.items():
            if stem not in merged.tones:
                merged.tones[stem] = []
            merged.tones[stem].extend(samples)

    # Sort all by quality (best first) and cap at 8 per class
    for label in merged.drums:
        merged.drums[label] = sorted(
            merged.drums[label], key=lambda s: s.quality, reverse=True
        )[:8]

    for stem in merged.tones:
        merged.tones[stem] = sorted(
            merged.tones[stem], key=lambda s: s.quality, reverse=True
        )[:8]

    # Deduplicate source tracks
    merged.source_tracks = list(dict.fromkeys(merged.source_tracks))

    return merged


# ── Utilities ────────────────────────────────────────────


def _find_stem(stems_dir: Path, stem_name: str) -> Path | None:
    """Find a stem file in the directory, trying common naming patterns."""
    patterns = [
        f"{stem_name}.wav",
        f"{stem_name}.mp3",
        f"*{stem_name}*.wav",
        f"*{stem_name}*.mp3",
    ]
    for pattern in patterns:
        matches = list(stems_dir.glob(pattern))
        if matches:
            return matches[0]
    return None
