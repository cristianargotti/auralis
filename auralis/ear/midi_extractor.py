"""MIDI extraction from audio using basic-pitch (Spotify).

Extracts polyphonic MIDI from tonal audio stems (bass, melody, pads).
Supports graceful degradation when basic-pitch is not installed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class MIDIExtractionResult:
    """Result of MIDI extraction from audio."""

    source_audio: str
    midi_path: Path | None = None
    notes_count: int = 0
    pitch_range: tuple[int, int] = (0, 0)
    duration: float = 0.0
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_audio": self.source_audio,
            "midi_path": str(self.midi_path) if self.midi_path else None,
            "notes_count": self.notes_count,
            "pitch_range": list(self.pitch_range),
            "duration": self.duration,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


def _check_basic_pitch_available() -> bool:
    """Check if basic-pitch is installed."""
    try:
        import basic_pitch  # noqa: F401

        return True
    except ImportError:
        return False


def extract_midi(
    audio_path: str | Path,
    output_dir: str | Path,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length: float = 58.0,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
    progress_callback: Any | None = None,
) -> MIDIExtractionResult:
    """Extract MIDI from an audio file using basic-pitch.

    Args:
        audio_path: Path to audio file (ideally a single-instrument stem).
        output_dir: Directory to save MIDI file.
        onset_threshold: Onset detection sensitivity (0-1).
        frame_threshold: Frame detection sensitivity (0-1).
        min_note_length: Minimum note length in ms.
        min_frequency: Minimum frequency to transcribe (Hz).
        max_frequency: Maximum frequency to transcribe (Hz).
        progress_callback: Optional callable(step, total, message).

    Returns:
        MIDIExtractionResult with path to MIDI file and statistics.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        msg = f"Audio file not found: {audio_path}"
        raise FileNotFoundError(msg)

    if not _check_basic_pitch_available():
        msg = "basic-pitch is not installed. Install on EC2: pip install basic-pitch"
        raise ImportError(msg)

    from basic_pitch.inference import predict

    if progress_callback:
        progress_callback(1, 3, "Running MIDI extraction model...")

    # Run basic-pitch inference
    _model_output, midi_data, note_events = predict(
        str(audio_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length,
        minimum_frequency=min_frequency,
        maximum_frequency=max_frequency,
    )

    if progress_callback:
        progress_callback(2, 3, "Saving MIDI file...")

    # Save MIDI file
    stem_name = audio_path.stem
    midi_path = output_dir / f"{stem_name}.mid"
    midi_data.write(str(midi_path))

    # Analyze results
    notes_count = len(note_events)
    if notes_count > 0:
        pitches = [n[2] for n in note_events]  # MIDI note numbers
        pitch_range = (int(min(pitches)), int(max(pitches)))
        [n[1] - n[0] for n in note_events]  # end - start times
        duration = float(max(n[1] for n in note_events))
        confidences = [n[3] for n in note_events]  # confidence
        avg_confidence = float(np.mean(confidences))
    else:
        pitch_range = (0, 0)
        duration = 0.0
        avg_confidence = 0.0

    if progress_callback:
        progress_callback(3, 3, "Done!")

    logger.info(
        "MIDI extraction complete",
        source=str(audio_path),
        notes=notes_count,
        midi=str(midi_path),
    )

    return MIDIExtractionResult(
        source_audio=str(audio_path),
        midi_path=midi_path,
        notes_count=notes_count,
        pitch_range=pitch_range,
        duration=duration,
        confidence=avg_confidence,
        metadata={
            "onset_threshold": onset_threshold,
            "frame_threshold": frame_threshold,
            "min_note_length": min_note_length,
        },
    )


def extract_midi_from_stems(
    stems_dir: str | Path,
    output_dir: str | Path,
    exclude_stems: list[str] | None = None,
    progress_callback: Any | None = None,
) -> dict[str, MIDIExtractionResult]:
    """Extract MIDI from all tonal stems in a directory.

    Args:
        stems_dir: Directory containing separated stems.
        output_dir: Directory to save MIDI files.
        exclude_stems: Stem names to skip (e.g., ['drums', 'vocals']).
        progress_callback: Optional callable for progress.

    Returns:
        Dict mapping stem name to extraction result.
    """
    stems_dir = Path(stems_dir)
    output_dir = Path(output_dir)
    exclude = set(exclude_stems or ["drums", "vocals"])

    results: dict[str, MIDIExtractionResult] = {}

    stem_files = sorted(stems_dir.glob("*.wav"))
    total = len(stem_files)

    for idx, stem_file in enumerate(stem_files):
        stem_name = stem_file.stem
        if stem_name in exclude:
            logger.info("Skipping non-tonal stem", stem=stem_name)
            continue

        if progress_callback:
            progress_callback(idx + 1, total, f"Extracting MIDI from {stem_name}...")

        result = extract_midi(
            audio_path=stem_file,
            output_dir=output_dir,
        )
        results[stem_name] = result

    # Save combined metadata
    meta_path = output_dir / "midi_extraction_metadata.json"
    meta_path.write_text(
        json.dumps(
            {k: v.to_dict() for k, v in results.items()},
            indent=2,
        )
    )

    return results
