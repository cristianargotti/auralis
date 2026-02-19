"""MIDI extraction from audio — hybrid local + Replicate strategy.

Strategy:
  1. Try basic-pitch (ONNX backend) locally — fast, no API calls
  2. Fall back to Replicate API for GPU-powered transcription
  3. Final fallback: librosa.pyin for simple pitch tracking

Extracts polyphonic MIDI from tonal audio stems (bass, melody, pads).
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
    method: str = "none"
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
            "method": self.method,
            "metadata": self.metadata,
        }


def _check_basic_pitch_available() -> bool:
    """Check if basic-pitch is installed (any backend)."""
    try:
        import basic_pitch  # noqa: F401
        return True
    except ImportError:
        return False


def _check_replicate_available() -> bool:
    """Check if Replicate client is configured."""
    try:
        from auralis.hands.midi_transcribe import ReplicateMIDIClient
        client = ReplicateMIDIClient()
        return client.available
    except Exception:
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
    use_replicate_fallback: bool = True,
) -> MIDIExtractionResult:
    """Extract MIDI from an audio file.

    Tries methods in order:
      1. basic-pitch (local, ONNX backend preferred)
      2. Replicate API (external GPU)
      3. librosa.pyin (monophonic fallback)

    Args:
        audio_path: Path to audio file (ideally a single-instrument stem).
        output_dir: Directory to save MIDI file.
        onset_threshold: Onset detection sensitivity (0-1).
        frame_threshold: Frame detection sensitivity (0-1).
        min_note_length: Minimum note length in ms.
        min_frequency: Minimum frequency to transcribe (Hz).
        max_frequency: Maximum frequency to transcribe (Hz).
        progress_callback: Optional callable(step, total, message).
        use_replicate_fallback: Try Replicate if local fails.

    Returns:
        MIDIExtractionResult with path to MIDI file and statistics.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        msg = f"Audio file not found: {audio_path}"
        raise FileNotFoundError(msg)

    # ── Method 1: basic-pitch (local ONNX) ──
    if _check_basic_pitch_available():
        try:
            return _extract_with_basic_pitch(
                audio_path, output_dir,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                min_note_length=min_note_length,
                min_frequency=min_frequency,
                max_frequency=max_frequency,
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.warning("midi.basic_pitch_failed", error=str(e))

    # ── Method 2: Replicate API (external GPU) ──
    if use_replicate_fallback and _check_replicate_available():
        try:
            return _extract_with_replicate(
                audio_path, output_dir,
                onset_threshold=onset_threshold,
                frame_threshold=frame_threshold,
                min_note_length=min_note_length,
                progress_callback=progress_callback,
            )
        except Exception as e:
            logger.warning("midi.replicate_failed", error=str(e))

    # ── Method 3: librosa.pyin (monophonic fallback) ──
    try:
        return _extract_with_pyin(
            audio_path, output_dir,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            progress_callback=progress_callback,
        )
    except Exception as e:
        logger.warning("midi.pyin_failed", error=str(e))

    # All methods failed
    return MIDIExtractionResult(
        source_audio=str(audio_path),
        method="none",
        metadata={"error": "All extraction methods failed"},
    )


def _extract_with_basic_pitch(
    audio_path: Path,
    output_dir: Path,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length: float = 58.0,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
    progress_callback: Any | None = None,
) -> MIDIExtractionResult:
    """Extract MIDI using basic-pitch (local, ONNX preferred)."""
    from basic_pitch.inference import predict

    logger.info("midi.basic_pitch_start", audio=str(audio_path))

    if progress_callback:
        progress_callback(1, 3, "Running basic-pitch MIDI extraction...")

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

    stem_name = audio_path.stem
    midi_path = output_dir / f"{stem_name}.mid"
    midi_data.write(str(midi_path))

    notes_count = len(note_events)
    if notes_count > 0:
        pitches = [n[2] for n in note_events]
        pitch_range = (int(min(pitches)), int(max(pitches)))
        duration = float(max(n[1] for n in note_events))
        confidences = [n[3] for n in note_events]
        avg_confidence = float(np.mean(confidences))
    else:
        pitch_range = (0, 0)
        duration = 0.0
        avg_confidence = 0.0

    if progress_callback:
        progress_callback(3, 3, "Done!")

    logger.info(
        "midi.basic_pitch_success",
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
        method="basic-pitch-onnx",
        metadata={
            "onset_threshold": onset_threshold,
            "frame_threshold": frame_threshold,
            "min_note_length": min_note_length,
        },
    )


def _extract_with_replicate(
    audio_path: Path,
    output_dir: Path,
    *,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length: float = 58.0,
    progress_callback: Any | None = None,
) -> MIDIExtractionResult:
    """Extract MIDI using Replicate API (external GPU)."""
    from auralis.hands.midi_transcribe import ReplicateMIDIClient

    logger.info("midi.replicate_start", audio=str(audio_path))

    if progress_callback:
        progress_callback(1, 3, "Sending to Replicate for MIDI extraction...")

    client = ReplicateMIDIClient()
    midi_path = client.transcribe(
        audio_path=audio_path,
        output_dir=output_dir,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        min_note_length=min_note_length,
    )

    if not midi_path or not midi_path.exists():
        raise RuntimeError("Replicate transcription returned no MIDI file")

    if progress_callback:
        progress_callback(2, 3, "Analyzing MIDI output...")

    # Read MIDI to get stats
    notes_count = 0
    pitch_range = (0, 0)
    duration = 0.0

    try:
        import mido
        mid = mido.MidiFile(str(midi_path))
        all_notes = []
        current_time = 0.0
        for track in mid.tracks:
            current_time = 0.0
            for msg in track:
                current_time += msg.time
                if msg.type == "note_on" and msg.velocity > 0:
                    all_notes.append((msg.note, current_time))

        notes_count = len(all_notes)
        if notes_count > 0:
            pitches = [n[0] for n in all_notes]
            pitch_range = (min(pitches), max(pitches))
            duration = max(n[1] for n in all_notes)
    except Exception as e:
        logger.warning("midi.replicate_stats_failed", error=str(e))

    if progress_callback:
        progress_callback(3, 3, "Done!")

    logger.info(
        "midi.replicate_success",
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
        confidence=0.8,  # Replicate models are generally high quality
        method="replicate-api",
        metadata={"model": "basic-pitch"},
    )


def _extract_with_pyin(
    audio_path: Path,
    output_dir: Path,
    *,
    min_frequency: float | None = None,
    max_frequency: float | None = None,
    progress_callback: Any | None = None,
) -> MIDIExtractionResult:
    """Extract MIDI using librosa.pyin (monophonic pitch tracking)."""
    import librosa
    import mido

    logger.info("midi.pyin_start", audio=str(audio_path))

    if progress_callback:
        progress_callback(1, 3, "Running pitch tracking (librosa.pyin)...")

    fmin = min_frequency or 65.0  # C2
    fmax = max_frequency or 2093.0  # C7

    y, sr = librosa.load(str(audio_path), sr=22050)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr,
    )

    if progress_callback:
        progress_callback(2, 3, "Converting to MIDI...")

    # Convert F0 to MIDI notes
    midi_notes = librosa.hz_to_midi(f0)
    times = librosa.times_like(f0, sr=sr)

    # Build MIDI file from pitch contour
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=audio_path.stem))

    ticks_per_beat = mid.ticks_per_beat
    tempo = 500000  # 120 BPM default
    track.append(mido.MetaMessage("set_tempo", tempo=tempo))

    # Group continuous pitches into notes
    notes = []
    current_note = None
    current_start = None

    for i, (t, note, voiced) in enumerate(zip(times, midi_notes, voiced_flag)):
        if voiced and not np.isnan(note):
            rounded_note = int(round(note))
            if current_note is None or abs(rounded_note - current_note) >= 1:
                if current_note is not None:
                    notes.append((current_note, current_start, t))
                current_note = rounded_note
                current_start = t
        else:
            if current_note is not None:
                notes.append((current_note, current_start, t))
                current_note = None
                current_start = None

    if current_note is not None:
        notes.append((current_note, current_start, times[-1]))

    # Filter very short notes
    min_dur = 0.05  # 50ms
    notes = [(n, s, e) for n, s, e in notes if (e - s) >= min_dur]

    # Write notes to MIDI
    prev_tick = 0
    for note, start, end in notes:
        start_tick = int(start * ticks_per_beat * 1_000_000 / tempo)
        end_tick = int(end * ticks_per_beat * 1_000_000 / tempo)
        delta_on = max(0, start_tick - prev_tick)
        delta_off = max(1, end_tick - start_tick)

        track.append(mido.Message("note_on", note=note, velocity=80, time=delta_on))
        track.append(mido.Message("note_off", note=note, velocity=0, time=delta_off))
        prev_tick = end_tick

    stem_name = audio_path.stem
    midi_path = output_dir / f"{stem_name}.mid"
    mid.save(str(midi_path))

    if progress_callback:
        progress_callback(3, 3, "Done!")

    notes_count = len(notes)
    pitch_range = (
        (min(n[0] for n in notes), max(n[0] for n in notes)) if notes else (0, 0)
    )
    duration = float(notes[-1][2]) if notes else 0.0

    logger.info(
        "midi.pyin_success",
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
        confidence=0.5,  # pyin is less accurate than ML models
        method="librosa-pyin",
        metadata={"monophonic": True},
    )


def extract_midi_from_stems(
    stems_dir: str | Path,
    output_dir: str | Path,
    exclude_stems: list[str] | None = None,
    progress_callback: Any | None = None,
    use_replicate_fallback: bool = True,
) -> dict[str, MIDIExtractionResult]:
    """Extract MIDI from all tonal stems in a directory.

    Args:
        stems_dir: Directory containing separated stems.
        output_dir: Directory to save MIDI files.
        exclude_stems: Stem names to skip (e.g., ['drums', 'vocals']).
        progress_callback: Optional callable for progress.
        use_replicate_fallback: Try Replicate if local fails.

    Returns:
        Dict mapping stem name to extraction result.
    """
    stems_dir = Path(stems_dir)
    output_dir = Path(output_dir)
    exclude = set(exclude_stems or ["drums"])

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
            use_replicate_fallback=use_replicate_fallback,
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

    logger.info(
        "midi.batch_complete",
        stems=len(results),
        methods={v.method for v in results.values()},
    )

    return results
