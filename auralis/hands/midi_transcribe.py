"""MIDI transcription via Replicate API (external GPU).

Provides access to GPU-heavy transcription models like MT3
when local basic-pitch is insufficient or unavailable.
Falls back gracefully if Replicate is not configured.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import httpx
import structlog

from auralis.config import settings

logger = structlog.get_logger()

# Available models on Replicate for audio-to-MIDI
# We use basic-pitch via cog container for best compatibility
MODELS = {
    "basic-pitch": "spotify/basic-pitch:bbc97d65a95f0636bf1b7187fb18a1671ba52b0b",
    # Future: MT3, Aria-AMT, etc.
}

DEFAULT_MODEL = "basic-pitch"


class ReplicateMIDIClient:
    """Client for MIDI transcription via Replicate API.

    Uses GPU-powered models externally when local inference
    is unavailable or a more powerful model is needed.
    """

    def __init__(self) -> None:
        self.api_token = settings.replicate_api_token or os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            logger.warning("midi_transcribe.missing_token", msg="REPLICATE_API_TOKEN not set")
        else:
            os.environ["REPLICATE_API_TOKEN"] = self.api_token

    @property
    def available(self) -> bool:
        """Check if Replicate is configured."""
        return bool(self.api_token)

    def transcribe(
        self,
        audio_path: str | Path,
        output_dir: str | Path,
        model: str = DEFAULT_MODEL,
        onset_threshold: float = 0.5,
        frame_threshold: float = 0.3,
        min_note_length: float = 58.0,
    ) -> Path | None:
        """Transcribe audio to MIDI using Replicate.

        Args:
            audio_path: Path to audio file.
            output_dir: Directory to save MIDI output.
            model: Model key from MODELS dict.
            onset_threshold: Onset detection sensitivity.
            frame_threshold: Frame detection sensitivity.
            min_note_length: Minimum note length in ms.

        Returns:
            Path to the downloaded MIDI file, or None if failed.
        """
        import replicate

        if not self.available:
            logger.error("midi_transcribe.token_missing")
            return None

        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_id = MODELS.get(model)
        if not model_id:
            logger.error("midi_transcribe.unknown_model", model=model)
            return None

        try:
            logger.info(
                "midi_transcribe.start",
                audio=str(audio_path),
                model=model,
            )

            # Upload audio file and run model
            output = None
            for attempt in range(2):
                try:
                    with open(audio_path, "rb") as f:
                        output = replicate.run(
                            model_id,
                            input={
                                "audio": f,
                                "onset_threshold": onset_threshold,
                                "frame_threshold": frame_threshold,
                                "minimum_note_length": min_note_length,
                            },
                        )
                    break
                except Exception as retry_err:
                    if attempt == 0:
                        logger.warning("midi_transcribe.retry", error=str(retry_err))
                        import time
                        time.sleep(2)
                    else:
                        raise

            if not output:
                logger.error("midi_transcribe.empty_response")
                return None

            # Handle output — could be URL, FileOutput, or dict
            midi_url = None
            if isinstance(output, dict):
                midi_url = output.get("midi") or output.get("midi_file")
            elif hasattr(output, "url"):
                midi_url = output.url
            elif isinstance(output, str):
                midi_url = output

            if not midi_url:
                logger.error("midi_transcribe.no_midi_url", output=str(output)[:200])
                return None

            # Download MIDI file
            stem_name = audio_path.stem
            midi_path = output_dir / f"{stem_name}.mid"

            with httpx.Client(timeout=30.0) as client:
                resp = client.get(str(midi_url))
                resp.raise_for_status()
                midi_path.write_bytes(resp.content)

            size_kb = midi_path.stat().st_size / 1024
            logger.info(
                "midi_transcribe.success",
                path=str(midi_path),
                size_kb=f"{size_kb:.1f}",
                model=model,
            )
            return midi_path

        except Exception as e:
            logger.error("midi_transcribe.failed", error=str(e), audio=str(audio_path))
            return None

    def transcribe_stems(
        self,
        stems_dir: str | Path,
        output_dir: str | Path,
        exclude_stems: list[str] | None = None,
        model: str = DEFAULT_MODEL,
    ) -> dict[str, Path | None]:
        """Transcribe all tonal stems in a directory.

        Args:
            stems_dir: Directory containing stems.
            output_dir: Directory to save MIDI files.
            exclude_stems: Stem names to skip.
            model: Model key.

        Returns:
            Dict mapping stem name to MIDI file path (or None if failed).
        """
        stems_dir = Path(stems_dir)
        output_dir = Path(output_dir)
        exclude = set(exclude_stems or ["drums"])

        results: dict[str, Path | None] = {}

        for stem_file in sorted(stems_dir.glob("*.wav")):
            stem_name = stem_file.stem
            if stem_name in exclude:
                logger.info("midi_transcribe.skip_stem", stem=stem_name)
                continue

            midi_path = self.transcribe(
                audio_path=stem_file,
                output_dir=output_dir,
                model=model,
            )
            results[stem_name] = midi_path

        return results
