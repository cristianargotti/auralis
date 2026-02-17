"""Stable Audio Open — text-to-audio generation (Replicate)."""

from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime

import httpx
import replicate
import structlog

from auralis.config import settings

logger = structlog.get_logger()

# Stable Audio Open max generation: 47 seconds
MODEL_ID = "stability-ai/stable-audio-open-1.0"
MAX_DURATION_S = 47.0  # Stable Audio Open hard limit


class StableAudioClient:
    """Client for generating audio via Replicate API.

    Stable Audio Open can generate up to 47 seconds of audio.
    For longer stems, the post-processor in stem_generator handles
    looping with crossfade.
    """

    def __init__(self) -> None:
        # Prioritize config, fallback to env var
        self.api_token = settings.replicate_api_token or os.getenv("REPLICATE_API_TOKEN")
        if not self.api_token:
            logger.warning("stable_audio.missing_token", msg="REPLICATE_API_TOKEN not set")
        else:
            # Ensure environment variable is set for the library
            os.environ["REPLICATE_API_TOKEN"] = self.api_token

    def generate_loop(
        self,
        prompt: str,
        bpm: float,
        seconds: float = 8.0,
        output_dir: Path | None = None,
    ) -> Path | None:
        """Generate a looping audio sample from a text prompt.

        Args:
            prompt: Description of the sound (e.g. "Deep house bass loop")
            bpm: Tempo in BPM
            seconds: Duration in seconds (capped at 47s)
            output_dir: Directory to save the file (optional)

        Returns:
            Path to the generated WAV file, or None if failed.
        """
        if not self.api_token:
            logger.error("stable_audio.token_missing")
            return None

        # Cap at Stable Audio max (post-processor handles looping to full duration)
        actual_seconds = min(seconds, MAX_DURATION_S)

        try:
            logger.info(
                "stable_audio.generate_start",
                prompt=prompt, bpm=bpm,
                requested=seconds, actual=actual_seconds,
            )

            # Run inference with retry for transient errors
            output = None
            for attempt in range(2):
                try:
                    output = replicate.run(
                        MODEL_ID,
                        input={
                            "prompt": prompt,
                            "seconds_total": actual_seconds,
                        }
                    )
                    break
                except Exception as retry_err:
                    if attempt == 0:
                        logger.warning("stable_audio.retry", error=str(retry_err))
                        import time
                        time.sleep(2)
                    else:
                        raise

            if not output:
                logger.error("stable_audio.empty_response")
                return None

            # Replicate may return a FileOutput object, URL string, or iterator
            if hasattr(output, "url"):
                audio_url = output.url
            elif isinstance(output, str):
                audio_url = output
            else:
                # Could be an iterator or FileOutput — try to get URL
                audio_url = str(output)

            # Download audio if output_dir provided
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                slug = "".join(
                    c for c in prompt[:20] if c.isalnum() or c in (' ', '_')
                ).strip().replace(' ', '_')
                filename = f"gen_{slug}_{timestamp}.wav"
                filepath = output_dir / filename

                with httpx.Client(timeout=60.0) as client:
                    resp = client.get(audio_url)
                    resp.raise_for_status()
                    filepath.write_bytes(resp.content)

                size_kb = filepath.stat().st_size / 1024
                logger.info(
                    "stable_audio.generate_success",
                    path=str(filepath), size_kb=f"{size_kb:.0f}",
                )
                return filepath

            return None

        except Exception as e:
            logger.error("stable_audio.generate_failed", error=str(e))
            return None
