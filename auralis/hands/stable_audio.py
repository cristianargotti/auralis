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

# Model: stability-ai/stable-audio-open-1.0
# Version hash might change, but this is a common one. Better to use "stability-ai/stable-audio-open-1.0" as model name if allowed.
# Replicate python client allows "owner/model" or "owner/model:version".
MODEL_ID = "stability-ai/stable-audio-open-1.0"

class StableAudioClient:
    """Client for generating audio via Replicate API."""

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
            seconds: Duration in seconds
            output_dir: Directory to save the file (optional)

        Returns:
            Path to the generated WAV file, or None if failed.
        """
        if not self.api_token:
            logger.error("stable_audio.token_missing")
            return None

        try:
            logger.info("stable_audio.generate_start", prompt=prompt, bpm=bpm, seconds=seconds)
            
            # Run inference
            output = replicate.run(
                MODEL_ID,
                input={
                    "prompt": prompt,
                    "seconds_total": seconds,
                    # Optional tuning
                    # "steps": 100,
                    # "cfg_scale": 7,
                }
            )
            
            # The output is usually a URI string or file object
            if not output:
                logger.error("stable_audio.empty_response")
                return None

            audio_url = str(output)
            
            # Download audio if output_dir provided
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Sanitized prompt for filename
                slug = "".join(c for c in prompt[:20] if c.isalnum() or c in (' ', '_')).strip().replace(' ', '_')
                filename = f"gen_{slug}_{timestamp}.wav"
                filepath = output_dir / filename
                
                with httpx.Client() as client:
                    resp = client.get(audio_url)
                    resp.raise_for_status()
                    filepath.write_bytes(resp.content)
                
                logger.info("stable_audio.generate_success", path=str(filepath))
                return filepath
            
            return None

        except Exception as e:
            logger.error("stable_audio.generate_failed", error=str(e))
            return None
