"""AURALIS Texture Generator — LLM-guided neural texture orchestration.

Uses the LLM's texture_prompts (from ProductionPlan) to generate
atmospheric layers via Stable Audio (Replicate API).

The LLM decides WHAT textures to generate and WHERE they go.
Stable Audio provides the sonic fidelity.

Pipeline:
  LLM → texture_prompts → Stable Audio → download → trim → crossfade → stem
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import structlog

logger = structlog.get_logger()


@dataclass
class TextureRequest:
    """A single texture generation request from the LLM."""

    section: str  # "breakdown", "intro", "outro"
    prompt: str  # "ethereal dark ambient pad, reverb tail, 128bpm"
    duration_s: float = 16.0  # Desired duration
    start_beat: float = 0.0  # Where in the arrangement
    volume: float = 0.5  # Mix volume (0-1)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TextureRequest:
        return cls(
            section=d.get("section", ""),
            prompt=d.get("prompt", ""),
            duration_s=d.get("duration_s", 16.0),
            start_beat=d.get("start_beat", 0.0),
            volume=d.get("volume", 0.5),
        )


@dataclass
class GeneratedTexture:
    """Result of a texture generation."""

    path: Path
    prompt: str
    duration_s: float
    section: str
    volume: float


def generate_textures(
    texture_prompts: list[dict[str, Any]],
    output_dir: str | Path,
    bpm: float = 128.0,
) -> list[GeneratedTexture]:
    """Generate all textures from LLM plan via Stable Audio.

    Args:
        texture_prompts: List of texture request dicts from ProductionPlan.
        output_dir: Directory to save generated audio.
        bpm: Track BPM (passed to Stable Audio).

    Returns:
        List of GeneratedTexture results.
    """
    output_dir = Path(output_dir) / "textures"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — only loaded when textures are needed
    try:
        from auralis.hands.stable_audio import StableAudioClient
    except ImportError:
        logger.warning("texture_gen.stable_audio_unavailable")
        return []

    client = StableAudioClient()
    results: list[GeneratedTexture] = []

    for i, prompt_dict in enumerate(texture_prompts):
        req = TextureRequest.from_dict(prompt_dict)

        if not req.prompt:
            continue

        logger.info(
            "texture_gen.generating",
            index=i,
            section=req.section,
            prompt=req.prompt[:60],
            duration=req.duration_s,
        )

        generated_path = client.generate_loop(
            prompt=req.prompt,
            bpm=bpm,
            seconds=req.duration_s,
            output_dir=output_dir,
        )

        if generated_path and generated_path.exists():
            # Verify the generated file is valid audio
            try:
                info = sf.info(str(generated_path))
                actual_duration = info.duration
            except Exception as e:
                logger.warning(
                    "texture_gen.invalid_audio",
                    path=str(generated_path),
                    error=str(e),
                )
                continue

            texture = GeneratedTexture(
                path=generated_path,
                prompt=req.prompt,
                duration_s=actual_duration,
                section=req.section,
                volume=req.volume,
            )
            results.append(texture)

            logger.info(
                "texture_gen.generated",
                section=req.section,
                path=str(generated_path),
                duration=round(actual_duration, 2),
            )
        else:
            logger.warning(
                "texture_gen.generation_failed",
                section=req.section,
                prompt=req.prompt[:40],
            )

    logger.info("texture_gen.complete", total=len(results))
    return results


def load_texture_audio(
    texture: GeneratedTexture,
    target_duration_s: float | None = None,
    sr: int = 44100,
) -> np.ndarray:
    """Load texture audio and optionally trim/loop to target duration.

    If the generated audio is shorter than target, it loops with crossfade.
    If longer, it trims with smooth fade-out.

    Returns mono audio array.
    """
    if not texture.path.exists():
        return np.zeros(int(sr * (target_duration_s or 1.0)))

    y, sr_actual = sf.read(str(texture.path), dtype="float64")

    # Convert to mono if stereo
    if y.ndim > 1:
        y = np.mean(y, axis=1)

    # Resample if needed
    if sr_actual != sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr_actual, target_sr=sr)

    if target_duration_s is None:
        return y * texture.volume

    target_samples = int(target_duration_s * sr)

    if len(y) >= target_samples:
        # Trim with fade-out
        result = y[:target_samples].copy()
        fade_len = min(int(0.05 * sr), target_samples // 4)
        if fade_len > 0:
            result[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)
    else:
        # Loop with crossfade
        result = _loop_with_crossfade(y, target_samples, sr)

    return result * texture.volume


def _loop_with_crossfade(
    audio: np.ndarray,
    target_samples: int,
    sr: int,
    crossfade_ms: float = 50.0,
) -> np.ndarray:
    """Loop audio to fill target length with smooth crossfades."""
    crossfade_samples = int(crossfade_ms / 1000 * sr)
    result = np.zeros(target_samples)

    pos = 0
    while pos < target_samples:
        chunk_len = min(len(audio), target_samples - pos)
        chunk = audio[:chunk_len].copy()

        # Crossfade at loop boundary
        if pos > 0 and crossfade_samples > 0:
            fade_len = min(crossfade_samples, chunk_len, pos)
            if fade_len > 0:
                fade_in = np.linspace(0.0, 1.0, fade_len)
                fade_out = np.linspace(1.0, 0.0, fade_len)
                chunk[:fade_len] *= fade_in
                result[pos : pos + fade_len] *= fade_out

        end = min(pos + chunk_len, target_samples)
        actual = end - pos
        result[pos:end] += chunk[:actual]
        pos += chunk_len

    return result
