"""Source separation engine — Demucs v4 and Mel-Band RoFormer.

Supports graceful degradation: if ML models aren't installed,
provides clear error messages. Full power on EC2 with GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import structlog

logger = structlog.get_logger()


class SeparationModel(StrEnum):
    """Available separation models."""

    HTDEMUCS = "htdemucs"
    HTDEMUCS_FT = "htdemucs_ft"
    DEMUCS_V4 = "mdx_extra"


class StemType(StrEnum):
    """Standard stem types from separation."""

    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"


@dataclass
class SeparationResult:
    """Result of source separation."""

    model_used: str
    stems: dict[str, Path] = field(default_factory=dict)
    sample_rate: int = 44100
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "model_used": self.model_used,
            "stems": {k: str(v) for k, v in self.stems.items()},
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "metadata": self.metadata,
        }
        return result

    def save_metadata(self, output_dir: Path) -> Path:
        """Save separation metadata to JSON."""
        meta_path = output_dir / "separation_metadata.json"
        meta_path.write_text(json.dumps(self.to_dict(), indent=2))
        return meta_path


def _check_demucs_available() -> bool:
    """Check if demucs is installed."""
    try:
        import demucs  # noqa: F401

        return True
    except ImportError:
        return False


def _check_torch_available() -> bool:
    """Check if PyTorch is available and detect GPU."""
    try:
        import torch

        return bool(torch.cuda.is_available() or torch.backends.mps.is_available())
    except ImportError:
        return False


def separate_track(
    audio_path: str | Path,
    output_dir: str | Path,
    model: SeparationModel = SeparationModel.HTDEMUCS_FT,
    device: str = "auto",
    shifts: int = 1,
    overlap: float = 0.25,
    progress_callback: Any | None = None,
) -> SeparationResult:
    """Separate a track into stems using Demucs.

    Args:
        audio_path: Path to input audio file.
        output_dir: Directory to save separated stems.
        model: Which separation model to use.
        device: Device for inference ('auto', 'cuda', 'cpu', 'mps').
        shifts: Number of random shifts for better quality (more = slower).
        overlap: Overlap between chunks for smoothing.
        progress_callback: Optional callable(step, total, message) for progress.

    Returns:
        SeparationResult with paths to all separated stems.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        msg = f"Audio file not found: {audio_path}"
        raise FileNotFoundError(msg)

    if not _check_demucs_available():
        msg = "Demucs is not installed. Install ML dependencies: uv pip install auralis[ml]"
        raise ImportError(msg)

    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    # Device selection
    if device == "auto":
        if torch.cuda.is_available():
            selected_device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            selected_device = torch.device("mps")
        else:
            selected_device = torch.device("cpu")
    else:
        selected_device = torch.device(device)

    logger.info(
        "Starting separation",
        model=model.value,
        device=str(selected_device),
        audio=str(audio_path),
    )

    if progress_callback:
        progress_callback(1, 4, f"Loading model {model.value}...")

    # Load model
    demucs_model = get_model(model.value)
    demucs_model.to(selected_device)

    if progress_callback:
        progress_callback(2, 4, "Loading audio...")

    # Load audio
    y, sr = sf.read(str(audio_path))
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    elif y.ndim == 2:
        y = y.T

    # Convert to torch tensor
    wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(selected_device)

    if progress_callback:
        progress_callback(3, 4, "Separating stems (this takes a while)...")

    # Apply model
    sources = apply_model(
        demucs_model,
        wav,
        shifts=shifts,
        overlap=overlap,
        device=selected_device,
    )

    if progress_callback:
        progress_callback(4, 4, "Saving stems...")

    # Save stems
    stem_paths: dict[str, Path] = {}
    source_names = demucs_model.sources

    for idx, source_name in enumerate(source_names):
        stem_audio = sources[0, idx].cpu().numpy()
        stem_path = output_dir / f"{source_name}.wav"
        sf.write(str(stem_path), stem_audio.T, sr)
        stem_paths[source_name] = stem_path
        logger.info("Saved stem", name=source_name, path=str(stem_path))

    duration = float(y.shape[1]) / sr if y.ndim > 1 else float(len(y)) / sr

    return SeparationResult(
        model_used=model.value,
        stems=stem_paths,
        sample_rate=sr,
        duration=duration,
        metadata={
            "device": str(selected_device),
            "shifts": shifts,
            "overlap": overlap,
            "source_names": list(source_names),
        },
    )


def get_available_models() -> list[dict[str, str]]:
    """Get list of available separation models."""
    models = [
        {
            "id": SeparationModel.HTDEMUCS.value,
            "name": "HTDemucs",
            "description": "Hybrid Transformer Demucs — best general purpose",
            "stems": "vocals, drums, bass, other",
        },
        {
            "id": SeparationModel.HTDEMUCS_FT.value,
            "name": "HTDemucs Fine-Tuned",
            "description": "Fine-tuned variant — highest quality vocals",
            "stems": "vocals, drums, bass, other",
        },
        {
            "id": SeparationModel.DEMUCS_V4.value,
            "name": "MDX Extra",
            "description": "MDX competition model — competitive quality",
            "stems": "vocals, drums, bass, other",
        },
    ]
    return models
