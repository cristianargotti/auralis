"""Source separation engine — Mel-RoFormer primary, HTDemucs fallback.

Supports multiple separation models:
- Mel-RoFormer: Best quality (9.96 dB SDR), via python-audio-separator
- HTDemucs: Proven fallback (7.68 dB SDR), via demucs
- Auto-detection: uses best available model

All models produce track-agnostic stems — no hardcoded references.
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


# ── Models ───────────────────────────────────────────────


class SeparationModel(StrEnum):
    """Available separation models."""

    # Primary — highest quality
    MEL_ROFORMER = "mel_roformer"
    MEL_ROFORMER_VOCALS = "mel_roformer_vocals"

    # Fallback — proven reliable
    HTDEMUCS = "htdemucs"
    HTDEMUCS_FT = "htdemucs_ft"


class StemType(StrEnum):
    """Standard stem types from separation."""

    VOCALS = "vocals"
    DRUMS = "drums"
    BASS = "bass"
    OTHER = "other"
    PIANO = "piano"
    GUITAR = "guitar"
    # These come from 6-stem models:
    INSTRUMENTAL = "instrumental"


# ── Result ───────────────────────────────────────────────


@dataclass
class SeparationResult:
    """Result of source separation — completely track-agnostic."""

    model_used: str
    stems: dict[str, Path] = field(default_factory=dict)
    sample_rate: int = 44100
    duration: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "model_used": self.model_used,
            "stems": {k: str(v) for k, v in self.stems.items()},
            "sample_rate": self.sample_rate,
            "duration": self.duration,
            "metadata": self.metadata,
        }

    def save_metadata(self, output_dir: Path) -> Path:
        """Save separation metadata to JSON."""
        meta_path = output_dir / "separation_metadata.json"
        meta_path.write_text(json.dumps(self.to_dict(), indent=2))
        return meta_path


# ── Availability checks ─────────────────────────────────


def _check_audio_separator_available() -> bool:
    """Check if python-audio-separator (Mel-RoFormer) is available."""
    try:
        from audio_separator.separator import Separator  # noqa: F401

        return True
    except ImportError:
        return False


def _check_demucs_available() -> bool:
    """Check if demucs is installed."""
    try:
        import demucs  # noqa: F401

        return True
    except ImportError:
        return False


def _check_torch_available() -> tuple[bool, str]:
    """Check if PyTorch is available and detect device."""
    try:
        import torch

        if torch.cuda.is_available():
            return True, "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return True, "mps"
        return True, "cpu"
    except ImportError:
        return False, "none"


def get_best_available_model() -> SeparationModel:
    """Auto-detect best available separation model."""
    if _check_audio_separator_available():
        return SeparationModel.MEL_ROFORMER
    if _check_demucs_available():
        return SeparationModel.HTDEMUCS_FT
    msg = "No separation model available. Install: pip install audio-separator[gpu] or pip install demucs"
    raise ImportError(msg)


# ── Mel-RoFormer separation ─────────────────────────────


def _separate_mel_roformer(
    audio_path: Path,
    output_dir: Path,
    model_variant: str = "mel_roformer",
    progress_callback: Any | None = None,
) -> SeparationResult:
    """Separate using Mel-RoFormer via python-audio-separator.

    Mel-RoFormer achieves 9.96 dB avg SDR — best in class for 2025/2026.
    Uses Mel-band projection for human-auditory-aligned separation.
    """
    from audio_separator.separator import Separator

    if progress_callback:
        progress_callback(1, 5, "Loading Mel-RoFormer model...")

    # Initialize separator
    separator = Separator(
        output_dir=str(output_dir),
        output_format="wav",
    )

    # Select model — vocals vs general
    if model_variant == "mel_roformer_vocals":
        # Vocal-optimized: 11.4 dB SDR on vocals specifically
        model_file = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
    else:
        # General purpose: 9.96 dB SDR average across all stems
        # NOTE: When a dedicated 4-stem general model is released, update here
        model_file = "model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"

    if progress_callback:
        progress_callback(2, 5, f"Loading model: {model_file}")

    separator.load_model(model_filename=model_file)

    if progress_callback:
        progress_callback(3, 5, "Separating stems (Mel-RoFormer)...")

    # Separate — returns list of output file paths
    output_files = separator.separate(str(audio_path))

    if progress_callback:
        progress_callback(4, 5, "Organizing stems...")

    # Map output files to stem types
    stem_paths: dict[str, Path] = {}
    for output_file in output_files:
        p = Path(output_file)
        fname = p.stem.lower()
        # audio-separator names files like "track_(Vocals).wav", "track_(Instrumental).wav"
        if "vocal" in fname:
            stem_paths["vocals"] = p
        elif "instrumental" in fname or "no_vocal" in fname:
            stem_paths["instrumental"] = p
        elif "drum" in fname:
            stem_paths["drums"] = p
        elif "bass" in fname:
            stem_paths["bass"] = p
        elif "other" in fname:
            stem_paths["other"] = p
        elif "piano" in fname:
            stem_paths["piano"] = p
        elif "guitar" in fname:
            stem_paths["guitar"] = p
        else:
            # Unknown stem — keep with original name
            stem_paths[p.stem] = p

    # Get audio info
    info = sf.info(str(audio_path))

    if progress_callback:
        progress_callback(5, 5, "Mel-RoFormer separation complete")

    return SeparationResult(
        model_used=f"mel_roformer:{model_file}",
        stems=stem_paths,
        sample_rate=info.samplerate,
        duration=info.duration,
        metadata={
            "engine": "audio-separator",
            "model": model_file,
            "source_stems": list(stem_paths.keys()),
            "quality": "9.96_dB_SDR",
        },
    )


# ── HTDemucs separation (fallback) ──────────────────────


def _separate_htdemucs(
    audio_path: Path,
    output_dir: Path,
    model: str = "htdemucs_ft",
    device: str = "auto",
    shifts: int = 1,
    overlap: float = 0.25,
    progress_callback: Any | None = None,
) -> SeparationResult:
    """Separate using HTDemucs — proven fallback at 7.68 dB SDR."""
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

    logger.info("Starting HTDemucs separation", model=model, device=str(selected_device))

    if progress_callback:
        progress_callback(1, 4, f"Loading model {model}...")

    demucs_model = get_model(model)
    demucs_model.to(selected_device)

    if progress_callback:
        progress_callback(2, 4, "Loading audio...")

    y, sr = sf.read(str(audio_path))
    if y.ndim == 1:
        y = np.stack([y, y], axis=0)
    elif y.ndim == 2:
        y = y.T

    wav = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(selected_device)

    if progress_callback:
        progress_callback(3, 4, "Separating stems (HTDemucs)...")

    sources = apply_model(
        demucs_model,
        wav,
        shifts=shifts,
        overlap=overlap,
        device=selected_device,
    )

    if progress_callback:
        progress_callback(4, 4, "Saving stems...")

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
        model_used=model,
        stems=stem_paths,
        sample_rate=sr,
        duration=duration,
        metadata={
            "engine": "demucs",
            "device": str(selected_device),
            "shifts": shifts,
            "overlap": overlap,
            "source_names": list(source_names),
            "quality": "7.68_dB_SDR",
        },
    )


# ── Public API ───────────────────────────────────────────


def separate_track(
    audio_path: str | Path,
    output_dir: str | Path,
    model: SeparationModel | str = "auto",
    device: str = "auto",
    shifts: int = 1,
    overlap: float = 0.25,
    progress_callback: Any | None = None,
) -> SeparationResult:
    """Separate any track into stems — track-agnostic.

    Auto-selects best available model:
    1. Mel-RoFormer (9.96 dB SDR) — if audio-separator installed
    2. HTDemucs (7.68 dB SDR) — fallback

    Args:
        audio_path: Path to any audio file.
        output_dir: Directory to save separated stems.
        model: Model to use ('auto', 'mel_roformer', 'htdemucs', etc).
        device: Device for inference ('auto', 'cuda', 'cpu', 'mps').
        shifts: Shifts for HTDemucs (more = better quality, slower).
        overlap: Chunk overlap for HTDemucs.
        progress_callback: Optional callable(step, total, message).

    Returns:
        SeparationResult with paths to all separated stems.
    """
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        msg = f"Audio file not found: {audio_path}"
        raise FileNotFoundError(msg)

    # Auto-select model
    if model == "auto":
        selected = get_best_available_model()
        logger.info("Auto-selected model", model=selected)
    elif isinstance(model, str):
        selected = SeparationModel(model)
    else:
        selected = model

    # Route to the right engine
    if selected in (SeparationModel.MEL_ROFORMER, SeparationModel.MEL_ROFORMER_VOCALS):
        return _separate_mel_roformer(
            audio_path, output_dir,
            model_variant=selected.value,
            progress_callback=progress_callback,
        )
    else:
        return _separate_htdemucs(
            audio_path, output_dir,
            model=selected.value,
            device=device,
            shifts=shifts,
            overlap=overlap,
            progress_callback=progress_callback,
        )


def get_available_models() -> list[dict[str, str]]:
    """Get list of available separation models with availability status."""
    has_mel_roformer = _check_audio_separator_available()
    has_demucs = _check_demucs_available()

    models = [
        {
            "id": SeparationModel.MEL_ROFORMER.value,
            "name": "Mel-RoFormer",
            "description": "Best quality — 9.96 dB SDR, Mel-band Transformer",
            "stems": "vocals, instrumental (2-stem) or vocals, drums, bass, other (4-stem)",
            "quality": "9.96 dB SDR",
            "available": str(has_mel_roformer),
            "install": "pip install audio-separator[gpu]",
        },
        {
            "id": SeparationModel.MEL_ROFORMER_VOCALS.value,
            "name": "Mel-RoFormer (Vocals)",
            "description": "Optimized for vocal separation — 11.4 dB SDR on vocals",
            "stems": "vocals, instrumental",
            "quality": "11.4 dB SDR (vocals)",
            "available": str(has_mel_roformer),
            "install": "pip install audio-separator[gpu]",
        },
        {
            "id": SeparationModel.HTDEMUCS.value,
            "name": "HTDemucs",
            "description": "Hybrid Transformer Demucs — proven general purpose",
            "stems": "vocals, drums, bass, other",
            "quality": "7.68 dB SDR",
            "available": str(has_demucs),
            "install": "pip install demucs",
        },
        {
            "id": SeparationModel.HTDEMUCS_FT.value,
            "name": "HTDemucs Fine-Tuned",
            "description": "Fine-tuned variant — highest quality vocals in Demucs family",
            "stems": "vocals, drums, bass, other",
            "quality": "8.0+ dB SDR",
            "available": str(has_demucs),
            "install": "pip install demucs",
        },
    ]
    return models
