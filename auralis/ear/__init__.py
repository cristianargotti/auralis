"""EAR — Analysis & Deconstruction layer.

Track-agnostic audio intelligence:
- Separator: Mel-RoFormer (primary) + HTDemucs (fallback)
- MIDI: Basic-Pitch (Spotify) polyphonic extraction
- Profiler: BPM, key, sections, energy, spectral fingerprint
"""

from auralis.ear.separator import (
    SeparationModel,
    SeparationResult,
    StemType,
    separate_track,
    get_available_models,
    get_best_available_model,
)
from auralis.ear.midi_extractor import (
    MIDIExtractionResult,
    extract_midi,
    extract_midi_from_stems,
)
from auralis.ear.profiler import (
    TrackDNA,
    Section,
    profile_track,
)

__all__ = [
    "SeparationModel",
    "SeparationResult",
    "StemType",
    "separate_track",
    "get_available_models",
    "get_best_available_model",
    "MIDIExtractionResult",
    "extract_midi",
    "extract_midi_from_stems",
    "TrackDNA",
    "Section",
    "profile_track",
]
