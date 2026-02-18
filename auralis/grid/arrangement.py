"""AURALIS Arrangement Engine — Song structure, sections, and track layout.

Manages song arrangement: intro, verse, chorus, bridge, outro.
Generates complete multi-track arrangements from patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from auralis.grid.midi import (
    Pattern,
    Note,
    generate_bassline,
    generate_chord_progression,
    generate_drum_pattern,
    generate_melody,
)

SectionType = Literal["intro", "verse", "chorus", "bridge", "breakdown", "drop", "outro"]


# ── Data Types ───────────────────────────────────────────


@dataclass
class SectionTemplate:
    """Template for a song section."""

    name: SectionType
    bars: int = 8
    energy: float = 0.5  # 0-1 energy level
    has_drums: bool = True
    has_bass: bool = True
    has_chords: bool = True
    has_melody: bool = True
    drum_style: str = "four_on_floor"
    bass_style: str = "simple"
    melody_density: float = 0.5


@dataclass
class SectionInstance:
    """A rendered section with all track patterns."""

    template: SectionTemplate
    start_bar: int
    patterns: dict[str, Pattern] = field(default_factory=dict)

    @property
    def end_bar(self) -> int:
        return self.start_bar + self.template.bars


@dataclass
class ArrangementConfig:
    """Configuration for a song arrangement."""

    key: str = "C"
    scale: str = "minor"
    bpm: float = 120.0
    structure: list[SectionType] | None = None
    genre: Literal["house", "techno", "ambient", "pop", "hip_hop"] = "house"


@dataclass
class Arrangement:
    """Complete song arrangement with all sections."""

    config: ArrangementConfig
    sections: list[SectionInstance] = field(default_factory=list)

    @property
    def total_bars(self) -> int:
        if not self.sections:
            return 0
        return self.sections[-1].end_bar

    @property
    def total_beats(self) -> float:
        return self.total_bars * 4.0

    @property
    def duration_s(self) -> float:
        return self.total_beats * 60.0 / self.config.bpm


# ── Genre Templates ──────────────────────────────────────


GENRE_STRUCTURES: dict[str, list[SectionTemplate]] = {
    "house": [
        SectionTemplate("intro", bars=8, energy=0.3, has_melody=False, has_chords=False, drum_style="minimal"),
        SectionTemplate("verse", bars=16, energy=0.5, drum_style="four_on_floor", bass_style="simple"),
        SectionTemplate("breakdown", bars=8, energy=0.2, has_drums=False, melody_density=0.3),
        SectionTemplate("drop", bars=16, energy=1.0, drum_style="four_on_floor", bass_style="syncopated", melody_density=0.7),
        SectionTemplate("verse", bars=16, energy=0.6, drum_style="four_on_floor", bass_style="walking"),
        SectionTemplate("breakdown", bars=8, energy=0.2, has_drums=False, melody_density=0.4),
        SectionTemplate("drop", bars=16, energy=1.0, drum_style="four_on_floor", bass_style="syncopated"),
        SectionTemplate("outro", bars=8, energy=0.3, has_melody=False, drum_style="minimal"),
    ],
    "techno": [
        SectionTemplate("intro", bars=16, energy=0.3, has_melody=False, has_chords=False, drum_style="minimal"),
        SectionTemplate("verse", bars=16, energy=0.6, drum_style="four_on_floor", bass_style="syncopated", has_melody=False),
        SectionTemplate("breakdown", bars=8, energy=0.1, has_drums=False, has_bass=False),
        SectionTemplate("drop", bars=16, energy=1.0, drum_style="four_on_floor", bass_style="syncopated"),
        SectionTemplate("verse", bars=16, energy=0.7, drum_style="four_on_floor"),
        SectionTemplate("outro", bars=16, energy=0.3, drum_style="minimal", has_melody=False),
    ],
    "ambient": [
        SectionTemplate("intro", bars=8, energy=0.1, has_drums=False, has_bass=False, melody_density=0.2),
        SectionTemplate("verse", bars=16, energy=0.3, has_drums=False, bass_style="simple", melody_density=0.3),
        SectionTemplate("chorus", bars=8, energy=0.4, drum_style="minimal", melody_density=0.4),
        SectionTemplate("verse", bars=16, energy=0.3, has_drums=False, melody_density=0.3),
        SectionTemplate("outro", bars=8, energy=0.1, has_drums=False, has_bass=False),
    ],
    "pop": [
        SectionTemplate("intro", bars=4, energy=0.4, has_melody=False, drum_style="minimal"),
        SectionTemplate("verse", bars=8, energy=0.5, drum_style="breakbeat", bass_style="simple"),
        SectionTemplate("chorus", bars=8, energy=0.8, drum_style="four_on_floor", bass_style="walking", melody_density=0.7),
        SectionTemplate("verse", bars=8, energy=0.5, drum_style="breakbeat"),
        SectionTemplate("chorus", bars=8, energy=0.9, drum_style="four_on_floor", melody_density=0.8),
        SectionTemplate("bridge", bars=4, energy=0.3, drum_style="minimal"),
        SectionTemplate("chorus", bars=8, energy=1.0, drum_style="four_on_floor"),
        SectionTemplate("outro", bars=4, energy=0.3, has_melody=False),
    ],
    "hip_hop": [
        SectionTemplate("intro", bars=4, energy=0.3, drum_style="trap", has_melody=False),
        SectionTemplate("verse", bars=16, energy=0.6, drum_style="trap", bass_style="syncopated"),
        SectionTemplate("chorus", bars=8, energy=0.8, drum_style="trap", melody_density=0.6),
        SectionTemplate("verse", bars=16, energy=0.6, drum_style="trap", bass_style="syncopated"),
        SectionTemplate("chorus", bars=8, energy=0.8, drum_style="trap"),
        SectionTemplate("outro", bars=4, energy=0.3, drum_style="trap", has_melody=False),
    ],
}


# ── Arrangement Generator ────────────────────────────────


def generate_arrangement(config: ArrangementConfig) -> Arrangement:
    """Generate a complete song arrangement."""
    # Get structure from genre or custom
    if config.structure:
        templates = []
        for section_type in config.structure:
            defaults = next(
                (s for s in GENRE_STRUCTURES.get(config.genre, GENRE_STRUCTURES["house"]) if s.name == section_type),
                SectionTemplate(section_type),
            )
            templates.append(defaults)
    else:
        templates = GENRE_STRUCTURES.get(config.genre, GENRE_STRUCTURES["house"])

    arrangement = Arrangement(config=config)
    current_bar = 0

    for i, template in enumerate(templates):
        section = SectionInstance(template=template, start_bar=current_bar)
        seed = i * 100 + 42

        # Generate track patterns
        if template.has_drums:
            section.patterns["drums"] = generate_drum_pattern(
                style=template.drum_style,  # type: ignore[arg-type]
                bars=template.bars,
                velocity=int(70 + template.energy * 57),
                energy=template.energy,  # Pass narrative energy
            )

        if template.has_bass:
            section.patterns["bass"] = generate_bassline(
                root=config.key,
                scale=config.scale,
                octave=2,
                pattern_type=template.bass_style,  # type: ignore[arg-type]
                bars=template.bars,
                velocity=int(80 + template.energy * 47),
                energy=template.energy,  # Pass narrative energy
            )

        if template.has_chords:
            section.patterns["chords"] = generate_chord_progression(
                root=config.key,
                scale=config.scale,
                octave=3,
                bars=min(template.bars, 4),
                velocity=int(60 + template.energy * 40),
            )

        if template.has_melody:
            section.patterns["melody"] = generate_melody(
                root=config.key,
                scale=config.scale,
                octave=4,
                bars=template.bars,
                density=template.melody_density,
                velocity=int(70 + template.energy * 50),
                seed=seed,
            )

        arrangement.sections.append(section)
        current_bar += template.bars

    return arrangement


def arrangement_summary(arrangement: Arrangement) -> dict:
    """Get a summary of the arrangement."""
    sections_info = []
    for section in arrangement.sections:
        sections_info.append({
            "type": section.template.name,
            "bars": section.template.bars,
            "start_bar": section.start_bar,
            "end_bar": section.end_bar,
            "energy": section.template.energy,
            "tracks": list(section.patterns.keys()),
        })

    return {
        "key": arrangement.config.key,
        "scale": arrangement.config.scale,
        "bpm": arrangement.config.bpm,
        "genre": arrangement.config.genre,
        "total_bars": arrangement.total_bars,
        "duration_s": round(arrangement.duration_s, 1),
        "sections": sections_info,
    }
