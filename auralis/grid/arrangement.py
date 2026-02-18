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
    # Humanization
    humanize_velocity,
    humanize_timing,
    add_ghost_notes,
    add_drum_fill,
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


def _evolve_template(
    template: SectionTemplate,
    occurrence: int,
) -> SectionTemplate:
    """Apply progressive evolution to repeated sections.

    Each occurrence of the same section type gets subtle additions:
      - 2nd time: slightly higher energy, more ghost notes
      - 3rd time: different drum fills, more melodic density
      - 4th+: energy caps, maximum layering
    """
    if occurrence <= 0:
        return template

    # Create a copy with evolved parameters
    evolved = SectionTemplate(
        name=template.name,
        bars=template.bars,
        energy=min(1.0, template.energy + occurrence * 0.05),
        has_drums=template.has_drums,
        has_bass=template.has_bass,
        has_chords=template.has_chords,
        has_melody=template.has_melody,
        drum_style=template.drum_style,
        bass_style=template.bass_style,
        melody_density=min(0.9, template.melody_density + occurrence * 0.08),
    )

    # 3rd+ occurrence: upgrade drum style for variety
    if occurrence >= 2 and template.drum_style == "four_on_floor":
        evolved.bass_style = "syncopated"

    return evolved


# ── Track entry order for staggered intros ──
_ENTRY_ORDER = ["drums", "bass", "chords", "melody"]


def _apply_micro_arrangement(
    section: "SectionInstance",
) -> None:
    """Trim element entry/exit within a section for natural build.

    Modifies section.patterns in-place:
      - intro:     Stagger entry every 2 bars (drums first, melody last)
      - outro:     Stagger exit in reverse (melody drops first)
      - breakdown: Strip drums for first 2 bars, bass for first bar
      - drop/chorus/verse: No trimming (full impact)
    """
    sec_type = section.template.name
    sec_bars = section.template.bars

    if sec_type == "intro" and sec_bars >= 4:
        # Stagger entry: each element starts 2 bars later
        bar_delay = 2
        for idx, track_name in enumerate(_ENTRY_ORDER):
            if track_name not in section.patterns:
                continue
            start_bar = idx * bar_delay
            if start_bar <= 0:
                continue  # First element plays from bar 0
            start_beat = start_bar * 4.0
            pattern = section.patterns[track_name]
            pattern.notes = [n for n in pattern.notes if n.start_beat >= start_beat]

    elif sec_type == "outro" and sec_bars >= 4:
        # Stagger exit: reverse order, each drops 2 bars earlier
        bar_delay = 2
        for idx, track_name in enumerate(reversed(_ENTRY_ORDER)):
            if track_name not in section.patterns:
                continue
            cutoff_bar = sec_bars - (idx * bar_delay)
            if cutoff_bar >= sec_bars:
                continue  # Last element plays the whole section
            cutoff_beat = cutoff_bar * 4.0
            pattern = section.patterns[track_name]
            pattern.notes = [n for n in pattern.notes if n.start_beat < cutoff_beat]

    elif sec_type == "breakdown" and sec_bars >= 4:
        # Remove drums for first 2 bars, bass for first bar
        if "drums" in section.patterns:
            section.patterns["drums"].notes = [
                n for n in section.patterns["drums"].notes if n.start_beat >= 8.0
            ]
        if "bass" in section.patterns:
            section.patterns["bass"].notes = [
                n for n in section.patterns["bass"].notes if n.start_beat >= 4.0
            ]


def generate_arrangement(config: ArrangementConfig) -> Arrangement:
    """Generate a complete song arrangement with section variation.

    Key improvement: sections of the same type (e.g. verse 1 vs verse 2)
    share the same base seed so they use the same motif, but pass an
    occurrence counter for progressive variation (more layers, higher
    energy, different motif arrangements each time).
    """
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

    # ── Occurrence tracking ──
    # Same section type shares base seed → same motif, different variations
    occurrence_count: dict[str, int] = {}
    base_seeds: dict[str, int] = {}

    for i, template in enumerate(templates):
        section_type = template.name
        occurrence = occurrence_count.get(section_type, 0)
        occurrence_count[section_type] = occurrence + 1

        # First occurrence establishes the seed; later ones reuse it
        if occurrence == 0:
            base_seeds[section_type] = i * 100 + 42
        seed = base_seeds[section_type]

        # Evolve template for repeated sections
        active_template = _evolve_template(template, occurrence)

        section = SectionInstance(template=active_template, start_bar=current_bar)

        # Generate track patterns
        if active_template.has_drums:
            section.patterns["drums"] = generate_drum_pattern(
                style=active_template.drum_style,  # type: ignore[arg-type]
                bars=active_template.bars,
                velocity=int(70 + active_template.energy * 57),
                energy=active_template.energy,
            )

        if active_template.has_bass:
            section.patterns["bass"] = generate_bassline(
                root=config.key,
                scale=config.scale,
                octave=2,
                pattern_type=active_template.bass_style,  # type: ignore[arg-type]
                bars=active_template.bars,
                velocity=int(80 + active_template.energy * 47),
                energy=active_template.energy,
            )

        if active_template.has_chords:
            section.patterns["chords"] = generate_chord_progression(
                root=config.key,
                scale=config.scale,
                octave=3,
                bars=min(active_template.bars, 4),
                velocity=int(60 + active_template.energy * 40),
                energy=active_template.energy,
                seed=seed + 200,
                section_type=active_template.name,
            )

        if active_template.has_melody:
            # Map section type to melodic contour
            _contour_map = {
                "intro": "release",
                "verse": "arc",
                "chorus": "climax",
                "drop": "climax",
                "breakdown": "release",
                "bridge": "tension",
                "outro": "release",
            }
            contour = _contour_map.get(active_template.name, "arc")

            section.patterns["melody"] = generate_melody(
                root=config.key,
                scale=config.scale,
                octave=4,
                bars=active_template.bars,
                density=active_template.melody_density,
                velocity=int(70 + active_template.energy * 50),
                seed=seed,  # Shared seed → same motif
                energy=active_template.energy,
                contour=contour,
                section_type=active_template.name,
                occurrence=occurrence,  # Progressive variation
                chord_progression=section.patterns.get("chords"),
            )

        # ── Micro-arrangement: stagger element entry/exit ──
        _apply_micro_arrangement(section)

        # ── Humanize all patterns ──
        for track_name, pattern in section.patterns.items():
            section.patterns[track_name] = humanize_velocity(
                pattern, amount=12, seed=seed + hash(track_name) % 1000,
            )
            swing = 0.01 + active_template.energy * 0.03
            section.patterns[track_name] = humanize_timing(
                section.patterns[track_name],
                swing=swing,
                seed=seed + hash(track_name) % 1000 + 1,
            )

        # Drum-specific humanization
        if "drums" in section.patterns and active_template.energy > 0.4:
            ghost_prob = 0.1 + active_template.energy * 0.25
            # More ghosts on repeated sections
            ghost_prob = min(0.5, ghost_prob + occurrence * 0.05)
            section.patterns["drums"] = add_ghost_notes(
                section.patterns["drums"],
                bars=active_template.bars,
                probability=ghost_prob,
                velocity=int(25 + active_template.energy * 20),
                seed=seed + 50,
            )
            # Alternate fill types on repeats
            fill_type = "snare_roll" if active_template.energy < 0.7 else "buildup"
            if occurrence >= 1 and fill_type == "snare_roll":
                fill_type = "buildup"  # Upgrade fills on repeat
            section.patterns["drums"] = add_drum_fill(
                section.patterns["drums"],
                every_n_bars=4,
                total_bars=active_template.bars,
                fill_type=fill_type,
                energy=active_template.energy,
                seed=seed + 99,
            )

        arrangement.sections.append(section)
        current_bar += active_template.bars

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
