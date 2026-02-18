"""AURALIS MIDI Engine — Read, write, generate, quantize MIDI data.

Uses mido for MIDI I/O. Supports pattern generation, chord progressions,
drum patterns, and quantization.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mido

# ── Music Theory Constants ───────────────────────────────

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic_major": [0, 2, 4, 7, 9],
    "pentatonic_minor": [0, 3, 5, 7, 10],
    "blues": [0, 3, 5, 6, 7, 10],
    "chromatic": list(range(12)),
}

CHORD_TYPES: dict[str, list[int]] = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "9": [0, 4, 7, 10, 14],
}


# ── Data Types ───────────────────────────────────────────


@dataclass
class Note:
    """A single MIDI note event."""

    pitch: int  # MIDI note number (0-127)
    start_beat: float  # Start time in beats
    duration_beats: float  # Duration in beats
    velocity: int = 100  # 0-127

    @property
    def name(self) -> str:
        """Get note name like 'C4'."""
        octave = self.pitch // 12 - 1
        return f"{NOTE_NAMES[self.pitch % 12]}{octave}"


@dataclass
class Pattern:
    """A collection of notes forming a musical pattern."""

    name: str
    notes: list[Note] = field(default_factory=list)
    length_beats: float = 4.0  # Pattern length


@dataclass
class DrumHit:
    """A drum hit event (General MIDI drum map)."""

    drum: int  # GM drum note (36=kick, 38=snare, 42=hihat, etc.)
    beat: float
    velocity: int = 100


# ── Helper Functions ─────────────────────────────────────


def note_name_to_midi(name: str) -> int:
    """Convert note name like 'C4' to MIDI number."""
    note_part = name[:-1].upper()
    octave = int(name[-1])
    idx = NOTE_NAMES.index(note_part)
    return (octave + 1) * 12 + idx


def get_scale_notes(root: str, scale: str = "minor", octave: int = 4) -> list[int]:
    """Get MIDI notes for a scale."""
    root_midi = note_name_to_midi(f"{root}{octave}")
    intervals = SCALES.get(scale, SCALES["minor"])
    return [root_midi + i for i in intervals]


def get_chord_notes(root: int, chord_type: str = "minor") -> list[int]:
    """Get MIDI notes for a chord."""
    intervals = CHORD_TYPES.get(chord_type, CHORD_TYPES["minor"])
    return [root + i for i in intervals]


# ── Pattern Generators ───────────────────────────────────


def generate_chord_progression(
    root: str = "C",
    scale: str = "minor",
    octave: int = 3,
    progression: list[int] | None = None,
    bars: int = 4,
    velocity: int = 80,
) -> Pattern:
    """Generate a chord progression pattern.

    Default progression: i-iv-v-i (minor) or I-IV-V-I (major).
    """
    if progression is None:
        if scale in ("minor", "dorian", "phrygian", "harmonic_minor"):
            progression = [0, 3, 4, 0]  # i-iv-v-i
        else:
            progression = [0, 3, 4, 0]  # I-IV-V-I

    scale_notes = get_scale_notes(root, scale, octave)
    notes: list[Note] = []

    for bar, degree in enumerate(progression[:bars]):
        root_note = scale_notes[degree % len(scale_notes)]
        # Determine chord quality from scale
        chord = get_chord_notes(root_note, "minor" if scale in ("minor", "dorian") else "major")
        for note_pitch in chord:
            notes.append(Note(
                pitch=note_pitch,
                start_beat=bar * 4.0,
                duration_beats=3.8,
                velocity=velocity,
            ))

    return Pattern(
        name=f"Chord Progression ({root} {scale})",
        notes=notes,
        length_beats=bars * 4.0,
    )


def generate_bassline(
    root: str = "C",
    scale: str = "minor",
    octave: int = 2,
    pattern_type: Literal["simple", "walking", "syncopated"] = "simple",
    bars: int = 4,
    velocity: int = 100,
    energy: float = 0.5,  # 0-1 narrative energy
) -> Pattern:
    """Generate a bassline pattern."""
    scale_notes = get_scale_notes(root, scale, octave)
    notes: list[Note] = []
    rng = random.Random(42 + int(energy * 100))

    # Narrative Override: Force simple patterns for low energy
    if energy < 0.4:
        pattern_type = "simple"
    elif energy > 0.8:
        pattern_type = "syncopated"

    for bar in range(bars):
        root_note = scale_notes[0]

        if pattern_type == "simple":
            # Low Energy: Sustained whole/half notes
            if energy < 0.3:
                notes.append(Note(
                    pitch=root_note, start_beat=bar * 4.0,
                    duration_beats=4.0, velocity=velocity,
                ))
            else:
                # Quarter notes on root
                for beat in range(4):
                    notes.append(Note(
                        pitch=root_note, start_beat=bar * 4.0 + beat,
                        duration_beats=0.8, velocity=velocity,
                    ))
        elif pattern_type == "walking":
            # Walking bass — different scale notes per beat
            for beat in range(4):
                note = rng.choice(scale_notes)
                notes.append(Note(
                    pitch=note, start_beat=bar * 4.0 + beat,
                    duration_beats=0.9, velocity=velocity - rng.randint(0, 20),
                ))
        elif pattern_type == "syncopated":
            # Syncopated pattern (16th note drive)
            density = 6 if energy > 0.9 else 4
            hits = [0.0, 0.75, 1.5, 2.5, 3.0, 3.5]
            if energy > 0.9:
                hits = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # Rolling bass
            
            for hit in hits:
                note = rng.choice(scale_notes[:3])
                notes.append(Note(
                    pitch=note, start_beat=bar * 4.0 + hit,
                    duration_beats=0.5, velocity=velocity - rng.randint(0, 15),
                ))

    return Pattern(
        name=f"Bassline ({root} {scale} {pattern_type})",
        notes=notes,
        length_beats=bars * 4.0,
    )


def generate_melody(
    root: str = "C",
    scale: str = "minor",
    octave: int = 4,
    bars: int = 4,
    density: float = 0.6,
    velocity: int = 90,
    seed: int = 42,
) -> Pattern:
    """Generate a melodic pattern using scale-based random walk."""
    scale_notes = get_scale_notes(root, scale, octave)
    # Extend to octave above
    scale_notes += [n + 12 for n in scale_notes]
    notes: list[Note] = []
    rng = random.Random(seed)

    position = len(scale_notes) // 2  # Start in middle
    beat = 0.0
    total_beats = bars * 4.0

    while beat < total_beats:
        if rng.random() < density:
            # Step: move up/down in scale
            step = rng.choice([-2, -1, -1, 0, 1, 1, 2])
            position = max(0, min(len(scale_notes) - 1, position + step))
            dur = rng.choice([0.25, 0.5, 0.5, 1.0, 1.0, 2.0])
            vel = velocity + rng.randint(-15, 10)

            notes.append(Note(
                pitch=scale_notes[position],
                start_beat=beat,
                duration_beats=min(dur, total_beats - beat),
                velocity=max(40, min(127, vel)),
            ))
            beat += dur
        else:
            beat += rng.choice([0.25, 0.5])

    return Pattern(
        name=f"Melody ({root} {scale})",
        notes=notes,
        length_beats=total_beats,
    )


def generate_drum_pattern(
    style: Literal["four_on_floor", "breakbeat", "trap", "minimal"] = "four_on_floor",
    bars: int = 4,
    velocity: int = 100,
    energy: float = 0.5,  # 0-1 narrative intensity
) -> Pattern:
    """Generate a drum pattern using GM drum map."""
    KICK = 36
    SNARE = 38
    HIHAT_C = 42  # Closed
    HIHAT_O = 46  # Open
    CLAP = 39
    RIDE = 51

    notes: list[Note] = []

    for bar in range(bars):
        offset = bar * 4.0

        # narrative intensity control
        has_kick = True
        has_hats = energy > 0.3
        has_snare = energy > 0.4
        has_rides = energy > 0.8
        has_ghosts = energy > 0.6

        if style == "four_on_floor":
            # Kick on every beat
            if has_kick:
                for beat in range(4):
                    notes.append(Note(KICK, offset + beat, 0.1, velocity))
            
            # Snare/clap on 2 and 4
            if has_snare:
                notes.append(Note(CLAP, offset + 1, 0.1, velocity - 10))
                notes.append(Note(CLAP, offset + 3, 0.1, velocity - 10))
            
            # Hi-hats on 8ths
            if has_hats:
                for i in range(8):
                    vel = velocity - 20 if i % 2 == 0 else velocity - 35
                    notes.append(Note(HIHAT_C, offset + i * 0.5, 0.05, vel))
            
            # Rides for high energy
            if has_rides:
                for i in range(8):
                    notes.append(Note(RIDE, offset + i * 0.5, 0.05, velocity - 40))

        elif style == "breakbeat":
            if has_kick:
                notes.append(Note(KICK, offset + 0, 0.1, velocity))
                notes.append(Note(KICK, offset + 1.5, 0.1, velocity - 10))
                notes.append(Note(KICK, offset + 2.75, 0.1, velocity - 5))
            if has_snare:
                notes.append(Note(SNARE, offset + 1, 0.1, velocity))
                notes.append(Note(SNARE, offset + 3, 0.1, velocity))
            if has_hats:
                for i in range(8):
                    notes.append(Note(HIHAT_C, offset + i * 0.5, 0.05, velocity - 25))

        elif style == "trap":
            if has_kick:
                notes.append(Note(KICK, offset + 0, 0.2, velocity))
                notes.append(Note(KICK, offset + 2.25, 0.2, velocity - 5))
            if has_snare:
                notes.append(Note(SNARE, offset + 1, 0.1, velocity))
                notes.append(Note(SNARE, offset + 3, 0.1, velocity))
            if has_hats:
                # Rapid hi-hats
                sub = 4 if energy > 0.7 else 2
                for i in range(4 * sub):
                    vel = velocity - 30 + random.randint(0, 15)
                    notes.append(Note(HIHAT_C, offset + i * (1/sub), 0.05, vel))

        elif style == "minimal":
            if has_kick:
                notes.append(Note(KICK, offset + 0, 0.1, velocity))
                notes.append(Note(KICK, offset + 2, 0.1, velocity - 10))
            if has_hats:
                notes.append(Note(HIHAT_C, offset + 1, 0.05, velocity - 30))
                notes.append(Note(HIHAT_O, offset + 3, 0.1, velocity - 25))

    return Pattern(
        name=f"Drums ({style})",
        notes=notes,
        length_beats=bars * 4.0,
    )


# ── Quantization ─────────────────────────────────────────


def quantize_pattern(
    pattern: Pattern,
    grid: float = 0.25,  # 1/16th note
    strength: float = 1.0,  # 0-1
) -> Pattern:
    """Quantize pattern notes to a grid."""
    quantized_notes = []
    for note in pattern.notes:
        nearest = round(note.start_beat / grid) * grid
        new_start = note.start_beat + (nearest - note.start_beat) * strength
        quantized_notes.append(Note(
            pitch=note.pitch,
            start_beat=new_start,
            duration_beats=note.duration_beats,
            velocity=note.velocity,
        ))
    return Pattern(name=pattern.name, notes=quantized_notes, length_beats=pattern.length_beats)


# ── MIDI I/O ─────────────────────────────────────────────


def pattern_to_midi(
    pattern: Pattern,
    bpm: float = 120.0,
    channel: int = 0,
) -> mido.MidiFile:
    """Convert a Pattern to a MIDI file."""
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    ticks_per_beat = mid.ticks_per_beat

    # Sort notes by start time
    sorted_notes = sorted(pattern.notes, key=lambda n: n.start_beat)

    # Build events
    events: list[tuple[int, mido.Message]] = []
    for note in sorted_notes:
        on_tick = int(note.start_beat * ticks_per_beat)
        off_tick = int((note.start_beat + note.duration_beats) * ticks_per_beat)
        events.append((on_tick, mido.Message("note_on", note=note.pitch, velocity=note.velocity, channel=channel)))
        events.append((off_tick, mido.Message("note_off", note=note.pitch, velocity=0, channel=channel)))

    events.sort(key=lambda e: e[0])

    # Convert to delta times
    last_tick = 0
    for tick, msg in events:
        delta = tick - last_tick
        msg.time = max(0, delta)
        track.append(msg)
        last_tick = tick

    return mid


def save_midi(pattern: Pattern, path: str | Path, bpm: float = 120.0) -> Path:
    """Save Pattern as MIDI file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mid = pattern_to_midi(pattern, bpm)
    mid.save(str(p))
    return p


def load_midi(path: str | Path) -> Pattern:
    """Load a MIDI file as a Pattern."""
    mid = mido.MidiFile(str(path))
    notes: list[Note] = []
    ticks_per_beat = mid.ticks_per_beat

    for track in mid.tracks:
        current_tick = 0
        active: dict[int, tuple[int, int]] = {}  # pitch -> (start_tick, velocity)

        for msg in track:
            current_tick += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = (current_tick, msg.velocity)
            elif msg.type in ("note_off", "note_on") and (msg.type == "note_off" or msg.velocity == 0):
                if msg.note in active:
                    start_tick, vel = active.pop(msg.note)
                    notes.append(Note(
                        pitch=msg.note,
                        start_beat=start_tick / ticks_per_beat,
                        duration_beats=(current_tick - start_tick) / ticks_per_beat,
                        velocity=vel,
                    ))

    return Pattern(name=Path(path).stem, notes=notes, length_beats=max((n.start_beat + n.duration_beats for n in notes), default=4.0))


def pattern_to_note_events(pattern: Pattern, bpm: float = 120.0) -> list[dict[str, float]]:
    """Convert Pattern to note events for synth rendering."""
    beat_duration = 60.0 / bpm
    return [
        {
            "note": float(n.pitch),
            "start": n.start_beat * beat_duration,
            "duration": n.duration_beats * beat_duration,
            "velocity": n.velocity / 127.0,
        }
        for n in pattern.notes
    ]
