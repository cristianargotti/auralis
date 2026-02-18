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
    energy: float = 0.5,
    seed: int = 42,
) -> Pattern:
    """Generate a chord progression with energy-aware complexity.

    Low energy  → simple triads, whole notes, gentle dynamics
    High energy → 7th/9th chords, rhythmic variation, wide dynamics
    """
    rng = random.Random(seed)
    is_minor = scale in ("minor", "dorian", "phrygian", "harmonic_minor")

    # Energy-tiered progressions (the AI already suggests structure,
    # but this gives musical defaults when the AI doesn't specify)
    if progression is None:
        if is_minor:
            pool = [
                [0, 3, 4, 0],      # i-iv-v-i (classic)
                [0, 5, 3, 4],      # i-vi-iv-v (emotional)
                [0, 2, 5, 4],      # i-iii-vi-v (cinematic)
                [0, 5, 2, 4],      # i-vi-iii-v (melancholic)
                [0, 3, 6, 4],      # i-iv-VII-v (dark)
            ]
        else:
            pool = [
                [0, 3, 4, 0],      # I-IV-V-I (classic)
                [0, 4, 5, 3],      # I-V-vi-IV (pop)
                [0, 5, 3, 4],      # I-vi-IV-V (50s)
                [0, 2, 5, 4],      # I-iii-vi-V (jazzy)
                [0, 3, 1, 4],      # I-IV-ii-V (soul)
            ]
        progression = rng.choice(pool)

    # Chord quality gets richer at higher energy
    def _chord_type_for_degree(degree: int) -> str:
        if energy > 0.8:
            return rng.choice(["min7", "7", "9"]) if is_minor else rng.choice(["maj7", "7", "9"])
        elif energy > 0.5:
            return rng.choice(["minor", "min7"]) if is_minor else rng.choice(["major", "maj7"])
        else:
            return "minor" if is_minor else "major"

    scale_notes = get_scale_notes(root, scale, octave)
    notes: list[Note] = []

    for bar in range(bars):
        degree = progression[bar % len(progression)]
        root_note = scale_notes[degree % len(scale_notes)]
        chord_type = _chord_type_for_degree(degree)
        chord = get_chord_notes(root_note, chord_type)

        # Velocity follows energy arc
        bar_energy = energy + rng.uniform(-0.1, 0.1)
        bar_vel = int(velocity * (0.7 + 0.3 * bar_energy))

        if energy > 0.7:
            # High energy: rhythmic chord stabs
            positions = [0.0, 1.0, 2.0, 3.0] if energy > 0.9 else [0.0, 2.0]
            for pos in positions:
                stab_vel = bar_vel + rng.randint(-8, 8)
                for note_pitch in chord:
                    notes.append(Note(
                        pitch=note_pitch,
                        start_beat=bar * 4.0 + pos,
                        duration_beats=0.8 if energy > 0.9 else 1.8,
                        velocity=max(40, min(127, stab_vel)),
                    ))
        else:
            # Low/mid energy: sustained pads
            for note_pitch in chord:
                notes.append(Note(
                    pitch=note_pitch,
                    start_beat=bar * 4.0,
                    duration_beats=3.8,
                    velocity=max(40, min(127, bar_vel)),
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
    energy: float = 0.5,
    contour: str = "arc",
) -> Pattern:
    """Generate a melodic pattern with narrative contour.

    Contours:
        arc:      Rises to 2/3, then descends (natural phrasing)
        climax:   Builds to highest register at the end (pre-drop)
        tension:  Gradually rises, never resolves (build-up)
        release:  Starts high, resolves downward (post-drop calm)
        flat:     No contour bias (ambient, textural)

    Energy controls range, leap probability, and density.
    """
    scale_notes = get_scale_notes(root, scale, octave)
    # Range expands with energy: low=1 octave, high=2+ octaves
    n_octaves = 1 if energy < 0.3 else (2 if energy < 0.7 else 3)
    extended = scale_notes.copy()
    for o in range(1, n_octaves):
        extended += [n + 12 * o for n in scale_notes]

    notes: list[Note] = []
    rng = random.Random(seed)

    position = len(extended) // 2  # Start in middle
    beat = 0.0
    total_beats = bars * 4.0

    while beat < total_beats:
        progress = beat / total_beats  # 0.0 → 1.0

        # Contour shapes the target register
        if contour == "arc":
            # Peak at 2/3, then descend
            target = 1.0 - abs(progress - 0.66) * 2.0
        elif contour == "climax":
            # Exponential rise to the top
            target = progress ** 1.5
        elif contour == "tension":
            # Linear rise, no resolution
            target = progress * 0.9
        elif contour == "release":
            # Start high, resolve down
            target = 1.0 - progress ** 0.7
        else:  # flat
            target = 0.5

        # Target position in the extended scale
        target_pos = int(target * (len(extended) - 1))

        # Bias the random walk toward the target
        if rng.random() < density:
            # Step toward target with some randomness
            toward_target = 1 if target_pos > position else (-1 if target_pos < position else 0)
            # Higher energy = bigger leaps
            leap_chance = 0.1 + energy * 0.3  # 0.1-0.4
            if rng.random() < leap_chance:
                step = toward_target * rng.choice([2, 3, 4])
            else:
                step = rng.choice([-1, 0, toward_target, toward_target, 1])

            position = max(0, min(len(extended) - 1, position + step))

            # Duration: shorter at high energy (busier melodies)
            if energy > 0.7:
                dur = rng.choice([0.25, 0.25, 0.5, 0.5, 1.0])
            elif energy > 0.4:
                dur = rng.choice([0.25, 0.5, 0.5, 1.0, 1.0, 2.0])
            else:
                dur = rng.choice([0.5, 1.0, 1.0, 2.0, 2.0, 4.0])

            # Velocity follows contour (louder at peak)
            contour_vel = int(velocity * (0.7 + 0.3 * target))
            vel = contour_vel + rng.randint(-12, 8)

            notes.append(Note(
                pitch=extended[position],
                start_beat=beat,
                duration_beats=min(dur, total_beats - beat),
                velocity=max(40, min(127, vel)),
            ))
            beat += dur
        else:
            # Rest — duration also energy-dependent
            beat += rng.choice([0.25, 0.5]) if energy > 0.5 else rng.choice([0.5, 1.0])

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


# ── Humanization ─────────────────────────────────────────
# These break the mechanical grid and make patterns feel alive.


def humanize_velocity(
    pattern: Pattern,
    amount: int = 15,
    seed: int = 42,
) -> Pattern:
    """Add random velocity variation to every note.

    A real musician never hits at exactly the same velocity twice.
    amount=10 is subtle, 20 is noticeable, 30+ is sloppy.
    """
    rng = random.Random(seed)
    new_notes = []
    for note in pattern.notes:
        delta = rng.randint(-amount, amount)
        new_vel = max(20, min(127, note.velocity + delta))
        new_notes.append(Note(
            pitch=note.pitch,
            start_beat=note.start_beat,
            duration_beats=note.duration_beats,
            velocity=new_vel,
        ))
    return Pattern(name=pattern.name, notes=new_notes, length_beats=pattern.length_beats)


def humanize_timing(
    pattern: Pattern,
    swing: float = 0.02,
    seed: int = 42,
) -> Pattern:
    """Apply micro-timing offsets (swing) to off-beat notes.

    Breaks the metronomic grid. A real drummer is slightly ahead or behind.
    swing=0.01 is tight, 0.03 is groovy, 0.05+ is loose/drunken.
    """
    rng = random.Random(seed)
    new_notes = []
    for note in pattern.notes:
        # Only humanize off-beat notes (not downbeats)
        is_downbeat = abs(note.start_beat - round(note.start_beat)) < 0.01
        if is_downbeat:
            offset = rng.uniform(-swing * 0.3, swing * 0.3)  # Subtle on downbeats
        else:
            offset = rng.uniform(-swing, swing)  # More on offbeats
        new_start = max(0.0, note.start_beat + offset)
        new_notes.append(Note(
            pitch=note.pitch,
            start_beat=new_start,
            duration_beats=note.duration_beats,
            velocity=note.velocity,
        ))
    return Pattern(name=pattern.name, notes=new_notes, length_beats=pattern.length_beats)


def add_ghost_notes(
    pattern: Pattern,
    bars: int = 4,
    probability: float = 0.3,
    velocity: int = 35,
    seed: int = 42,
) -> Pattern:
    """Insert subtle low-velocity snare ghost notes between main beats.

    Ghost notes are the secret sauce of groove. They fill the space
    between kick and snare with barely-audible texture.
    probability=0.3 means 30% chance per 16th note slot.
    """
    SNARE = 38
    rng = random.Random(seed)
    new_notes = list(pattern.notes)

    # Find existing hit positions to avoid doubling
    existing_positions = {round(n.start_beat * 4) / 4 for n in pattern.notes if n.pitch == SNARE}

    for bar in range(bars):
        for sixteenth in range(16):
            beat_pos = bar * 4.0 + sixteenth * 0.25
            # Skip if there's already a hit here
            if round(beat_pos * 4) / 4 in existing_positions:
                continue
            # Skip downbeats (those should be intentional, not ghosts)
            if sixteenth % 4 == 0:
                continue
            if rng.random() < probability:
                vel = velocity + rng.randint(-10, 10)
                new_notes.append(Note(
                    pitch=SNARE,
                    start_beat=beat_pos,
                    duration_beats=0.05,
                    velocity=max(15, min(60, vel)),
                ))

    return Pattern(name=pattern.name, notes=new_notes, length_beats=pattern.length_beats)


def add_drum_fill(
    pattern: Pattern,
    every_n_bars: int = 4,
    total_bars: int = 4,
    fill_type: str = "snare_roll",
    energy: float = 0.5,
    seed: int = 42,
) -> Pattern:
    """Add a drum fill at the end of every N bars.

    Fills mark phrase boundaries — a human drummer ALWAYS marks them.
    fill_type: 'snare_roll' (classic) or 'buildup' (ascending).
    energy controls fill intensity: 0.3=subtle, 0.8=aggressive.
    """
    SNARE = 38
    TOM_HIGH = 50
    TOM_MID = 47
    TOM_LOW = 45
    CRASH = 49

    rng = random.Random(seed)
    new_notes = list(pattern.notes)

    for bar in range(total_bars):
        # Only fill at phrase boundaries
        if (bar + 1) % every_n_bars != 0:
            continue

        fill_start = bar * 4.0 + 3.0  # Last beat of the bar
        n_hits = 4 if energy < 0.5 else 8  # 16ths or 32nds

        if fill_type == "snare_roll":
            for i in range(n_hits):
                beat_pos = fill_start + i * (1.0 / n_hits)
                vel = int(70 + energy * 57 - rng.randint(0, 15))
                # Crescendo: velocity rises toward the end
                vel = int(vel * (0.6 + 0.4 * i / n_hits))
                new_notes.append(Note(
                    pitch=SNARE,
                    start_beat=beat_pos,
                    duration_beats=0.05,
                    velocity=max(40, min(127, vel)),
                ))
        elif fill_type == "buildup":
            # Tom cascade: high -> mid -> low -> crash
            toms = [TOM_HIGH, TOM_HIGH, TOM_MID, TOM_MID, TOM_LOW, TOM_LOW, SNARE, SNARE]
            for i in range(min(n_hits, len(toms))):
                beat_pos = fill_start + i * (1.0 / n_hits)
                vel = int(80 + energy * 47)
                new_notes.append(Note(
                    pitch=toms[i],
                    start_beat=beat_pos,
                    duration_beats=0.05,
                    velocity=max(50, min(127, vel)),
                ))

        # Crash on the 1 of the next bar (if not at end)
        if bar + 1 < total_bars and energy > 0.4:
            new_notes.append(Note(
                pitch=CRASH,
                start_beat=(bar + 1) * 4.0,
                duration_beats=0.5,
                velocity=int(90 + energy * 37),
            ))

    return Pattern(name=pattern.name, notes=new_notes, length_beats=pattern.length_beats)
