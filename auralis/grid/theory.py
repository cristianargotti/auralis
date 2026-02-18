"""AURALIS Music Theory Engine — Voice leading, inversions, cadences, progressions.

Provides musically intelligent chord and scale operations that turn
random note selection into real harmonic language.

Key features:
  - Voice leading: minimize movement between chords (common tones hold)
  - Inversions: root, 1st, 2nd, 3rd position
  - Cadences: authentic, plagal, deceptive, half
  - Section-aware progressions: verse/chorus/bridge/drop have distinct harmonic character
  - Chord-tone classification: stable vs passing tones for melody targeting
"""

from __future__ import annotations

from dataclasses import dataclass

# ── Scale + Chord Constants ──────────────────────────────

# Intervals from root for each scale type
SCALE_INTERVALS: dict[str, list[int]] = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "pentatonic_minor": [0, 3, 5, 7, 10],
}

# Diatonic triads built on each scale degree (for minor scale)
# degree → (semitone offset from root, chord quality)
MINOR_DIATONIC: list[tuple[int, str]] = [
    (0, "minor"),   # i
    (2, "dim"),     # ii°
    (3, "major"),   # III
    (5, "minor"),   # iv
    (7, "minor"),   # v   (natural minor)
    (8, "major"),   # VI
    (10, "major"),  # VII
]

MAJOR_DIATONIC: list[tuple[int, str]] = [
    (0, "major"),   # I
    (2, "minor"),   # ii
    (4, "minor"),   # iii
    (5, "major"),   # IV
    (7, "major"),   # V
    (9, "minor"),   # vi
    (11, "dim"),    # vii°
]

CHORD_INTERVALS: dict[str, list[int]] = {
    "major": [0, 4, 7],
    "minor": [0, 3, 7],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "sus2": [0, 2, 7],
    "sus4": [0, 5, 7],
    "9": [0, 4, 7, 10, 14],
}


# ── Data Types ───────────────────────────────────────────


@dataclass
class ChordVoicing:
    """A specific voicing of a chord."""

    root: int           # MIDI note of the root
    quality: str        # "major", "minor", "dim", etc.
    notes: list[int]    # Actual MIDI notes in this voicing
    inversion: int      # 0=root, 1=first, 2=second
    degree: int         # Scale degree (0-6) — which chord in the key

    @property
    def bass(self) -> int:
        return min(self.notes) if self.notes else self.root


# ── Inversions ───────────────────────────────────────────


def get_inversions(root: int, quality: str = "minor") -> list[list[int]]:
    """Return all inversions of a chord.

    Args:
        root: MIDI note of the root.
        quality: Chord quality (major, minor, dim, etc.)

    Returns:
        List of voicings: [root_position, 1st_inversion, 2nd_inversion, ...]
    """
    intervals = CHORD_INTERVALS.get(quality, CHORD_INTERVALS["minor"])
    root_pos = [root + i for i in intervals]

    inversions = [root_pos]
    current = list(root_pos)
    for _ in range(len(intervals) - 1):
        # Move lowest note up an octave
        lowest = current[0]
        current = current[1:] + [lowest + 12]
        inversions.append(list(current))

    return inversions


# ── Voice Leading ────────────────────────────────────────


def voice_lead(
    chord_a: list[int],
    chord_b_root: int,
    chord_b_quality: str = "minor",
) -> list[int]:
    """Find the voicing of chord_b that minimizes movement from chord_a.

    Rules (from classical harmony):
      1. Common tones between chords stay in the same voice
      2. Other voices move by the smallest possible interval
      3. Prefer contrary motion to bass
      4. No voice crosses another voice

    Args:
        chord_a: Current chord voicing (MIDI notes, sorted low→high).
        chord_b_root: Root MIDI note of the target chord.
        chord_b_quality: Quality of the target chord.

    Returns:
        Best voicing of chord_b (same number of voices as chord_a).
    """
    if not chord_a:
        intervals = CHORD_INTERVALS.get(chord_b_quality, [0, 3, 7])
        return [chord_b_root + i for i in intervals]

    # Generate all inversions of chord_b
    inversions = get_inversions(chord_b_root, chord_b_quality)

    # Also try octave-shifted versions (±12) for better voice leading
    candidates: list[list[int]] = []
    for inv in inversions:
        candidates.append(inv)
        candidates.append([n - 12 for n in inv])
        candidates.append([n + 12 for n in inv])

    # Normalize to same voice count as chord_a
    n_voices = len(chord_a)
    normalized: list[list[int]] = []
    for cand in candidates:
        if len(cand) >= n_voices:
            normalized.append(sorted(cand[:n_voices]))
        else:
            # Pad by doubling notes in different octaves
            padded = list(cand)
            while len(padded) < n_voices:
                padded.append(cand[0] + 12 * (len(padded) // len(cand) + 1))
            normalized.append(sorted(padded[:n_voices]))

    # Score each candidate: lower total movement = better
    sorted_a = sorted(chord_a)
    best_score = float("inf")
    best_voicing = normalized[0] if normalized else sorted_a

    for voicing in normalized:
        # Total semitone movement across all voices
        movement = sum(abs(b - a) for a, b in zip(sorted_a, voicing))

        # Penalty: voices should stay in a reasonable range (MIDI 36-84)
        range_penalty = sum(
            5 for n in voicing if n < 36 or n > 84
        )

        # Penalty: voice crossing (higher voice goes below lower voice)
        crossing_penalty = 0
        for i in range(len(voicing) - 1):
            if voicing[i] > voicing[i + 1]:
                crossing_penalty += 10

        total = movement + range_penalty + crossing_penalty
        if total < best_score:
            best_score = total
            best_voicing = voicing

    return best_voicing


# ── Cadences ─────────────────────────────────────────────


def resolve_cadence(
    root_midi: int,
    scale: str = "minor",
    cadence_type: str = "authentic",
) -> list[list[int]]:
    """Generate a cadence chord pair.

    Args:
        root_midi: MIDI note of the key root (e.g. 60 for C4).
        scale: Scale type.
        cadence_type: authentic (V→I), plagal (IV→I), deceptive (V→vi), half (→V).

    Returns:
        List of two chords [penultimate, final], each as MIDI note list.
    """
    diatonic = MAJOR_DIATONIC if scale == "major" else MINOR_DIATONIC

    def _build(degree_idx: int) -> list[int]:
        offset, quality = diatonic[degree_idx % len(diatonic)]
        intervals = CHORD_INTERVALS.get(quality, [0, 3, 7])
        return [root_midi + offset + i for i in intervals]

    if cadence_type == "authentic":
        # V → I (strongest resolution)
        return [_build(4), _build(0)]
    elif cadence_type == "plagal":
        # IV → I ("Amen" cadence)
        return [_build(3), _build(0)]
    elif cadence_type == "deceptive":
        # V → vi (surprise — expected I, got vi)
        return [_build(4), _build(5)]
    elif cadence_type == "half":
        # I → V (suspense — doesn't resolve)
        return [_build(0), _build(4)]
    else:
        return [_build(4), _build(0)]


# ── Chord-Tone Classification ───────────────────────────


def classify_scale_tones(
    root: str,
    scale: str = "minor",
    octave: int = 4,
) -> dict[str, list[int]]:
    """Classify scale tones as stable, tension, or passing.

    Used by melody engine to target chord tones on strong beats
    and passing tones on weak beats.

    Returns:
        {"stable": [...], "tension": [...], "passing": [...]}
        where each list contains MIDI note numbers.
    """
    from auralis.grid.midi import get_scale_notes

    notes = get_scale_notes(root, scale, octave)

    if len(notes) < 5:
        return {"stable": notes, "tension": [], "passing": []}

    # In any diatonic scale:
    # Stable: 1 (root), 3 (mediant), 5 (dominant) — chord tones
    # Tension: 7 (leading), 4 (subdominant) — want to resolve
    # Passing: 2 (supertonic), 6 (submediant) — decorative
    return {
        "stable": [notes[0], notes[2], notes[4]],    # 1, 3, 5
        "tension": [notes[6], notes[3]],               # 7, 4
        "passing": [notes[1], notes[5]],               # 2, 6
    }


# ── Section-Aware Progressions ──────────────────────────

# Common progressions by section type
# Each is a list of scale degree indices (0-based)
SECTION_PROGRESSIONS: dict[str, list[list[int]]] = {
    "verse": [
        [0, 3, 4, 0],     # i → iv → v → i (stable, cyclical)
        [0, 5, 3, 4],     # i → VI → iv → v (wandering)
        [0, 6, 5, 4],     # i → VII → VI → v (descending)
    ],
    "chorus": [
        [0, 5, 2, 6],     # i → VI → III → VII (anthemic, ascending)
        [0, 2, 5, 6],     # i → III → VI → VII (bright, uplifting)
        [5, 6, 0, 4],     # VI → VII → i → v (powerful)
    ],
    "bridge": [
        [3, 4, 5, 3],     # iv → v → VI → iv (searching, unresolved)
        [5, 3, 0, 4],     # VI → iv → i → v (surprise)
        [2, 5, 3, 6],     # III → VI → iv → VII (wandering far)
    ],
    "drop": [
        [0, 0, 0, 0],     # i → i → i → i (power, minimal)
        [0, 6, 0, 6],     # i → VII → i → VII (pulsing)
        [0, 5, 0, 5],     # i → VI → i → VI (anthemic oscillation)
    ],
    "breakdown": [
        [3, 0, 5, 0],     # iv → i → VI → i (gentle, resolving)
        [5, 3, 0, 0],     # VI → iv → i → i (settling)
    ],
    "intro": [
        [0, 0, 3, 4],     # i → i → iv → v (establishing key)
        [0, 5, 0, 5],     # i → VI → i → VI (simple)
    ],
    "outro": [
        [3, 0, 3, 0],     # iv → i → iv → i (plagal loop — closure)
        [5, 3, 0, 0],     # VI → iv → i → i (resolving home)
    ],
}


def suggest_progression(
    root: str,
    scale: str = "minor",
    octave: int = 3,
    bars: int = 4,
    energy: float = 0.5,
    section_type: str = "verse",
    seed: int = 42,
) -> list[ChordVoicing]:
    """Generate a harmonically intelligent chord progression.

    Uses section-appropriate progressions with voice leading
    between successive chords.

    Args:
        root: Key root (e.g. "C").
        scale: Scale type.
        octave: Base octave.
        bars: Number of bars.
        energy: 0-1 energy level (affects complexity).
        section_type: Section context for progression choice.
        seed: Random seed for reproducibility.

    Returns:
        List of ChordVoicing objects with voice-led voicings.
    """
    import random as rng_mod
    from auralis.grid.midi import note_name_to_midi

    rng = rng_mod.Random(seed)

    root_midi = note_name_to_midi(f"{root}{octave}")
    diatonic = MAJOR_DIATONIC if scale == "major" else MINOR_DIATONIC

    # Pick a progression template for this section type
    templates = SECTION_PROGRESSIONS.get(
        section_type,
        SECTION_PROGRESSIONS["verse"],
    )
    prog_degrees = rng.choice(templates)

    # Extend or truncate to match bar count
    full_degrees: list[int] = []
    while len(full_degrees) < bars:
        full_degrees.extend(prog_degrees)
    full_degrees = full_degrees[:bars]

    # Build voicings with voice leading
    voicings: list[ChordVoicing] = []
    prev_notes: list[int] = []

    for i, degree in enumerate(full_degrees):
        offset, quality = diatonic[degree % len(diatonic)]
        chord_root = root_midi + offset

        # At higher energy, upgrade triads to 7th chords
        if energy > 0.7 and quality in ("major", "minor"):
            quality = "maj7" if quality == "major" else "min7"
        elif energy > 0.5 and i == len(full_degrees) - 1:
            # Dominant 7th on last chord for tension
            if degree == 4:  # V chord
                quality = "7"

        # Choose inversion: prefer voice-led voicing
        if prev_notes:
            notes = voice_lead(prev_notes, chord_root, quality)
            # Determine which inversion was chosen
            inversions = get_inversions(chord_root, quality)
            inversion = 0
            for inv_idx, inv in enumerate(inversions):
                inv_set = {n % 12 for n in inv}
                notes_set = {n % 12 for n in notes}
                if inv_set == notes_set:
                    inversion = inv_idx
                    break
        else:
            # First chord: root position
            intervals = CHORD_INTERVALS.get(quality, [0, 3, 7])
            notes = [chord_root + iv for iv in intervals]
            inversion = 0

        voicing = ChordVoicing(
            root=chord_root,
            quality=quality,
            notes=notes,
            inversion=inversion,
            degree=degree,
        )
        voicings.append(voicing)
        prev_notes = notes

    # Add cadence at the end if we have room
    if len(voicings) >= 2 and section_type in ("chorus", "verse", "outro"):
        # Replace last two chords with a cadence
        cadence_type = "authentic" if section_type != "outro" else "plagal"
        cadence = resolve_cadence(root_midi, scale, cadence_type)
        if cadence and len(cadence) == 2:
            # Voice-lead into the cadence
            voicings[-2].notes = voice_lead(
                voicings[-3].notes if len(voicings) > 2 else voicings[-2].notes,
                cadence[0][0],
                voicings[-2].quality,
            )
            voicings[-1].notes = voice_lead(
                voicings[-2].notes,
                cadence[1][0],
                voicings[-1].quality,
            )

    return voicings
