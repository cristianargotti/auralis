"""AURALIS Production AI — Full pipeline: description → finished track.

Orchestrates: Brain (LLM) → Grid (arrangement) → Hands (synth+fx+mix) → Console (master+QC).
This is the core of AURALIS — from text description to finished music.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from auralis.brain.agent import (
    BrainConfig,
    ProductionPlan,
    generate_production_plan,
    plan_to_render_config,
)
from auralis.grid.arrangement import (
    Arrangement,
    ArrangementConfig,
    generate_arrangement,
    arrangement_summary,
)
from auralis.grid.midi import (
    pattern_to_note_events,
    save_midi,
)
from auralis.hands.synth import (
    PRESETS,
    VoiceConfig,
    render_midi_to_audio,
    save_audio,
)
from auralis.hands.effects import (
    PRESET_CHAINS,
    EffectChain,
    process_chain,
)
from auralis.hands.mixer import (
    Mixer,
    MixConfig,
    MixResult,
    SendConfig,
)


# ── Result Types ─────────────────────────────────────────


@dataclass
class RenderProgress:
    """Progress of a render operation."""

    stage: str
    progress: float  # 0-1
    message: str


@dataclass
class TrackRender:
    """Result of rendering a complete track."""

    plan: ProductionPlan
    arrangement_info: dict[str, Any]
    mix_result: MixResult
    stems: dict[str, str]  # track name -> file path
    midi_files: dict[str, str]  # track name -> MIDI path
    output_path: str
    duration_s: float

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "title": self.plan.title,
            "genre": self.plan.genre,
            "bpm": self.plan.bpm,
            "key": f"{self.plan.key} {self.plan.scale}",
            "duration_s": round(self.duration_s, 1),
            "sections": len(self.arrangement_info.get("sections", [])),
            "tracks": list(self.stems.keys()),
            "output": self.output_path,
        }


# ── Pipeline ─────────────────────────────────────────────


def render_track(
    description: str,
    output_dir: str | Path | None = None,
    brain_config: BrainConfig | None = None,
    sample_rate: int = 44100,
    on_progress: Any = None,
) -> TrackRender:
    """Full pipeline: description → finished track.

    Steps:
    1. Brain: Generate production plan from description
    2. Grid: Create arrangement from plan
    3. Hands: Render each track (synth + effects)
    4. Hands: Mix all tracks to stereo
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="auralis_"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _progress(stage: str, pct: float, msg: str) -> None:
        if on_progress:
            on_progress(RenderProgress(stage=stage, progress=pct, message=msg))

    # ── Step 1: Brain — Production Plan ──
    _progress("brain", 0.0, "Generating production plan...")
    plan = generate_production_plan(description, brain_config)
    _progress("brain", 1.0, f"Plan ready: {plan.title} ({plan.genre}, {plan.bpm}bpm, {plan.key} {plan.scale})")

    # ── Step 2: Grid — Arrangement ──
    _progress("grid", 0.0, "Creating arrangement...")
    arr_config = ArrangementConfig(
        key=plan.key,
        scale=plan.scale,
        bpm=plan.bpm,
        genre=plan.genre,  # type: ignore[arg-type]
        structure=plan.structure,  # type: ignore[arg-type]
    )
    arrangement = generate_arrangement(arr_config)
    arr_info = arrangement_summary(arrangement)
    _progress("grid", 1.0, f"Arrangement: {arrangement.total_bars} bars, {len(arrangement.sections)} sections")

    # ── Step 3: Hands — Render Each Track (Narrative Mode) ──
    _progress("hands", 0.0, "Rendering tracks...")
    stems: dict[str, str] = {}
    midi_files: dict[str, str] = {}
    track_audio: dict[str, NDArray[np.float64]] = {}

    # Determine which tracks we need to render
    all_track_names: set[str] = set()
    for section in arrangement.sections:
        all_track_names.update(section.patterns.keys())

    total_tracks = len(all_track_names)
    beat_duration = 60.0 / plan.bpm

    for t_idx, track_name in enumerate(sorted(all_track_names)):
        _progress("hands", t_idx / total_tracks, f"Rendering {track_name}...")

        # 🧠 Narrative Rendering: Combine sections with evolving sound design
        # Instead of one giant render, we render section by section
        # allowing the "Drop" to sound different from the "Intro".
        
        full_track_audio = []
        current_time_s = 0.0

        for section_idx, section in enumerate(arrangement.sections):
            section_duration_s = section.template.bars * 4.0 * beat_duration
            
            # 1. Get notes for just this section
            if track_name not in section.patterns:
                # Silence for this section
                sr_len = int(section_duration_s * sample_rate)
                section_audio = np.zeros(sr_len, dtype=np.float64)
            else:
                pattern = section.patterns[track_name]
                events = pattern_to_note_events(pattern, plan.bpm)
                
                # 2. Select narrative-aware patch
                # "What does the track need HERE?"
                # e.g. Drop = Acid, Intro = Deep
                from auralis.hands.synth import get_patch_for_stem
                
                # Special handling for drums (still procedural for now)
                if track_name == "drums":
                     voice = VoiceConfig() 
                else:
                    # Ask the narrative engine for the right sound
                    patch = get_patch_for_stem(
                        stem_name=track_name,
                        style=section.template.name,  # "drop", "intro", "verse"
                        bpm=plan.bpm,
                        synth_patch=plan.synth_presets.get(track_name, "")
                    )
                    voice = patch.voice
                
                # 3. Render section
                # Note: events are relative to 0.0 here, which is correct for section render
                section_audio = render_midi_to_audio(events, sr=sample_rate, voice=voice)
                
                # Ensure exact length match
                target_len = int(section_duration_s * sample_rate)
                if len(section_audio) < target_len:
                    # Pad
                    section_audio = np.pad(section_audio, (0, target_len - len(section_audio)))
                elif len(section_audio) > target_len:
                    # Crop (carefully, maybe fade out? For now just crop)
                    section_audio = section_audio[:target_len]

            full_track_audio.append(section_audio)

        # 4. Stitch sections (Crossfade Concat would be better, but simple concat for MVP)
        # TODO: Implement crossfade_concat for seamless transitions
        combined_audio = np.concatenate(full_track_audio)
        
        # Apply effect chain (Global for the track)
        # In V2 we could have per-section FX too!
        chain_name = plan.effect_chains.get(track_name, "")
        if chain_name in PRESET_CHAINS:
            combined_audio = process_chain(combined_audio, PRESET_CHAINS[chain_name], sample_rate, plan.bpm)

        track_audio[track_name] = combined_audio

        # Save stem
        stem_path = out / f"stem_{track_name}.wav"
        save_audio(combined_audio, stem_path, sample_rate)
        stems[track_name] = str(stem_path)

        # Save MIDI (full linear MIDI for DAW export)
        if track_name != "drums":
            midi_path = out / f"midi_{track_name}.mid"
            from auralis.grid.midi import Pattern, Note
            combined = Pattern(name=track_name, notes=[], length_beats=arrangement.total_beats)
            for section in arrangement.sections:
                if track_name in section.patterns:
                    for note in section.patterns[track_name].notes:
                        combined.notes.append(Note(
                            pitch=note.pitch,
                            start_beat=note.start_beat + section.start_bar * 4.0,
                            duration_beats=note.duration_beats,
                            velocity=note.velocity,
                        ))
            save_midi(combined, midi_path, plan.bpm)
            midi_files[track_name] = str(midi_path)

    _progress("hands", 1.0, f"Rendered {len(stems)} tracks with narrative evolution")

    # ── Step 4: Mix ──
    _progress("mix", 0.0, "Mixing tracks...")
    mixer = Mixer(MixConfig(sample_rate=sample_rate, bpm=plan.bpm))

    # Add reverb bus
    from auralis.hands.effects import ReverbConfig
    reverb_chain = EffectChain(
        name="reverb_bus",
        reverb=ReverbConfig(room_size=0.7, damping=0.4, wet=0.8),
    )
    mixer.add_bus("reverb", effects=reverb_chain, volume_db=-3)

    # Auto-pan and add tracks
    pan_positions = {"drums": 0.0, "bass": 0.0, "chords": -0.3, "melody": 0.3}
    volume_offsets = {"drums": -2.0, "bass": 0.0, "chords": -4.0, "melody": -1.0}

    for name, audio in track_audio.items():
        pan = pan_positions.get(name, 0.0)
        vol = volume_offsets.get(name, -3.0)
        sends = [SendConfig(bus_name="reverb", amount=0.2)] if name not in ("drums", "bass") else []
        mixer.add_track(name, audio, volume_db=vol, pan=pan, sends=sends)

    output_path = out / "mix_final.wav"
    mix_result = mixer.mix(output_path)
    _progress("mix", 1.0, f"Mix complete: {mix_result.peak_db:.1f}dB peak")

    return TrackRender(
        plan=plan,
        arrangement_info=arr_info,
        mix_result=mix_result,
        stems=stems,
        midi_files=midi_files,
        output_path=str(output_path),
        duration_s=arrangement.duration_s,
    )


def render_track_offline(
    genre: str = "house",
    key: str = "C",
    scale: str = "minor",
    bpm: float = 120.0,
    output_dir: str | Path | None = None,
    sample_rate: int = 44100,
) -> TrackRender:
    """Render a track without LLM — using defaults and genre templates.

    Useful for testing or when no OpenAI API key is available.
    """
    # Bypass LLM — create plan directly
    plan = ProductionPlan(
        title=f"{genre.title()} Track",
        genre=genre,
        bpm=bpm,
        key=key,
        scale=scale,
        energy="high",
        mood=f"Energetic {genre} vibes",
        structure=[],
        synth_presets={
            "bass": "bass_808",
            "melody": "supersaw",
            "chords": "pad_warm",
        },
        effect_chains={
            "bass": "deep_house_bass",
            "melody": "edm_lead",
            "chords": "ambient_pad",
        },
        description=f"Auto-generated {genre} track",
    )

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="auralis_"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Generate arrangement
    arr_config = ArrangementConfig(key=key, scale=scale, bpm=bpm, genre=genre)  # type: ignore[arg-type]
    arrangement = generate_arrangement(arr_config)
    arr_info = arrangement_summary(arrangement)

    # Render tracks
    stems: dict[str, str] = {}
    midi_files: dict[str, str] = {}
    track_audio: dict[str, NDArray[np.float64]] = {}
    beat_duration = 60.0 / bpm

    all_track_names: set[str] = set()
    for section in arrangement.sections:
        all_track_names.update(section.patterns.keys())

    for track_name in sorted(all_track_names):
        all_notes: list[dict[str, float]] = []
        for section in arrangement.sections:
            if track_name not in section.patterns:
                continue
            events = pattern_to_note_events(section.patterns[track_name], bpm)
            section_start_s = section.start_bar * 4.0 * beat_duration
            for event in events:
                event["start"] += section_start_s
            all_notes.extend(events)

        if not all_notes:
            continue

        preset_name = plan.synth_presets.get(track_name, "supersaw")
        voice = PRESETS.get(preset_name, PRESETS["supersaw"]).voice if preset_name in PRESETS else VoiceConfig()
        if track_name == "drums":
            voice = VoiceConfig()

        audio = render_midi_to_audio(all_notes, sr=sample_rate, voice=voice)

        chain_name = plan.effect_chains.get(track_name, "")
        if chain_name in PRESET_CHAINS:
            audio = process_chain(audio, PRESET_CHAINS[chain_name], sample_rate, bpm)

        track_audio[track_name] = audio
        stem_path = out / f"stem_{track_name}.wav"
        save_audio(audio, stem_path, sample_rate)
        stems[track_name] = str(stem_path)

    # Mix
    mixer = Mixer(MixConfig(sample_rate=sample_rate, bpm=bpm))
    pan_positions = {"drums": 0.0, "bass": 0.0, "chords": -0.3, "melody": 0.3}
    for name, audio in track_audio.items():
        mixer.add_track(name, audio, pan=pan_positions.get(name, 0.0))

    output_path = out / "mix_final.wav"
    mix_result = mixer.mix(output_path)

    return TrackRender(
        plan=plan,
        arrangement_info=arr_info,
        mix_result=mix_result,
        stems=stems,
        midi_files=midi_files,
        output_path=str(output_path),
        duration_s=arrangement.duration_s,
    )
