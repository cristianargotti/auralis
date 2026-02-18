"""AURALIS Production AI — Full pipeline: description → finished track.

Orchestrates: Brain (LLM) → Grid (arrangement) → Hands (synth+fx+mix) → Console (master+QC).
This is the core of AURALIS — from text description to finished music.

With reference tracks: extracts cloned palette (drums + timbres) and uses it
for rendering, achieving 100% AI-driven production with cloned sonic identity.
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
    SectionTemplate,
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
    # Creative FX (AI-Driven)
    apply_shimmer_reverb,
    apply_filter_sweep,
    apply_stereo_width,
    apply_tape_stop,
    apply_pitch_riser,
    apply_ring_mod,
    automation_curve,
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


def _crossfade_concat(
    chunks: list[NDArray[np.float64]],
    crossfade_ms: int = 50,
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Concatenate audio chunks with cosine crossfade."""
    if not chunks:
        return np.array([], dtype=np.float64)
    if len(chunks) == 1:
        return chunks[0]

    fade_len = int(sr * crossfade_ms / 1000)
    result = chunks[0].copy()

    for chunk in chunks[1:]:
        if fade_len > 0 and len(result) >= fade_len and len(chunk) >= fade_len:
            fade_out = np.cos(np.linspace(0, np.pi / 2, fade_len)) ** 2
            fade_in = np.sin(np.linspace(0, np.pi / 2, fade_len)) ** 2
            result[-fade_len:] *= fade_out
            chunk = chunk.copy()
            chunk[:fade_len] *= fade_in
            result[-fade_len:] += chunk[:fade_len]
            result = np.concatenate([result, chunk[fade_len:]])
        else:
            result = np.concatenate([result, chunk])

    return result


def _generate_transition_fx(
    from_energy: float,
    to_energy: float,
    to_section_type: str,
    section_duration_s: float,
    sr: int = 44100,
    bpm: float = 120.0,
) -> NDArray[np.float64] | None:
    """Generate transition FX audio to overlay on a section's tail.

    Returns a full-section-length array (mostly silence) with FX
    overlaid on the last N bars.  Returns None if no transition needed.

    Rules:
      - Riser before drops and choruses (energy jump >= 0.2)
      - Short impact hit at the downbeat of high-energy sections
      - No transition between low-energy sections
    """
    energy_jump = to_energy - from_energy
    needs_riser = energy_jump >= 0.15 and to_section_type in (
        "drop", "chorus",
    )
    needs_impact = to_section_type == "drop" and to_energy >= 0.7

    if not needs_riser and not needs_impact:
        return None

    total_samples = int(section_duration_s * sr)
    fx_audio = np.zeros(total_samples, dtype=np.float64)

    if needs_riser:
        # Filtered noise riser over last 2 bars
        beat_dur = 60.0 / bpm
        riser_dur_s = min(2 * 4 * beat_dur, section_duration_s * 0.5)
        riser_len = int(riser_dur_s * sr)

        # White noise
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(riser_len) * 0.12

        # HP filter sweep: 200Hz → 8000Hz (simulated with time-varying filter)
        from scipy.signal import butter, sosfilt
        n_segments = 16
        seg_len = riser_len // n_segments
        filtered = np.zeros(riser_len, dtype=np.float64)
        for seg_i in range(n_segments):
            t = seg_i / n_segments
            cutoff = 200 + (8000 - 200) * (t ** 2)  # Exponential sweep
            nyq = sr / 2
            cutoff_norm = min(cutoff / nyq, 0.95)
            sos = butter(2, cutoff_norm, btype="high", output="sos")
            start = seg_i * seg_len
            end = start + seg_len if seg_i < n_segments - 1 else riser_len
            filtered[start:end] = sosfilt(sos, noise[start:end])

        # Volume envelope: fade in from silence
        env = np.linspace(0.0, 1.0, riser_len) ** 1.5
        riser = filtered * env * (0.15 + energy_jump * 0.3)

        # Place at end of section
        fx_audio[-riser_len:] += riser

    if needs_impact:
        # Short noise burst (20ms) at the very end
        impact_len = int(0.02 * sr)
        rng = np.random.default_rng(99)
        impact = rng.standard_normal(impact_len) * 0.2
        # Sharp decay envelope
        impact *= np.exp(-np.linspace(0, 5, impact_len))
        fx_audio[-impact_len:] += impact

    return fx_audio

def _apply_narrative_fx(
    audio: NDArray[np.float64],
    track_name: str,
    arrangement: Arrangement,
    plan: ProductionPlan,
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Apply AI-generated narrative FX to a track.

    The LLM's fx_plan tells us exactly what effect to apply,
    on which section, with which parameters, and why.
    """
    if not plan.fx_plan:
        return audio

    result = audio.copy()
    total_len = len(result)

    for fx_entry in plan.fx_plan:
        target = fx_entry.get("target", "")
        if target != track_name:
            continue

        section_name = fx_entry.get("section", "").lower()
        effect_name = fx_entry.get("effect", "")
        params = fx_entry.get("params", {})

        # Find the section(s) that match this name
        for section in arrangement.sections:
            if section.name.lower() != section_name:
                continue

            # Calculate sample range for this section
            bar_samples = int(60.0 / plan.bpm * 4 * sr)
            start_sample = section.start_bar * bar_samples
            end_sample = min(start_sample + section.bars * bar_samples, total_len)
            if start_sample >= total_len:
                continue

            segment = result[start_sample:end_sample].copy()

            try:
                if effect_name == "shimmer_reverb":
                    segment = apply_shimmer_reverb(
                        segment,
                        decay_s=params.get("decay_s", 3.0),
                        pitch_shift_semitones=int(params.get("pitch_shift_semitones", 12)),
                        wet=params.get("wet", 0.4),
                        damping=params.get("damping", 0.5),
                        sr=sr,
                    )
                elif effect_name == "filter_sweep":
                    curve = fx_entry.get("automation", {}).get("curve", "exponential")
                    segment = apply_filter_sweep(
                        segment,
                        filter_type=params.get("filter_type", "highpass"),
                        start_hz=params.get("start_hz", 200),
                        end_hz=params.get("end_hz", 20000),
                        curve_shape=curve,
                        resonance=params.get("resonance", 0.707),
                        sr=sr,
                    )
                elif effect_name == "stereo_width":
                    segment = apply_stereo_width(
                        segment,
                        width=params.get("width", 1.5),
                    )
                elif effect_name == "tape_stop":
                    segment = apply_tape_stop(
                        segment,
                        duration_ms=params.get("duration_ms", 500),
                        sr=sr,
                    )
                elif effect_name == "pitch_riser":
                    curve = fx_entry.get("automation", {}).get("curve", "ease_in")
                    segment = apply_pitch_riser(
                        segment,
                        semitones=params.get("semitones", 12),
                        curve_shape=curve,
                        sr=sr,
                    )
                elif effect_name == "ring_mod":
                    segment = apply_ring_mod(
                        segment,
                        freq_hz=params.get("freq_hz", 440),
                        wet=params.get("wet", 0.3),
                        sr=sr,
                    )
            except Exception:
                # If an FX fails, skip it — don't crash the render
                continue

            # Write back (handle length changes from reverb tails etc.)
            actual_len = min(len(segment), end_sample - start_sample)
            result[start_sample:start_sample + actual_len] = segment[:actual_len]

    return result


def render_track(
    description: str,
    output_dir: str | Path | None = None,
    brain_config: BrainConfig | None = None,
    sample_rate: int = 44100,
    on_progress: Any = None,
    reference_paths: list[str | Path] | None = None,
) -> TrackRender:
    """Full pipeline: description → finished track.

    Steps:
    1. Clone: Extract palette from reference tracks (if provided)
    2. Brain: Generate production plan from description + DNA
    3. Grid: Create arrangement from plan
    4. Hands: Render each track (with cloned palette if available)
    5. Textures: Generate neural textures via Stable Audio
    6. Hands: Mix all tracks to stereo
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="auralis_"))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    def _progress(stage: str, pct: float, msg: str) -> None:
        if on_progress:
            on_progress(RenderProgress(stage=stage, progress=pct, message=msg))

    # ── Step 0: Clone — Extract palette from references ──
    palette = None
    ref_dna = None

    if reference_paths:
        _progress("clone", 0.0, "Extracting sonic DNA from references...")
        try:
            from auralis.ear.sample_extractor import (
                ClonedPalette,
                build_cloned_palette,
                merge_palettes,
            )
            from auralis.ear.separator import separate_track
            from auralis.hands.timbre_cloner import build_timbre_models

            palettes: list[ClonedPalette] = []
            for ref_idx, ref_path in enumerate(reference_paths):
                ref_path = Path(ref_path)
                if not ref_path.exists():
                    continue
                _progress(
                    "clone",
                    ref_idx / len(reference_paths),
                    f"Separating {ref_path.name}...",
                )
                # Separate reference into stems
                stems_dir = out / f"ref_{ref_idx}_stems"
                sep_result = separate_track(ref_path, stems_dir)

                # Build cloned palette from separated stems
                ref_palette = build_cloned_palette(
                    stems_dir, source_name=ref_path.stem
                )
                palettes.append(ref_palette)

            if palettes:
                palette = (
                    palettes[0]
                    if len(palettes) == 1
                    else merge_palettes(*palettes)
                )

                # Build timbre models from tonal samples
                timbre_models = build_timbre_models(palette.tones)
                # Attach to palette for renderer access
                palette._timbre_models = timbre_models  # type: ignore[attr-defined]

                _progress(
                    "clone",
                    1.0,
                    f"Palette ready: {palette.total_samples()} samples, "
                    f"{len(timbre_models)} timbres",
                )
        except Exception as e:
            _progress("clone", 1.0, f"Clone extraction failed: {e}, continuing with synthesis")
            palette = None

        # Get reference DNA for LLM context
        try:
            from auralis.ear.reference_bank import ReferenceBank
            ref_bank = ReferenceBank()
            for ref_path in reference_paths:
                ref_path = Path(ref_path)
                if ref_path.exists():
                    ref_bank.add(ref_path)
            ref_dna = ref_bank.get_deep_profile()
        except Exception:
            ref_dna = None

    # ── Step 1: Brain — Production Plan ──
    _progress("brain", 0.0, "Generating production plan...")
    plan = generate_production_plan(description, brain_config, reference_dna=ref_dna)
    _progress("brain", 1.0, f"Plan ready: {plan.title} ({plan.genre}, {plan.bpm}bpm, {plan.key} {plan.scale})")

    # ── Step 2: Arrangement ──
    _progress("grid", 0.0, "Creating arrangement...")
    arr_config = ArrangementConfig(
        key=plan.key,
        scale=plan.scale,
        bpm=plan.bpm,
        structure=plan.structure,
        genre=plan.genre if plan.genre in ("house", "techno", "ambient", "pop", "hip_hop") else "house",
    )
    arrangement = generate_arrangement(arr_config)

    # Override section energy/bars if the LLM specified section_details
    if plan.section_details:
        detail_map = {d.get("name", "").lower(): d for d in plan.section_details}
        for section in arrangement.sections:
            detail = detail_map.get(section.name.lower())
            if detail:
                if "energy" in detail:
                    section.template = SectionTemplate(
                        name=section.template.name,
                        bars=detail.get("bars", section.template.bars),
                        energy=float(detail["energy"]),
                        has_drums=section.template.has_drums,
                        has_bass=section.template.has_bass,
                        has_chords=section.template.has_chords,
                        has_melody=section.template.has_melody,
                        drum_style=section.template.drum_style,
                        bass_style=section.template.bass_style,
                        melody_density=section.template.melody_density,
                    )
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
                
                # 3. Render section (with clone palette if available)
                # Note: events are relative to 0.0 here, which is correct for section render
                section_audio = render_midi_to_audio(
                    events,
                    sr=sample_rate,
                    voice=voice,
                    palette=palette,
                    stem_name=track_name,
                )
                
                # Ensure exact length match
                target_len = int(section_duration_s * sample_rate)
                if len(section_audio) < target_len:
                    # Pad
                    section_audio = np.pad(section_audio, (0, target_len - len(section_audio)))
                elif len(section_audio) > target_len:
                    # Crop (carefully, maybe fade out? For now just crop)
                    section_audio = section_audio[:target_len]

            full_track_audio.append(section_audio)

        # Stitch sections with crossfade
        combined_audio = _crossfade_concat(full_track_audio, crossfade_ms=50, sr=sample_rate)
        
        # Apply effect chain (Global for the track)
        chain_name = plan.effect_chains.get(track_name, "")
        if chain_name in PRESET_CHAINS:
            combined_audio = process_chain(combined_audio, PRESET_CHAINS[chain_name], sample_rate, plan.bpm)

        # Apply AI-generated FX plan (per-section creative FX)
        combined_audio = _apply_narrative_fx(
            combined_audio, track_name, arrangement, plan, sample_rate,
        )

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

    # ── Step 3a: Transition FX — risers and impacts between sections ──
    if len(arrangement.sections) > 1:
        transition_audio_chunks: list[NDArray[np.float64]] = []
        for si, section in enumerate(arrangement.sections):
            section_dur_s = section.template.bars * 4.0 * beat_duration
            if si < len(arrangement.sections) - 1:
                next_section = arrangement.sections[si + 1]
                fx = _generate_transition_fx(
                    from_energy=section.template.energy,
                    to_energy=next_section.template.energy,
                    to_section_type=next_section.template.name,
                    section_duration_s=section_dur_s,
                    sr=sample_rate,
                    bpm=plan.bpm,
                )
                if fx is not None:
                    transition_audio_chunks.append(fx)
                else:
                    transition_audio_chunks.append(
                        np.zeros(int(section_dur_s * sample_rate), dtype=np.float64)
                    )
            else:
                transition_audio_chunks.append(
                    np.zeros(int(section_dur_s * sample_rate), dtype=np.float64)
                )

        transitions_audio = _crossfade_concat(
            transition_audio_chunks, crossfade_ms=50, sr=sample_rate,
        )
        if np.any(transitions_audio != 0):
            track_audio["transitions"] = transitions_audio
            t_path = out / "stem_transitions.wav"
            save_audio(transitions_audio, t_path, sample_rate)
            stems["transitions"] = str(t_path)

    # ── Step 3b: Textures — Neural generation via Stable Audio ──
    if plan.texture_prompts:
        _progress("textures", 0.0, f"Generating {len(plan.texture_prompts)} neural textures...")
        try:
            from auralis.hands.texture_gen import generate_textures, load_texture_audio
            textures = generate_textures(plan.texture_prompts, out, bpm=plan.bpm)
            for t_idx, texture in enumerate(textures):
                texture_audio = load_texture_audio(
                    texture,
                    target_duration_s=arrangement.duration_s,
                    sr=sample_rate,
                )
                tex_name = f"texture_{texture.section}_{t_idx}"
                track_audio[tex_name] = texture_audio
                tex_path = out / f"stem_{tex_name}.wav"
                save_audio(texture_audio, tex_path, sample_rate)
                stems[tex_name] = str(tex_path)
                _progress(
                    "textures",
                    (t_idx + 1) / len(textures),
                    f"Texture {t_idx + 1}/{len(textures)}: {texture.section}",
                )
        except Exception as e:
            _progress("textures", 1.0, f"Texture generation failed: {e}")

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

    # Use AI mix_plan if available, otherwise defaults
    default_pan = {"drums": 0.0, "bass": 0.0, "chords": -0.3, "melody": 0.3}
    default_vol = {"drums": -2.0, "bass": 0.0, "chords": -4.0, "melody": -1.0}

    for name, audio in track_audio.items():
        track_mix = plan.mix_plan.get(name, {})
        pan = track_mix.get("pan", default_pan.get(name, 0.0))
        vol = track_mix.get("volume_db", default_vol.get(name, -3.0))
        rev_send = track_mix.get("reverb_send", 0.2 if name not in ("drums", "bass") else 0.0)
        sends = [SendConfig(bus_name="reverb", amount=rev_send)] if rev_send > 0 else []
        mixer.add_track(name, audio, volume_db=vol, pan=pan, sends=sends)

    output_path = out / "mix_final.wav"
    mix_result = mixer.mix(output_path)
    _progress("mix", 1.0, f"Mix complete: {mix_result.peak_db:.1f}dB peak")

    # ── Step 5: Adaptive Mastering (if reference DNA available) ──
    if ref_dna:
        _progress("master", 0.0, "Applying adaptive mastering from reference DNA...")
        try:
            from auralis.console.dna_brain import think as dna_think, Evidence
            from auralis.console.mastering import MasterConfig, master_audio

            # Build evidence from reference DNA
            evidence = Evidence(
                ref_lufs=ref_dna.get("spectral", {}).get("lufs", -8.0),
                bass_type=ref_dna.get("bass", {}).get("type", ""),
                percussion_density=ref_dna.get("drums", {}).get("density", 2.0),
                sidechain_ratio=ref_dna.get("dynamics", {}).get("sidechain_ratio", 0.0),
                instruments=ref_dna.get("instruments", []),
            )
            brain_result = dna_think(evidence)
            master_plan = brain_result.master_plan

            master_config = MasterConfig(
                target_lufs=master_plan.target_lufs,
                brain_plan=master_plan,
            )
            mastered_path = out / "master_final.wav"
            master_audio(str(output_path), mastered_path, config=master_config)
            output_path = mastered_path
            _progress("master", 1.0, f"Adaptive mastering complete: {master_plan.target_lufs:.1f} LUFS target")
        except Exception as e:
            _progress("master", 1.0, f"Adaptive mastering skipped: {e}")

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
