"""AURALIS Stem Generator — Hybrid one-shot extraction + synthesis.

Two generation strategies:
  1. ORGANIC (drums/percussion) → Extract one-shots from reference bank
     stems via onset detection, then sequence into new patterns.
  2. SYNTH (bass/pads/leads) → Generate via grid/midi patterns +
     hands/synth engine, matched to the user track's BPM & key.

The engine NEVER copies full loops.  It extracts individual hits (kick,
snare, hat) and re-sequences them in original patterns.  For tonal
elements, it generates new audio from scratch via synthesis.

Future upgrades:
  - DOSE (arxiv 2025): Neural one-shot extraction from full mixes
  - ACE-Step 1.5 (Feb 2026): Commercial-grade generation on consumer HW
  - Stable Audio Open: Text-to-audio for FX/ambience
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import structlog
from numpy.typing import NDArray

from auralis.console.stem_decisions import StemDecision
from auralis.grid.midi import (
    generate_bassline,
    generate_chord_progression,
    generate_drum_pattern,
    pattern_to_note_events,
)
from auralis.hands.synth import (
    PRESETS,
    VoiceConfig,
    render_midi_to_audio,
    save_audio,
)

logger = structlog.get_logger()

# Global client cache
_AI_CLIENT = None

def _get_ai_client() -> Any:
    global _AI_CLIENT
    if _AI_CLIENT is None:
        try:
            from auralis.hands.stable_audio import StableAudioClient
            _AI_CLIENT = StableAudioClient()
        except ImportError:
            _AI_CLIENT = False
    return _AI_CLIENT if _AI_CLIENT is not False else None



# ── One-Shot Extraction (Organic) ───────────────────────


def extract_oneshots(
    stem_path: str | Path,
    max_hits: int = 16,
    min_length_ms: float = 50.0,
    max_length_ms: float = 500.0,
) -> list[NDArray[np.float64]]:
    """Extract individual one-shot hits from a drum stem via onset detection.

    Uses librosa onset detection to find transients, then slices the audio
    into individual hits.  Each hit is trimmed, normalized, and fade-out
    applied to avoid clicks.

    Args:
        stem_path: Path to the drum stem WAV file.
        max_hits: Maximum number of one-shots to extract.
        min_length_ms: Minimum hit length in milliseconds.
        max_length_ms: Maximum hit length in milliseconds.

    Returns:
        List of numpy arrays, each containing one hit.
    """
    import librosa

    y, sr = librosa.load(str(stem_path), sr=44100, mono=True)

    # Detect onsets
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=512, backtrack=True, units="frames"
    )

    if len(onset_frames) == 0:
        logger.warning("oneshot_extraction.no_onsets", path=str(stem_path))
        return []

    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=512)

    min_samples = int(min_length_ms * sr / 1000)
    max_samples = int(max_length_ms * sr / 1000)

    hits: list[NDArray[np.float64]] = []
    seen_rms: list[float] = []  # track uniqueness via RMS

    for i, start in enumerate(onset_samples):
        if len(hits) >= max_hits:
            break

        # Determine end point
        if i + 1 < len(onset_samples):
            end = min(onset_samples[i + 1], start + max_samples)
        else:
            end = min(start + max_samples, len(y))

        segment = y[start:end].astype(np.float64)

        # Skip too-short hits
        if len(segment) < min_samples:
            continue

        # Skip near-silent hits
        rms = float(np.sqrt(np.mean(segment**2)))
        if rms < 0.01:
            continue

        # Skip duplicates (similar RMS = probably same sound)
        is_duplicate = any(abs(rms - prev) < 0.005 for prev in seen_rms[-4:])
        if is_duplicate and len(hits) > 4:
            continue
        seen_rms.append(rms)

        # Normalize
        peak = np.max(np.abs(segment))
        if peak > 0:
            segment = segment / peak * 0.9

        # Apply fade-out (last 10ms) to avoid clicks
        fade_samples = min(int(0.01 * sr), len(segment) // 4)
        if fade_samples > 0:
            fade = np.linspace(1.0, 0.0, fade_samples)
            segment[-fade_samples:] *= fade

        hits.append(segment)

    logger.info(
        "oneshot_extraction.complete",
        path=str(stem_path),
        hits=len(hits),
        onsets=len(onset_samples),
    )
    return hits


def _classify_oneshot(hit: NDArray[np.float64], sr: int = 44100) -> str:
    """Classify a one-shot as kick, snare, hat, or perc using spectral features."""
    # Simple spectral centroid-based classification
    fft = np.abs(np.fft.rfft(hit))
    freqs = np.fft.rfftfreq(len(hit), 1.0 / sr)

    # Weighted average frequency
    if np.sum(fft) > 0:
        centroid = float(np.sum(freqs * fft) / np.sum(fft))
    else:
        centroid = 1000.0

    # Energy in sub-bass (20-150Hz)
    sub_mask = (freqs >= 20) & (freqs <= 150)
    sub_energy = float(np.sum(fft[sub_mask])) / max(float(np.sum(fft)), 1e-10)

    duration_ms = len(hit) / sr * 1000

    if centroid < 300 and sub_energy > 0.3:
        return "kick"
    elif centroid < 2000 and duration_ms > 100:
        return "snare"
    elif centroid > 5000:
        return "hat"
    else:
        return "perc"


# ── Organic Drum Sequence from One-Shots ────────────────


def sequence_oneshots(
    hits: list[NDArray[np.float64]],
    bpm: float,
    duration_s: float,
    style: str = "four_on_floor",
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Sequence extracted one-shots into a drum pattern.

    Classifies each hit (kick/snare/hat/perc), then places them in a
    musically coherent pattern at the given BPM.

    Args:
        hits: List of one-shot audio arrays.
        bpm: Target BPM.
        duration_s: Total duration in seconds.
        style: Pattern style (four_on_floor, breakbeat, trap, minimal).
        sr: Sample rate.

    Returns:
        Mixed drum pattern as numpy array.
    """
    if not hits:
        return np.zeros(int(duration_s * sr), dtype=np.float64)

    # Classify hits
    classified: dict[str, list[NDArray[np.float64]]] = {
        "kick": [], "snare": [], "hat": [], "perc": [],
    }
    for hit in hits:
        category = _classify_oneshot(hit, sr)
        classified[category].append(hit)

    # Fallback: if we have no kicks, use the lowest-pitched hit
    if not classified["kick"] and hits:
        classified["kick"] = [hits[0]]
    if not classified["snare"] and len(hits) > 1:
        classified["snare"] = [hits[1]]
    if not classified["hat"] and len(hits) > 2:
        classified["hat"] = [hits[2]]

    # Generate pattern from grid/midi
    beats_total = duration_s * bpm / 60.0
    bars = max(1, int(beats_total / 4))
    pattern = generate_drum_pattern(style=style, bars=bars)

    # Map GM drum numbers to our classified hits
    GM_KICK = 36
    GM_SNARE = 38
    GM_CLAP = 39
    GM_HIHAT_C = 42
    GM_HIHAT_O = 46

    total_samples = int(duration_s * sr)
    output = np.zeros(total_samples, dtype=np.float64)
    beat_duration = 60.0 / bpm
    rng = np.random.default_rng(42)

    for note in pattern.notes:
        # Map pitch to hit category
        if note.pitch in (GM_KICK,):
            pool = classified["kick"]
            vel_scale = 1.0
        elif note.pitch in (GM_SNARE, GM_CLAP):
            pool = classified["snare"]
            vel_scale = 0.9
        elif note.pitch in (GM_HIHAT_C, GM_HIHAT_O):
            pool = classified["hat"]
            vel_scale = 0.6
        else:
            pool = classified["perc"]
            vel_scale = 0.7

        if not pool:
            continue

        # Pick a random hit from the pool for variation
        hit = pool[int(rng.integers(0, len(pool)))]
        start_sample = int(note.start_beat * beat_duration * sr)
        velocity = (note.velocity / 127.0) * vel_scale

        if start_sample >= total_samples:
            continue

        end_sample = min(start_sample + len(hit), total_samples)
        actual_len = end_sample - start_sample
        output[start_sample:end_sample] += hit[:actual_len] * velocity

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.85

    return output


# ── Synth Generation (Tonal) ───────────────────────────


def generate_bass_stem(
    bpm: float,
    key: str,
    scale: str,
    duration_s: float,
    style: str = "simple",
    sr: int = 44100,
    stem_plan: Any | None = None,
) -> NDArray[np.float64]:
    """Generate a bass stem via synthesis."""
    # Generate pattern
    pattern = generate_bassline(
        root=key, scale=scale, octave=2,
        pattern_type=style if style in ("simple", "walking", "syncopated") else "simple",
        bars=max(1, int(duration_s * bpm / 60 / 4)),
    )
    note_events = pattern_to_note_events(pattern, bpm)

    # Select voice — brain-guided or BPM fallback
    if stem_plan and getattr(stem_plan, 'patch', ''):
        patch_name = stem_plan.patch
        logger.info("stem_generator.bass_brain_patch", patch=patch_name)
    else:
        patch_name = "bass_808" if bpm < 130 else "acid_303"
    patch = PRESETS.get(patch_name)
    voice = patch.voice if patch else VoiceConfig()

    audio = render_midi_to_audio(note_events, sr=sr, voice=voice)

    # Trim or pad to exact duration
    target_len = int(duration_s * sr)
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))

    return audio


def generate_pad_stem(
    bpm: float,
    key: str,
    scale: str,
    duration_s: float,
    sr: int = 44100,
    stem_plan: Any | None = None,
) -> NDArray[np.float64]:
    """Generate a pad/harmony stem via synthesis."""
    # Generate chord progression
    pattern = generate_chord_progression(
        root=key, scale=scale, octave=3,
        bars=max(1, int(duration_s * bpm / 60 / 4)),
    )
    note_events = pattern_to_note_events(pattern, bpm)

    # Select voice — brain-guided or BPM fallback
    if stem_plan and getattr(stem_plan, 'patch', ''):
        patch_name = stem_plan.patch
        logger.info("stem_generator.pad_brain_patch", patch=patch_name)
    else:
        patch_name = "pad_warm" if bpm < 125 else "supersaw"
    patch = PRESETS.get(patch_name)
    voice = patch.voice if patch else VoiceConfig()

    audio = render_midi_to_audio(note_events, sr=sr, voice=voice)

    # Trim/pad
    target_len = int(duration_s * sr)
    if len(audio) > target_len:
        audio = audio[:target_len]
    elif len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))

    return audio


# ── Main Generator Entry Point ──────────────────────────


def generate_stem(
    decision: StemDecision,
    bpm: float,
    key: str,
    scale: str,
    duration_s: float,
    output_dir: Path,
    reference_stem_path: str | None = None,
    sr: int = 44100,
    stem_plan: Any | None = None,
) -> Path | None:
    """Generate a replacement or enhancement stem based on the decision.

    For drums: extracts one-shots from reference stem and re-sequences.
    For bass/other: generates via MIDI patterns + synth engine.

    Args:
        decision: The StemDecision with patch/style info.
        bpm: Track BPM.
        key: Musical key (e.g. "C", "Am").
        scale: Scale type (e.g. "minor", "major").
        duration_s: Target duration in seconds.
        output_dir: Where to save the generated stem.
        reference_stem_path: Path to reference drum stem (for one-shot extraction).
        sr: Sample rate.
        stem_plan: Optional StemPlan from DNABrain for enriched prompts.

    Returns:
        Path to generated WAV file, or None if generation failed.
    """
    stem_name = decision.stem_name
    action = decision.action

    if action not in ("enhance", "replace"):
        return None

    logger.info(
        "stem_generator.start",
        stem=stem_name,
        action=action,
        patch=decision.synth_patch,
        style=decision.pattern_style,
    )

    # ── 🌿 ORGANIC-FIRST: check real sample pack ────────
    # When brain detects organic instruments, try pack BEFORE AI/synthesis
    if stem_plan and getattr(stem_plan, 'use_organic', False):
        organic_cat = getattr(stem_plan, 'organic_category', '')
        if organic_cat:
            try:
                from auralis.console.organic_pack import OrganicPack
                pack = OrganicPack.load()
                sample = pack.find_best(
                    category=organic_cat,
                    bpm=bpm,
                    bpm_tolerance=5.0,
                    key=key if key else None,
                )
                if sample:
                    audio_data = pack.load_audio(sample, sr=sr)
                    if audio_data is not None:
                        # Trim/loop to target duration
                        target_samples = int(duration_s * sr)
                        if len(audio_data) > target_samples:
                            audio_data = audio_data[:target_samples]
                        elif len(audio_data) < target_samples:
                            # Loop the organic sample
                            repeats = (target_samples // len(audio_data)) + 1
                            audio_data = np.tile(audio_data, repeats)[:target_samples]

                        out_path = output_dir / f"{stem_name}_organic.wav"
                        sf.write(str(out_path), audio_data, sr, subtype="FLOAT")
                        logger.info(
                            "stem_generator.organic_hit",
                            category=organic_cat,
                            sample=sample.filename,
                            pack=sample.pack,
                        )
                        return out_path
                else:
                    logger.info(
                        "stem_generator.organic_miss",
                        category=organic_cat,
                        msg="No matching organic sample, falling through to AI/synth",
                    )
            except Exception as e:
                logger.warning("stem_generator.organic_error", error=str(e))

    # ── AI GENERATION (Phase 5) ─────────────────────────
    # For tonal elements, try Stable Audio first
    if stem_name in ("bass", "other", "vocals") and action in ("replace", "enhance"):
        ai = _get_ai_client()
        if ai and ai.api_token:
            # Build DNA-enriched prompt from brain plan
            if stem_plan and getattr(stem_plan, 'ai_prompt_hints', []):
                hints = ", ".join(stem_plan.ai_prompt_hints)
                prompt = (
                    f"{hints}, "
                    f"high fidelity, pro production"
                ).strip()
                logger.info("stem_generator.ai_brain_prompt", prompt=prompt)
            else:
                # Fallback prompt — derive genre from brain hints or keep neutral
                genre_hint = "music"
                if stem_plan and getattr(stem_plan, 'style', ''):
                    genre_hint = stem_plan.style
                style = decision.pattern_style or "modern"
                prompt = (
                    f"{style} {stem_name} loop, {genre_hint}, "
                    f"{bpm} BPM, key of {key} {scale}, "
                    f"high fidelity, pro production, {decision.synth_patch or ''}"
                ).strip()
            
            # For ENHANCE, we might want a "texture" or "layer"
            if action == "enhance":
                prompt += ", atmospheric texture, layering tool"

            logger.info("stem_generator.ai_attempt", prompt=prompt)
            
            ai_path = ai.generate_loop(
                prompt=prompt,
                bpm=bpm,
                seconds=duration_s,
                output_dir=output_dir,
            )
            
            if ai_path:
                logger.info("stem_generator.ai_success", path=str(ai_path))
                return ai_path
            
            logger.warning("stem_generator.ai_failed", msg="Falling back to synth")

    # ── ALGORITHMIC FALLBACK ────────────────────────────

    try:
        audio: NDArray[np.float64] | None = None

        if stem_name == "drums":
            # ORGANIC: Extract one-shots from reference and re-sequence
            if reference_stem_path and Path(reference_stem_path).exists():
                hits = extract_oneshots(reference_stem_path)
                if hits:
                    style = decision.pattern_style or "four_on_floor"
                    audio = sequence_oneshots(hits, bpm, duration_s, style, sr)
                    logger.info(
                        "stem_generator.organic_drums",
                        hits=len(hits),
                        style=style,
                    )

            # Fallback: generate synthetic drums (FM kick + noise hat)
            if audio is None:
                logger.info("stem_generator.fallback_synth_drums")
                audio = _generate_synth_drums(bpm, duration_s, sr)

        elif stem_name == "bass":
            style = decision.pattern_style or "simple"
            audio = generate_bass_stem(bpm, key, scale, duration_s, style, sr)

        elif stem_name == "other":
            audio = generate_pad_stem(bpm, key, scale, duration_s, sr, stem_plan=stem_plan)

        else:
            logger.warning("stem_generator.unsupported", stem=stem_name)
            return None

        if audio is None or len(audio) == 0:
            logger.warning("stem_generator.empty_audio", stem=stem_name)
            return None

        # Make stereo
        stereo = np.column_stack([audio, audio])

        # Save
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = "gen" if action == "replace" else "layer"
        out_path = output_dir / f"{prefix}_{stem_name}.wav"
        sf.write(str(out_path), stereo, sr, subtype="PCM_24")

        logger.info(
            "stem_generator.saved",
            stem=stem_name,
            path=str(out_path),
            duration=f"{duration_s:.1f}s",
        )
        return out_path

    except Exception as e:
        logger.error("stem_generator.failed", stem=stem_name, error=str(e))
        return None


def _generate_synth_drums(
    bpm: float,
    duration_s: float,
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Fallback: generate synthetic drums using FM synthesis + noise."""
    from auralis.hands.synth import fm_synth, ADSREnvelope

    total_samples = int(duration_s * sr)
    output = np.zeros(total_samples, dtype=np.float64)
    beat_duration = 60.0 / bpm

    bars = max(1, int(duration_s * bpm / 60 / 4))

    for bar in range(bars):
        for beat in range(4):
            t = (bar * 4 + beat) * beat_duration
            sample = int(t * sr)
            if sample >= total_samples:
                break

            # FM kick on every beat
            kick_dur = min(0.15, beat_duration * 0.4)
            kick = fm_synth(50.0, 200.0, 8.0, kick_dur, sr)
            env = ADSREnvelope(attack_s=0.001, decay_s=0.1, sustain=0.0, release_s=0.02)
            kick *= env.generate(kick_dur, sr)[:len(kick)] * 0.8

            end = min(sample + len(kick), total_samples)
            output[sample:end] += kick[:end - sample]

            # Noise hat on offbeats
            hat_t = int((t + beat_duration * 0.5) * sr)
            if hat_t < total_samples:
                hat_dur = 0.03
                hat_samples = int(hat_dur * sr)
                hat = np.random.default_rng(bar * 4 + beat).uniform(-0.3, 0.3, hat_samples)
                hat_env = np.linspace(1.0, 0.0, hat_samples)
                hat *= hat_env
                hat_end = min(hat_t + hat_samples, total_samples)
                output[hat_t:hat_end] += hat[:hat_end - hat_t]

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.8

    return output


def find_reference_stem_path(
    stem_name: str,
    bank_entries: list[dict[str, Any]],
    projects_dir: Path,
    target_bpm: float | None = None,
) -> str | None:
    """Find the best reference stem file to extract one-shots from.

    Picks the reference track closest in BPM to the target.

    Args:
        stem_name: Which stem to find (e.g. "drums").
        bank_entries: List of reference entries from the bank.
        projects_dir: Base projects directory.
        target_bpm: Target BPM for matching.

    Returns:
        Path to the reference stem WAV file, or None.
    """
    if not bank_entries:
        return None

    # Sort by BPM proximity
    if target_bpm:
        entries = sorted(
            bank_entries,
            key=lambda e: abs(e.get("bpm", 120) - target_bpm),
        )
    else:
        entries = bank_entries

    for entry in entries:
        track_id = entry.get("track_id", "")
        if not track_id:
            continue

        # Check common stem file locations
        for pattern in [
            projects_dir / track_id / "stems" / f"{stem_name}.wav",
            projects_dir / track_id / f"{stem_name}.wav",
        ]:
            if pattern.exists():
                logger.info(
                    "reference_stem.found",
                    stem=stem_name,
                    track=entry.get("name", track_id),
                    path=str(pattern),
                )
                return str(pattern)

    logger.info("reference_stem.not_found", stem=stem_name)
    return None
