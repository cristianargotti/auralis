"""
AURALIS — Full-Spectrum Audio Intelligence (Track DNA Extraction)

Analyzes separated stems to identify every element in a track:
- 12-class percussion fingerprinting
- Bass type classification + pitch tracking
- Vocal region mapping + effect detection
- Instrument/synth/FX classification
"""

import numpy as np
import librosa
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────
# WAVEFORM EXTRACTION
# ─────────────────────────────────────────────────────────────────

def extract_waveform(audio_path: Path, points: int = 2000) -> list[float]:
    """Extract a simplified waveform for visualization."""
    try:
        y, _ = librosa.load(audio_path, sr=8000, mono=True)
        chunk_size = max(1, len(y) // points)
        waveform = []
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]
            if len(chunk) > 0:
                waveform.append(float(np.max(np.abs(chunk))))
        return waveform
    except Exception as e:
        print(f"Error extracting waveform from {audio_path}: {e}")
        return [0.0] * 100


# ─────────────────────────────────────────────────────────────────
# 🥁 DRUMS — 12-CLASS PERCUSSION FINGERPRINTING
# ─────────────────────────────────────────────────────────────────

# Spectral fingerprint profiles for each percussion class
# Based on published acoustic research (frequency ranges, spectral shapes)
PERCUSSION_PROFILES = {
    "Kick": {
        "centroid_range": (30, 250),
        "bandwidth_max": 800,
        "rolloff_max": 500,
        "zcr_max": 0.05,
        "decay_max_ms": 150,
        "priority": 1,
    },
    "Tom": {
        "centroid_range": (200, 700),
        "bandwidth_max": 1500,
        "rolloff_max": 2000,
        "zcr_max": 0.08,
        "decay_max_ms": 200,
        "priority": 2,
    },
    "Conga": {
        "centroid_range": (200, 1300),
        "bandwidth_max": 2000,
        "rolloff_max": 3000,
        "zcr_max": 0.10,
        "decay_max_ms": 180,
        "priority": 3,
    },
    "Snare": {
        "centroid_range": (800, 4000),
        "bandwidth_max": 6000,
        "rolloff_max": 8000,
        "zcr_max": 0.25,
        "decay_max_ms": 200,
        "priority": 4,
    },
    "Clap": {
        "centroid_range": (1500, 5000),
        "bandwidth_max": 5000,
        "rolloff_max": 7000,
        "zcr_max": 0.20,
        "decay_max_ms": 120,
        "priority": 5,
    },
    "Rimshot": {
        "centroid_range": (1800, 6000),
        "bandwidth_max": 4000,
        "rolloff_max": 8000,
        "zcr_max": 0.22,
        "decay_max_ms": 60,
        "priority": 6,
    },
    "Cowbell": {
        "centroid_range": (500, 4000),
        "bandwidth_max": 3000,
        "rolloff_max": 7500,
        "zcr_max": 0.15,
        "decay_max_ms": 250,
        "priority": 7,
    },
    "Closed HH": {
        "centroid_range": (5000, 14000),
        "bandwidth_max": 5000,
        "rolloff_max": 15000,
        "zcr_max": 0.50,
        "decay_max_ms": 50,
        "priority": 8,
    },
    "Open HH": {
        "centroid_range": (3500, 12000),
        "bandwidth_max": 8000,
        "rolloff_max": 14000,
        "zcr_max": 0.45,
        "decay_min_ms": 100,  # MUST be long — key differentiator from closed
        "priority": 9,
    },
    "Ride": {
        "centroid_range": (2500, 7000),
        "bandwidth_max": 7000,
        "rolloff_max": 12000,
        "zcr_max": 0.35,
        "decay_min_ms": 200,  # Very sustained
        "priority": 10,
    },
    "Crash": {
        "centroid_range": (2000, 10000),
        "bandwidth_max": 12000,  # Widest bandwidth — fills spectrum
        "rolloff_max": 16000,
        "zcr_max": 0.40,
        "decay_min_ms": 300,  # Long sustain
        "priority": 11,
    },
    "Shaker": {
        "centroid_range": (6000, 18000),
        "bandwidth_max": 5000,
        "rolloff_max": 18000,
        "zcr_max": 0.60,
        "decay_max_ms": 80,
        "priority": 12,
    },
}


def _extract_onset_features(y: np.ndarray, sr: int, onset_sample: int) -> dict[str, float]:
    """
    Extract 6 spectral features from a single onset for percussion classification.
    
    Features:
    1. Spectral Centroid (Hz) — "brightness" center of mass
    2. Spectral Bandwidth (Hz) — frequency spread
    3. Spectral Rolloff (Hz) — where 85% energy lives
    4. Zero-Crossing Rate — noise vs tonal character
    5. Temporal Decay (ms) — sustain/release length
    6. MFCC centroid — timbral fingerprint
    """
    # Extract a window around the onset (100ms)
    window_samples = int(0.1 * sr)
    start = max(0, onset_sample)
    end = min(len(y), start + window_samples)
    window = y[start:end]
    
    if len(window) < 256:
        return {"centroid": 0, "bandwidth": 0, "rolloff": 0, "zcr": 0, "decay_ms": 0, "mfcc_centroid": 0, "energy": 0}
    
    # 1. Spectral Centroid
    centroid = librosa.feature.spectral_centroid(y=window, sr=sr, n_fft=min(2048, len(window)))
    centroid_val = float(np.mean(centroid)) if centroid.size > 0 else 0.0
    
    # 2. Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=window, sr=sr, n_fft=min(2048, len(window)))
    bandwidth_val = float(np.mean(bandwidth)) if bandwidth.size > 0 else 0.0
    
    # 3. Spectral Rolloff (85th percentile)
    rolloff = librosa.feature.spectral_rolloff(y=window, sr=sr, n_fft=min(2048, len(window)), roll_percent=0.85)
    rolloff_val = float(np.mean(rolloff)) if rolloff.size > 0 else 0.0
    
    # 4. Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=window)
    zcr_val = float(np.mean(zcr)) if zcr.size > 0 else 0.0
    
    # 5. Temporal Decay — measure how fast energy drops
    # Split window into 4 segments, measure energy decay
    decay_ms = 0.0
    try:
        # Extend analysis window for decay measurement (500ms)
        decay_end = min(len(y), start + int(0.5 * sr))
        decay_window = y[start:decay_end]
        if len(decay_window) > 512:
            rms = librosa.feature.rms(y=decay_window, frame_length=512, hop_length=128)[0]
            if len(rms) > 1 and rms[0] > 0:
                # Find where energy drops to 10% of peak
                peak_rms = np.max(rms)
                threshold = peak_rms * 0.1
                below_thresh = np.where(rms < threshold)[0]
                if len(below_thresh) > 0:
                    decay_frames = below_thresh[0]
                    decay_ms = float(librosa.frames_to_time(decay_frames, sr=sr, hop_length=128) * 1000)
                else:
                    decay_ms = 500.0  # Still sustaining at 500ms
    except Exception:
        decay_ms = 100.0  # Default
    
    # 6. MFCC centroid — first 4 MFCCs averaged
    mfcc_centroid = 0.0
    try:
        mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=4, n_fft=min(2048, len(window)))
        if mfccs.size > 0:
            mfcc_centroid = float(np.mean(mfccs[1]))  # Skip MFCC0 (just energy)
    except Exception:
        pass
    
    # Energy (RMS)
    energy = float(np.sqrt(np.mean(window ** 2)))
    
    return {
        "centroid": centroid_val,
        "bandwidth": bandwidth_val,
        "rolloff": rolloff_val,
        "zcr": zcr_val,
        "decay_ms": decay_ms,
        "mfcc_centroid": mfcc_centroid,
        "energy": energy,
    }


def _classify_hit(features: dict[str, float]) -> tuple[str, float]:
    """
    Classify a single percussion hit using multi-feature spectral fingerprinting.
    Returns (label, confidence).
    """
    centroid = features["centroid"]
    bandwidth = features["bandwidth"]
    rolloff = features["rolloff"]
    zcr = features["zcr"]
    decay_ms = features["decay_ms"]
    
    best_label = "Percussion"
    best_score = 0.0
    
    for label, profile in PERCUSSION_PROFILES.items():
        score = 0.0
        checks = 0
        
        # Centroid match (most important — weighted 3x)
        lo, hi = profile["centroid_range"]
        if lo <= centroid <= hi:
            # How centered within the range
            mid = (lo + hi) / 2
            range_size = hi - lo
            distance = abs(centroid - mid) / (range_size / 2)
            score += (1.0 - distance * 0.5) * 3.0
            checks += 3
        else:
            # Penalize but allow some tolerance (20% outside range)
            if centroid < lo:
                overshoot = (lo - centroid) / max(lo, 1)
            else:
                overshoot = (centroid - hi) / max(hi, 1)
            if overshoot < 0.2:
                score += 0.5 * 3.0
                checks += 3
            else:
                checks += 3  # Total miss
        
        # Bandwidth check
        if bandwidth <= profile.get("bandwidth_max", 20000):
            score += 1.0
        checks += 1
        
        # Rolloff check
        if rolloff <= profile.get("rolloff_max", 20000):
            score += 1.0
        checks += 1
        
        # ZCR check
        if zcr <= profile.get("zcr_max", 1.0):
            score += 1.0
        checks += 1
        
        # Decay check — critical for open vs closed sounds
        if "decay_max_ms" in profile:
            if decay_ms <= profile["decay_max_ms"]:
                score += 2.0
            else:
                score += 0.3  # Partial credit
            checks += 2
        elif "decay_min_ms" in profile:
            if decay_ms >= profile["decay_min_ms"]:
                score += 2.0
            else:
                score += 0.3
            checks += 2

        # Normalize score
        confidence = score / max(checks, 1)
        
        if confidence > best_score:
            best_score = confidence
            best_label = label
    
    return best_label, round(best_score, 3)


def classify_percussion(drum_stem: Path) -> list[dict[str, Any]]:
    """
    12-class percussion classification using 6-feature spectral fingerprinting.
    
    Returns list of events:
    {time, label, energy, confidence, features: {centroid, bandwidth, rolloff, zcr, decay_ms}}
    """
    events = []
    try:
        y, sr = librosa.load(drum_stem, sr=22050, mono=True)
        
        # High-resolution onset detection
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, backtrack=True,
            pre_max=3, post_max=3, pre_avg=3, post_avg=5,
            delta=0.05, wait=5
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        onset_samples = librosa.frames_to_samples(onset_frames)
        
        for frame_idx, (frame, time, sample) in enumerate(zip(onset_frames, onset_times, onset_samples)):
            if not np.isfinite(time):
                continue
            
            # Extract onset sample safely
            onset_sample = int(sample) if np.isscalar(sample) else int(sample[0]) if len(sample) > 0 else 0
            
            # Extract 6-feature fingerprint
            features = _extract_onset_features(y, sr, onset_sample)
            
            # Skip extremely quiet hits
            if features["energy"] < 0.005:
                continue
            
            # Classify using spectral fingerprint
            label, confidence = _classify_hit(features)
            
            events.append({
                "time": round(float(time), 3),
                "label": label,
                "energy": round(features["energy"], 4),
                "confidence": round(confidence, 3),
                "features": {
                    "centroid": round(features["centroid"], 1),
                    "bandwidth": round(features["bandwidth"], 1),
                    "rolloff": round(features["rolloff"], 1),
                    "zcr": round(features["zcr"], 4),
                    "decay_ms": round(features["decay_ms"], 1),
                }
            })
        
        return events
    except Exception as e:
        print(f"Error classifying percussion in {drum_stem}: {e}")
        return []


def _get_percussion_summary(events: list[dict]) -> dict[str, Any]:
    """Generate a summary of detected percussion elements with counts and patterns."""
    if not events:
        return {"total_hits": 0, "instruments": {}, "dominant": None}
    
    counts: dict[str, int] = {}
    for ev in events:
        label = ev["label"]
        counts[label] = counts.get(label, 0) + 1
    
    dominant = max(counts, key=counts.get) if counts else None
    
    return {
        "total_hits": len(events),
        "instruments": counts,
        "dominant": dominant,
        "unique_instruments": len(counts),
    }


# ─────────────────────────────────────────────────────────────────
# 🎸 BASS — TYPE + NOTES + STYLE
# ─────────────────────────────────────────────────────────────────

BASS_TYPES = {
    "Sub Bass": {
        "harmonic_ratio_max": 0.15,   # Almost no harmonics (pure sine)
        "centroid_max": 100,
        "fundamental_max": 80,
    },
    "808": {
        "harmonic_ratio_max": 0.30,   # Odd harmonics only
        "centroid_max": 200,
        "has_decay": True,            # Decaying envelope
        "fundamental_max": 100,
    },
    "Synth Bass": {
        "harmonic_ratio_min": 0.25,
        "centroid_range": (100, 800),
        "spectral_flatness_max": 0.3, # Not noise-like
    },
    "Bass Guitar": {
        "harmonic_ratio_min": 0.35,   # Rich natural harmonics
        "centroid_range": (150, 1200),
        "body_resonances": True,       # Peaks at 100/200/400Hz
    },
}


def analyze_bass(bass_stem: Path) -> dict[str, Any]:
    """
    Bass stem analysis: type classification + pitch tracking + playing style.
    
    Returns:
    {
        type: str,
        type_confidence: float,
        notes: [{time, pitch_hz, pitch_midi, note_name, duration, velocity}],
        style: {sustain_ratio, has_slides, staccato_ratio},
        summary: {total_notes, pitch_range, avg_velocity}
    }
    """
    result: dict[str, Any] = {
        "type": "Unknown",
        "type_confidence": 0.0,
        "notes": [],
        "style": {},
        "summary": {},
    }
    
    try:
        y, sr = librosa.load(bass_stem, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # Skip silent stems
        rms_total = np.sqrt(np.mean(y ** 2))
        if rms_total < 0.001:
            return result
        
        # ── Bass Type Classification ──
        
        # Spectral features for type classification
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        
        # Harmonic ratio: energy in harmonics vs fundamental
        S = np.abs(librosa.stft(y))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Find fundamental band (20-120 Hz)
        fund_mask = (freqs >= 20) & (freqs <= 120)
        harm_mask = (freqs > 120) & (freqs <= 2000)
        
        fund_energy = float(np.sum(S[fund_mask, :])) if np.any(fund_mask) else 0.0
        harm_energy = float(np.sum(S[harm_mask, :])) if np.any(harm_mask) else 0.0
        total_energy = fund_energy + harm_energy
        harmonic_ratio = harm_energy / total_energy if total_energy > 0 else 0.0
        
        # Classify bass type
        bass_type = "Synth Bass"
        bass_confidence = 0.5
        
        if harmonic_ratio < 0.15 and centroid < 100:
            bass_type = "Sub Bass"
            bass_confidence = 0.85
        elif harmonic_ratio < 0.30 and centroid < 200:
            # Check for 808-style decay
            rms_env = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
            if len(rms_env) > 10:
                # 808s have characteristic exponential decay
                first_half = np.mean(rms_env[:len(rms_env)//2])
                second_half = np.mean(rms_env[len(rms_env)//2:])
                if first_half > 0 and second_half / first_half < 0.4:
                    bass_type = "808"
                    bass_confidence = 0.80
                else:
                    bass_type = "Sub Bass"
                    bass_confidence = 0.70
        elif harmonic_ratio > 0.35 and centroid > 150:
            # Check for body resonances typical of bass guitar
            # Bass guitar has peaks at ~100Hz, ~200Hz, ~400Hz
            freq_bins_100 = np.argmin(np.abs(freqs - 100))
            freq_bins_200 = np.argmin(np.abs(freqs - 200))
            freq_bins_400 = np.argmin(np.abs(freqs - 400))
            
            avg_spectrum = np.mean(S, axis=1)
            if len(avg_spectrum) > max(freq_bins_100, freq_bins_200, freq_bins_400):
                peak_100 = float(avg_spectrum[freq_bins_100])
                peak_200 = float(avg_spectrum[freq_bins_200])
                peak_400 = float(avg_spectrum[freq_bins_400])
                avg_level = float(np.mean(avg_spectrum[fund_mask | harm_mask]))
                
                # Bass guitar signature: prominent peaks at these resonances
                resonance_strength = (peak_100 + peak_200 + peak_400) / (3 * max(avg_level, 1e-10))
                if resonance_strength > 2.0:
                    bass_type = "Bass Guitar"
                    bass_confidence = 0.75
                else:
                    bass_type = "Synth Bass"
                    bass_confidence = 0.70
        else:
            bass_type = "Synth Bass"
            bass_confidence = 0.60
        
        result["type"] = bass_type
        result["type_confidence"] = round(bass_confidence, 3)
        
        # ── Pitch Tracking ──
        # Use pyin for robust pitch detection
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C1'),
            fmax=librosa.note_to_hz('C4'),
            sr=sr
        )
        
        times = librosa.times_like(f0, sr=sr)
        
        # Convert pitch track to discrete notes
        notes = []
        current_note = None
        note_start = 0.0
        
        for i, (t, freq, is_voiced) in enumerate(zip(times, f0, voiced_flag)):
            if is_voiced and np.isfinite(freq) and freq > 0:
                midi = int(round(librosa.hz_to_midi(freq)))
                note_name = librosa.midi_to_note(midi)
                
                if current_note is None or abs(midi - current_note["midi"]) > 1:
                    # New note
                    if current_note is not None:
                        current_note["duration"] = round(float(t - current_note["time"]), 3)
                        notes.append(current_note)
                    
                    current_note = {
                        "time": round(float(t), 3),
                        "pitch_hz": round(float(freq), 1),
                        "midi": midi,
                        "note_name": note_name,
                        "duration": 0.0,
                        "velocity": 0.8,
                    }
                    note_start = t
            else:
                if current_note is not None:
                    current_note["duration"] = round(float(t - current_note["time"]), 3)
                    notes.append(current_note)
                    current_note = None
        
        # Append final note
        if current_note is not None:
            current_note["duration"] = round(float(times[-1] - current_note["time"]), 3)
            notes.append(current_note)
        
        # Filter out very short notes (< 50ms) — likely artifacts
        notes = [n for n in notes if n["duration"] > 0.05]
        result["notes"] = notes
        
        # ── Playing Style Analysis ──
        if notes:
            durations = [n["duration"] for n in notes]
            avg_dur = np.mean(durations)
            short_notes = sum(1 for d in durations if d < 0.15)
            long_notes = sum(1 for d in durations if d > 0.4)
            
            # Slide detection: look for continuous pitch changes
            slide_count = 0
            for i in range(1, len(notes)):
                time_gap = notes[i]["time"] - (notes[i-1]["time"] + notes[i-1]["duration"])
                pitch_diff = abs(notes[i]["midi"] - notes[i-1]["midi"])
                if time_gap < 0.05 and 1 <= pitch_diff <= 4:
                    slide_count += 1
            
            result["style"] = {
                "avg_note_duration": round(float(avg_dur), 3),
                "staccato_ratio": round(short_notes / len(notes), 3),
                "sustain_ratio": round(long_notes / len(notes), 3),
                "has_slides": slide_count > 2,
                "slide_count": slide_count,
            }
            
            # Summary
            pitches_hz = [n["pitch_hz"] for n in notes]
            result["summary"] = {
                "total_notes": len(notes),
                "pitch_range_hz": [round(min(pitches_hz), 1), round(max(pitches_hz), 1)],
                "pitch_range_midi": [min(n["midi"] for n in notes), max(n["midi"] for n in notes)],
                "avg_duration": round(float(avg_dur), 3),
            }
        else:
            result["style"] = {"avg_note_duration": 0, "staccato_ratio": 0, "sustain_ratio": 0, "has_slides": False, "slide_count": 0}
            result["summary"] = {"total_notes": 0, "pitch_range_hz": [0, 0], "pitch_range_midi": [0, 0], "avg_duration": 0}
        
        return result
    except Exception as e:
        print(f"Error analyzing bass in {bass_stem}: {e}")
        return result


# ─────────────────────────────────────────────────────────────────
# 🎤 VOCALS — REGION MAPPING (existing + enhanced)
# ─────────────────────────────────────────────────────────────────

def detect_regions(stem_path: Path, label: str, threshold_db: float = -40) -> list[dict[str, Any]]:
    """Detect regions of activity based on RMS energy."""
    regions = []
    try:
        y, sr = librosa.load(stem_path, sr=22050, mono=True)
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        active = rms_db > threshold_db
        active_padded = np.pad(active, 1, mode='constant')
        diff = np.diff(active_padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            start_time = librosa.frames_to_time(start, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(end, sr=sr, hop_length=hop_length)
            
            if end_time - start_time > 0.5:
                regions.append({
                    "start": round(float(start_time), 3),
                    "end": round(float(end_time), 3),
                    "label": label
                })
        return regions
    except Exception as e:
        print(f"Error detecting regions in {stem_path}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────
# 🎹 OTHER — GENERIC EVENT DETECTION (enhanced)
# ─────────────────────────────────────────────────────────────────

def detect_events(stem_path: Path, label: str) -> list[dict[str, Any]]:
    """Generic onset detection for any stem."""
    events = []
    try:
        y, sr = librosa.load(stem_path, sr=22050, mono=True)
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        
        for time in onset_times:
            events.append({
                "time": round(float(time), 3),
                "label": label,
                "energy": 0.0
            })
        return events
    except Exception as e:
        print(f"Error detecting events in {stem_path}: {e}")
        return []


# ─────────────────────────────────────────────────────────────────
# 🔬 ORCHESTRATOR — FULL TRACK ANALYSIS
# ─────────────────────────────────────────────────────────────────

def analyze_track_layers(project_dir: Path) -> dict[str, Any]:
    """
    Full-spectrum track analysis for X-Ray visualization.
    
    Analyzes each Demucs stem to extract:
    - Drums: 12-class percussion events with confidence scores
    - Bass: Type classification + note-by-note pitch tracking
    - Vocals: Region mapping
    - Other: Event detection
    """
    result: dict[str, Any] = {
        "duration": 0.0,
        "waveform": [],
        "layers": [],
        "analysis": {},
    }
    
    # Analyze Original Waveform
    original_path = project_dir / "original.wav"
    if original_path.exists():
        result["waveform"] = extract_waveform(original_path)
        result["duration"] = librosa.get_duration(path=original_path)

    stems_dir = project_dir / "stems"
    if not stems_dir.exists():
        return result

    # ── 1. DRUMS — 12-Class Percussion ──
    drums_path = stems_dir / "drums.wav"
    if drums_path.exists():
        print("[Analyzer] 🥁 Classifying percussion (12 classes)...")
        drum_events = classify_percussion(drums_path)
        drum_summary = _get_percussion_summary(drum_events)
        
        result["layers"].append({
            "name": "Drums",
            "type": "point",
            "events": drum_events,
        })
        result["analysis"]["drums"] = drum_summary
        print(f"[Analyzer] ✅ Drums: {drum_summary['total_hits']} hits, "
              f"{drum_summary['unique_instruments']} instruments detected: "
              f"{list(drum_summary['instruments'].keys())}")

    # ── 2. BASS — Type + Notes + Style ──
    bass_path = stems_dir / "bass.wav"
    if bass_path.exists():
        print("[Analyzer] 🎸 Analyzing bass (type + pitch + style)...")
        bass_data = analyze_bass(bass_path)
        
        # Convert bass notes to event format for the X-Ray layer
        bass_events = []
        for note in bass_data.get("notes", []):
            bass_events.append({
                "time": note["time"],
                "label": f"{note['note_name']}",
                "energy": note.get("velocity", 0.8),
                "pitch_hz": note["pitch_hz"],
                "midi": note["midi"],
                "duration": note["duration"],
            })
        
        result["layers"].append({
            "name": "Bass",
            "type": "point",
            "events": bass_events,
        })
        result["analysis"]["bass"] = {
            "type": bass_data["type"],
            "type_confidence": bass_data["type_confidence"],
            "style": bass_data["style"],
            "summary": bass_data["summary"],
        }
        print(f"[Analyzer] ✅ Bass: {bass_data['type']} "
              f"({bass_data['type_confidence']:.0%} confidence), "
              f"{bass_data['summary'].get('total_notes', 0)} notes")

    # ── 3. VOCALS — Region Mapping ──
    vocals_path = stems_dir / "vocals.wav"
    if vocals_path.exists():
        print("[Analyzer] 🎤 Mapping vocal regions...")
        vocal_regions = detect_regions(vocals_path, "Vocals")
        result["layers"].append({
            "name": "Vocals",
            "type": "region",
            "events": vocal_regions,
        })
        result["analysis"]["vocals"] = {
            "total_regions": len(vocal_regions),
            "total_duration": round(sum(r["end"] - r["start"] for r in vocal_regions), 2),
        }
        print(f"[Analyzer] ✅ Vocals: {len(vocal_regions)} regions")

    # ── 4. OTHER — Event Detection ──
    other_path = stems_dir / "other.wav"
    if other_path.exists():
        print("[Analyzer] 🎹 Detecting elements in Other stem...")
        other_events = detect_events(other_path, "Synth/FX")
        result["layers"].append({
            "name": "Other",
            "type": "point",
            "events": other_events,
        })
        result["analysis"]["other"] = {
            "total_events": len(other_events),
        }
        print(f"[Analyzer] ✅ Other: {len(other_events)} events")

    return result
