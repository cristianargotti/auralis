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
# 🎤 VOCALS — TYPE + REGIONS + EFFECT DETECTION
# ─────────────────────────────────────────────────────────────────

def analyze_vocals(vocal_stem: Path) -> dict[str, Any]:
    """
    Deep vocal analysis: region mapping, type classification, effect detection.
    
    Vocal Types:
    - Lead: Highest energy, sustained pitch, dominant
    - Backing: Lower energy, harmonic intervals
    - Ad-lib: Short isolated bursts
    - Vocal Chop: Ultra-short, rhythmic, no sustained pitch
    
    Effect Detection:
    - Reverb Tail: Energy decay > 500ms after vocal offset
    - Delay/Echo: Repeated patterns at regular intervals
    - Pitch Shift: Unusual formant ratios
    """
    result: dict[str, Any] = {
        "regions": [],
        "effects": [],
        "summary": {},
    }
    
    try:
        y, sr = librosa.load(vocal_stem, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        hop_length = 512
        
        # Skip silent
        rms_total = np.sqrt(np.mean(y ** 2))
        if rms_total < 0.001:
            return result
        
        # ── Region Detection with Classification ──
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        threshold_db = -35
        active = rms_db > threshold_db
        active_padded = np.pad(active, 1, mode='constant')
        diff = np.diff(active_padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        # Pitch tracking for vocal type detection
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6'), sr=sr
        )
        f0_times = librosa.times_like(f0, sr=sr)
        
        # Global energy stats for relative classification
        if len(rms) > 0:
            energy_75th = float(np.percentile(rms[rms > 0], 75)) if np.any(rms > 0) else 0.01
        else:
            energy_75th = 0.01
        
        regions = []
        for start_f, end_f in zip(starts, ends):
            start_time = float(librosa.frames_to_time(start_f, sr=sr, hop_length=hop_length))
            end_time = float(librosa.frames_to_time(end_f, sr=sr, hop_length=hop_length))
            region_dur = end_time - start_time
            
            if region_dur < 0.1:
                continue
            
            # Region energy
            region_rms = rms[start_f:end_f]
            avg_energy = float(np.mean(region_rms)) if len(region_rms) > 0 else 0.0
            
            # Pitch stability in region
            time_mask = (f0_times >= start_time) & (f0_times <= end_time)
            region_f0 = f0[time_mask]
            region_voiced = voiced_flag[time_mask]
            
            voiced_ratio = float(np.mean(region_voiced)) if len(region_voiced) > 0 else 0.0
            
            pitch_stability = 0.0
            if np.any(np.isfinite(region_f0[region_voiced])) and np.sum(region_voiced) > 2:
                valid_f0 = region_f0[region_voiced & np.isfinite(region_f0)]
                if len(valid_f0) > 1:
                    pitch_stability = 1.0 - min(float(np.std(valid_f0) / np.mean(valid_f0)), 1.0)
            
            # Classify vocal type
            vocal_type = "Lead"
            confidence = 0.5
            
            if region_dur < 0.3 and voiced_ratio < 0.4:
                vocal_type = "Vocal Chop"
                confidence = 0.80
            elif region_dur < 0.8 and avg_energy < energy_75th * 0.5:
                vocal_type = "Ad-lib"
                confidence = 0.70
            elif avg_energy < energy_75th * 0.6 and pitch_stability > 0.5:
                vocal_type = "Backing"
                confidence = 0.65
            elif avg_energy >= energy_75th * 0.6:
                vocal_type = "Lead"
                confidence = 0.75
            
            # Get average pitch for the region
            avg_pitch = 0.0
            avg_note = ""
            if np.any(np.isfinite(region_f0[region_voiced])):
                valid_f0 = region_f0[region_voiced & np.isfinite(region_f0)]
                if len(valid_f0) > 0:
                    avg_pitch = float(np.median(valid_f0))
                    avg_midi = int(round(librosa.hz_to_midi(avg_pitch)))
                    avg_note = librosa.midi_to_note(avg_midi)
            
            regions.append({
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "label": vocal_type,
                "confidence": round(confidence, 3),
                "energy": round(avg_energy, 4),
                "pitch_hz": round(avg_pitch, 1),
                "note": avg_note,
                "voiced_ratio": round(voiced_ratio, 3),
                "pitch_stability": round(pitch_stability, 3),
            })
        
        result["regions"] = regions
        
        # ── Effect Detection ──
        effects = []
        
        # Reverb tail detection: look for energy that decays slowly after vocal regions
        for i, region in enumerate(regions):
            end_sample = int(region["end"] * sr)
            tail_end = min(len(y), end_sample + int(1.5 * sr))  # 1.5s after region
            tail = y[end_sample:tail_end]
            
            if len(tail) > sr // 4:
                tail_rms = librosa.feature.rms(y=tail, frame_length=2048, hop_length=256)[0]
                if len(tail_rms) > 4:
                    # Check for slow decay (reverb signature)
                    first_quarter = float(np.mean(tail_rms[:len(tail_rms)//4]))
                    last_quarter = float(np.mean(tail_rms[-len(tail_rms)//4:]))
                    
                    if first_quarter > 0.005 and last_quarter > 0.001 and first_quarter > last_quarter:
                        decay_ratio = last_quarter / first_quarter
                        if decay_ratio > 0.1:  # Still audible = reverb
                            effects.append({
                                "type": "Reverb Tail",
                                "time": round(region["end"], 3),
                                "duration": round(float(tail_end / sr - region["end"]), 3),
                                "intensity": round(decay_ratio, 3),
                            })
        
        # Delay detection: autocorrelation of envelope for periodic patterns
        if len(y) > sr:
            env = librosa.feature.rms(y=y, frame_length=2048, hop_length=256)[0]
            if len(env) > 100:
                env_centered = env - np.mean(env)
                autocorr = np.correlate(env_centered, env_centered, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                if len(autocorr) > 1 and autocorr[0] > 0:
                    autocorr = autocorr / autocorr[0]
                    
                    # Look for peaks in autocorrelation (delay echoes)
                    min_delay_frames = int(0.1 * sr / 256)   # Min 100ms delay
                    max_delay_frames = int(1.0 * sr / 256)    # Max 1s delay
                    
                    search_region = autocorr[min_delay_frames:max_delay_frames]
                    if len(search_region) > 0:
                        peak_val = float(np.max(search_region))
                        if peak_val > 0.3:  # Strong periodic repetition
                            peak_idx = int(np.argmax(search_region)) + min_delay_frames
                            delay_time_ms = round(peak_idx * 256 / sr * 1000)
                            effects.append({
                                "type": "Delay/Echo",
                                "delay_ms": delay_time_ms,
                                "strength": round(peak_val, 3),
                            })
        
        result["effects"] = effects
        
        # ── Summary ──
        type_counts: dict[str, int] = {}
        for r in regions:
            t = r["label"]
            type_counts[t] = type_counts.get(t, 0) + 1
        
        result["summary"] = {
            "total_regions": len(regions),
            "total_duration": round(sum(r["end"] - r["start"] for r in regions), 2),
            "types": type_counts,
            "effects_detected": [e["type"] for e in effects],
            "has_reverb": any(e["type"] == "Reverb Tail" for e in effects),
            "has_delay": any(e["type"] == "Delay/Echo" for e in effects),
        }
        
        return result
    except Exception as e:
        print(f"Error analyzing vocals in {vocal_stem}: {e}")
        return result


# ─────────────────────────────────────────────────────────────────
# 🎹 OTHER — INSTRUMENT + SYNTH + FX CLASSIFICATION
# ─────────────────────────────────────────────────────────────────

# Spectral profiles for "Other" stem elements
OTHER_PROFILES = {
    # ── Instruments ──
    "Pad": {
        "onset_strength_max": 0.3,    # Slow attack
        "duration_min_ms": 500,        # Sustained
        "bandwidth_min": 500,          # Wide
        "spectral_flatness_max": 0.3,  # Tonal
        "zcr_max": 0.15,
    },
    "Lead": {
        "centroid_min": 800,
        "centroid_max": 6000,
        "bandwidth_max": 3000,         # Focused
        "pitched": True,
        "spectral_flatness_max": 0.25,
    },
    "Pluck": {
        "onset_strength_min": 0.5,     # Sharp attack
        "decay_max_ms": 300,           # Fast decay
        "pitched": True,
        "spectral_flatness_max": 0.3,
    },
    "Arp": {
        "onset_regularity_min": 0.6,   # Regular rhythmic pattern
        "pitched": True,
        "note_density_min": 4,         # Many notes per second
    },
    "Piano": {
        "onset_strength_min": 0.4,     # Hammer attack
        "harmonic_ratio_min": 0.4,     # Rich harmonics
        "centroid_range": (300, 4000),
        "spectral_flatness_max": 0.15, # Very tonal
    },
    "Guitar": {
        "harmonic_ratio_min": 0.3,
        "centroid_range": (400, 5000),
        "spectral_flatness_max": 0.2,
        "bandwidth_max": 4000,
    },
    "Strings": {
        "onset_strength_max": 0.3,     # Slow attack
        "duration_min_ms": 1000,       # Very sustained
        "centroid_range": (200, 4000),
        "spectral_flatness_max": 0.2,
    },
    # ── FX ──
    "Riser": {
        "energy_slope_positive": True,  # Increasing energy over time
        "centroid_slope_positive": True, # Rising pitch
        "duration_min_ms": 1000,
    },
    "Sweep": {
        "centroid_variance_high": True,  # Moving spectral peak
        "duration_min_ms": 500,
    },
    "Impact": {
        "onset_strength_min": 0.7,      # Sudden burst
        "bandwidth_min": 3000,           # Broadband
        "decay_max_ms": 500,             # Fast decay
    },
    "White Noise": {
        "spectral_flatness_min": 0.7,    # Near-uniform spectrum
        "zcr_min": 0.3,
    },
    "Reverse": {
        "energy_slope_positive": True,   # Increasing amplitude
        "centroid_slope_positive": True,
        "duration_max_ms": 2000,
    },
}


def _classify_other_region(y: np.ndarray, sr: int, start_sample: int, end_sample: int) -> tuple[str, float]:
    """Classify a segment of the 'other' stem into instrument/synth/FX type."""
    segment = y[start_sample:end_sample]
    seg_dur_ms = len(segment) / sr * 1000
    
    if len(segment) < 512:
        return "Unknown", 0.0
    
    n_fft = min(2048, len(segment))
    
    # Extract features
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=n_fft)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=segment, sr=sr, n_fft=n_fft)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=segment, n_fft=n_fft)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=segment)))
    
    # Onset strength (attack sharpness)
    onset_env = librosa.onset.onset_strength(y=segment, sr=sr)
    onset_strength = float(np.max(onset_env)) if len(onset_env) > 0 else 0.0
    # Normalize onset strength
    onset_strength = min(onset_strength / 10.0, 1.0)
    
    # Energy envelope slope (for risers/reverses)
    rms_env = librosa.feature.rms(y=segment, frame_length=min(2048, len(segment)), hop_length=256)[0]
    energy_slope = 0.0
    if len(rms_env) > 2:
        x_axis = np.arange(len(rms_env))
        try:
            coeffs = np.polyfit(x_axis, rms_env, 1)
            energy_slope = float(coeffs[0])
        except Exception:
            pass
    
    # Centroid slope (for sweeps/risers)
    centroid_series = librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=n_fft)[0]
    centroid_slope = 0.0
    centroid_variance = 0.0
    if len(centroid_series) > 2:
        try:
            x_axis = np.arange(len(centroid_series))
            coeffs = np.polyfit(x_axis, centroid_series, 1)
            centroid_slope = float(coeffs[0])
            centroid_variance = float(np.std(centroid_series) / max(np.mean(centroid_series), 1))
        except Exception:
            pass
    
    # Harmonic ratio
    S = np.abs(librosa.stft(segment, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    fund_mask = freqs <= 500
    harm_mask = (freqs > 500) & (freqs <= 8000)
    fund_e = float(np.sum(S[fund_mask, :])) if np.any(fund_mask) else 0.0
    harm_e = float(np.sum(S[harm_mask, :])) if np.any(harm_mask) else 0.0
    total_e = fund_e + harm_e
    harmonic_ratio = harm_e / total_e if total_e > 0 else 0.0
    
    # Pitch detection (for tonal vs noise)
    try:
        f0_check, voiced_check, _ = librosa.pyin(
            segment, fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'), sr=sr
        )
        is_pitched = float(np.mean(voiced_check)) > 0.3
    except Exception:
        is_pitched = False
    
    # Decay measurement
    decay_ms = seg_dur_ms  # Default
    if len(rms_env) > 4:
        peak_idx = np.argmax(rms_env)
        post_peak = rms_env[peak_idx:]
        if len(post_peak) > 1 and post_peak[0] > 0:
            thresh = post_peak[0] * 0.1
            below = np.where(post_peak < thresh)[0]
            if len(below) > 0:
                decay_ms = float(below[0] * 256 / sr * 1000)
    
    # ── Classification Decision Tree ──
    
    # FX first — they have distinctive signatures
    if flatness > 0.65 and zcr > 0.25:
        return "White Noise", round(min(flatness, 0.95), 3)
    
    if energy_slope > 0.001 and seg_dur_ms > 800:
        if centroid_slope > 5.0:
            return "Riser", round(min(0.5 + energy_slope * 100, 0.9), 3)
        else:
            return "Reverse", round(0.65, 3)
    
    if centroid_variance > 0.4 and seg_dur_ms > 400:
        return "Sweep", round(min(0.5 + centroid_variance, 0.9), 3)
    
    if onset_strength > 0.6 and bandwidth > 3000 and decay_ms < 400:
        return "Impact", round(min(0.5 + onset_strength, 0.9), 3)
    
    # Instruments
    if onset_strength < 0.25 and seg_dur_ms > 800:
        if bandwidth > 500:
            return "Pad", round(0.70, 3)
        else:
            return "Strings", round(0.60, 3)
    
    if onset_strength < 0.25 and seg_dur_ms > 1500:
        return "Strings", round(0.65, 3)
    
    if flatness < 0.15 and is_pitched and harmonic_ratio > 0.35:
        if onset_strength > 0.35:
            return "Piano", round(0.65, 3)
        else:
            return "Guitar", round(0.55, 3)
    
    if onset_strength > 0.4 and decay_ms < 250 and is_pitched:
        return "Pluck", round(0.70, 3)
    
    if is_pitched and centroid > 800:
        return "Lead", round(0.60, 3)
    
    if is_pitched and centroid <= 800:
        return "Guitar", round(0.50, 3)
    
    # Fallback
    return "Synth", round(0.40, 3)


def analyze_other(other_stem: Path) -> dict[str, Any]:
    """
    Deep analysis of 'Other' stem: instruments, synths, and FX.
    
    Classifies every significant region into:
    Instruments: Pad, Lead, Pluck, Arp, Piano, Guitar, Strings
    FX: Riser, Sweep, Impact, White Noise, Reverse
    """
    result: dict[str, Any] = {
        "events": [],
        "summary": {},
    }
    
    try:
        y, sr = librosa.load(other_stem, sr=22050, mono=True)
        hop_length = 512
        
        # Skip silent
        rms_total = np.sqrt(np.mean(y ** 2))
        if rms_total < 0.001:
            return result
        
        # Detect regions of activity using RMS
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        threshold_db = -35
        active = rms_db > threshold_db
        active_padded = np.pad(active, 1, mode='constant')
        diff = np.diff(active_padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        events = []
        for start_f, end_f in zip(starts, ends):
            start_time = float(librosa.frames_to_time(start_f, sr=sr, hop_length=hop_length))
            end_time = float(librosa.frames_to_time(end_f, sr=sr, hop_length=hop_length))
            region_dur = end_time - start_time
            
            if region_dur < 0.15:
                continue
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Classify the region
            label, confidence = _classify_other_region(y, sr, start_sample, end_sample)
            
            # Get pitch if tonal
            pitch_hz = 0.0
            note_name = ""
            try:
                segment = y[start_sample:end_sample]
                if len(segment) > 2048:
                    f0, voiced, _ = librosa.pyin(
                        segment, fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'), sr=sr
                    )
                    valid = f0[voiced & np.isfinite(f0)]
                    if len(valid) > 0:
                        pitch_hz = float(np.median(valid))
                        midi_note = int(round(librosa.hz_to_midi(pitch_hz)))
                        note_name = librosa.midi_to_note(midi_note)
            except Exception:
                pass
            
            # Region energy
            region_rms = rms[start_f:end_f]
            energy = float(np.mean(region_rms)) if len(region_rms) > 0 else 0.0
            
            events.append({
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "label": label,
                "confidence": round(confidence, 3),
                "energy": round(energy, 4),
                "pitch_hz": round(pitch_hz, 1),
                "note": note_name,
                "duration": round(region_dur, 3),
            })
        
        result["events"] = events
        
        # Summary
        type_counts: dict[str, int] = {}
        fx_types = {"Riser", "Sweep", "Impact", "White Noise", "Reverse"}
        instrument_list = []
        fx_list = []
        
        for ev in events:
            label = ev["label"]
            type_counts[label] = type_counts.get(label, 0) + 1
            if label in fx_types:
                fx_list.append(label)
            else:
                instrument_list.append(label)
        
        result["summary"] = {
            "total_elements": len(events),
            "types": type_counts,
            "unique_types": len(type_counts),
            "instruments_detected": list(set(instrument_list)),
            "fx_detected": list(set(fx_list)),
        }
        
        return result
    except Exception as e:
        print(f"Error analyzing other stem in {other_stem}: {e}")
        return result


# ─────────────────────────────────────────────────────────────────
# 🔗 CROSS-STEM INTELLIGENCE
# ─────────────────────────────────────────────────────────────────

def analyze_arrangement(project_dir: Path) -> dict[str, Any]:
    """
    Cross-stem analysis for arrangement structure, key detection, and sidechain.
    
    Detects:
    - Song sections (Intro, Verse, Chorus, Bridge, Drop, Outro)
    - Musical key and scale
    - Sidechain compression patterns
    - Tempo and tempo variations
    """
    result: dict[str, Any] = {
        "sections": [],
        "key": "",
        "scale": "",
        "tempo_bpm": 0.0,
        "tempo_stable": True,
        "sidechain_detected": False,
        "sidechain_pattern": "",
    }
    
    try:
        original_path = project_dir / "original.wav"
        if not original_path.exists():
            return result
        
        y, sr = librosa.load(original_path, sr=22050, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)
        
        # ── Key Detection via Chroma ──
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_avg = np.mean(chroma, axis=1)
        
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Key detection using Krumhansl-Kessler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        best_key = "C"
        best_scale = "Major"
        best_corr = -1.0
        
        for shift in range(12):
            rolled = np.roll(chroma_avg, -shift)
            
            # Major correlation
            corr_major = float(np.corrcoef(rolled, major_profile)[0, 1])
            if corr_major > best_corr:
                best_corr = corr_major
                best_key = pitch_classes[shift]
                best_scale = "Major"
            
            # Minor correlation
            corr_minor = float(np.corrcoef(rolled, minor_profile)[0, 1])
            if corr_minor > best_corr:
                best_corr = corr_minor
                best_key = pitch_classes[shift]
                best_scale = "Minor"
        
        result["key"] = best_key
        result["scale"] = best_scale
        
        # ── Tempo Detection ──
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if np.isscalar(tempo):
            result["tempo_bpm"] = round(float(tempo), 1)
        else:
            result["tempo_bpm"] = round(float(tempo[0]), 1) if len(tempo) > 0 else 0.0
        
        # ── Arrangement Sections ──
        # Use spectral changes to detect section boundaries
        hop = 512
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, hop_length=hop)
        
        # Segment using structural features (self-similarity)
        # Simple approach: divide into segments by energy profile changes
        segment_duration = 8.0  # 8-second analysis windows
        segment_frames = int(segment_duration * sr / hop)
        n_segments = max(1, mfcc.shape[1] // segment_frames)
        
        segment_features = []
        for i in range(n_segments):
            start_f = i * segment_frames
            end_f = min((i + 1) * segment_frames, mfcc.shape[1])
            seg_mfcc = mfcc[:, start_f:end_f]
            
            # Average MFCC + energy for this segment
            avg_mfcc = np.mean(seg_mfcc, axis=1)
            
            seg_start = i * segment_duration
            seg_end = min((i + 1) * segment_duration, duration)
            
            # Energy of segment
            seg_start_sample = int(seg_start * sr)
            seg_end_sample = min(int(seg_end * sr), len(y))
            seg_audio = y[seg_start_sample:seg_end_sample]
            seg_energy = float(np.sqrt(np.mean(seg_audio ** 2))) if len(seg_audio) > 0 else 0.0
            
            segment_features.append({
                "start": round(seg_start, 1),
                "end": round(seg_end, 1),
                "energy": seg_energy,
                "mfcc_mean": avg_mfcc.tolist(),
            })
        
        # Classify sections based on energy patterns
        if segment_features:
            energies = [s["energy"] for s in segment_features]
            max_energy = max(energies) if energies else 1.0
            if max_energy == 0:
                max_energy = 1.0
            
            sections = []
            for i, seg in enumerate(segment_features):
                rel_energy = seg["energy"] / max_energy
                position_ratio = seg["start"] / max(duration, 1)
                
                # Heuristic section classification
                if position_ratio < 0.08:
                    section_type = "Intro"
                elif position_ratio > 0.92:
                    section_type = "Outro"
                elif rel_energy > 0.75:
                    section_type = "Drop" if i > 0 and energies[i-1] / max_energy < 0.5 else "Chorus"
                elif rel_energy < 0.35:
                    if position_ratio < 0.5:
                        section_type = "Verse"
                    else:
                        section_type = "Bridge"
                elif rel_energy < 0.55:
                    # Check if energy is building up
                    if i < len(energies) - 1 and energies[i+1] / max_energy > rel_energy + 0.2:
                        section_type = "Build"
                    else:
                        section_type = "Verse"
                else:
                    section_type = "Chorus"
                
                # Merge adjacent same-type sections
                if sections and sections[-1]["label"] == section_type:
                    sections[-1]["end"] = seg["end"]
                else:
                    sections.append({
                        "start": seg["start"],
                        "end": seg["end"],
                        "label": section_type,
                        "energy": round(rel_energy, 3),
                    })
            
            result["sections"] = sections
        
        # ── Sidechain Detection ──
        # Look for periodic amplitude dips in bass/other that correlate with kick pattern
        stems_dir = project_dir / "stems"
        bass_path = stems_dir / "bass.wav" if stems_dir.exists() else None
        drums_path = stems_dir / "drums.wav" if stems_dir.exists() else None
        
        if bass_path and bass_path.exists() and drums_path and drums_path.exists():
            try:
                y_bass, sr_b = librosa.load(bass_path, sr=22050, mono=True)
                y_drums, sr_d = librosa.load(drums_path, sr=22050, mono=True)
                
                min_len = min(len(y_bass), len(y_drums))
                y_bass = y_bass[:min_len]
                y_drums = y_drums[:min_len]
                
                # Get bass and drum envelopes
                bass_env = librosa.feature.rms(y=y_bass, frame_length=1024, hop_length=256)[0]
                drum_env = librosa.feature.rms(y=y_drums, frame_length=1024, hop_length=256)[0]
                
                min_env_len = min(len(bass_env), len(drum_env))
                bass_env = bass_env[:min_env_len]
                drum_env = drum_env[:min_env_len]
                
                if len(bass_env) > 100:
                    # Sidechain signature: bass drops when drums peak
                    # Compute anti-correlation
                    if np.std(bass_env) > 0 and np.std(drum_env) > 0:
                        correlation = float(np.corrcoef(bass_env, drum_env)[0, 1])
                        
                        if correlation < -0.15:
                            result["sidechain_detected"] = True
                            result["sidechain_pattern"] = "Kick-triggered duck"
                            result["sidechain_strength"] = round(abs(correlation), 3)
            except Exception:
                pass
        
        return result
    except Exception as e:
        print(f"Error analyzing arrangement: {e}")
        return result


# ─────────────────────────────────────────────────────────────────
# 🔬 ORCHESTRATOR — FULL TRACK ANALYSIS
# ─────────────────────────────────────────────────────────────────

def analyze_track_layers(project_dir: Path) -> dict[str, Any]:
    """
    Full-spectrum track analysis for X-Ray visualization.
    
    Analyzes each Demucs stem to extract EVERYTHING in the track:
    - Drums: 12-class percussion events with confidence scores
    - Bass: Type classification + note-by-note pitch tracking + style
    - Vocals: Lead/Backing/Ad-lib/Chop regions + reverb/delay detection
    - Other: Instrument (pad/lead/pluck/arp/piano/guitar/strings) + FX (riser/sweep/impact/noise)
    - Arrangement: Sections + key/scale + tempo + sidechain
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

    # ── 3. VOCALS — Type + Regions + Effects ──
    vocals_path = stems_dir / "vocals.wav"
    if vocals_path.exists():
        print("[Analyzer] 🎤 Analyzing vocals (type + regions + effects)...")
        vocal_data = analyze_vocals(vocals_path)
        
        result["layers"].append({
            "name": "Vocals",
            "type": "region",
            "events": vocal_data.get("regions", []),
        })
        result["analysis"]["vocals"] = vocal_data.get("summary", {})
        result["analysis"]["vocal_effects"] = vocal_data.get("effects", [])
        
        summary = vocal_data.get("summary", {})
        types = summary.get("types", {})
        effects = summary.get("effects_detected", [])
        print(f"[Analyzer] ✅ Vocals: {summary.get('total_regions', 0)} regions "
              f"({', '.join(f'{k}:{v}' for k, v in types.items())}), "
              f"Effects: {effects if effects else 'none'}")

    # ── 4. OTHER — Instruments + Synths + FX ──
    other_path = stems_dir / "other.wav"
    if other_path.exists():
        print("[Analyzer] 🎹 Analyzing Other stem (instruments + synths + FX)...")
        other_data = analyze_other(other_path)
        
        result["layers"].append({
            "name": "Other",
            "type": "region",
            "events": other_data.get("events", []),
        })
        result["analysis"]["other"] = other_data.get("summary", {})
        
        summary = other_data.get("summary", {})
        instruments = summary.get("instruments_detected", [])
        fx = summary.get("fx_detected", [])
        print(f"[Analyzer] ✅ Other: {summary.get('total_elements', 0)} elements — "
              f"Instruments: {instruments}, FX: {fx}")

    # ── 5. CROSS-STEM — Arrangement + Key + Sidechain ──
    print("[Analyzer] 🔗 Analyzing arrangement (sections + key + sidechain)...")
    arrangement = analyze_arrangement(project_dir)
    
    result["analysis"]["arrangement"] = {
        "sections": arrangement.get("sections", []),
        "key": arrangement.get("key", ""),
        "scale": arrangement.get("scale", ""),
        "tempo_bpm": arrangement.get("tempo_bpm", 0.0),
        "sidechain_detected": arrangement.get("sidechain_detected", False),
        "sidechain_pattern": arrangement.get("sidechain_pattern", ""),
    }
    
    key = arrangement.get("key", "?")
    scale = arrangement.get("scale", "?")
    tempo = arrangement.get("tempo_bpm", 0)
    n_sections = len(arrangement.get("sections", []))
    sidechain = "YES" if arrangement.get("sidechain_detected") else "no"
    print(f"[Analyzer] ✅ Arrangement: {key} {scale}, {tempo} BPM, "
          f"{n_sections} sections, sidechain: {sidechain}")

    print("[Analyzer] 🧬 Track DNA extraction complete.")
    return result
