import numpy as np
import librosa
from pathlib import Path
from typing import Any

def extract_waveform(audio_path: Path, points: int = 2000) -> list[float]:
    """
    Extract a simplified waveform for visualization.
    Returns a list of amplitude values (0-1) of length `points`.
    """
    try:
        y, _ = librosa.load(audio_path, sr=8000, mono=True)
        # Resample logic: take max amplitude in chunks
        chunk_size = max(1, len(y) // points)
        waveform = []
        for i in range(0, len(y), chunk_size):
            chunk = y[i:i + chunk_size]
            if len(chunk) > 0:
                waveform.append(float(np.max(np.abs(chunk))))
        return waveform
    except Exception as e:
        print(f"Error extracting waveform from {audio_path}: {e}")
        return [0.0] * 100  # Return dummy data to prevent client crash

def detect_drum_events(drum_stem: Path) -> list[dict[str, Any]]:
    """
    Detect and classify drum hits (Kick, Snare, HiHat) from a drum stem.
    Returns list of {time: float, label: str, energy: float}
    """
    events = []
    try:
        y, sr = librosa.load(drum_stem, sr=22050, mono=True)
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Spectral centroid for classification
        centroids_raw = librosa.feature.spectral_centroid(y=y, sr=sr)
        if centroids_raw.ndim == 2 and centroids_raw.shape[1] > 0:
            centroids = centroids_raw[0]
        else:
             centroids = np.zeros(len(onset_frames))

        for frame, time in zip(onset_frames, onset_times):
            # Get centroid at this frame
            idx = min(frame, len(centroids)-1)
            if idx < 0:
                continue
            centroid = centroids[idx]
            
            label = "Percussion"
            if centroid < 150:
                label = "Kick"
            elif centroid > 5000:
                label = "HiHat"
            elif 150 <= centroid <= 5000:
                label = "Snare" # Simple heuristic
            
            # Get energy (RMS) around the onset
            start_sample = librosa.frames_to_samples(frame)[0]
            # Analyze a small window (50ms)
            window = y[start_sample : start_sample + int(0.05 * sr)]
            energy = float(np.sqrt(np.mean(window**2))) if len(window) > 0 else 0.0

            events.append({
                "time": round(float(time), 3),
                "label": label,
                "energy": round(energy, 3)
            })
            
        return events
    except Exception as e:
        print(f"Error detecting drums in {drum_stem}: {e}")
        return []

def detect_events(stem_path: Path, label: str) -> list[dict[str, Any]]:
    """
    Generic onset detection for Bass/Other stems.
    """
    events = []
    try:
        y, sr = librosa.load(stem_path, sr=22050, mono=True)
        onset_times = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        
        for time in onset_times:
             events.append({
                "time": round(float(time), 3),
                "label": label,
                "energy": 0.0 # Placeholder/calculate if needed
            })
        return events
    except Exception as e:
        print(f"Error detecting events in {stem_path}: {e}")
        return []

def detect_regions(stem_path: Path, label: str, threshold_db: float = -40) -> list[dict[str, Any]]:
    """
    Detect regions of activity (e.g. vocal phrases) based on RMS energy.
    Returns list of {start: float, end: float, label: str}
    """
    regions = []
    try:
        y, sr = librosa.load(stem_path, sr=22050, mono=True)
        # Calculate RMS energy
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Boolean mask of active regions
        active = rms_db > threshold_db
        
        # Find continuous regions
        # Pad with False to handle edges
        active_padded = np.pad(active, 1, mode='constant')
        diff = np.diff(active_padded.astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            start_time = librosa.frames_to_time(start, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(end, sr=sr, hop_length=hop_length)
            
            # Filter short regions (< 0.5s)
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

def analyze_track_layers(project_dir: Path) -> dict[str, Any]:
    """
    Orchestrate full track analysis for "X-Ray" visualization.
    """
    result = {
        "duration": 0.0,
        "waveform": [],
        "layers": []
    }
    
    # Analyze Original Waveform (Main view)
    original_path = project_dir / "original.wav"
    if original_path.exists():
        result["waveform"] = extract_waveform(original_path)
        result["duration"] = librosa.get_duration(path=original_path)

    stems_dir = project_dir / "stems"
    if not stems_dir.exists():
        return result

    # 1. Drums X-Ray
    drums_path = stems_dir / "drums.wav"
    if drums_path.exists():
        drum_events = detect_drum_events(drums_path)
        result["layers"].append({
            "name": "Drums",
            "type": "point",
            "events": drum_events
        })

    # 2. Bass X-Ray
    bass_path = stems_dir / "bass.wav"
    if bass_path.exists():
        bass_events = detect_events(bass_path, "Bass Note")
        result["layers"].append({
            "name": "Bass",
            "type": "point",
            "events": bass_events
        })
    
    # 3. Vocals X-Ray (Regions)
    vocals_path = stems_dir / "vocals.wav"
    if vocals_path.exists():
        vocal_regions = detect_regions(vocals_path, "Vocals")
        result["layers"].append({
            "name": "Vocals",
            "type": "region",
            "events": vocal_regions
        })

    # 4. Other X-Ray
    other_path = stems_dir / "other.wav"
    if other_path.exists():
        other_events = detect_events(other_path, "Synth/Fx")
        result["layers"].append({
            "name": "Other",
            "type": "point",
            "events": other_events
        })
        
    return result
