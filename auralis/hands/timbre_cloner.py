"""AURALIS Timbre Cloner — Spectral morphing for tonal timbre cloning.

Extracts the harmonic + noise profile from a tonal audio segment and
builds a TimbreModel that can resynthesize new notes with the same
sonic character.

Technique: Additive synthesis (harmonic decomposition) + filtered noise.
  1. STFT → identify harmonic peaks relative to fundamental
  2. Extract noise floor (residual after harmonic removal)
  3. Capture amplitude envelope (temporal shape)
  4. Rendering: additive harmonics + shaped noise, envelope applied

Pure numpy/scipy — no GPU, no training, deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt

import structlog

logger = structlog.get_logger()

# Max harmonics to track
MAX_HARMONICS = 32


# ── Data Structures ──────────────────────────────────────


@dataclass
class TimbreModel:
    """Spectral fingerprint of a tonal sound — enough to resynthesize it.

    harmonic_amps: Relative amplitude of each harmonic (1=fundamental).
                   Length up to MAX_HARMONICS.
    harmonic_frames: Time-varying harmonic amps — N frames capturing how
                     the harmonic content evolves over the sound's duration.
                     Each frame is a list[float] of the same length as
                     harmonic_amps. Empty = use static harmonic_amps.
    noise_profile: Spectral shape of the noise component (FFT magnitudes).
    noise_ratio:   How much noise vs harmonic (0=pure tone, 1=pure noise).
                   Used as fallback when noise_envelope is empty.
    noise_envelope: Time-varying noise ratio — N points matching the
                    temporal evolution of noise vs harmonics. Empty = use
                    static noise_ratio.
    amp_envelope:  Temporal amplitude shape, normalized 0-1 (256 points).
    transient_profile: Spectral shape of the first ~20ms (attack character).
                       Empty = no special transient handling.
    transient_duration_ms: Duration of the transient capture window.
    formants:      Resonance peaks [(freq_hz, bandwidth_hz), ...].
    source_pitch:  MIDI note of the source sound (for pitch-shifting ref).
    name:          Human label for this timbre.
    """

    name: str = "unnamed"
    harmonic_amps: list[float] = field(default_factory=list)
    harmonic_frames: list[list[float]] = field(default_factory=list)
    noise_profile: list[float] = field(default_factory=list)
    noise_ratio: float = 0.1
    noise_envelope: list[float] = field(default_factory=list)
    amp_envelope: list[float] = field(default_factory=list)
    transient_profile: list[float] = field(default_factory=list)
    transient_duration_ms: float = 20.0
    formants: list[tuple[float, float]] = field(default_factory=list)
    source_pitch: int = 60  # MIDI note
    sample_rate: int = 22050

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "name": self.name,
            "harmonic_amps": self.harmonic_amps,
            "harmonic_frames": self.harmonic_frames,
            "noise_profile": self.noise_profile,
            "noise_ratio": self.noise_ratio,
            "noise_envelope": self.noise_envelope,
            "amp_envelope": self.amp_envelope,
            "transient_profile": self.transient_profile,
            "transient_duration_ms": self.transient_duration_ms,
            "formants": self.formants,
            "source_pitch": self.source_pitch,
            "sample_rate": self.sample_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimbreModel:
        """Deserialize from dict."""
        return cls(
            name=data.get("name", "unnamed"),
            harmonic_amps=data.get("harmonic_amps", []),
            harmonic_frames=data.get("harmonic_frames", []),
            noise_profile=data.get("noise_profile", []),
            noise_ratio=data.get("noise_ratio", 0.1),
            noise_envelope=data.get("noise_envelope", []),
            amp_envelope=data.get("amp_envelope", []),
            transient_profile=data.get("transient_profile", []),
            transient_duration_ms=data.get("transient_duration_ms", 20.0),
            formants=[tuple(f) for f in data.get("formants", [])],
            source_pitch=data.get("source_pitch", 60),
            sample_rate=data.get("sample_rate", 22050),
        )

    def save(self, path: Path) -> None:
        """Save model to JSON."""
        import json
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> TimbreModel:
        """Load model from JSON."""
        import json
        return cls.from_dict(json.loads(path.read_text()))


# ── Timbre Analysis ──────────────────────────────────────


def analyze_timbre(
    audio: NDArray[np.floating[Any]],
    sr: int = 22050,
    n_harmonics: int = MAX_HARMONICS,
    name: str = "analyzed",
) -> TimbreModel:
    """Extract a TimbreModel from a tonal audio segment.

    Pipeline:
      1. Estimate fundamental frequency (pYIN)
      2. STFT → find harmonic peaks relative to F0
      3. Extract noise residual (total - harmonics)
      4. Capture temporal amplitude envelope
      5. Detect formants (spectral envelope peaks)

    Args:
        audio: Mono audio signal (numpy array).
        sr: Sample rate.
        n_harmonics: Number of harmonics to track.
        name: Label for this timbre.

    Returns:
        TimbreModel ready for resynthesis.
    """
    if len(audio) < 2048:
        logger.warning("timbre_cloner.audio_too_short", length=len(audio))
        return TimbreModel(name=name, sample_rate=sr)

    # ── 1. Estimate fundamental frequency ──
    f0, voiced_flag, _voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C1"),
        fmax=librosa.note_to_hz("C6"),
        sr=sr,
    )
    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) == 0:
        # Can't analyze — return default model
        logger.warning("timbre_cloner.no_f0_detected", name=name)
        return TimbreModel(name=name, sample_rate=sr)

    median_f0 = float(np.median(valid_f0))
    source_midi = int(round(librosa.hz_to_midi(median_f0)))

    # ── 2. STFT → harmonic peak amplitudes ──
    n_fft = 4096  # High resolution for harmonic detection
    S = np.abs(librosa.stft(audio, n_fft=n_fft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Average spectrum across time (kept for backward compat)
    avg_spectrum = np.mean(S, axis=1)

    # ── 2a. Time-varying harmonics (N temporal frames) ──
    n_frames = 8  # 8 temporal slices captures attack → sustain → release
    frame_boundaries = np.linspace(0, S.shape[1], n_frames + 1, dtype=int)
    harmonic_frames: list[list[float]] = []

    def _extract_harmonics_from_spectrum(
        spectrum: NDArray[np.floating[Any]],
        f0: float,
        n_harm: int,
    ) -> tuple[list[float], float]:
        """Extract harmonic amplitudes from a spectrum, normalized to fundamental."""
        amps: list[float] = []
        fund_amp = 0.0
        for h in range(1, n_harm + 1):
            target_freq = f0 * h
            if target_freq > sr / 2:
                break
            bin_idx = int(round(target_freq / (sr / n_fft)))
            if bin_idx >= len(spectrum):
                break
            search_lo = max(0, bin_idx - 2)
            search_hi = min(len(spectrum), bin_idx + 3)
            peak_amp = float(np.max(spectrum[search_lo:search_hi]))
            if h == 1:
                fund_amp = max(peak_amp, 1e-10)
            amps.append(round(peak_amp / fund_amp, 4))
        return amps, fund_amp

    # Static (averaged) harmonics — backward compatible
    harmonic_amps, fundamental_amp = _extract_harmonics_from_spectrum(
        avg_spectrum, median_f0, n_harmonics
    )

    # Per-frame harmonics — captures temporal evolution
    for fi in range(n_frames):
        frame_start = frame_boundaries[fi]
        frame_end = frame_boundaries[fi + 1]
        if frame_end <= frame_start:
            harmonic_frames.append(list(harmonic_amps))
            continue
        frame_spectrum = np.mean(S[:, frame_start:frame_end], axis=1)
        frame_amps, _ = _extract_harmonics_from_spectrum(
            frame_spectrum, median_f0, n_harmonics
        )
        # Pad to same length as harmonic_amps
        while len(frame_amps) < len(harmonic_amps):
            frame_amps.append(0.0)
        harmonic_frames.append(frame_amps[:len(harmonic_amps)])

    # ── 2b. Transient capture (first ~20ms) ──
    transient_duration_ms = 20.0
    transient_samples = int(transient_duration_ms / 1000 * sr)
    transient_profile: list[float] = []
    if len(audio) > transient_samples:
        transient_audio = audio[:transient_samples]
        trans_fft = np.abs(np.fft.rfft(transient_audio))
        # Downsample to 64 bins
        if len(trans_fft) > 64:
            chunk_size = len(trans_fft) // 64
            transient_profile = [
                round(float(np.mean(trans_fft[i * chunk_size : (i + 1) * chunk_size])), 6)
                for i in range(64)
            ]
        else:
            transient_profile = [round(float(v), 6) for v in trans_fft]
        # Normalize
        tp_max = max(transient_profile) if transient_profile else 1.0
        if tp_max > 0:
            transient_profile = [round(v / tp_max, 6) for v in transient_profile]

    # ── 3. Noise residual ──
    # Estimate noise by computing harmonic-to-noise ratio
    harmonic_energy = sum(a**2 for a in harmonic_amps) * fundamental_amp**2
    total_energy = float(np.sum(avg_spectrum**2))
    noise_energy = max(0, total_energy - harmonic_energy)
    noise_ratio = noise_energy / max(total_energy, 1e-10)
    noise_ratio = min(1.0, max(0.0, noise_ratio))

    # ── 3a. Noise envelope (time-varying noise ratio) ──
    noise_envelope: list[float] = []
    for fi in range(n_frames):
        frame_start = frame_boundaries[fi]
        frame_end = frame_boundaries[fi + 1]
        if frame_end <= frame_start:
            noise_envelope.append(noise_ratio)
            continue
        frame_spectrum = np.mean(S[:, frame_start:frame_end], axis=1)
        frame_h_amps, frame_f_amp = _extract_harmonics_from_spectrum(
            frame_spectrum, median_f0, n_harmonics
        )
        frame_h_energy = sum(a**2 for a in frame_h_amps) * frame_f_amp**2
        frame_total = float(np.sum(frame_spectrum**2))
        frame_noise = max(0, frame_total - frame_h_energy)
        frame_nr = frame_noise / max(frame_total, 1e-10)
        noise_envelope.append(round(min(1.0, max(0.0, frame_nr)), 4))

    # Noise spectral shape: smooth the residual spectrum
    noise_profile_raw = avg_spectrum.copy()
    # Zero out harmonic bins
    for h in range(1, len(harmonic_amps) + 1):
        target_freq = median_f0 * h
        bin_idx = int(round(target_freq / (sr / n_fft)))
        lo = max(0, bin_idx - 3)
        hi = min(len(noise_profile_raw), bin_idx + 4)
        noise_profile_raw[lo:hi] = 0

    # Downsample noise profile to 64 bins for storage
    if len(noise_profile_raw) > 64:
        chunk = len(noise_profile_raw) // 64
        noise_profile = [
            round(float(np.mean(noise_profile_raw[i * chunk : (i + 1) * chunk])), 6)
            for i in range(64)
        ]
    else:
        noise_profile = [round(float(v), 6) for v in noise_profile_raw]

    # ── 4. Amplitude envelope (temporal shape) ──
    rms = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
    if len(rms) > 0:
        peak_rms = np.max(rms)
        if peak_rms > 0:
            envelope_norm = rms / peak_rms
        else:
            envelope_norm = rms
        # Resample to 256 points for consistent storage
        from scipy.interpolate import interp1d

        if len(envelope_norm) > 1:
            x_old = np.linspace(0, 1, len(envelope_norm))
            x_new = np.linspace(0, 1, 256)
            interpolator = interp1d(x_old, envelope_norm, kind="linear")
            amp_envelope = [round(float(v), 4) for v in interpolator(x_new)]
        else:
            amp_envelope = [1.0] * 256
    else:
        amp_envelope = [1.0] * 256

    # ── 5. Formant detection (spectral envelope peaks) ──
    formants = _detect_formants(avg_spectrum, freqs)

    model = TimbreModel(
        name=name,
        harmonic_amps=harmonic_amps,
        harmonic_frames=harmonic_frames,
        noise_profile=noise_profile,
        noise_ratio=round(noise_ratio, 4),
        noise_envelope=noise_envelope,
        amp_envelope=amp_envelope,
        transient_profile=transient_profile,
        transient_duration_ms=transient_duration_ms,
        formants=formants,
        source_pitch=source_midi,
        sample_rate=sr,
    )

    logger.info(
        "timbre_cloner.analysis_complete",
        name=name,
        f0=round(median_f0, 1),
        midi=source_midi,
        harmonics=len(harmonic_amps),
        frames=len(harmonic_frames),
        transient_bins=len(transient_profile),
        noise_ratio=round(noise_ratio, 3),
        noise_env_range=f"{min(noise_envelope):.3f}-{max(noise_envelope):.3f}" if noise_envelope else "static",
        formants=len(formants),
    )

    return model


def _detect_formants(
    spectrum: NDArray[np.floating[Any]],
    freqs: NDArray[np.floating[Any]],
    max_formants: int = 5,
) -> list[tuple[float, float]]:
    """Detect spectral envelope peaks (formants) from an average spectrum.

    Returns list of (center_freq_hz, bandwidth_hz) tuples.
    """
    if len(spectrum) < 32:
        return []

    # Smooth spectrum to find envelope (not individual harmonics)
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(spectrum, size=15)

    # Find local maxima
    formants: list[tuple[float, float]] = []
    for i in range(1, len(smoothed) - 1):
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
            freq = float(freqs[i]) if i < len(freqs) else 0
            if freq < 100 or freq > 10000:
                continue  # Skip sub-bass and ultra-high

            # Estimate bandwidth: find -3dB points
            peak_val = smoothed[i]
            threshold = peak_val * 0.707  # -3dB
            # Search left
            left_idx = i
            while left_idx > 0 and smoothed[left_idx] > threshold:
                left_idx -= 1
            # Search right
            right_idx = i
            while right_idx < len(smoothed) - 1 and smoothed[right_idx] > threshold:
                right_idx += 1

            bw = float(freqs[min(right_idx, len(freqs) - 1)] - freqs[max(left_idx, 0)])
            if bw > 0:
                formants.append((round(freq, 1), round(bw, 1)))

    # Sort by prominence (amplitude) and keep top N
    # Re-check amplitudes at formant frequencies
    formants_with_amp = []
    for freq, bw in formants:
        bin_idx = int(round(freq / (freqs[1] - freqs[0]))) if len(freqs) > 1 else 0
        amp = float(smoothed[min(bin_idx, len(smoothed) - 1)])
        formants_with_amp.append((freq, bw, amp))

    formants_with_amp.sort(key=lambda x: x[2], reverse=True)
    return [(f, bw) for f, bw, _ in formants_with_amp[:max_formants]]


# ── Timbre Resynthesis ───────────────────────────────────


def clone_render(
    model: TimbreModel,
    midi_note: int,
    duration_s: float,
    velocity: float = 0.8,
    sr: int = 44100,
) -> NDArray[np.float64]:
    """Synthesize a new note using a cloned TimbreModel.

    Technique: additive synthesis (harmonics) + shaped noise.
      1. Generate each harmonic sine wave at target pitch
      2. Apply cloned harmonic amplitude ratios
      3. Generate noise with the cloned spectral shape
      4. Mix harmonics + noise at the cloned ratio
      5. Apply the cloned amplitude envelope
      6. Apply formant filtering for character

    Args:
        model: TimbreModel from analyze_timbre().
        midi_note: MIDI note to render (0-127).
        duration_s: Duration in seconds.
        velocity: Note velocity (0-1).
        sr: Output sample rate.

    Returns:
        Audio signal as numpy array.
    """
    if not model.harmonic_amps:
        # Fallback: simple sine wave
        n = int(duration_s * sr)
        freq = librosa.midi_to_hz(midi_note)
        t = np.linspace(0, duration_s, n, endpoint=False)
        return np.sin(2 * np.pi * freq * t) * velocity * 0.5

    n_samples = int(duration_s * sr)
    target_freq = float(librosa.midi_to_hz(midi_note))
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    from scipy.interpolate import interp1d

    # ── 1. Additive harmonics (time-varying if available) ──
    harmonic_signal = np.zeros(n_samples, dtype=np.float64)
    use_frames = (
        model.harmonic_frames
        and len(model.harmonic_frames) >= 2
        and all(len(f) > 0 for f in model.harmonic_frames)
    )

    if use_frames:
        # Time-varying: interpolate harmonic amplitudes across N frames
        n_harm = len(model.harmonic_frames[0])
        n_fr = len(model.harmonic_frames)
        # Build amplitude curves per harmonic
        frame_times = np.linspace(0, duration_s, n_fr)
        for h_idx in range(n_harm):
            harmonic_num = h_idx + 1
            freq = target_freq * harmonic_num
            if freq >= sr / 2:
                break
            # Amplitude curve for this harmonic across frames
            amp_curve_pts = [model.harmonic_frames[fi][h_idx] for fi in range(n_fr)]
            if len(amp_curve_pts) > 1:
                interp_fn = interp1d(frame_times, amp_curve_pts, kind="linear",
                                     fill_value="extrapolate")
                amp_curve = interp_fn(t)
            else:
                amp_curve = np.full(n_samples, amp_curve_pts[0])
            harmonic_signal += amp_curve * np.sin(2 * np.pi * freq * t)
    else:
        # Static harmonics (backward compatible)
        for h_idx, amp in enumerate(model.harmonic_amps):
            harmonic_num = h_idx + 1
            freq = target_freq * harmonic_num
            if freq >= sr / 2:
                break
            harmonic_signal += amp * np.sin(2 * np.pi * freq * t)

    # Normalize harmonic signal
    h_peak = np.max(np.abs(harmonic_signal))
    if h_peak > 0:
        harmonic_signal /= h_peak

    # ── 1b. Transient layer (first ~20ms) ──
    transient_layer = np.zeros(n_samples, dtype=np.float64)
    if model.transient_profile and len(model.transient_profile) > 0:
        trans_dur_s = model.transient_duration_ms / 1000.0
        trans_samples = min(int(trans_dur_s * sr), n_samples)
        if trans_samples > 0:
            # Generate noise burst shaped by transient spectral profile
            trans_noise = np.random.randn(trans_samples) * 0.5
            trans_fft = np.fft.rfft(trans_noise)
            profile = np.array(model.transient_profile, dtype=np.float64)
            target_len = len(trans_fft)
            if len(profile) > 1:
                x_old = np.linspace(0, 1, len(profile))
                x_new = np.linspace(0, 1, target_len)
                trans_shape = interp1d(x_old, profile, kind="linear")(x_new)
            else:
                trans_shape = np.ones(target_len)
            ts_peak = np.max(trans_shape)
            if ts_peak > 0:
                trans_shape /= ts_peak
            trans_fft *= trans_shape
            trans_burst = np.fft.irfft(trans_fft, n=trans_samples)
            # Envelope: sharp attack, fast decay
            trans_env = np.exp(-np.linspace(0, 8, trans_samples))
            trans_burst *= trans_env
            # Normalize and mix
            tb_peak = np.max(np.abs(trans_burst))
            if tb_peak > 0:
                trans_burst /= tb_peak
            transient_layer[:trans_samples] = trans_burst * 0.3  # subtle blend

    # ── 2. Shaped noise ──
    noise = np.random.randn(n_samples) * 0.5

    # Apply noise spectral shape via FFT filtering
    if model.noise_profile and len(model.noise_profile) > 0:
        noise_fft = np.fft.rfft(noise)
        profile = np.array(model.noise_profile, dtype=np.float64)
        if len(profile) > 0:
            target_len = len(noise_fft)
            if len(profile) > 1:
                x_old = np.linspace(0, 1, len(profile))
                x_new = np.linspace(0, 1, target_len)
                noise_shape = interp1d(x_old, profile, kind="linear")(x_new)
            else:
                noise_shape = np.ones(target_len)
            ns_peak = np.max(noise_shape)
            if ns_peak > 0:
                noise_shape /= ns_peak
            noise_fft *= noise_shape
            noise = np.fft.irfft(noise_fft, n=n_samples)

    # Normalize noise
    n_peak = np.max(np.abs(noise))
    if n_peak > 0:
        noise /= n_peak

    # ── 3. Mix harmonics + noise with time-varying ratio ──
    if model.noise_envelope and len(model.noise_envelope) >= 2:
        # Interpolate noise envelope to sample resolution
        ne = np.array(model.noise_envelope, dtype=np.float64)
        x_old = np.linspace(0, 1, len(ne))
        x_new = np.linspace(0, 1, n_samples)
        noise_weight = interp1d(x_old, ne, kind="linear")(x_new)
        harmonic_weight = 1.0 - noise_weight
        signal = harmonic_signal * harmonic_weight + noise * noise_weight
    else:
        # Static ratio (backward compatible)
        harmonic_weight = 1.0 - model.noise_ratio
        noise_weight = model.noise_ratio
        signal = harmonic_signal * harmonic_weight + noise * noise_weight

    # Add transient layer
    signal += transient_layer

    # ── 4. Apply amplitude envelope ──
    if model.amp_envelope and len(model.amp_envelope) > 1:
        env = np.array(model.amp_envelope, dtype=np.float64)
        x_old = np.linspace(0, 1, len(env))
        x_new = np.linspace(0, 1, n_samples)
        envelope = interp1d(x_old, env, kind="linear")(x_new)
        signal *= envelope
    else:
        # Simple ADSR fallback
        attack = min(int(0.01 * sr), n_samples // 4)
        release = min(int(0.05 * sr), n_samples // 4)
        env = np.ones(n_samples)
        if attack > 0:
            env[:attack] = np.linspace(0, 1, attack)
        if release > 0:
            env[-release:] = np.linspace(1, 0, release)
        signal *= env

    # ── 5. Formant filtering (character) ──
    if model.formants:
        signal = _apply_formants(signal, model.formants, sr)

    # ── 6. Final gain + velocity ──
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * velocity * 0.85

    return signal.astype(np.float64)


def _apply_formants(
    audio: NDArray[np.float64],
    formants: list[tuple[float, float]],
    sr: int,
) -> NDArray[np.float64]:
    """Apply formant resonances via parallel bandpass filters.

    Each formant adds a resonance peak at its center frequency,
    giving the sound its characteristic "voice" or "body".
    """
    result = audio * 0.3  # Keep some of the original (dry)

    for center_hz, bandwidth_hz in formants:
        if center_hz <= 0 or bandwidth_hz <= 0:
            continue
        if center_hz >= sr / 2:
            continue

        # Bandpass filter centered at formant frequency
        low = max(20.0, center_hz - bandwidth_hz / 2) / (sr / 2)
        high = min(0.99, (center_hz + bandwidth_hz / 2) / (sr / 2))

        if low >= high or low <= 0 or high >= 1:
            continue

        try:
            b, a = butter(2, [low, high], btype="band")
            filtered = filtfilt(b, a, audio)
            # Weight by bandwidth (narrower formants = more resonant)
            weight = min(1.0, 500.0 / max(bandwidth_hz, 1.0)) * 0.15
            result += filtered * weight
        except Exception:
            continue

    # Normalize
    peak = np.max(np.abs(result))
    if peak > 0:
        result /= peak

    return result


# ── Timbre Morphing ──────────────────────────────────────


def morph_timbres(
    model_a: TimbreModel,
    model_b: TimbreModel,
    blend: float = 0.5,
    name: str = "morphed",
) -> TimbreModel:
    """Blend two timbres into a new one.

    blend=0.0 → fully model_a, blend=1.0 → fully model_b.
    """
    blend = max(0.0, min(1.0, blend))
    inv = 1.0 - blend

    # Harmonic amplitudes: linear interpolation
    len_a = len(model_a.harmonic_amps)
    len_b = len(model_b.harmonic_amps)
    max_len = max(len_a, len_b)
    amps_a = model_a.harmonic_amps + [0.0] * (max_len - len_a)
    amps_b = model_b.harmonic_amps + [0.0] * (max_len - len_b)
    harmonic_amps = [
        round(a * inv + b * blend, 4) for a, b in zip(amps_a, amps_b)
    ]

    # Noise ratio (static fallback)
    noise_ratio = model_a.noise_ratio * inv + model_b.noise_ratio * blend

    # Noise envelope (time-varying)
    noise_envelope: list[float] = []
    if model_a.noise_envelope and model_b.noise_envelope:
        max_ne = max(len(model_a.noise_envelope), len(model_b.noise_envelope))
        ne_a = model_a.noise_envelope + [model_a.noise_ratio] * (max_ne - len(model_a.noise_envelope))
        ne_b = model_b.noise_envelope + [model_b.noise_ratio] * (max_ne - len(model_b.noise_envelope))
        noise_envelope = [round(a * inv + b * blend, 4) for a, b in zip(ne_a, ne_b)]

    # Harmonic frames (time-varying)
    harmonic_frames: list[list[float]] = []
    if model_a.harmonic_frames and model_b.harmonic_frames:
        max_fr = max(len(model_a.harmonic_frames), len(model_b.harmonic_frames))
        for fi in range(max_fr):
            fa = model_a.harmonic_frames[min(fi, len(model_a.harmonic_frames) - 1)]
            fb = model_b.harmonic_frames[min(fi, len(model_b.harmonic_frames) - 1)]
            max_h = max(len(fa), len(fb))
            fa_pad = fa + [0.0] * (max_h - len(fa))
            fb_pad = fb + [0.0] * (max_h - len(fb))
            harmonic_frames.append([
                round(a * inv + b * blend, 4) for a, b in zip(fa_pad, fb_pad)
            ])

    # Transient profile
    transient_profile: list[float] = []
    if model_a.transient_profile and model_b.transient_profile:
        max_tp = max(len(model_a.transient_profile), len(model_b.transient_profile))
        tp_a = model_a.transient_profile + [0.0] * (max_tp - len(model_a.transient_profile))
        tp_b = model_b.transient_profile + [0.0] * (max_tp - len(model_b.transient_profile))
        transient_profile = [round(a * inv + b * blend, 6) for a, b in zip(tp_a, tp_b)]

    # Noise profile
    len_na = len(model_a.noise_profile)
    len_nb = len(model_b.noise_profile)
    max_n = max(len_na, len_nb)
    np_a = model_a.noise_profile + [0.0] * (max_n - len_na)
    np_b = model_b.noise_profile + [0.0] * (max_n - len_nb)
    noise_profile = [
        round(a * inv + b * blend, 6) for a, b in zip(np_a, np_b)
    ]

    # Amplitude envelope
    len_ea = len(model_a.amp_envelope)
    len_eb = len(model_b.amp_envelope)
    max_e = max(len_ea, len_eb, 1)
    env_a = model_a.amp_envelope + [0.0] * (max_e - len_ea) if len_ea > 0 else [1.0] * 256
    env_b = model_b.amp_envelope + [0.0] * (max_e - len_eb) if len_eb > 0 else [1.0] * 256
    amp_envelope = [
        round(a * inv + b * blend, 4) for a, b in zip(env_a, env_b)
    ]

    # Formants: weighted merge (take all unique, average overlapping)
    formants = _merge_formants(
        model_a.formants, model_b.formants, blend
    )

    # Source pitch: weighted
    source_pitch = int(
        round(model_a.source_pitch * inv + model_b.source_pitch * blend)
    )

    # Transient duration: weighted
    transient_duration_ms = model_a.transient_duration_ms * inv + model_b.transient_duration_ms * blend

    return TimbreModel(
        name=name,
        harmonic_amps=harmonic_amps,
        harmonic_frames=harmonic_frames,
        noise_profile=noise_profile,
        noise_ratio=round(noise_ratio, 4),
        noise_envelope=noise_envelope,
        amp_envelope=amp_envelope,
        transient_profile=transient_profile,
        transient_duration_ms=round(transient_duration_ms, 1),
        formants=formants,
        source_pitch=source_pitch,
        sample_rate=model_a.sample_rate,
    )


def _merge_formants(
    formants_a: list[tuple[float, float]],
    formants_b: list[tuple[float, float]],
    blend: float,
) -> list[tuple[float, float]]:
    """Merge two sets of formants with blending."""
    inv = 1.0 - blend

    # If one is empty, return the other
    if not formants_a:
        return formants_b
    if not formants_b:
        return formants_a

    # Simple approach: interpolate matching pairs by index
    max_len = max(len(formants_a), len(formants_b))
    result: list[tuple[float, float]] = []

    for i in range(max_len):
        if i < len(formants_a) and i < len(formants_b):
            f = formants_a[i][0] * inv + formants_b[i][0] * blend
            bw = formants_a[i][1] * inv + formants_b[i][1] * blend
            result.append((round(f, 1), round(bw, 1)))
        elif i < len(formants_a):
            result.append(formants_a[i])
        else:
            result.append(formants_b[i])

    return result[:5]  # Cap at 5 formants


# ── Batch Analysis ───────────────────────────────────────


def analyze_timbre_from_file(
    audio_path: str | Path,
    sr: int = 22050,
    name: str | None = None,
) -> TimbreModel:
    """Convenience: analyze timbre from an audio file."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        logger.warning("timbre_cloner.file_not_found", path=str(audio_path))
        return TimbreModel(name=name or "unknown", sample_rate=sr)

    y, sr_actual = librosa.load(audio_path, sr=sr, mono=True)
    return analyze_timbre(y, sr_actual, name=name or audio_path.stem)


def build_timbre_models(
    palette_tones: dict[str, list[Any]],
    sr: int = 22050,
) -> dict[str, TimbreModel]:
    """Build TimbreModels from all tonal samples in a ClonedPalette.

    For each stem (bass, other), loads the best-quality sample and
    analyzes its timbre.

    Args:
        palette_tones: Dict from ClonedPalette.tones (stem → samples).
        sr: Sample rate.

    Returns:
        Dict mapping stem name → TimbreModel.
    """
    models: dict[str, TimbreModel] = {}

    for stem_name, samples in palette_tones.items():
        if not samples:
            continue

        # Use the highest-quality sample
        best = max(samples, key=lambda s: s.quality)

        if not Path(best.path).exists():
            logger.warning(
                "timbre_cloner.sample_missing",
                stem=stem_name,
                path=str(best.path),
            )
            continue

        model = analyze_timbre_from_file(
            best.path, sr=sr, name=f"{stem_name}_clone"
        )

        # Override source pitch if the sample has MIDI pitch
        if best.pitch_midi is not None:
            model.source_pitch = best.pitch_midi

        models[stem_name] = model
        logger.info(
            "timbre_cloner.model_built",
            stem=stem_name,
            source=str(best.path),
            harmonics=len(model.harmonic_amps),
        )

    return models
