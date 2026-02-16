"""AURALIS Mixer — Bus routing, sends, panning, and mix rendering.

Manages multi-track mixing with stereo buses, send effects,
panning, and final mix-down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from auralis.hands.effects import EffectChain, process_chain


# ── Data Types ───────────────────────────────────────────


@dataclass
class PanConfig:
    """Stereo panning."""

    position: float = 0.0  # -1 (left) to +1 (right)

    @property
    def left_gain(self) -> float:
        """Left channel gain (constant power panning)."""
        angle = (self.position + 1) * 0.25 * np.pi
        return float(np.cos(angle))

    @property
    def right_gain(self) -> float:
        """Right channel gain (constant power panning)."""
        angle = (self.position + 1) * 0.25 * np.pi
        return float(np.sin(angle))


@dataclass
class SendConfig:
    """Effect send configuration."""

    bus_name: str  # Target bus name
    amount: float = 0.3  # 0-1 send level
    pre_fader: bool = False


@dataclass
class Track:
    """A single mixer track."""

    name: str
    audio: NDArray[np.float64] | None = None  # Mono or stereo
    volume_db: float = 0.0
    pan: PanConfig = field(default_factory=PanConfig)
    mute: bool = False
    solo: bool = False
    effects: EffectChain = field(default_factory=EffectChain)
    sends: list[SendConfig] = field(default_factory=list)

    @property
    def gain(self) -> float:
        """Linear gain from dB volume."""
        return 10 ** (self.volume_db / 20.0)


@dataclass
class Bus:
    """Effect bus (for sends/returns)."""

    name: str
    effects: EffectChain = field(default_factory=EffectChain)
    volume_db: float = 0.0
    pan: PanConfig = field(default_factory=PanConfig)

    @property
    def gain(self) -> float:
        return 10 ** (self.volume_db / 20.0)


@dataclass
class MixConfig:
    """Master mix configuration."""

    sample_rate: int = 44100
    bpm: float = 120.0
    master_volume_db: float = 0.0
    master_effects: EffectChain = field(default_factory=EffectChain)


@dataclass
class MixResult:
    """Result of a mix operation."""

    output_path: str
    duration_s: float
    peak_db: float
    tracks_mixed: int
    buses_used: int
    sample_rate: int


# ── Mixer Engine ─────────────────────────────────────────


class Mixer:
    """Multi-track mixer with buses and send effects."""

    def __init__(self, config: MixConfig | None = None) -> None:
        self.config = config or MixConfig()
        self.tracks: dict[str, Track] = {}
        self.buses: dict[str, Bus] = {}

    def add_track(
        self,
        name: str,
        audio: NDArray[np.float64],
        volume_db: float = 0.0,
        pan: float = 0.0,
        effects: EffectChain | None = None,
        sends: list[SendConfig] | None = None,
    ) -> Track:
        """Add a track to the mixer."""
        track = Track(
            name=name,
            audio=audio,
            volume_db=volume_db,
            pan=PanConfig(position=pan),
            effects=effects or EffectChain(),
            sends=sends or [],
        )
        self.tracks[name] = track
        return track

    def add_bus(
        self,
        name: str,
        effects: EffectChain | None = None,
        volume_db: float = 0.0,
    ) -> Bus:
        """Add an effect bus."""
        bus = Bus(name=name, effects=effects or EffectChain(), volume_db=volume_db)
        self.buses[name] = bus
        return bus

    def _to_mono(self, audio: NDArray[np.float64]) -> NDArray[np.float64]:
        """Ensure audio is mono."""
        if audio.ndim == 2:
            return audio.mean(axis=1)
        return audio

    def _apply_pan(
        self, mono: NDArray[np.float64], pan: PanConfig
    ) -> NDArray[np.float64]:
        """Apply stereo panning to mono audio → stereo (N, 2)."""
        left = mono * pan.left_gain
        right = mono * pan.right_gain
        return np.column_stack([left, right])

    def mix(self, output_path: str | Path | None = None) -> MixResult:
        """Render the full mix to stereo audio."""
        sr = self.config.sample_rate
        bpm = self.config.bpm

        # Determine max length
        max_len = 0
        for track in self.tracks.values():
            if track.audio is not None:
                mono = self._to_mono(track.audio)
                max_len = max(max_len, len(mono))

        if max_len == 0:
            max_len = sr  # 1 second minimum

        # Check for solo tracks
        has_solo = any(t.solo for t in self.tracks.values())

        # Mix stereo bus
        stereo_mix = np.zeros((max_len, 2), dtype=np.float64)
        bus_inputs: dict[str, NDArray[np.float64]] = {
            name: np.zeros(max_len, dtype=np.float64) for name in self.buses
        }

        for track in self.tracks.values():
            if track.audio is None:
                continue
            if track.mute:
                continue
            if has_solo and not track.solo:
                continue

            # Get mono audio, pad to max length
            mono = self._to_mono(track.audio)
            padded = np.zeros(max_len, dtype=np.float64)
            padded[: len(mono)] = mono

            # Apply track effects
            processed = process_chain(padded, track.effects, sr, bpm)

            # Collect sends (pre-fader)
            for send in track.sends:
                if send.pre_fader and send.bus_name in bus_inputs:
                    bus_inputs[send.bus_name] += processed * send.amount

            # Apply volume
            processed *= track.gain

            # Collect sends (post-fader)
            for send in track.sends:
                if not send.pre_fader and send.bus_name in bus_inputs:
                    bus_inputs[send.bus_name] += processed * send.amount

            # Apply panning → stereo
            stereo = self._apply_pan(processed, track.pan)
            stereo_mix += stereo

        # Process buses and add to mix
        buses_used = 0
        for bus_name, bus in self.buses.items():
            if bus_name not in bus_inputs:
                continue
            bus_audio = bus_inputs[bus_name]
            if np.max(np.abs(bus_audio)) < 1e-10:
                continue

            bus_processed = process_chain(bus_audio, bus.effects, sr, bpm)
            bus_processed *= bus.gain
            bus_stereo = self._apply_pan(bus_processed, bus.pan)
            stereo_mix += bus_stereo
            buses_used += 1

        # Master effects
        master_gain = 10 ** (self.config.master_volume_db / 20.0)
        for ch in range(2):
            stereo_mix[:, ch] = process_chain(
                stereo_mix[:, ch], self.config.master_effects, sr, bpm
            )
        stereo_mix *= master_gain

        # Normalize to avoid clipping
        peak = np.max(np.abs(stereo_mix))
        if peak > 0.99:
            stereo_mix = stereo_mix / peak * 0.99

        peak_db = 20.0 * np.log10(max(peak, 1e-10))
        duration_s = max_len / sr

        # Save
        if output_path is None:
            output_path = Path("mix_output.wav")
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(p), stereo_mix, sr, subtype="PCM_24")

        return MixResult(
            output_path=str(p),
            duration_s=duration_s,
            peak_db=peak_db,
            tracks_mixed=len([t for t in self.tracks.values() if t.audio is not None]),
            buses_used=buses_used,
            sample_rate=sr,
        )


# ── Quick Mix Helper ─────────────────────────────────────


def quick_mix(
    tracks: dict[str, NDArray[np.float64]],
    output_path: str | Path,
    sr: int = 44100,
    bpm: float = 120.0,
) -> MixResult:
    """Quick mix multiple tracks with auto-panning."""
    mixer = Mixer(MixConfig(sample_rate=sr, bpm=bpm))

    # Auto-pan tracks across the stereo field
    n = len(tracks)
    for i, (name, audio) in enumerate(tracks.items()):
        if n == 1:
            pan = 0.0
        else:
            pan = (i / (n - 1)) * 1.4 - 0.7  # Spread -0.7 to +0.7
        mixer.add_track(name, audio, pan=pan)

    return mixer.mix(output_path)
