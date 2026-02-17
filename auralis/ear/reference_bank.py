"""AURALIS Reference DNA Bank — Professional track intelligence database.

Stores DNA fingerprints from reference tracks. When reconstructing a user's
track, the system compares stem-by-stem against averaged reference profiles
to identify gaps and drive intelligent processing decisions.

v2: Deep DNA — stores full percussion maps, bass types, vocal effects,
instrument palettes, FX palettes, and arrangement sections.

Storage: projects/_reference_bank/bank.json (simple JSON, no external DB).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


# ── Data Types ───────────────────────────────────────────


@dataclass
class StemFingerprint:
    """Spectral + dynamic + deep fingerprint of a single stem."""

    # ── Core levels (always present) ──
    rms_db: float = -20.0
    peak_db: float = -6.0
    dynamic_range_db: float = 10.0
    energy_pct: float = 25.0
    freq_bands: dict[str, float] = field(default_factory=lambda: {
        "low": 33.3, "mid": 33.3, "high": 33.3,
    })
    spectral_centroid_avg: float = 2000.0
    instrument_types: list[str] = field(default_factory=list)

    # ── Deep analysis (v2) ──
    # Drums: percussion instrument hit counts {"Kick": 245, "Hi-Hat": 512, ...}
    percussion_map: dict[str, int] = field(default_factory=dict)
    percussion_density: float = 0.0         # hits per second
    percussion_dominant: str = ""           # most common hit type

    # Bass: type classification + style profile
    bass_type: str = ""                     # "Sub Bass", "808", "Analog Synth", etc.
    bass_type_confidence: float = 0.0
    bass_style: dict[str, float] = field(default_factory=dict)  # sustain_ratio, etc.
    bass_note_count: int = 0
    bass_pitch_range: str = ""              # "C1-G2"

    # Vocals: types + effects
    vocal_types: dict[str, int] = field(default_factory=dict)   # {"Lead": 4, "Ad-lib": 12}
    vocal_effects: list[str] = field(default_factory=list)      # ["Reverb Tail", "Delay"]
    vocal_region_count: int = 0

    # Other: instruments + FX
    instruments_detected: list[str] = field(default_factory=list)  # ["Pad", "Lead"]
    fx_detected: list[str] = field(default_factory=list)          # ["Riser", "Sweep"]
    element_count: int = 0


@dataclass
class ReferenceEntry:
    """Complete DNA entry for one reference track."""

    track_id: str = ""
    name: str = "Unknown"
    bpm: float = 120.0
    key: str = "C"
    scale: str = "minor"
    lufs: float = -14.0
    stereo_width: float = 1.0
    duration: float = 0.0
    stems: dict[str, dict[str, Any]] = field(default_factory=dict)
    sections: list[dict[str, Any]] = field(default_factory=list)
    spectral_summary: dict[str, float] = field(default_factory=dict)
    created_at: str = ""

    # ── Deep DNA (v2) ──
    percussion_palette: dict[str, int] = field(default_factory=dict)
    instrument_palette: list[str] = field(default_factory=list)
    fx_palette: list[str] = field(default_factory=list)
    bass_type: str = ""
    vocal_effects: list[str] = field(default_factory=list)
    arrangement_sections: list[dict[str, Any]] = field(default_factory=list)
    sidechain_detected: bool = False
    deep_version: int = 0  # 0 = legacy, 2 = deep DNA

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReferenceEntry:
        """Deserialize from JSON dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ── Reference Bank Engine ────────────────────────────────


class ReferenceBank:
    """Database of professional track DNA fingerprints.

    Stores all reference entries in a single JSON file.
    Provides averaging across all references for gap analysis.
    """

    def __init__(self, projects_dir: Path | str = Path("./projects")) -> None:
        self._bank_dir = Path(projects_dir) / "_reference_bank"
        self._bank_file = self._bank_dir / "bank.json"
        self._entries: dict[str, ReferenceEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load bank from disk."""
        if self._bank_file.exists():
            try:
                data = json.loads(self._bank_file.read_text())
                for entry_data in data.get("references", []):
                    entry = ReferenceEntry.from_dict(entry_data)
                    self._entries[entry.track_id] = entry
                logger.info("reference_bank.loaded", count=len(self._entries))
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("reference_bank.load_failed", error=str(e))

    def _save(self) -> None:
        """Persist bank to disk."""
        self._bank_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 2,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "count": len(self._entries),
            "references": [e.to_dict() for e in self._entries.values()],
        }
        self._bank_file.write_text(json.dumps(data, indent=2, default=str))
        logger.info("reference_bank.saved", count=len(self._entries))

    # ── CRUD ──

    def add_reference(
        self,
        track_id: str,
        name: str,
        ear_analysis: dict[str, Any],
        stem_analysis: dict[str, dict[str, Any]],
        layers_analysis: dict[str, Any] | None = None,
    ) -> ReferenceEntry:
        """Add a processed track to the reference bank with DEEP DNA extraction.

        Args:
            track_id: Unique identifier (job_id or project_id)
            name: Human-readable track name
            ear_analysis: Full EAR analysis data (BPM, key, LUFS, etc.)
            stem_analysis: Per-stem analysis from EAR stage
            layers_analysis: Deep analysis from analyze_track_layers (optional)
        """
        # ── 1. Build stem fingerprints with deep data ──
        stem_fps: dict[str, dict[str, Any]] = {}
        for stem_name, sa in stem_analysis.items():
            if isinstance(sa, dict) and "error" not in sa:
                stem_fps[stem_name] = {
                    "rms_db": sa.get("rms_db", -20.0),
                    "peak_db": sa.get("peak_db", -6.0),
                    "dynamic_range_db": abs(
                        sa.get("peak_db", -6.0) - sa.get("rms_db", -20.0)
                    ),
                    "energy_pct": sa.get("energy_pct", 25.0),
                    "freq_bands": sa.get("freq_bands", {
                        "low": 33.3, "mid": 33.3, "high": 33.3,
                    }),
                }

        # ── 2. Extract deep analysis from layers ──
        analysis = layers_analysis.get("analysis", {}) if layers_analysis else {}

        # Drums: percussion map
        drums_analysis = analysis.get("drums", {})
        percussion_map = drums_analysis.get("instruments", {})
        percussion_dominant = drums_analysis.get("dominant", "")
        total_hits = drums_analysis.get("total_hits", 0)
        duration = float(ear_analysis.get("duration", 1.0)) or 1.0
        percussion_density = round(total_hits / duration, 2) if total_hits else 0.0

        if "drums" in stem_fps:
            stem_fps["drums"]["percussion_map"] = percussion_map
            stem_fps["drums"]["percussion_density"] = percussion_density
            stem_fps["drums"]["percussion_dominant"] = percussion_dominant

        # Bass: type + style
        bass_analysis = analysis.get("bass", {})
        bass_type = bass_analysis.get("type", "")
        bass_type_confidence = bass_analysis.get("type_confidence", 0.0)
        bass_style = bass_analysis.get("style", {})
        bass_summary = bass_analysis.get("summary", {})

        if "bass" in stem_fps:
            stem_fps["bass"]["bass_type"] = bass_type
            stem_fps["bass"]["bass_type_confidence"] = bass_type_confidence
            stem_fps["bass"]["bass_style"] = bass_style
            stem_fps["bass"]["bass_note_count"] = bass_summary.get("total_notes", 0)
            stem_fps["bass"]["bass_pitch_range"] = bass_summary.get("pitch_range", "")

        # Vocals: types + effects
        vocals_analysis = analysis.get("vocals", {})
        vocal_types = vocals_analysis.get("types", {})
        vocal_effects_detected = vocals_analysis.get("effects_detected", [])
        vocal_region_count = vocals_analysis.get("total_regions", 0)

        if "vocals" in stem_fps:
            stem_fps["vocals"]["vocal_types"] = vocal_types
            stem_fps["vocals"]["vocal_effects"] = vocal_effects_detected
            stem_fps["vocals"]["vocal_region_count"] = vocal_region_count

        # Other: instruments + FX
        other_analysis = analysis.get("other", {})
        instruments_detected = other_analysis.get("instruments_detected", [])
        fx_detected = other_analysis.get("fx_detected", [])

        if "other" in stem_fps:
            stem_fps["other"]["instruments_detected"] = instruments_detected
            stem_fps["other"]["fx_detected"] = fx_detected
            stem_fps["other"]["element_count"] = other_analysis.get("total_elements", 0)

        # Arrangement
        arrangement = analysis.get("arrangement", {})
        arrangement_sections = arrangement.get("sections", [])
        sidechain_detected = arrangement.get("sidechain_detected", False)

        # ── 3. Build spectral summary ──
        spectral_summary: dict[str, float] = {}
        if stem_fps:
            all_bands = [s.get("freq_bands", {}) for s in stem_fps.values()]
            for band_key in ("low", "mid", "high"):
                vals = [b.get(band_key, 33.3) for b in all_bands]
                spectral_summary[band_key] = round(sum(vals) / len(vals), 1)

        # ── 4. Build entry with all deep data ──
        entry = ReferenceEntry(
            track_id=track_id,
            name=name,
            bpm=float(ear_analysis.get("bpm", 120.0)),
            key=str(ear_analysis.get("key", "C")),
            scale=str(ear_analysis.get("scale", "minor")),
            lufs=float(ear_analysis.get("integrated_lufs", -14.0)),
            duration=float(ear_analysis.get("duration", 0.0)),
            stems=stem_fps,
            sections=ear_analysis.get("sections", []),
            spectral_summary=spectral_summary,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            # Deep DNA v2
            percussion_palette=percussion_map,
            instrument_palette=instruments_detected,
            fx_palette=fx_detected,
            bass_type=bass_type,
            vocal_effects=vocal_effects_detected,
            arrangement_sections=arrangement_sections,
            sidechain_detected=sidechain_detected,
            deep_version=2,
        )

        self._entries[track_id] = entry
        self._save()

        deep_items = []
        if percussion_map:
            deep_items.append(f"perc={list(percussion_map.keys())}")
        if bass_type:
            deep_items.append(f"bass={bass_type}")
        if instruments_detected:
            deep_items.append(f"inst={instruments_detected}")
        if fx_detected:
            deep_items.append(f"fx={fx_detected}")
        if vocal_effects_detected:
            deep_items.append(f"vfx={vocal_effects_detected}")

        logger.info(
            "reference_bank.added_deep",
            track_id=track_id,
            name=name,
            stems=list(stem_fps.keys()),
            deep_data=", ".join(deep_items) if deep_items else "no deep data",
        )
        return entry

    def remove_reference(self, track_id: str) -> bool:
        """Remove a track from the reference bank."""
        if track_id in self._entries:
            del self._entries[track_id]
            self._save()
            return True
        return False

    def list_references(self) -> list[dict[str, Any]]:
        """List all references (summary only)."""
        return [
            {
                "track_id": e.track_id,
                "name": e.name,
                "bpm": e.bpm,
                "key": f"{e.key} {e.scale}",
                "lufs": e.lufs,
                "stems": list(e.stems.keys()),
                "created_at": e.created_at,
                "deep": e.deep_version >= 2,
            }
            for e in self._entries.values()
        ]

    def count(self) -> int:
        """Number of references in the bank."""
        return len(self._entries)

    # ── Averaging (for gap analysis) ──

    def get_stem_averages(self) -> dict[str, StemFingerprint]:
        """Compute averaged fingerprints across all references per stem type.

        Returns:
            Dict mapping stem name (drums, bass, vocals, other)
            to averaged StemFingerprint.
        """
        if not self._entries:
            return {}

        # Collect all data per stem name
        stem_data: dict[str, list[dict[str, Any]]] = {}
        for entry in self._entries.values():
            for stem_name, fp in entry.stems.items():
                if stem_name not in stem_data:
                    stem_data[stem_name] = []
                stem_data[stem_name].append(fp)

        # Average each stem type
        averages: dict[str, StemFingerprint] = {}
        for stem_name, fps in stem_data.items():
            n = len(fps)
            if n == 0:
                continue

            avg_rms = sum(f.get("rms_db", -20) for f in fps) / n
            avg_peak = sum(f.get("peak_db", -6) for f in fps) / n
            avg_dr = sum(f.get("dynamic_range_db", 10) for f in fps) / n
            avg_energy = sum(f.get("energy_pct", 25) for f in fps) / n

            # Average freq bands
            avg_bands: dict[str, float] = {}
            all_band_keys = set()
            for f in fps:
                all_band_keys.update(f.get("freq_bands", {}).keys())
            for bk in all_band_keys:
                vals = [f.get("freq_bands", {}).get(bk, 33.3) for f in fps]
                avg_bands[bk] = round(sum(vals) / len(vals), 1)

            averages[stem_name] = StemFingerprint(
                rms_db=round(avg_rms, 1),
                peak_db=round(avg_peak, 1),
                dynamic_range_db=round(avg_dr, 1),
                energy_pct=round(avg_energy, 1),
                freq_bands=avg_bands,
            )

        return averages

    def get_master_averages(self) -> dict[str, float]:
        """Get averaged master-level stats across all references."""
        if not self._entries:
            return {}

        entries = list(self._entries.values())
        n = len(entries)
        return {
            "lufs": round(sum(e.lufs for e in entries) / n, 1),
            "bpm": round(sum(e.bpm for e in entries) / n, 1),
            "count": n,
        }

    def get_full_averages(self) -> dict[str, Any]:
        """Get complete averaged profile (stems + master)."""
        stem_avgs = self.get_stem_averages()
        master_avgs = self.get_master_averages()
        return {
            "stems": {k: asdict(v) for k, v in stem_avgs.items()},
            "master": master_avgs,
            "reference_count": self.count(),
        }

    # ── Deep DNA Profile (v2) ──

    def get_deep_profile(self) -> dict[str, Any]:
        """Get aggregated deep DNA profile across ALL references.

        Merges percussion palettes, instrument lists, FX lists, bass types,
        vocal effects, and arrangement patterns from all references into a
        single unified "style fingerprint" that Auralis uses for reconstruction.
        """
        if not self._entries:
            return {"empty": True, "message": "No references yet"}

        entries = list(self._entries.values())
        n = len(entries)
        deep_count = sum(1 for e in entries if e.deep_version >= 2)

        # ── Percussion: merge all hit counts ──
        merged_perc: dict[str, int] = {}
        total_density = 0.0
        perc_count = 0
        for e in entries:
            for label, count in e.percussion_palette.items():
                merged_perc[label] = merged_perc.get(label, 0) + count
            # Average density from stem data
            drums_fp = e.stems.get("drums", {})
            if drums_fp.get("percussion_density", 0):
                total_density += drums_fp["percussion_density"]
                perc_count += 1

        # Sort by frequency
        sorted_perc = dict(sorted(merged_perc.items(), key=lambda x: x[1], reverse=True))
        dominant_percussion = list(sorted_perc.keys())[:3] if sorted_perc else []

        # ── Bass: most common type ──
        bass_types: dict[str, int] = {}
        for e in entries:
            if e.bass_type:
                bass_types[e.bass_type] = bass_types.get(e.bass_type, 0) + 1
        dominant_bass = max(bass_types, key=bass_types.get) if bass_types else "Unknown"

        # ── Instruments: merge all palettes ──
        all_instruments: dict[str, int] = {}
        for e in entries:
            for inst in e.instrument_palette:
                all_instruments[inst] = all_instruments.get(inst, 0) + 1
        sorted_instruments = sorted(all_instruments.keys(),
                                     key=lambda x: all_instruments[x], reverse=True)

        # ── FX: merge all palettes ──
        all_fx: dict[str, int] = {}
        for e in entries:
            for fx in e.fx_palette:
                all_fx[fx] = all_fx.get(fx, 0) + 1
        sorted_fx = sorted(all_fx.keys(), key=lambda x: all_fx[x], reverse=True)

        # ── Vocal effects: most common ──
        all_vfx: dict[str, int] = {}
        for e in entries:
            for vfx in e.vocal_effects:
                all_vfx[vfx] = all_vfx.get(vfx, 0) + 1
        sorted_vfx = sorted(all_vfx.keys(), key=lambda x: all_vfx[x], reverse=True)

        # ── Arrangement: average section count, sidechain ratio ──
        section_counts = [len(e.arrangement_sections) for e in entries if e.arrangement_sections]
        sidechain_ratio = sum(1 for e in entries if e.sidechain_detected) / n if n else 0

        # ── Keys: most common ──
        keys: dict[str, int] = {}
        for e in entries:
            k = f"{e.key} {e.scale}"
            keys[k] = keys.get(k, 0) + 1
        dominant_key = max(keys, key=keys.get) if keys else "Unknown"

        return {
            "reference_count": n,
            "deep_count": deep_count,
            "master": self.get_master_averages(),
            "dominant_key": dominant_key,
            "percussion": {
                "palette": sorted_perc,
                "dominant": dominant_percussion,
                "avg_density": round(total_density / perc_count, 2) if perc_count else 0,
                "total_hits_across_refs": sum(sorted_perc.values()),
            },
            "bass": {
                "dominant_type": dominant_bass,
                "types_found": bass_types,
            },
            "instruments": {
                "palette": sorted_instruments,
                "frequency": all_instruments,
            },
            "fx": {
                "palette": sorted_fx,
                "frequency": all_fx,
            },
            "vocals": {
                "effects": sorted_vfx,
                "frequency": all_vfx,
            },
            "arrangement": {
                "avg_sections": round(sum(section_counts) / len(section_counts), 1) if section_counts else 0,
                "sidechain_ratio": round(sidechain_ratio, 2),
            },
        }
