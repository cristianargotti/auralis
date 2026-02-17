"""AURALIS Reference DNA Bank — Professional track intelligence database.

Stores DNA fingerprints from reference tracks. When reconstructing a user's
track, the system compares stem-by-stem against averaged reference profiles
to identify gaps and drive intelligent processing decisions.

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
    """Spectral + dynamic fingerprint of a single stem."""

    rms_db: float = -20.0
    peak_db: float = -6.0
    dynamic_range_db: float = 10.0
    energy_pct: float = 25.0
    freq_bands: dict[str, float] = field(default_factory=lambda: {
        "low": 33.3, "mid": 33.3, "high": 33.3,
    })
    spectral_centroid_avg: float = 2000.0
    instrument_types: list[str] = field(default_factory=list)


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
            "version": 1,
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
    ) -> ReferenceEntry:
        """Add a processed track to the reference bank.

        Args:
            track_id: Unique identifier (job_id or project_id)
            name: Human-readable track name
            ear_analysis: Full EAR analysis data (BPM, key, LUFS, etc.)
            stem_analysis: Per-stem analysis from EAR stage
        """
        # Build stem fingerprints from analysis
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

        # Build spectral summary (average across all stems)
        spectral_summary: dict[str, float] = {}
        if stem_fps:
            all_bands = [s.get("freq_bands", {}) for s in stem_fps.values()]
            for band_key in ("low", "mid", "high"):
                vals = [b.get(band_key, 33.3) for b in all_bands]
                spectral_summary[band_key] = round(sum(vals) / len(vals), 1)

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
        )

        self._entries[track_id] = entry
        self._save()
        logger.info(
            "reference_bank.added",
            track_id=track_id,
            name=name,
            stems=list(stem_fps.keys()),
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
