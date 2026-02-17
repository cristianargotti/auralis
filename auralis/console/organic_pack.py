"""AURALIS Organic Pack — Intelligent sample selection from the arsenal.

Auto-classifies samples as organic (AI can't generate) vs synthetic
(AI handles well) based on metadata analysis. The brain queries this
at runtime to prefer real samples for organic textures.

Intelligence rules:
  - MIDI/presets → always synthetic (AI generates these)
  - Percussion with organic subcategory → organic
  - Vocals/chops → organic (unique human recordings)
  - Atmospheres → organic (recorded ambiences AI can't replicate)
  - Kicks/snares/claps/hats → synthetic (AI generates perfectly)
  - Bass/pad/lead/synth/arp → synthetic (AI generates perfectly)

Usage:
    pack = OrganicPack.load()
    sample = pack.find_best("percussion", "conga", bpm=128)
    if sample:
        audio = pack.load_audio(sample)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from auralis.config import settings

logger = structlog.get_logger()

# ── Path to the sample database ──
# Searches multiple locations so it works locally and on EC2/Docker
_DB_SEARCH_PATHS = [
    settings.samples_dir / "sample_database.json",        # EC2/Docker: /app/samples/
    Path("/Users/cristian.reyes/code/produce/tools/sample_database.json"),  # Local Mac
    Path(__file__).parent.parent.parent / "sample_database.json",          # Repo root
    Path("/data/samples/sample_database.json"),            # Docker volume mount
]


@dataclass
class OrganicSample:
    """A single organic sample with metadata."""

    id: int
    pack: str
    filename: str
    path: str
    category: str
    subcategory: str
    sample_type: str  # "loop" or "one-shot"
    description: str
    bpm: float | None = None
    key: str | None = None

    @property
    def exists(self) -> bool:
        return Path(self.path).exists()


@dataclass
class OrganicPack:
    """Intelligent organic sample library.

    Auto-classifies samples from the arsenal database.
    No hardcoded lists — classification is rule-based on audio metadata.
    """

    samples: list[OrganicSample] = field(default_factory=list)
    _index: dict[str, list[OrganicSample]] = field(
        default_factory=dict, repr=False
    )

    # ── Classification Intelligence ──

    @staticmethod
    def _ai_can_generate(category: str, subcategory: str, fmt: str) -> bool:
        """Return True if AI/synthesis handles this sound well.

        This is the core intelligence: it decides what to EXCLUDE.
        Everything else is considered organic and kept.
        """
        # MIDI presets = always synthetic
        if fmt == "midi" or category == "midi_preset":
            return True

        # Reference mixes / construction kits = not samples
        if category in ("reference", "construction_kit"):
            return True

        # Drum machine sounds AI generates perfectly
        if subcategory in (
            "kick", "snare", "clap", "hat", "hihat", "hi_hat",
            "ride", "crash", "cymbal",
        ):
            return True

        # Tonal elements AI generates perfectly
        if subcategory in (
            "bass", "bassline", "sub", "pad", "lead", "synth",
            "arp", "chord", "stab", "pluck", "keys", "piano",
        ):
            return True

        # MIDI versions of anything
        if subcategory.startswith("midi_"):
            return True

        # Category-level synthetic
        if category == "melodic" and subcategory in (
            "bass", "lead", "pad", "synth", "arp", "chord",
        ):
            return True

        return False

    @staticmethod
    def _classify_organic_type(sample: dict) -> str | None:
        """Classify an organic sample into a semantic type.

        Returns a normalized type or None if not classifiable.
        """
        subcat = sample.get("subcategory", "").lower()
        cat = sample.get("category", "").lower()
        fname = sample.get("filename", "").lower()
        desc = sample.get("description", "").lower()

        # Congas
        if "conga" in subcat or "conga" in fname:
            return "conga"

        # Bongos
        if "bongo" in subcat or "bongo" in fname:
            return "bongo"

        # Shakers
        if "shaker" in subcat or "shaker" in fname:
            return "shaker"

        # Toms (ethnic/organic)
        if "tom" in subcat or ("tom" in fname and "atom" not in fname):
            return "tom"

        # Bells
        if "bell" in subcat or "bell" in fname:
            return "bell"

        # Vocals / Chops
        if cat == "vocals" or subcat in ("vocal", "chop", "hook"):
            return "vocal"

        # Atmospheres
        if cat == "atmosphere" or subcat in ("atmosphere", "ambient", "texture"):
            return "atmosphere"

        # Top loops (layered percussion)
        if subcat == "top_loop":
            return "top_loop"

        # General percussion
        if cat == "percussion" or subcat == "percussion":
            return "percussion"

        # FX
        if cat == "fx" or subcat in ("misc_fx", "fx"):
            return "fx"

        # Kit elements (mixed organic sounds)
        if subcat == "kit_element":
            return "kit_element"

        return "other_organic"

    # ── Loading ──

    @classmethod
    def load(cls, db_path: str | Path | None = None) -> OrganicPack:
        """Load and auto-classify samples from the arsenal database."""
        path = None

        if db_path:
            path = Path(db_path)
        else:
            for candidate in _DB_SEARCH_PATHS:
                if candidate.exists():
                    path = candidate
                    break

        if not path or not path.exists():
            logger.warning("organic_pack.no_database", searched=str(_DB_SEARCH_PATHS))
            return cls()

        with open(path) as f:
            db = json.load(f)

        samples_raw = db.get("samples", [])
        organic: list[OrganicSample] = []

        for s in samples_raw:
            cat = s.get("category", "").lower()
            subcat = s.get("subcategory", "").lower()
            fmt = s.get("format", "")

            # Skip what AI handles
            if cls._ai_can_generate(cat, subcat, fmt):
                continue

            organic_type = cls._classify_organic_type(s)

            organic.append(OrganicSample(
                id=s.get("id", 0),
                pack=s.get("pack", ""),
                filename=s.get("filename", ""),
                path=s.get("path", ""),
                category=organic_type or subcat,
                subcategory=subcat,
                sample_type=s.get("type", "loop"),
                description=s.get("description", ""),
                bpm=s.get("bpm"),
                key=s.get("key"),
            ))

        # Build index by category
        index: dict[str, list[OrganicSample]] = {}
        for sample in organic:
            index.setdefault(sample.category, []).append(sample)

        pack = cls(samples=organic, _index=index)

        logger.info(
            "organic_pack.loaded",
            total=len(organic),
            categories=list(index.keys()),
            by_type={k: len(v) for k, v in index.items()},
        )

        return pack

    # ── Querying ──

    def find_best(
        self,
        category: str,
        subcategory: str | None = None,
        bpm: float | None = None,
        bpm_tolerance: float = 5.0,
        key: str | None = None,
    ) -> OrganicSample | None:
        """Find the best matching organic sample.

        Scoring:
          - Exact subcategory match: +50
          - BPM within tolerance: +30
          - Key match: +20
          - File exists on disk: required (skips missing)

        Returns the highest-scoring sample, or None.
        """
        candidates = self._index.get(category, [])

        if not candidates:
            # Try broader search across all categories
            for cat_name, cat_samples in self._index.items():
                if subcategory and subcategory in cat_name:
                    candidates = cat_samples
                    break

        if not candidates:
            return None

        scored: list[tuple[float, OrganicSample]] = []

        for sample in candidates:
            if not sample.exists:
                continue

            score = 0.0

            # Subcategory match
            if subcategory and subcategory in sample.subcategory:
                score += 50
            elif subcategory and subcategory in sample.filename.lower():
                score += 30

            # BPM proximity
            if bpm and sample.bpm:
                bpm_diff = abs(bpm - sample.bpm)
                if bpm_diff <= bpm_tolerance:
                    score += 30 * (1 - bpm_diff / bpm_tolerance)

            # Key match
            if key and sample.key:
                if key.lower() == sample.key.lower():
                    score += 20

            # Small random tie-breaker for variety
            score += random.random() * 2

            scored.append((score, sample))

        if not scored:
            return None

        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]

        logger.info(
            "organic_pack.found",
            category=category,
            subcategory=subcategory,
            selected=best.filename,
            score=scored[0][0],
            candidates=len(scored),
        )

        return best

    def find_all(
        self,
        category: str,
        limit: int = 10,
    ) -> list[OrganicSample]:
        """Get all samples of a category (for one-shot extraction)."""
        candidates = self._index.get(category, [])
        existing = [s for s in candidates if s.exists]
        return existing[:limit]

    def has_organic(self, category: str) -> bool:
        """Check if we have organic samples for this category."""
        return bool(self._index.get(category))

    # ── Audio Loading ──

    def load_audio(
        self, sample: OrganicSample, sr: int = 44100
    ) -> Any:
        """Load audio from an organic sample.

        Returns numpy array or None if loading fails.
        """
        import numpy as np
        import soundfile as sf

        try:
            data, file_sr = sf.read(sample.path, dtype="float64")

            # Convert to mono if needed
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            # Resample if needed
            if file_sr != sr:
                import librosa
                data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)

            return data

        except Exception as e:
            logger.warning(
                "organic_pack.load_failed",
                path=sample.path,
                error=str(e),
            )
            return None

    # ── Summary ──

    def summary(self) -> dict[str, Any]:
        """Return a summary for logging/display."""
        return {
            "total_organic": len(self.samples),
            "categories": {k: len(v) for k, v in self._index.items()},
            "packs": list(set(s.pack for s in self.samples)),
        }
