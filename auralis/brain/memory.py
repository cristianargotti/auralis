"""AURALIS Session Memory — remembers what worked, learns from QC scores.

Provides cross-session learning so the DNABrain can:
  1. Track which candidate choices led to high QC scores
  2. Bias future selections based on proven successes
  3. Build long-term preference profiles per genre/style

Storage: JSON file at ~/.auralis/session_memory.json
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# ── Constants ──────────────────────────────────────────────

MEMORY_DIR = Path.home() / ".auralis"
MEMORY_FILE = MEMORY_DIR / "session_memory.json"
MAX_SESSIONS = 100  # Keep last N sessions to limit file size


# ── Data Structures ───────────────────────────────────────


@dataclass
class StemChoice:
    """A recorded choice for one stem in a session."""

    stem_name: str
    style: str = ""
    patch: str = ""
    fx_chain: list[str] = field(default_factory=list)
    qc_score: float = 0.0  # 0-100 quality score from QC pass
    was_kept: bool = True   # True if user kept this choice


@dataclass
class SessionRecord:
    """One production session's choices and outcomes."""

    session_id: str = ""
    timestamp: float = 0.0
    bpm: float = 120.0
    key: str = "C"
    scale: str = "minor"
    genre_hints: list[str] = field(default_factory=list)
    stem_choices: list[StemChoice] = field(default_factory=list)
    overall_qc: float = 0.0      # Overall QC score (0-100)
    mastering_lufs: float = -14.0


# ── Session Memory ────────────────────────────────────────


class SessionMemory:
    """Persistent memory across production sessions.

    Tracks which choices led to high QC scores and provides
    a 'preference bonus' to bias future candidate selection.
    """

    def __init__(self) -> None:
        self._sessions: list[SessionRecord] = []
        self._load()

    def _load(self) -> None:
        """Load from disk if available."""
        if MEMORY_FILE.exists():
            try:
                data = json.loads(MEMORY_FILE.read_text())
                for entry in data.get("sessions", []):
                    choices = [
                        StemChoice(**c) for c in entry.pop("stem_choices", [])
                    ]
                    self._sessions.append(
                        SessionRecord(**entry, stem_choices=choices)
                    )
                logger.info(
                    "session_memory.loaded",
                    sessions=len(self._sessions),
                )
            except Exception as e:
                logger.warning("session_memory.load_error", error=str(e))
                self._sessions = []

    def save(self) -> None:
        """Persist to disk."""
        try:
            MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "sessions": [asdict(s) for s in self._sessions[-MAX_SESSIONS:]]
            }
            MEMORY_FILE.write_text(json.dumps(data, indent=2))
            logger.info(
                "session_memory.saved",
                sessions=len(self._sessions),
            )
        except Exception as e:
            logger.warning("session_memory.save_error", error=str(e))

    def record_session(
        self,
        bpm: float,
        key: str,
        scale: str,
        stem_choices: list[dict[str, Any]],
        overall_qc: float = 0.0,
        genre_hints: list[str] | None = None,
        mastering_lufs: float = -14.0,
    ) -> None:
        """Record a completed session with outcomes."""
        choices = []
        for c in stem_choices:
            choices.append(StemChoice(
                stem_name=c.get("stem_name", ""),
                style=c.get("style", ""),
                patch=c.get("patch", ""),
                fx_chain=c.get("fx_chain", []),
                qc_score=c.get("qc_score", 0.0),
                was_kept=c.get("was_kept", True),
            ))

        session = SessionRecord(
            session_id=f"session_{int(time.time())}",
            timestamp=time.time(),
            bpm=bpm,
            key=key,
            scale=scale,
            genre_hints=genre_hints or [],
            stem_choices=choices,
            overall_qc=overall_qc,
            mastering_lufs=mastering_lufs,
        )
        self._sessions.append(session)
        self.save()

    def preference_bonus(
        self,
        stem_name: str,
        candidate_name: str,
        bpm: float = 120.0,
        min_qc: float = 70.0,
    ) -> float:
        """Return 0-20 bonus for a candidate based on past Success.

        Looks at past sessions where the same candidate was chosen for
        this stem and achieved high QC scores. Higher bonus = more
        proven history of success.

        Args:
            stem_name: "drums", "bass", etc.
            candidate_name: The style/patch name being scored.
            bpm: Current BPM (similar BPMs get more weight).
            min_qc: Minimum QC score to count as "successful".

        Returns:
            0.0-20.0 preference bonus.
        """
        if not self._sessions:
            return 0.0

        successes = 0
        weighted_score = 0.0

        for session in self._sessions[-50:]:  # Last 50 sessions
            for choice in session.stem_choices:
                if choice.stem_name != stem_name:
                    continue
                if choice.style != candidate_name and choice.patch != candidate_name:
                    continue
                if not choice.was_kept:
                    continue
                if choice.qc_score < min_qc:
                    continue

                # BPM proximity weight (closer BPM = more relevant)
                bpm_diff = abs(session.bpm - bpm)
                bpm_weight = max(0.2, 1.0 - bpm_diff / 40.0)

                successes += 1
                weighted_score += (choice.qc_score / 100) * bpm_weight

        if successes == 0:
            return 0.0

        # Scale to 0-20 range, with diminishing returns
        avg_score = weighted_score / successes
        count_factor = min(1.0, successes / 5)  # Max bonus after 5 successes
        return round(avg_score * count_factor * 20.0, 1)

    @property
    def session_count(self) -> int:
        """Number of recorded sessions."""
        return len(self._sessions)

    def summary(self) -> dict[str, Any]:
        """Brief summary of memory state."""
        if not self._sessions:
            return {"sessions": 0, "avg_qc": 0.0}

        qc_scores = [s.overall_qc for s in self._sessions if s.overall_qc > 0]
        return {
            "sessions": len(self._sessions),
            "avg_qc": round(sum(qc_scores) / max(len(qc_scores), 1), 1),
            "latest_qc": self._sessions[-1].overall_qc if self._sessions else 0.0,
        }
