"""AURALIS QC Convergence — Iterative refinement loop.

Run QC → identify weakest dimensions → adjust mastering → re-master → repeat.
Converges reconstruction quality toward target score.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from auralis.console.mastering import MasterConfig, MasterResult, master_audio
from auralis.qc.comparator import ComparisonResult, compare_full

logger = logging.getLogger(__name__)


@dataclass
class ConvergenceConfig:
    """Configuration for the convergence loop."""

    target_score: float = 90.0
    max_iterations: int = 5
    improvement_threshold: float = 0.5  # Minimum improvement per iteration
    save_intermediates: bool = True


@dataclass
class ConvergenceStep:
    """Result of a single convergence iteration."""

    iteration: int
    score: float
    weakest_dimension: str
    adjustment_made: str
    master_config: dict[str, Any]


@dataclass
class ConvergenceResult:
    """Full convergence loop result."""

    converged: bool
    final_score: float
    iterations_used: int
    max_iterations: int
    steps: list[ConvergenceStep] = field(default_factory=list)
    final_comparison: ComparisonResult | None = None
    output_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "converged": self.converged,
            "final_score": self.final_score,
            "iterations_used": self.iterations_used,
            "max_iterations": self.max_iterations,
            "steps": [
                {
                    "iteration": s.iteration,
                    "score": s.score,
                    "weakest": s.weakest_dimension,
                    "adjustment": s.adjustment_made,
                }
                for s in self.steps
            ],
            "output_path": self.output_path,
        }


# ── Adjustment Strategies ────────────────────────────────

_ADJUSTMENT_MAP: dict[str, Callable[[MasterConfig, float], tuple[MasterConfig, str]]] = {}


def _adjust_spectral(config: MasterConfig, score: float) -> tuple[MasterConfig, str]:
    """Adjust EQ to improve spectral match."""
    if config.custom_eq is None:
        config.custom_eq = {}
    # Reduce aggressive processing that distorts spectrum
    config.drive = max(1.0, config.drive - 0.2)
    return config, f"Reduced drive to {config.drive:.1f} for cleaner spectrum"


def _adjust_rms(config: MasterConfig, score: float) -> tuple[MasterConfig, str]:
    """Adjust target LUFS to match original RMS."""
    delta = (90 - score) / 30  # Scale adjustment to gap
    config.target_lufs -= delta
    return config, f"Adjusted target LUFS to {config.target_lufs:.1f}"


def _adjust_stereo(config: MasterConfig, score: float) -> tuple[MasterConfig, str]:
    """Adjust stereo width."""
    if score < 50:
        config.width = 1.0  # Reset to neutral
    else:
        config.width = max(0.8, config.width - 0.1)
    return config, f"Adjusted stereo width to {config.width:.1f}"


def _adjust_dynamics(config: MasterConfig, score: float) -> tuple[MasterConfig, str]:
    """Adjust dynamic range processing."""
    config.ceiling_db = min(-0.5, config.ceiling_db + 0.5)
    return config, f"Raised ceiling to {config.ceiling_db:.1f} dB for more dynamics"


def _adjust_bass(config: MasterConfig, score: float) -> tuple[MasterConfig, str]:
    """Adjust bass response."""
    if config.custom_eq is None:
        config.custom_eq = {}
    config.custom_eq["low_shelf_gain"] = config.custom_eq.get("low_shelf_gain", 0) + 1
    return config, f"Boosted low shelf +1 dB"


def _adjust_default(config: MasterConfig, score: float) -> tuple[MasterConfig, str]:
    """Default: reduce drive slightly."""
    config.drive = max(1.0, config.drive * 0.9)
    return config, f"Reduced overall drive to {config.drive:.2f}"


_ADJUSTMENT_MAP = {
    "spectral_similarity": _adjust_spectral,
    "rms_match": _adjust_rms,
    "stereo_width_match": _adjust_stereo,
    "dynamic_range": _adjust_dynamics,
    "bass_pattern_match": _adjust_bass,
    "reverb_match": _adjust_spectral,
    "timbre_similarity": _adjust_spectral,
}


# ── Main Convergence Loop ────────────────────────────────


def convergence_loop(
    original_path: str | Path,
    mix_path: str | Path,
    output_dir: str | Path,
    config: ConvergenceConfig | None = None,
    master_config: MasterConfig | None = None,
    on_progress: Callable[[int, float, str], None] | None = None,
) -> ConvergenceResult:
    """Run iterative convergence loop.

    Args:
        original_path: Path to original reference track
        mix_path: Path to mixed reconstruction (pre-master)
        output_dir: Directory for output files
        config: Convergence loop configuration
        master_config: Initial mastering configuration
        on_progress: Callback(iteration, score, message)

    Returns:
        ConvergenceResult with convergence history
    """
    cfg = config or ConvergenceConfig()
    m_cfg = master_config or MasterConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps: list[ConvergenceStep] = []
    current_path = str(mix_path)
    best_score = 0.0
    best_path = ""

    for i in range(cfg.max_iterations):
        # Master the mix
        iter_output = output_dir / f"master_iter_{i}.wav"
        try:
            result = master_audio(
                input_path=current_path,
                output_path=iter_output,
                config=m_cfg,
            )
            mastered_path = str(iter_output)
        except Exception as e:
            logger.error(f"Mastering failed at iteration {i}: {e}")
            mastered_path = current_path

        # Run QC comparison
        comparison = compare_full(
            original_path=original_path,
            reconstruction_path=mastered_path,
            target_score=cfg.target_score,
        )

        score = comparison.overall_score
        weakest = comparison.weakest

        if on_progress:
            on_progress(i, score, f"Iteration {i}: score={score:.1f}%, weakest={weakest}")

        logger.info(f"Convergence iteration {i}: score={score:.1f}%, weakest={weakest}")

        # Track best
        if score > best_score:
            best_score = score
            best_path = mastered_path

        # Make adjustment based on weakest dimension
        adjuster = _ADJUSTMENT_MAP.get(weakest, _adjust_default)
        m_cfg, adjustment_msg = adjuster(m_cfg, score)

        steps.append(ConvergenceStep(
            iteration=i,
            score=score,
            weakest_dimension=weakest,
            adjustment_made=adjustment_msg,
            master_config={"target_lufs": m_cfg.target_lufs, "drive": m_cfg.drive, "width": m_cfg.width},
        ))

        # Check convergence
        if score >= cfg.target_score:
            logger.info(f"Converged at iteration {i} with score {score:.1f}%")
            return ConvergenceResult(
                converged=True,
                final_score=score,
                iterations_used=i + 1,
                max_iterations=cfg.max_iterations,
                steps=steps,
                final_comparison=comparison,
                output_path=mastered_path,
            )

        # Check improvement stall
        if i > 0 and abs(score - steps[-2].score) < cfg.improvement_threshold:
            logger.info(f"Stalled at iteration {i} (improvement < {cfg.improvement_threshold}%)")
            break

        current_path = mastered_path

    # Return best result
    return ConvergenceResult(
        converged=False,
        final_score=best_score,
        iterations_used=len(steps),
        max_iterations=cfg.max_iterations,
        steps=steps,
        output_path=best_path,
    )
