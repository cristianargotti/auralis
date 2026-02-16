"""QC — Quality Assurance layer.

Modules:
  comparator: 12-dimension A/B audio comparison
  convergence: Iterative mastering refinement loop

Imports are lazy to avoid requiring all dependencies at import time.
"""


def compare_full(*args, **kwargs):
    """Run full 12-dimension A/B comparison."""
    from auralis.qc.comparator import compare_full as _compare_full
    return _compare_full(*args, **kwargs)


def compare_quick(*args, **kwargs):
    """Quick comparison — returns just scores."""
    from auralis.qc.comparator import compare_quick as _compare_quick
    return _compare_quick(*args, **kwargs)


def convergence_loop(*args, **kwargs):
    """Run iterative convergence loop."""
    from auralis.qc.convergence import convergence_loop as _convergence_loop
    return _convergence_loop(*args, **kwargs)


__all__ = [
    "compare_full",
    "compare_quick",
    "convergence_loop",
]
