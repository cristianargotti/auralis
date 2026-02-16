"""Mono Aullador — Track reconstruction & benchmark module.

This module contains:
- million_pieces.py: Complete DNA blueprint of the reference track
- Future: reconstruction engine, A/B comparison tools
"""

from auralis.mono_aullador.million_pieces import (
    MILLION_PIECES_SECTIONS,
    get_million_pieces_blueprint,
    get_reconstruction_config,
)

__all__ = [
    "MILLION_PIECES_SECTIONS",
    "get_million_pieces_blueprint",
    "get_reconstruction_config",
]
