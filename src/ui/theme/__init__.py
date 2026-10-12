"""Visual theme: tokens, fonts, icons and the application stylesheet."""

from pathlib import Path

from .tokens import DARK, LIGHT, MOTION, SPACE, TYPE, Palette, affiliation_colour

ASSETS = Path(__file__).resolve().parent.parent / "assets"

_current: Palette = DARK


def palette() -> Palette:
    """The active palette."""
    return _current


def set_palette(value: Palette) -> None:
    global _current
    _current = value


__all__ = [
    "ASSETS",
    "DARK",
    "LIGHT",
    "MOTION",
    "SPACE",
    "TYPE",
    "Palette",
    "affiliation_colour",
    "palette",
    "set_palette",
]
