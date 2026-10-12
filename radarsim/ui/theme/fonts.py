"""Registers the bundled IBM Plex fonts (SIL OFL 1.1) with Qt."""

import logging

from PySide6.QtGui import QFont, QFontDatabase

from . import ASSETS, TYPE

logger = logging.getLogger(__name__)

_loaded = False


def load_fonts() -> None:
    """Register bundled fonts once; Qt falls back to system fonts if this fails."""
    global _loaded
    if _loaded:
        return
    for path in sorted((ASSETS / "fonts").glob("*.ttf")):
        if QFontDatabase.addApplicationFont(str(path)) < 0:
            logger.warning("Could not load font %s", path.name)
    _loaded = True


def ui_font(size_px: int = TYPE.size_body, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont(TYPE.sans)
    font.setPixelSize(size_px)
    font.setWeight(weight)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return font


def mono_font(size_px: int = TYPE.size_body, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    font = QFont(TYPE.mono)
    font.setStyleHint(QFont.StyleHint.Monospace)
    font.setPixelSize(size_px)
    font.setWeight(weight)
    return font
