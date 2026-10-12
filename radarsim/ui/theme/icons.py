"""SVG line icons recoloured to the active palette."""

from functools import cache

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap
from PySide6.QtSvg import QSvgRenderer

from . import ASSETS, palette


@cache
def _svg(name: str) -> str:
    return (ASSETS / "icons" / f"{name}.svg").read_text(encoding="utf-8")


def _pixmap(name: str, colour: str, size: int) -> QPixmap:
    data = QByteArray(_svg(name).replace("currentColor", colour).encode("utf-8"))
    renderer = QSvgRenderer(data)
    scale = 2  # render at 2x so icons stay crisp on high-DPI screens
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    renderer.render(painter, QRectF(0, 0, size * scale, size * scale))
    painter.end()
    pixmap.setDevicePixelRatio(scale)
    return pixmap


def icon(name: str, size: int = 18, colour: str = None) -> QIcon:
    """Icon with normal, disabled and active (accent) states."""
    p = palette()
    result = QIcon()
    result.addPixmap(_pixmap(name, colour or p.text_muted, size), QIcon.Mode.Normal)
    result.addPixmap(_pixmap(name, p.text_faint, size), QIcon.Mode.Disabled)
    result.addPixmap(_pixmap(name, p.text, size), QIcon.Mode.Active)
    result.addPixmap(_pixmap(name, p.accent, size), QIcon.Mode.Normal, QIcon.State.On)
    return result
