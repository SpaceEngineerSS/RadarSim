"""Shared plumbing for the scope views."""

from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QVBoxLayout, QWidget

from radarsim.simulation.frame import FrameState

from ..theme import palette
from ..theme.fonts import mono_font

NICE_STEPS = (1.0, 2.0, 2.5, 5.0)


def nice_step(span: float, divisions: int = 4) -> float:
    """Round ``span / divisions`` to 1, 2, 2.5 or 5 times a power of ten."""
    raw = max(span / divisions, 1e-9)
    exponent = np.floor(np.log10(raw))
    for mantissa in NICE_STEPS + (10.0,):
        step = mantissa * 10**exponent
        if step >= raw:
            return float(step)
    return float(10 ** (exponent + 1))


def qcolor(hex_colour: str, alpha: float = 1.0) -> QColor:
    colour = QColor(hex_colour)
    colour.setAlphaF(max(0.0, min(1.0, alpha)))
    return colour


def pen(hex_colour: str, width: float = 1.0, alpha: float = 1.0, style=Qt.PenStyle.SolidLine) -> QPen:
    result = QPen(qcolor(hex_colour, alpha))
    result.setWidthF(width)
    result.setCosmetic(True)
    result.setStyle(style)
    return result


class ScopeViewBox(pg.ViewBox):
    """ViewBox with mouse pan/zoom disabled; wheel steps are forwarded instead."""

    def __init__(self, on_wheel: Optional[Callable[[int], None]] = None):
        super().__init__(enableMenu=False)
        self.setMouseEnabled(False, False)
        self._on_wheel = on_wheel

    def wheelEvent(self, event, axis=None):
        if self._on_wheel is not None:
            self._on_wheel(1 if event.delta() > 0 else -1)
            event.accept()
        else:
            event.ignore()


class ScopeView(QWidget):
    """Base class: a pyqtgraph plot styled for the scope palette."""

    target_clicked = Signal(int)

    title = "Scope"

    def __init__(self, parent: Optional[QWidget] = None, axes: bool = False):
        super().__init__(parent)
        p = palette()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view_box = ScopeViewBox(self.on_wheel)
        self.plot = pg.PlotWidget(viewBox=self.view_box, background=p.scope_background)
        self.plot.hideButtons()
        self.plot.setMenuEnabled(False)
        if axes:
            for name in ("left", "bottom"):
                axis = self.plot.getAxis(name)
                axis.setPen(pen(p.scope_grid_major))
                axis.setTextPen(pen(p.scope_label))
                axis.setTickFont(mono_font(10))
        else:
            self.plot.hideAxis("left")
            self.plot.hideAxis("bottom")
        layout.addWidget(self.plot)
        self.selected_id: Optional[int] = None

    # Subclasses implement these
    def render(self, frame: FrameState, extrapolate_s: float) -> None:  # noqa: D401
        raise NotImplementedError

    def reset(self) -> None:
        self.selected_id = None

    def on_wheel(self, direction: int) -> None:
        pass

    def set_selected(self, target_id: Optional[int]) -> None:
        self.selected_id = target_id

    def label(self, text: str = "", colour: Optional[str] = None, size: int = 10) -> pg.TextItem:
        item = pg.TextItem(text, color=colour or palette().scope_label, anchor=(0, 0.5))
        item.setFont(mono_font(size))
        self.plot.addItem(item)
        return item
