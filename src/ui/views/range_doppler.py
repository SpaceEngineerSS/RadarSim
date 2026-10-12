"""
Range-Doppler map from the coherent pulse-Doppler processor.

The map is the processor's real output (matched filter, MTI, windowed Doppler
FFT). When processing is off the view says so instead of showing an invented
picture; it can be switched on from here.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from src.simulation.frame import FrameState

from ..i18n import tr
from ..theme import SPACE, palette
from .base import ScopeView, pen


def _colormap() -> pg.ColorMap:
    p = palette()
    stops = [0.0, 0.35, 0.7, 1.0]
    colours = [p.scope_background, "#1d3a3d", p.sweep, "#f2efe2"]
    return pg.ColorMap(stops, [pg.mkColor(c) for c in colours])


class RangeDopplerView(ScopeView):
    title = "Range–Doppler"

    enable_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, axes=True)
        p = palette()
        self.plot.setLabel("bottom", tr("Range"), units="km")
        self.plot.setLabel("left", tr("Radial velocity"), units="m/s")
        self._image = pg.ImageItem()
        self._image.setLookupTable(_colormap().getLookupTable(nPts=256))
        self.plot.addItem(self._image)
        self._blind = self.plot.plot(pen=pen(p.threshold, 1, 0.7, Qt.PenStyle.DashLine), connect="finite")

        self._placeholder = QWidget(self)
        box = QVBoxLayout(self._placeholder)
        box.setContentsMargins(SPACE.xl, SPACE.xl, SPACE.xl, SPACE.xl)
        box.addStretch()
        message = QLabel(tr("Pulse-Doppler processing is off. Enabling it runs a coherent processing interval every few frames."))
        message.setWordWrap(True)
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setProperty("role", "caption")
        box.addWidget(message)
        row = QHBoxLayout()
        row.addStretch()
        button = QPushButton(tr("Enable pulse-Doppler"))
        button.clicked.connect(self.enable_requested)
        row.addWidget(button)
        row.addStretch()
        box.addLayout(row)
        box.addStretch()
        self.layout().addWidget(self._placeholder)
        self._shape = None

    def render(self, frame: FrameState, extrapolate_s: float) -> None:
        rd = frame.rd_map
        processing = frame.settings.get("pulse_doppler", False)
        self._placeholder.setVisible(not processing)
        self.plot.setVisible(processing)
        if rd is None:
            return
        data = rd.data_db - np.median(rd.data_db)
        self._image.setImage(data.T, autoLevels=False, levels=(0.0, 45.0))
        r0, r1 = rd.range_axis_m[0] / 1000, rd.range_axis_m[-1] / 1000
        v0, v1 = rd.velocity_axis_mps[0], rd.velocity_axis_mps[-1]
        self._image.setRect(r0, v0, r1 - r0, v1 - v0)
        shape = (r0, r1, v0, v1)
        if shape != self._shape:
            self.view_box.setRange(xRange=(r0, r1), yRange=(v0, v1), padding=0)
            self._shape = shape
        xs, ys = [], []
        for v in rd.blind_speeds_mps:
            for sign in (-1, 1):
                if v0 <= sign * v <= v1:
                    xs += [r0, r1, np.nan]
                    ys += [sign * v, sign * v, np.nan]
        self._blind.setData(xs, ys)
