"""
B-scope: azimuth (deg) against range (km), the usual display of sector-scanning
fighter radars. Shows detection plots, tracks and (optionally) ground truth.
"""

from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from radarsim.physics.geometry import polar_from_enu
from radarsim.simulation.frame import FrameState

from ..i18n import tr
from ..theme import affiliation_colour, palette
from .base import ScopeView, pen, qcolor
from .ppi import PLOT_FADE_LEVELS, RANGE_PRESETS_KM, SYMBOLS


class BScopeView(ScopeView):
    title = "B-Scope"

    def __init__(self, parent=None):
        super().__init__(parent, axes=True)
        p = palette()
        self.max_range_km = 100.0
        self.show_truth = True
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setLabel("bottom", tr("Azimuth"), units="deg")
        self.plot.setLabel("left", tr("Range"), units="km")
        self._sector = self.plot.plot(pen=pen(p.text_faint, 1, style=Qt.PenStyle.DashLine), connect="finite")
        self._beam = pg.LinearRegionItem(orientation="vertical", movable=False,
                                         brush=pg.mkBrush(qcolor(p.sweep, 0.10)),
                                         pen=pen(p.sweep, 1, 0.6))
        self.plot.addItem(self._beam)
        self._brushes = [pg.mkBrush(qcolor(p.echo, (i + 1) / PLOT_FADE_LEVELS)) for i in range(PLOT_FADE_LEVELS)]
        self._plots = deque(maxlen=4000)
        self._plot_items = pg.ScatterPlotItem(size=5, pen=None)
        self.plot.addItem(self._plot_items)
        self._tracks = pg.ScatterPlotItem(size=9, symbol="s")
        self.plot.addItem(self._tracks)
        self._truth = pg.ScatterPlotItem(size=9)
        self.plot.addItem(self._truth)
        self._last_frame = None
        self._limits = (-60.0, 60.0)
        self._apply_range()

    def on_wheel(self, direction: int) -> None:
        index = min(range(len(RANGE_PRESETS_KM)), key=lambda i: abs(RANGE_PRESETS_KM[i] - self.max_range_km))
        self.max_range_km = RANGE_PRESETS_KM[max(0, min(len(RANGE_PRESETS_KM) - 1, index - direction))]
        self._apply_range()

    def set_max_range(self, km: float) -> None:
        self.max_range_km = float(km)
        self._apply_range()

    def _apply_range(self) -> None:
        self.view_box.setRange(xRange=self._limits, yRange=(0, self.max_range_km), padding=0.02)

    def reset(self) -> None:
        super().reset()
        self._plots.clear()
        self._last_frame = None

    def render(self, frame: FrameState, extrapolate_s: float) -> None:
        p = palette()
        radar = frame.radar
        if radar.scan_type == "sector":
            limits = tuple(np.degrees(radar.sector))
        else:
            limits = (-180.0, 180.0)
        if limits != self._limits:
            self._limits = limits
            self._apply_range()
        lo, hi = limits
        self._sector.setData([lo, lo, np.nan, hi, hi], [0, self.max_range_km, np.nan, 0, self.max_range_km])
        az = np.degrees((radar.antenna_azimuth + np.pi) % (2 * np.pi) - np.pi)
        half = np.degrees(radar.beamwidth_az) / 2
        self._beam.setRegion((az - half, az + half))

        origin = np.asarray(radar.position)
        if frame is not self._last_frame:
            for target in frame.targets:
                if target.measured_position is not None:
                    rng, a, _ = polar_from_enu(np.asarray(target.measured_position) - origin)
                    self._plots.append((np.degrees(a), rng / 1000, frame.time))
            self._last_frame = frame
        persistence = 4.0
        now = frame.time + extrapolate_s
        while self._plots and now - self._plots[0][2] > persistence:
            self._plots.popleft()
        if self._plots:
            data = np.array(self._plots)
            age = np.clip((now - data[:, 2]) / persistence, 0, 0.999)
            levels = (PLOT_FADE_LEVELS - 1 - (age * PLOT_FADE_LEVELS).astype(int)).clip(0)
            self._plot_items.setData(data[:, 0], data[:, 1], brush=[self._brushes[i] for i in levels])
        else:
            self._plot_items.setData([], [])

        tx, ty = [], []
        for track in frame.tracks:
            position = np.array([track.position[0], track.position[1], origin[2]])
            rng, a, _ = polar_from_enu(position - origin)
            tx.append(np.degrees(a))
            ty.append(rng / 1000)
        self._tracks.setData(tx, ty, brush=pg.mkBrush(qcolor(p.text, 0.85)), pen=pen(p.text))

        xs, ys, symbols, pens = [], [], [], []
        if self.show_truth:
            for target in frame.targets:
                xs.append(np.degrees(target.azimuth))
                ys.append(target.range_m / 1000)
                symbols.append(SYMBOLS.get(target.affiliation, "p"))
                pens.append(pen(affiliation_colour(p, target.affiliation, target.target_type), 1.4))
        self._truth.setData(xs, ys, symbol=symbols, pen=pens, brush=None)
