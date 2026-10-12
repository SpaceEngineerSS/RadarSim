"""
Range-height indicator along one azimuth.

Shows the terrain profile (with the 4/3-Earth bulge applied relative to the
radar), the radar horizon for the antenna height, the elevation beam, and the
targets within a few beamwidths of the chosen azimuth. The azimuth follows the
selected target, or the antenna when nothing is selected.
"""

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt

from radarsim.physics.constants import EARTH_RADIUS_EFFECTIVE
from radarsim.physics.geometry import wrap_to_pi
from radarsim.simulation.frame import FrameState

from ..i18n import tr
from ..theme import affiliation_colour, palette
from .base import ScopeView, pen, qcolor
from .ppi import SYMBOLS

PROFILE_POINTS = 160


class RHIView(ScopeView):
    title = "RHI"

    def __init__(self, parent=None):
        super().__init__(parent, axes=True)
        p = palette()
        self.terrain = None
        self.max_range_km = 100.0
        self.plot.setLabel("bottom", tr("Ground range"), units="km")
        self.plot.setLabel("left", tr("Height"), units="km")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self._ground = pg.PlotDataItem(pen=pen(p.text_faint, 1), fillLevel=-1.0,
                                       brush=pg.mkBrush(qcolor(p.raised, 1.0)))
        self.plot.addItem(self._ground)
        self._horizon = self.plot.plot(pen=pen(p.text_faint, 1, style=Qt.PenStyle.DashLine))
        self._beam = self.plot.plot(pen=pen(p.sweep, 1, 0.5), connect="finite")
        self._targets = pg.ScatterPlotItem(size=9)
        self.plot.addItem(self._targets)
        self._info = self.label("", p.scope_label, 10)
        self._profile_key = None

    def set_max_range(self, km: float) -> None:
        self.max_range_km = float(km)
        self._profile_key = None

    def set_terrain(self, terrain) -> None:
        self.terrain = terrain
        self._profile_key = None

    def reset(self) -> None:
        super().reset()
        self._profile_key = None

    def render(self, frame: FrameState, extrapolate_s: float) -> None:
        p = palette()
        radar = frame.radar
        selected = frame.target(self.selected_id) if self.selected_id is not None else None
        azimuth = selected.azimuth if selected is not None else radar.antenna_azimuth
        r_max = self.max_range_km
        rx, ry, rz = radar.position

        key = (round(np.degrees(azimuth)), r_max, self.terrain is not None, rx, ry)
        if key != self._profile_key:
            ground = np.linspace(0.0, r_max * 1000, PROFILE_POINTS)
            heights = np.zeros_like(ground)
            if self.terrain is not None:
                heights = np.array(
                    [
                        self.terrain.get_elevation(rx + d * np.sin(azimuth), ry + d * np.cos(azimuth))
                        for d in ground
                    ]
                )
            bulge = ground**2 / (2 * EARTH_RADIUS_EFFECTIVE)
            self._ground.setData(ground / 1000, (heights - bulge) / 1000)
            # Heights are drawn relative to the radar's tangent plane, so a
            # horizontal ray stays at antenna height while the ground curves away.
            self._horizon.setData([0, r_max], [rz / 1000, rz / 1000])
            self._profile_key = key
            top = max(15.0, rz / 1000 + 2)
            self.view_box.setRange(xRange=(0, r_max), yRange=(-1.0, top), padding=0)

        half = radar.beamwidth_el / 2
        centre = selected.elevation if selected is not None else 0.0
        elevations = [centre + half, centre - half]
        bx, by = [], []
        for el in elevations:
            bx += [0, r_max, np.nan]
            by += [rz / 1000, rz / 1000 + r_max * np.tan(el), np.nan]
        self._beam.setData(bx, by)

        xs, ys, symbols, pens = [], [], [], []
        width = 3 * radar.beamwidth_az + np.radians(2)
        for target in frame.targets:
            if abs(wrap_to_pi(target.azimuth - azimuth)) > width and target.target_id != self.selected_id:
                continue
            ground_range = np.hypot(target.position[0] - rx, target.position[1] - ry)
            xs.append(ground_range / 1000)
            ys.append((target.altitude - ground_range**2 / (2 * EARTH_RADIUS_EFFECTIVE)) / 1000)
            symbols.append(SYMBOLS.get(target.affiliation, "p"))
            colour = p.accent if target.target_id == self.selected_id else affiliation_colour(p, target.affiliation, target.target_type)
            pens.append(pen(colour, 1.4, 0.5 if target.masked else 1.0))
        self._targets.setData(xs, ys, symbol=symbols, pen=pens, brush=None)
        self._info.setPos(0.02 * r_max, 0.0)
        self._info.setText(f"AZ {np.degrees(azimuth) % 360:05.1f}")
