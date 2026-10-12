"""
3D tactical view (OpenGL).

Loaded only when the tab is first opened, and replaced by a short message if
OpenGL is not available on the machine, so the rest of the application never
depends on a GPU.
"""

from typing import Optional

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.simulation.frame import FrameState

from ..i18n import tr
from ..theme import affiliation_colour, palette

TERRAIN_GRID = 60  # vertices per side; small enough for integrated GPUs


def _rgba(hex_colour: str, alpha: float = 1.0):
    c = hex_colour.lstrip("#")
    return tuple(int(c[i : i + 2], 16) / 255 for i in (0, 2, 4)) + (alpha,)


class Tactical3DView(QWidget):
    title = "3D"

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._gl = None
        self._failed = False
        self.terrain = None
        self.max_range_km = 100.0
        self.selected_id = None
        self._terrain_item = None

    def _ensure_gl(self) -> bool:
        if self._gl is not None or self._failed:
            return self._gl is not None
        try:
            import pyqtgraph.opengl as gl

            view = gl.GLViewWidget()
            view.setBackgroundColor(palette().scope_background)
            view.setCameraPosition(distance=self.max_range_km * 2.2, elevation=28, azimuth=-110)
            grid = gl.GLGridItem()
            grid.setSize(self.max_range_km * 2, self.max_range_km * 2)
            grid.setSpacing(self.max_range_km / 5, self.max_range_km / 5)
            grid.setColor(_rgba(palette().scope_grid_major, 0.8))
            view.addItem(grid)
            self._targets = gl.GLScatterPlotItem(pxMode=True, size=8)
            view.addItem(self._targets)
            self._tracks = gl.GLScatterPlotItem(pxMode=True, size=6)
            view.addItem(self._tracks)
            self._drop_lines = gl.GLLinePlotItem(mode="lines", antialias=True)
            view.addItem(self._drop_lines)
            self._gl_module = gl
            self._gl = view
            self._layout.addWidget(view)
            if self.terrain is not None:
                self._build_terrain()
        except Exception:  # no OpenGL context / driver
            self._failed = True
            message = QLabel(tr("3D view needs OpenGL, which is not available on this machine."))
            message.setAlignment(Qt.AlignmentFlag.AlignCenter)
            message.setProperty("role", "caption")
            self._layout.addWidget(message)
        return self._gl is not None

    def set_terrain(self, terrain) -> None:
        self.terrain = terrain
        if self._gl is not None:
            self._build_terrain()

    def set_max_range(self, km: float) -> None:
        self.max_range_km = float(km)

    def set_selected(self, target_id) -> None:
        self.selected_id = target_id

    def reset(self) -> None:
        self.selected_id = None

    def _build_terrain(self) -> None:
        gl = self._gl_module
        if self._terrain_item is not None:
            self._gl.removeItem(self._terrain_item)
            self._terrain_item = None
        if self.terrain is None:
            return
        extent = self.max_range_km
        axis = np.linspace(-extent, extent, TERRAIN_GRID)
        heights = np.array(
            [[self.terrain.get_elevation(x * 1000, y * 1000) / 1000 for y in axis] for x in axis]
        )
        colours = np.empty(heights.shape + (4,))
        top = max(heights.max(), 1e-3)
        shade = 0.18 + 0.35 * (heights / top)
        colours[..., 0], colours[..., 1], colours[..., 2], colours[..., 3] = shade, shade * 1.05, shade * 0.95, 1.0
        self._terrain_item = gl.GLSurfacePlotItem(x=axis, y=axis, z=heights, colors=colours, shader=None)
        self._gl.addItem(self._terrain_item)

    def render(self, frame: FrameState, extrapolate_s: float) -> None:
        if not self.isVisible() or not self._ensure_gl():
            return
        p = palette()
        if frame.targets:
            positions = np.array(
                [
                    [(t.position[i] + t.velocity[i] * extrapolate_s) / 1000 for i in range(3)]
                    for t in frame.targets
                ]
            )
            colours = np.array(
                [
                    _rgba(p.accent if t.target_id == self.selected_id else affiliation_colour(p, t.affiliation, t.target_type))
                    for t in frame.targets
                ]
            )
            self._targets.setData(pos=positions, color=colours, size=9)
            drops = np.repeat(positions, 2, axis=0)
            drops[1::2, 2] = 0.0
            self._drop_lines.setData(pos=drops, color=_rgba(p.text_faint, 0.6), width=1)
        else:
            self._targets.setData(pos=np.zeros((0, 3)))
            self._drop_lines.setData(pos=np.zeros((0, 3)))
        if frame.tracks:
            z = frame.radar.position[2] / 1000
            tracks = np.array([[t.position[0] / 1000, t.position[1] / 1000, z] for t in frame.tracks])
            self._tracks.setData(pos=tracks, color=_rgba(p.text, 0.9))
        else:
            self._tracks.setData(pos=np.zeros((0, 3)))
