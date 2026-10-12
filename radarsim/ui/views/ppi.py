"""
Plan position indicator.

North-up polar display centred on the radar (km). Layers, bottom to top:
range/bearing grid, sector limits, beam afterglow and sweep line, jamming
strobes, detection plots (fading over one revisit period), track symbols with
history and velocity leaders, ground truth (optional), selection ring.
Positions are extrapolated between simulation frames so motion stays smooth
at the display refresh rate.
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QPainterPath, QPolygonF
from PySide6.QtWidgets import QGraphicsPathItem

from radarsim.simulation.frame import FrameState, TargetFrame

from ..theme import affiliation_colour, palette
from .base import ScopeView, nice_step, pen, qcolor

RANGE_PRESETS_KM = (5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600)
PLOT_FADE_LEVELS = 8
LEADER_SECONDS = 60.0  # velocity leader length in time
SYMBOLS = {"friend": "o", "hostile": "d", "neutral": "s", "unknown": "p"}


def _xy_km(position) -> Tuple[float, float]:
    return position[0] / 1000.0, position[1] / 1000.0


class PPIView(ScopeView):
    title = "PPI"

    point_picked = Signal(float, float)  # metres, editor placement
    vector_dragged = Signal(float, float, float, float)  # metres, editor velocity

    def __init__(self, parent=None):
        super().__init__(parent)
        p = palette()
        self.plot.setAspectLocked(True)
        self.max_range_km = 100.0
        self.show_truth = True
        self.show_ellipses = True
        self.afterglow = True
        self.interaction = "select"  # or "place" / "vector" (scenario editor)
        self._radar_xy = (0.0, 0.0)
        self._plots: Deque[Tuple[float, float, float]] = deque(maxlen=4000)
        self._persistence_s = 10.0
        self._drag_start: Optional[Tuple[float, float]] = None

        self._grid = self.plot.plot(pen=pen(p.scope_grid, 1.0), connect="finite")
        self._grid_major = self.plot.plot(pen=pen(p.scope_grid_major, 1.0), connect="finite")
        self._sector = self.plot.plot(pen=pen(p.text_faint, 1.0, style=Qt.PenStyle.DashLine), connect="finite")
        self._glow = QGraphicsPathItem()
        self._glow.setPen(pen(p.sweep, 0.0, 0.0))
        self.plot.addItem(self._glow)
        self._strobes = QGraphicsPathItem()
        self._strobes.setPen(pen(p.hostile, 0.0, 0.0))
        self._strobes.setBrush(QBrush(qcolor(p.hostile, 0.10)))
        self.plot.addItem(self._strobes)
        self._sweep = self.plot.plot(pen=pen(p.sweep, 1.5, 0.85))
        self._plot_brushes = [
            pg.mkBrush(qcolor(p.echo, (i + 1) / PLOT_FADE_LEVELS)) for i in range(PLOT_FADE_LEVELS)
        ]
        self._plot_items = pg.ScatterPlotItem(size=4, pen=None, pxMode=True)
        self.plot.addItem(self._plot_items)
        self._false = pg.ScatterPlotItem(size=7, symbol="x", pen=pen(p.expendable, 1.2), brush=None)
        self.plot.addItem(self._false)

        self._track_history = pg.ScatterPlotItem(size=2.5, pen=None, brush=pg.mkBrush(qcolor(p.text_muted, 0.55)))
        self.plot.addItem(self._track_history)
        self._track_leaders = self.plot.plot(pen=pen(p.text_muted, 1.0), connect="finite")
        self._track_ellipses = self.plot.plot(pen=pen(p.text_faint, 1.0, 0.8), connect="finite")
        self._tracks = pg.ScatterPlotItem(size=9, symbol="s", pxMode=True)
        self.plot.addItem(self._tracks)

        self._truth_leaders = self.plot.plot(pen=pen(p.text_faint, 1.0, 0.9), connect="finite")
        self._truth = pg.ScatterPlotItem(size=9, pxMode=True)
        self.plot.addItem(self._truth)
        self._selection = pg.ScatterPlotItem(size=22, symbol="o", pen=pen(p.accent, 1.5), brush=None)
        self.plot.addItem(self._selection)

        self._range_labels: List[pg.TextItem] = []
        self._bearing_labels: List[pg.TextItem] = []
        for bearing in range(0, 360, 30):
            item = self.label(f"{bearing:03d}")
            item.setAnchor((0.5, 0.5))
            self._bearing_labels.append(item)
        self._name_labels: Dict[int, pg.TextItem] = {}
        self._track_labels: Dict[int, pg.TextItem] = {}

        self.plot.scene().sigMouseClicked.connect(self._on_click)
        self._last_frame: Optional[FrameState] = None
        self._rebuild_grid()

    # ----------------------------------------------------------------- zoom

    def on_wheel(self, direction: int) -> None:
        presets = RANGE_PRESETS_KM
        index = min(range(len(presets)), key=lambda i: abs(presets[i] - self.max_range_km))
        index = max(0, min(len(presets) - 1, index - direction))
        self.set_max_range(presets[index])

    def set_max_range(self, km: float) -> None:
        self.max_range_km = float(km)
        self._rebuild_grid()

    def fit_to(self, frame: FrameState) -> None:
        """Pick the smallest preset range that shows every object."""
        if not frame.targets:
            return
        reach = max(t.range_m for t in frame.targets) / 1000.0 * 1.1
        self.set_max_range(next((r for r in RANGE_PRESETS_KM if r >= reach), RANGE_PRESETS_KM[-1]))

    def _rebuild_grid(self) -> None:
        rx, ry = self._radar_xy
        r_max = self.max_range_km
        step = nice_step(r_max, 4)
        theta = np.linspace(0.0, 2 * np.pi, 181)
        minor, major = [], []
        ring = step
        while ring <= r_max + 1e-9:
            target = major if abs(ring - r_max) < 1e-6 or (ring / step) % 2 == 0 else minor
            target.append(np.column_stack([rx + ring * np.sin(theta), ry + ring * np.cos(theta)]))
            target.append(np.array([[np.nan, np.nan]]))
            ring += step
        outer = np.column_stack([rx + r_max * np.sin(theta), ry + r_max * np.cos(theta)])
        major.extend([outer, np.array([[np.nan, np.nan]])])
        for bearing in range(0, 360, 30):
            b = np.radians(bearing)
            spoke = np.array([[rx, ry], [rx + r_max * np.sin(b), ry + r_max * np.cos(b)], [np.nan, np.nan]])
            (major if bearing % 90 == 0 else minor).append(spoke)
        for bearing in range(0, 360, 5):
            b = np.radians(bearing)
            inner = r_max * (0.975 if bearing % 10 else 0.96)
            minor.append(
                np.array(
                    [
                        [rx + inner * np.sin(b), ry + inner * np.cos(b)],
                        [rx + r_max * np.sin(b), ry + r_max * np.cos(b)],
                        [np.nan, np.nan],
                    ]
                )
            )
        minor_xy = np.vstack(minor) if minor else np.empty((0, 2))
        major_xy = np.vstack(major)
        self._grid.setData(minor_xy[:, 0], minor_xy[:, 1])
        self._grid_major.setData(major_xy[:, 0], major_xy[:, 1])

        for item, bearing in zip(self._bearing_labels, range(0, 360, 30)):
            b = np.radians(bearing)
            item.setPos(rx + 1.06 * r_max * np.sin(b), ry + 1.06 * r_max * np.cos(b))
        for item in self._range_labels:
            self.plot.removeItem(item)
        self._range_labels = []
        ring = step
        b = np.radians(52.0)
        while ring <= r_max + 1e-9:
            text = f"{ring:g}" if ring != r_max else f"{ring:g} km"
            item = self.label(text, size=9)
            item.setPos(rx + ring * np.sin(b), ry + ring * np.cos(b))
            self._range_labels.append(item)
            ring += step
        pad = 0.12 * r_max
        self.view_box.setRange(
            xRange=(rx - r_max - pad, rx + r_max + pad),
            yRange=(ry - r_max - pad, ry + r_max + pad),
            padding=0,
        )

    # --------------------------------------------------------------- render

    def reset(self) -> None:
        super().reset()
        self._plots.clear()
        for item in list(self._name_labels.values()) + list(self._track_labels.values()):
            self.plot.removeItem(item)
        self._name_labels.clear()
        self._track_labels.clear()
        self._last_frame = None

    def render(self, frame: FrameState, extrapolate_s: float) -> None:
        p = palette()
        radar = frame.radar
        radar_xy = _xy_km(radar.position)
        if radar_xy != self._radar_xy:
            self._radar_xy = radar_xy
            self._rebuild_grid()
        rx, ry = radar_xy
        r_max = self.max_range_km
        revisit = 60.0 / radar.scan_rate_rpm if radar.scan_type == "circular" and radar.scan_rate_rpm > 0 else 4.0
        self._persistence_s = max(revisit, 2.0)

        if frame is not self._last_frame:
            for target in frame.targets:
                if target.measured_position is not None:
                    mx, my = _xy_km(target.measured_position)
                    self._plots.append((mx, my, frame.time))
            self._last_frame = frame

        # Sector limits
        if radar.scan_type == "sector":
            lo, hi = radar.sector
            xs = [rx, rx + r_max * np.sin(lo), np.nan, rx, rx + r_max * np.sin(hi)]
            ys = [ry, ry + r_max * np.cos(lo), np.nan, ry, ry + r_max * np.cos(hi)]
            self._sector.setData(xs, ys)
        else:
            self._sector.setData([], [])

        # Sweep and afterglow
        azimuth = radar.antenna_azimuth
        if radar.scan_type == "circular":
            azimuth += radar.scan_rate_rpm * 2 * np.pi / 60.0 * extrapolate_s
        self._sweep.setData([rx, rx + r_max * np.sin(azimuth)], [ry, ry + r_max * np.cos(azimuth)])
        self._update_afterglow(radar, azimuth, rx, ry, r_max, p)

        # Jamming strobes along the bearing of active noise jammers
        strobe = QPainterPath()
        for target in frame.targets:
            if target.jammer_active and target.jammer_loss_db > 3.0 and "noise" in target.ecm_type:
                half = radar.beamwidth_az * (1.5 if "barrage" in target.ecm_type else 0.8)
                polygon = QPolygonF(
                    [
                        QPointF(rx, ry),
                        QPointF(rx + r_max * np.sin(target.azimuth - half), ry + r_max * np.cos(target.azimuth - half)),
                        QPointF(rx + r_max * np.sin(target.azimuth + half), ry + r_max * np.cos(target.azimuth + half)),
                    ]
                )
                strobe.addPolygon(polygon)
                strobe.closeSubpath()
        self._strobes.setPath(strobe)

        # Plots fade with age over one revisit period
        now = frame.time + extrapolate_s
        while self._plots and now - self._plots[0][2] > self._persistence_s:
            self._plots.popleft()
        if self._plots:
            data = np.array(self._plots)
            age = np.clip((now - data[:, 2]) / self._persistence_s, 0.0, 0.999)
            levels = (PLOT_FADE_LEVELS - 1 - (age * PLOT_FADE_LEVELS).astype(int)).clip(0)
            brushes = [self._plot_brushes[level] for level in levels]
            self._plot_items.setData(data[:, 0], data[:, 1], brush=brushes)
        else:
            self._plot_items.setData([], [])

        self._false.setData(
            [ft.position[0] / 1000 + ft.velocity[0] / 1000 * extrapolate_s for ft in frame.false_targets],
            [ft.position[1] / 1000 + ft.velocity[1] / 1000 * extrapolate_s for ft in frame.false_targets],
        )
        self._render_tracks(frame, extrapolate_s, p)
        self._render_truth(frame, extrapolate_s, p)

    def _update_afterglow(self, radar, azimuth, rx, ry, r_max, p) -> None:
        path = QPainterPath()
        if self.afterglow and radar.scan_type != "stare":
            if radar.scan_type == "circular":
                trail = np.radians(18.0)
                angles = np.linspace(azimuth - trail, azimuth, 10)
            else:
                trail = radar.beamwidth_az * 3
                angles = np.linspace(azimuth - trail, azimuth + trail, 10)
            points = [QPointF(rx, ry)] + [
                QPointF(rx + r_max * np.sin(a), ry + r_max * np.cos(a)) for a in angles
            ]
            path.addPolygon(QPolygonF(points))
            path.closeSubpath()
        elif radar.scan_type == "stare":
            half = radar.beamwidth_az / 2
            points = [QPointF(rx, ry)] + [
                QPointF(rx + r_max * np.sin(a), ry + r_max * np.cos(a))
                for a in (azimuth - half, azimuth + half)
            ]
            path.addPolygon(QPolygonF(points))
            path.closeSubpath()
        self._glow.setBrush(QBrush(qcolor(p.sweep, 0.07)))
        self._glow.setPath(path)

    def _render_tracks(self, frame: FrameState, dt: float, p) -> None:
        xs, ys, brushes, pens = [], [], [], []
        leader_x, leader_y, hist_x, hist_y, ell_x, ell_y = [], [], [], [], [], []
        seen = set()
        circle = np.linspace(0, 2 * np.pi, 33)
        for track in frame.tracks:
            x = (track.position[0] + track.velocity[0] * dt) / 1000
            y = (track.position[1] + track.velocity[1] * dt) / 1000
            xs.append(x)
            ys.append(y)
            if track.status == "confirmed":
                brushes.append(pg.mkBrush(qcolor(p.text, 0.9)))
                pens.append(pen(p.text, 1.0))
            elif track.status == "coasting":
                brushes.append(pg.mkBrush(qcolor(p.accent, 0.15)))
                pens.append(pen(p.accent, 1.2))
            else:
                brushes.append(pg.mkBrush(qcolor(p.text, 0.0)))
                pens.append(pen(p.text_muted, 1.0))
            if track.status != "tentative":  # no reliable velocity yet
                leader_x += [x, x + track.velocity[0] * LEADER_SECONDS / 1000, np.nan]
                leader_y += [y, y + track.velocity[1] * LEADER_SECONDS / 1000, np.nan]
            for hx, hy in track.history[:-1]:
                hist_x.append(hx / 1000)
                hist_y.append(hy / 1000)
            if self.show_ellipses and track.status != "tentative":
                cov = np.array(track.covariance)
                values, vectors = np.linalg.eigh(cov)
                radii = 2.0 * np.sqrt(np.clip(values, 0, None)) / 1000  # 2-sigma
                points = vectors @ np.vstack([radii[0] * np.cos(circle), radii[1] * np.sin(circle)])
                ell_x += list(x + points[0]) + [np.nan]
                ell_y += list(y + points[1]) + [np.nan]
            label = self._track_labels.get(track.track_id)
            if label is None:
                label = self.label(f"T{track.track_id}", p.text_muted, 9)
                label.setAnchor((-0.25, 1.1))
                self._track_labels[track.track_id] = label
            label.setPos(x, y)
            seen.add(track.track_id)
        for track_id in [t for t in self._track_labels if t not in seen]:
            self.plot.removeItem(self._track_labels.pop(track_id))
        self._tracks.setData(xs, ys, brush=brushes, pen=pens)
        self._track_leaders.setData(leader_x, leader_y)
        self._track_history.setData(hist_x, hist_y)
        self._track_ellipses.setData(ell_x, ell_y)

    def _render_truth(self, frame: FrameState, dt: float, p) -> None:
        xs, ys, symbols, pens, leader_x, leader_y = [], [], [], [], [], []
        selected = None
        seen = set()
        show_names = len(frame.targets) <= 12
        for target in frame.targets:
            x = (target.position[0] + target.velocity[0] * dt) / 1000
            y = (target.position[1] + target.velocity[1] * dt) / 1000
            if target.target_id == self.selected_id:
                selected = (x, y)
            if not self.show_truth:
                continue
            colour = affiliation_colour(p, target.affiliation, target.target_type)
            xs.append(x)
            ys.append(y)
            symbols.append("x" if target.target_type in ("chaff", "decoy") else SYMBOLS.get(target.affiliation, "p"))
            pens.append(pen(colour, 1.4, 0.45 if target.masked else 1.0))
            if target.speed > 1.0:
                leader_x += [x, x + target.velocity[0] * LEADER_SECONDS / 1000, np.nan]
                leader_y += [y, y + target.velocity[1] * LEADER_SECONDS / 1000, np.nan]
            if show_names or target.target_id == self.selected_id:
                self._name_label(target, colour).setPos(x, y)
                seen.add(target.target_id)
        for target_id in [t for t in self._name_labels if t not in seen]:
            self.plot.removeItem(self._name_labels.pop(target_id))
        self._truth.setData(xs, ys, symbol=symbols, pen=pens, brush=None)
        self._truth_leaders.setData(leader_x, leader_y)
        self._selection.setData([selected[0]] if selected else [], [selected[1]] if selected else [])

    def _name_label(self, target: TargetFrame, colour: str) -> pg.TextItem:
        label = self._name_labels.get(target.target_id)
        if label is None:
            label = self.label(target.name, colour, 9)
            label.setAnchor((-0.15, -0.2))
            self._name_labels[target.target_id] = label
        return label

    # ---------------------------------------------------------- interaction

    def _scene_to_m(self, scene_pos) -> Tuple[float, float]:
        point = self.view_box.mapSceneToView(scene_pos)
        return point.x() * 1000.0, point.y() * 1000.0

    def _on_click(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x_m, y_m = self._scene_to_m(event.scenePos())
        if self.interaction == "place":
            self.point_picked.emit(x_m, y_m)
            return
        frame = self._last_frame
        if frame is None or not frame.targets:
            return
        pixel_km = self.view_box.viewPixelSize()[0]
        best, best_d = None, 14 * pixel_km
        for target in frame.targets:
            d = np.hypot(target.position[0] / 1000 - x_m / 1000, target.position[1] / 1000 - y_m / 1000)
            if d < best_d:
                best, best_d = target.target_id, d
        if best is not None:
            self.target_clicked.emit(best)
