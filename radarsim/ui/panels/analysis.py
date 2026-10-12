"""
Analysis tabs: detection coverage (Pd against range) and Monte Carlo check.

Both run on a background thread on a probe copy of the current radar, so the
simulation and the interface keep running while they compute.
"""

from typing import Callable

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QProgressBar, QPushButton, QWidget

from radarsim.simulation.analysis import coverage_curve, monte_carlo_pd, probe_engine

from ..i18n import tr
from ..theme import SPACE, palette
from ..theme.fonts import mono_font
from ..views.base import pen, qcolor
from ..widgets import ChoiceBox, NumberField, caption


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)


class _Job(QRunnable):
    def __init__(self, work: Callable, signals: _Signals):
        super().__init__()
        self._work = work
        self.signals = signals

    def run(self) -> None:
        try:
            self.signals.done.emit(self._work(self.signals))
        except Exception as exc:  # shown to the user
            self.signals.failed.emit(str(exc))


def _styled_plot(x_label: str, y_label: str) -> pg.PlotWidget:
    p = palette()
    plot = pg.PlotWidget(background=p.scope_background)
    plot.hideButtons()
    plot.setMenuEnabled(False)
    plot.showGrid(x=True, y=True, alpha=0.15)
    for name, text in (("bottom", x_label), ("left", y_label)):
        axis = plot.getAxis(name)
        axis.setPen(pen(p.scope_grid_major))
        axis.setTextPen(pen(p.scope_label))
        axis.setTickFont(mono_font(10))
        plot.setLabel(name, text)
    return plot


class _ProbeForm(QWidget):
    """Probe target parameters shared by both tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(SPACE.m)
        self.rcs = NumberField("m²", 1e-4, 1e5, 3)
        self.rcs.setValue(1.0)
        self.swerling = ChoiceBox([(f"Swerling {i}", i) for i in range(5)])
        self.swerling.setCurrentIndex(1)
        self.altitude = NumberField("m", 0, 30000, 0, step=100)
        self.altitude.setValue(3000)
        self.max_range = NumberField("km", 1, 2000, 0, 1e3, 10)
        self.max_range.setValue(150)
        grid.setVerticalSpacing(2)
        for index, (label, widget) in enumerate(
            ((tr("RCS"), self.rcs), (tr("Fluctuation"), self.swerling),
             (tr("Altitude"), self.altitude), (tr("Max range"), self.max_range))
        ):
            row, column = divmod(index, 2)
            grid.addWidget(caption(label), row * 2, column)
            grid.addWidget(widget, row * 2 + 1, column)

    def values(self) -> dict:
        return {
            "rcs_m2": self.rcs.value(),
            "swerling": self.swerling.currentData(),
            "altitude_m": self.altitude.value(),
            "max_range_m": self.max_range.value() * 1e3,
        }


class CoveragePanel(QWidget):
    """Single-look Pd and SNR against range for the current radar settings."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.c = controller
        p = palette()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE.m, SPACE.s, SPACE.m, SPACE.s)
        side = QWidget()
        side.setFixedWidth(280)
        form = QGridLayout(side)
        form.setContentsMargins(0, 0, 0, 0)
        self.probe = _ProbeForm()
        form.addWidget(self.probe, 0, 0)
        self.run = QPushButton(tr("Compute coverage"))
        self.run.setProperty("variant", "primary")
        self.run.clicked.connect(self.compute)
        form.addWidget(self.run, 1, 0)
        self.result = QLabel("")
        self.result.setProperty("role", "value")
        self.result.setWordWrap(True)
        form.addWidget(self.result, 2, 0)
        form.setRowStretch(3, 1)
        layout.addWidget(side)
        self.plot = _styled_plot(tr("Range [km]"), tr("Probability of detection"))
        self.plot.setYRange(0, 1.02, padding=0)
        self.curve = self.plot.plot(pen=pen(p.accent, 2))
        self.markers = self.plot.plot(pen=pen(p.text_faint, 1, style=Qt.PenStyle.DashLine), connect="finite")
        layout.addWidget(self.plot, 1)
        self._pool = QThreadPool.globalInstance()

    def compute(self) -> None:
        if self.c.engine is None:
            return
        params = self.probe.values()
        probe = self.c.call(probe_engine)
        self.run.setEnabled(False)
        self.result.setText(tr("Computing…"))
        signals = _Signals()
        signals.done.connect(self._show)
        signals.failed.connect(lambda m: (self.result.setText(m), self.run.setEnabled(True)))
        self._pool.start(_Job(lambda _s: coverage_curve(probe, points=160, **params), signals))
        self._signals = signals

    def _show(self, curve) -> None:
        self.run.setEnabled(True)
        km = curve.range_m / 1000
        self.curve.setData(km, curve.pd)
        xs, ys = [], []
        lines = []
        for level, value in ((0.9, curve.range_pd90_m), (0.5, curve.range_pd50_m)):
            if value is not None:
                xs += [value / 1000, value / 1000, np.nan]
                ys += [0, level, np.nan]
                lines.append(f"Pd {level:.0%}: {value / 1000:6.1f} km")
            else:
                lines.append(f"Pd {level:.0%}: –")
        self.markers.setData(xs, ys)
        self.result.setText("\n".join(lines))


class MonteCarloPanel(QWidget):
    """Empirical Pd from repeated looks, with 95% intervals, against theory."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.c = controller
        p = palette()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE.m, SPACE.s, SPACE.m, SPACE.s)
        side = QWidget()
        side.setFixedWidth(280)
        form = QGridLayout(side)
        form.setContentsMargins(0, 0, 0, 0)
        self.probe = _ProbeForm()
        form.addWidget(self.probe, 0, 0, 1, 2)
        form.addWidget(caption(tr("Looks per point")), 1, 0)
        self.looks = NumberField("", 20, 5000, 0, step=50)
        self.looks.setValue(300)
        form.addWidget(self.looks, 1, 1)
        form.addWidget(caption(tr("Range points")), 2, 0)
        self.points = NumberField("", 3, 60, 0, step=1)
        self.points.setValue(15)
        form.addWidget(self.points, 2, 1)
        self.run = QPushButton(tr("Run Monte Carlo"))
        self.run.setProperty("variant", "primary")
        self.run.clicked.connect(self.compute)
        form.addWidget(self.run, 3, 0, 1, 2)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        form.addWidget(self.progress, 4, 0, 1, 2)
        form.setRowStretch(5, 1)
        layout.addWidget(side)
        self.plot = _styled_plot(tr("Range [km]"), tr("Probability of detection"))
        self.plot.setYRange(0, 1.02, padding=0)
        self.theory = self.plot.plot(pen=pen(p.text_muted, 1.5, style=Qt.PenStyle.DashLine))
        self.empirical = pg.ErrorBarItem(pen=pen(p.accent, 1))
        self.plot.addItem(self.empirical)
        self.points_item = pg.ScatterPlotItem(size=6, brush=pg.mkBrush(qcolor(p.accent)), pen=None)
        self.plot.addItem(self.points_item)
        layout.addWidget(self.plot, 1)
        self._pool = QThreadPool.globalInstance()

    def compute(self) -> None:
        if self.c.engine is None:
            return
        params = self.probe.values()
        ranges = np.linspace(params["max_range_m"] / self.points.value(), params["max_range_m"], int(self.points.value()))
        looks = int(self.looks.value())
        probe = self.c.call(probe_engine)
        self.run.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(ranges))
        signals = _Signals()
        signals.progress.connect(lambda done, total: self.progress.setValue(done))
        signals.done.connect(self._show)
        signals.failed.connect(lambda m: self.run.setEnabled(True))

        def work(s):
            return monte_carlo_pd(
                probe, ranges, looks, params["rcs_m2"], params["swerling"], params["altitude_m"],
                progress=lambda done, total: s.progress.emit(done, total),
            )

        self._pool.start(_Job(work, signals))
        self._signals = signals

    def _show(self, result) -> None:
        self.run.setEnabled(True)
        self.progress.setVisible(False)
        km = result.range_m / 1000
        self.theory.setData(km, result.pd_theory)
        self.points_item.setData(km, result.pd_empirical)
        self.empirical.setData(
            x=km, y=result.pd_empirical,
            top=result.pd_high - result.pd_empirical, bottom=result.pd_empirical - result.pd_low,
            beam=0,
        )
