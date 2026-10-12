"""
RadarSim main window.

Layout: header strip (scenario, mode, simulated time, status), objects list on
the left, scope tabs in the centre, inspector and settings on the right, an
analysis/event drawer and the transport bar at the bottom.

Rendering is decoupled from the simulation: frames arrive from the worker at
up to 30 Hz, the active view is redrawn by a display timer that extrapolates
motion between frames and lowers its own rate when the machine is slow.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QByteArray, QLocale, QSettings, QStandardPaths, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.simulation.frame import FrameState

from . import motion
from .controller import SimulationController
from .i18n import LANGUAGES, language, tr
from .panels.analysis import CoveragePanel, MonteCarloPanel
from .panels.events import EventLog
from .panels.inspector import Inspector
from .panels.objects import ObjectsPanel
from .panels.settings import SettingsPanel, engine_snapshot
from .panels.transport import TransportBar
from .theme import SPACE
from .theme.icons import icon
from .views.ascope import AScopeView
from .views.bscope import BScopeView
from .views.ppi import PPIView
from .views.range_doppler import RangeDopplerView
from .views.rhi import RHIView
from .views.tactical3d import Tactical3DView
from .widgets import ToastHost, caption

logger = logging.getLogger(__name__)

APP_VERSION = "4.0.0"
DOCS_URL = "https://spaceengineerss.github.io/RadarSim/"

# Display refresh budgets [ms]; the loop steps down when frames take too long.
FRAME_INTERVALS_MS = (16, 33, 50)
PROFILES = ("auto", "quality", "light")


def scenarios_dir() -> Path:
    """Built-in scenarios: next to a frozen executable, else in the source tree."""
    candidates = []
    if getattr(sys, "frozen", False):
        candidates += [Path(getattr(sys, "_MEIPASS", "")) / "scenarios", Path(sys.executable).parent / "scenarios"]
    candidates += [Path(__file__).resolve().parents[2] / "scenarios", Path.cwd() / "scenarios"]
    return next((c for c in candidates if c.is_dir()), Path.cwd())


def data_dir() -> Path:
    base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppDataLocation)
    return Path(base or Path.home() / ".radarsim")


class HeaderStrip(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setProperty("rule", "bottom")
        self.setFixedHeight(40)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE.lg, 0, SPACE.lg, 0)
        layout.setSpacing(SPACE.m)
        mark = QLabel("RadarSim")
        mark.setProperty("role", "title")
        layout.addWidget(mark)
        divider = QFrame()
        divider.setFixedSize(1, 16)
        divider.setObjectName("Separator")
        divider.setStyleSheet("max-height: 16px; min-width: 1px;")
        layout.addWidget(divider)
        self.scenario = caption("")
        layout.addWidget(self.scenario)
        self.mode = QLabel("LIVE")
        self.mode.setProperty("role", "badge")
        self.mode.setFixedHeight(20)
        layout.addWidget(self.mode, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch()
        self.status = caption("", "faint")
        layout.addWidget(self.status)
        self.clock = QLabel("T+00:00:00")
        self.clock.setProperty("role", "value")
        layout.addWidget(self.clock)

    def set_mode(self, mode: str) -> None:
        self.mode.setText(tr(mode.upper()))
        self.mode.setProperty("state", mode)
        self.mode.style().unpolish(self.mode)
        self.mode.style().polish(self.mode)

    def set_time(self, seconds: float) -> None:
        h, rem = divmod(int(seconds), 3600)
        m, s = divmod(rem, 60)
        text = f"T+{h:02d}:{m:02d}:{s:02d}"
        if self.clock.text() != text:
            self.clock.setText(text)


class MainWindow(QMainWindow):
    def __init__(self, scenario_path: Optional[str] = None):
        super().__init__()
        self.setWindowTitle("RadarSim")
        self.setObjectName("Root")
        self.setMinimumSize(1024, 640)
        self.settings = QSettings("RadarSim", "RadarSim")
        self.controller = SimulationController(data_dir() / "recordings", self)
        self.selected_id: Optional[int] = None
        self.editing = False
        self.profile = self.settings.value("view/profile", "auto")
        self._frame_received = 0.0
        self._render_ms = 0.0
        self._interval_index = 0
        self._fps_counter = [0, time.perf_counter(), 0.0]
        self._settings_dirty = True

        self._build()
        self._build_menu()
        self._connect()
        self.toasts = ToastHost(self, bottom_margin=self.transport.sizeHint().height() + SPACE.lg)

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.timeout.connect(self._render)
        self._apply_profile(self.profile)
        self._restore_layout()

        path = scenario_path or self.settings.value("session/scenario", "")
        if not path or not Path(path).exists():
            path = str(scenarios_dir() / "basic_tracking.json")
        if not self.controller.load_scenario(path):
            self.controller.new_scenario()

    # ---------------------------------------------------------------- layout

    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.header = HeaderStrip()
        outer.addWidget(self.header)

        self.body = QSplitter(Qt.Orientation.Horizontal)
        self.body.setChildrenCollapsible(False)
        self.objects = ObjectsPanel()
        self.objects.setMinimumWidth(200)
        self.body.addWidget(self.objects)

        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.setContentsMargins(0, 0, 0, 0)
        centre_layout.setSpacing(0)
        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        tab_row = QFrame()
        tab_row.setObjectName("Panel")
        tab_row.setProperty("rule", "bottom")
        tab_layout = QHBoxLayout(tab_row)
        tab_layout.setContentsMargins(SPACE.s, 0, SPACE.s, 0)
        tab_layout.addWidget(self.tabs)
        tab_layout.addStretch()
        self.editor_badge = QPushButton(tr("Done editing"))
        self.editor_badge.setProperty("variant", "primary")
        self.editor_badge.setVisible(False)
        tab_layout.addWidget(self.editor_badge)
        self.add_object_button = QPushButton(icon("add-target", 16), tr("Place object"))
        self.add_object_button.setCheckable(True)
        self.add_object_button.setVisible(False)
        tab_layout.addWidget(self.add_object_button)
        centre_layout.addWidget(tab_row)

        self.views = {
            "ppi": PPIView(),
            "bscope": BScopeView(),
            "rhi": RHIView(),
            "ascope": AScopeView(),
            "rd": RangeDopplerView(),
            "3d": Tactical3DView(),
        }
        self.stack = QStackedWidget()
        for key, view in self.views.items():
            self.tabs.addTab(tr(view.title))
            self.stack.addWidget(view)
        self.vertical = QSplitter(Qt.Orientation.Vertical)
        self.vertical.setChildrenCollapsible(True)
        self.vertical.addWidget(self.stack)
        self.drawer = QTabWidget()
        self.drawer.setDocumentMode(True)
        self.event_log = EventLog()
        self.coverage = CoveragePanel(self.controller)
        self.monte_carlo = MonteCarloPanel(self.controller)
        self.drawer.addTab(self.event_log, tr("Events"))
        self.drawer.addTab(self.coverage, tr("Coverage"))
        self.drawer.addTab(self.monte_carlo, tr("Monte Carlo"))
        self.vertical.addWidget(self.drawer)
        self.vertical.setStretchFactor(0, 4)
        self.vertical.setStretchFactor(1, 1)
        centre_layout.addWidget(self.vertical, 1)
        self.body.addWidget(centre)

        right = QScrollArea()
        right.setWidgetResizable(True)
        right.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right.setFrameShape(QFrame.Shape.NoFrame)
        right.setMinimumWidth(290)
        right_body = QFrame()
        right_body.setObjectName("Panel")
        right_body.setProperty("rule", "left")
        right_layout = QVBoxLayout(right_body)
        right_layout.setContentsMargins(0, SPACE.xs, 0, SPACE.lg)
        right_layout.setSpacing(0)
        self.inspector = Inspector()
        self.settings_panel = SettingsPanel(self.controller)
        right_layout.addWidget(self.inspector)
        right_layout.addWidget(self.settings_panel)
        right.setWidget(right_body)
        self.body.addWidget(right)
        self.body.setStretchFactor(1, 1)
        self.body.setSizes([230, 900, 310])
        outer.addWidget(self.body, 1)

        self.transport = TransportBar()
        outer.addWidget(self.transport)
        self.setCentralWidget(root)

    def _build_menu(self) -> None:
        bar = self.menuBar()

        def action(menu, text, slot, shortcut=None, checkable=False, checked=False):
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            act.setCheckable(checkable)
            act.setChecked(checked)
            act.triggered.connect(slot)
            menu.addAction(act)
            return act

        file_menu = bar.addMenu(tr("&File"))
        action(file_menu, tr("New scenario"), self._new_scenario, "Ctrl+N")
        action(file_menu, tr("Open scenario…"), self._open_scenario, "Ctrl+O")
        builtin = file_menu.addMenu(tr("Built-in scenarios"))
        for path in sorted(scenarios_dir().glob("*.*")):
            if path.suffix.lower() in (".yaml", ".yml", ".json"):
                act = QAction(path.stem.replace("_", " ").capitalize(), self)
                act.triggered.connect(lambda _=False, p=str(path): self.controller.load_scenario(p))
                builtin.addAction(act)
        action(file_menu, tr("Save scenario as…"), self._save_scenario, "Ctrl+S")
        file_menu.addSeparator()
        action(file_menu, tr("Export report…"), self._export_report, "Ctrl+E")
        action(file_menu, tr("Open recording…"), self._open_recording)
        file_menu.addSeparator()
        action(file_menu, tr("Quit"), self.close, "Ctrl+Q")

        sim = bar.addMenu(tr("&Simulation"))
        action(sim, tr("Play / pause"), self.controller.toggle, "Space")
        action(sim, tr("Single step"), self.controller.step, ".")
        action(sim, tr("Restart scenario"), self._restart, "R")

        view = bar.addMenu(tr("&View"))
        for index, key in enumerate(self.views):
            action(view, tr(self.views[key].title), lambda _=False, i=index: self.tabs.setCurrentIndex(i), str(index + 1))
        view.addSeparator()
        action(view, tr("Zoom in"), lambda: self._zoom(1), "+")
        action(view, tr("Zoom out"), lambda: self._zoom(-1), "-")
        action(view, tr("Fit all objects"), self._fit, "F")
        view.addSeparator()
        self.truth_action = action(view, tr("Show ground truth"), self._toggle_truth, "T", True, True)
        action(view, tr("Show track uncertainty"), self._toggle_ellipses, None, True, True)
        action(view, tr("Toggle event drawer"), self._toggle_drawer, "Ctrl+J")
        view.addSeparator()
        perf = view.addMenu(tr("Performance"))
        group = QActionGroup(self)
        labels = {"auto": tr("Automatic"), "quality": tr("Best quality"), "light": tr("Light (slow machines)")}
        for key in PROFILES:
            act = action(perf, labels[key], lambda _=False, k=key: self._apply_profile(k), checkable=True,
                         checked=key == self.profile)
            group.addAction(act)
        reduced = self.settings.value("view/reduced_motion", False, type=bool)
        motion.set_reduced_motion(reduced)
        action(view, tr("Reduce motion"), self._toggle_motion, checkable=True, checked=reduced)
        lang_menu = view.addMenu(tr("Language"))
        lang_group = QActionGroup(self)
        for code, name in LANGUAGES.items():
            act = action(lang_menu, name, lambda _=False, c=code: self._set_language(c), checkable=True,
                         checked=code == language())
            lang_group.addAction(act)

        tools = bar.addMenu(tr("&Tools"))
        self.editor_action = action(tools, tr("Scenario editor"), self._toggle_editor, "Ctrl+Shift+E", True)
        action(tools, tr("Detection coverage"), lambda: self._open_drawer(1))
        action(tools, tr("Monte Carlo check"), lambda: self._open_drawer(2))
        self.record_action = action(tools, tr("Record session"), self._toggle_record, "Ctrl+R", True)

        help_menu = bar.addMenu(tr("&Help"))
        action(help_menu, tr("Keyboard shortcuts"), self._show_shortcuts, "F1")
        action(help_menu, tr("Documentation"), lambda: QDesktopServices.openUrl(QUrl(DOCS_URL)))
        action(help_menu, tr("About RadarSim"), self._show_about)

    def _connect(self) -> None:
        c = self.controller
        c.frame.connect(self._on_frame)
        c.scenario_loaded.connect(self._on_scenario_loaded)
        c.mode_changed.connect(self._on_mode)
        c.notice.connect(lambda text, kind: self.toasts.show(text, kind))
        c.recording_changed.connect(self._on_recording)
        c.replay_position.connect(self.transport.set_replay_position)
        self.tabs.currentChanged.connect(self._on_tab)
        self.objects.target_selected.connect(self._select)
        for view in self.views.values():
            if hasattr(view, "target_clicked"):
                view.target_clicked.connect(self._select)
        self.views["rd"].enable_requested.connect(lambda: c.set_pulse_doppler(True))
        self.views["ppi"].point_picked.connect(self._place_object)
        self.inspector.edited.connect(c.edit_object)
        self.inspector.delete_requested.connect(self._delete_object)
        t = self.transport
        t.play_toggled.connect(c.toggle)
        t.step_requested.connect(c.step)
        t.restart_requested.connect(self._restart)
        t.speed_changed.connect(c.set_speed)
        t.record_toggled.connect(lambda on: c.start_recording() if on else c.stop_recording())
        t.seek_requested.connect(c.seek)
        t.exit_replay.connect(lambda: c.close_replay())
        self.editor_badge.clicked.connect(lambda: self.editor_action.trigger())
        self.add_object_button.toggled.connect(self._place_mode)

    # ------------------------------------------------------------ simulation

    def _on_scenario_loaded(self, config) -> None:
        self.selected_id = None
        self.objects.set_scenario(config.name, config.description)
        self.header.scenario.setText(config.name)
        self.event_log.clear()
        self.inspector.set_target(None)
        for view in self.views.values():
            view.reset()
        terrain = self.controller.terrain
        self.views["rhi"].set_terrain(terrain)
        self.views["3d"].set_terrain(terrain)
        self.settings_panel.sync(self.controller.call(engine_snapshot))
        self._fit_pending = True
        if self.controller.scenario_path:
            self.settings.setValue("session/scenario", self.controller.scenario_path)

    def _on_frame(self, frame: FrameState) -> None:
        self._frame_received = time.perf_counter()
        if frame.events:
            self.event_log.add(frame.events)
        if getattr(self, "_fit_pending", False):
            self._fit_pending = False
            self.views["ppi"].fit_to(frame)
            for key in ("bscope", "rhi", "ascope", "3d"):
                self.views[key].set_max_range(self.views["ppi"].max_range_km)

    def _on_mode(self, mode: str) -> None:
        if self.editing:
            mode = "edit"
        self.header.set_mode(mode)
        self.transport.set_running(mode == "live" or (mode == "replay" and not self.controller.paused))
        self.transport.set_replay(self.controller.replaying)

    def _on_recording(self, recording: bool) -> None:
        self.transport.set_recording(recording)
        self.record_action.setChecked(recording)

    def _render(self) -> None:
        frame = self.controller.latest
        if frame is None:
            return
        started = time.perf_counter()
        running = not self.controller.paused and not self.editing
        extrapolate = 0.0
        if running and not self.controller.replaying:
            extrapolate = min((started - self._frame_received) * self.controller.speed, 0.25)
        view = self.stack.currentWidget()
        view.render(frame, extrapolate)
        self.header.set_time(frame.time + extrapolate)
        self._slow_updates(frame)
        self._measure(time.perf_counter() - started)

    def _slow_updates(self, frame: FrameState) -> None:
        """Panels refresh at ~5 Hz; they do not need display-rate updates."""
        now = time.perf_counter()
        if now - getattr(self, "_last_slow", 0.0) < 0.2:
            return
        self._last_slow = now
        self.objects.update_frame(frame)
        self.inspector.update_frame(frame)

    def _measure(self, seconds: float) -> None:
        self._render_ms = 0.9 * self._render_ms + 0.1 * seconds * 1000
        counter = self._fps_counter
        counter[0] += 1
        now = time.perf_counter()
        if now - counter[1] >= 1.0:
            counter[2] = counter[0] / (now - counter[1])
            counter[0], counter[1] = 0, now
            self.header.status.setText(f"{counter[2]:.0f} fps · {self._render_ms:.1f} ms")
            if self.profile == "auto":
                self._adapt_rate()

    def _adapt_rate(self) -> None:
        budget = FRAME_INTERVALS_MS[self._interval_index]
        if self._render_ms > 0.7 * budget and self._interval_index < len(FRAME_INTERVALS_MS) - 1:
            self._interval_index += 1
        elif self._interval_index > 0 and self._render_ms < 0.3 * FRAME_INTERVALS_MS[self._interval_index - 1]:
            self._interval_index -= 1
        self._timer.setInterval(FRAME_INTERVALS_MS[self._interval_index])

    def _apply_profile(self, profile: str) -> None:
        import pyqtgraph as pg

        self.profile = profile
        self.settings.setValue("view/profile", profile)
        light = profile == "light"
        pg.setConfigOptions(antialias=not light)
        self.views["ppi"].afterglow = not light
        self._interval_index = 1 if light else 0
        self._timer.start(FRAME_INTERVALS_MS[self._interval_index])

    # -------------------------------------------------------------- actions

    def _on_tab(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        motion.fade_in(self.stack.currentWidget())
        self.controller.set_include_rd_map(list(self.views)[index] == "rd")

    def _select(self, target_id: Optional[int]) -> None:
        self.selected_id = target_id
        for view in self.views.values():
            view.set_selected(target_id)
        self.objects.select(target_id)
        self.inspector.set_target(target_id)
        if self.controller.latest is not None:
            self.inspector.update_frame(self.controller.latest)

    def _zoom(self, direction: int) -> None:
        ppi = self.views["ppi"]
        ppi.on_wheel(direction)
        for key in ("bscope", "rhi", "ascope", "3d"):
            self.views[key].set_max_range(ppi.max_range_km)

    def _fit(self) -> None:
        if self.controller.latest is not None:
            self.views["ppi"].fit_to(self.controller.latest)
            self._zoom(0)

    def _toggle_truth(self, on: bool) -> None:
        self.views["ppi"].show_truth = on
        self.views["bscope"].show_truth = on

    def _toggle_ellipses(self, on: bool) -> None:
        self.views["ppi"].show_ellipses = on

    def _toggle_drawer(self) -> None:
        sizes = self.vertical.sizes()
        self.vertical.setSizes([sizes[0] + sizes[1], 0] if sizes[1] > 0 else [int(sum(sizes) * 0.72), int(sum(sizes) * 0.28)])

    def _open_drawer(self, index: int) -> None:
        self.drawer.setCurrentIndex(index)
        sizes = self.vertical.sizes()
        if sizes[1] < 160:
            self.vertical.setSizes([int(sum(sizes) * 0.62), int(sum(sizes) * 0.38)])

    def _toggle_motion(self, on: bool) -> None:
        motion.set_reduced_motion(on)
        self.settings.setValue("view/reduced_motion", on)

    def _set_language(self, code: str) -> None:
        self.settings.setValue("view/language", code)
        self.toasts.show(tr("The new language is used after a restart."), "info")

    def _restart(self) -> None:
        if self.controller.replaying:
            self.controller.seek(0.0)
        else:
            self.controller.reload()

    def _toggle_record(self, on: bool) -> None:
        if on:
            self.controller.start_recording()
        else:
            self.controller.stop_recording()

    # --------------------------------------------------------------- editor

    def _toggle_editor(self, on: bool) -> None:
        self.editing = on
        self.inspector.set_editing(on)
        self.editor_badge.setVisible(on)
        self.add_object_button.setVisible(on)
        if on:
            self.controller.begin_editing()
            self.tabs.setCurrentIndex(0)
            self.header.set_mode("edit")
            self.toasts.show(tr("Editing the scenario's start. Place objects on the PPI, then save."), "info")
        else:
            self.add_object_button.setChecked(False)
            self._on_mode("paused")

    def _place_mode(self, on: bool) -> None:
        self.views["ppi"].interaction = "place" if on else "select"
        self.views["ppi"].setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.ArrowCursor)

    def _place_object(self, x_m: float, y_m: float) -> None:
        target_id = self.controller.add_object(x_m, y_m)
        self.add_object_button.setChecked(False)
        self._select(target_id)

    def _delete_object(self, target_id: int) -> None:
        self.controller.remove_object(target_id)
        self._select(None)

    # ----------------------------------------------------------------- files

    def _new_scenario(self) -> None:
        self.controller.new_scenario()
        if not self.editing:
            self.editor_action.trigger()

    def _open_scenario(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Open scenario"), str(scenarios_dir()), tr("Scenarios (*.yaml *.yml *.json)")
        )
        if path:
            self.controller.load_scenario(path)

    def _save_scenario(self) -> None:
        start = self.controller.scenario_path or str(scenarios_dir() / "scenario.yaml")
        path, _ = QFileDialog.getSaveFileName(self, tr("Save scenario"), str(Path(start).with_suffix(".yaml")),
                                              tr("YAML scenario (*.yaml)"))
        if not path:
            return
        name, ok = QInputDialog.getText(self, tr("Scenario name"), tr("Name"),
                                        text=self.controller.config.name if self.controller.config else "")
        if ok:
            self.controller.export_scenario(path, name)

    def _export_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, tr("Export report"), str(Path.home() / "radarsim_report.csv"),
                                              tr("CSV (*.csv);;JSON (*.json)"))
        if path:
            self.controller.export_report(path)

    def _open_recording(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("Open recording"), str(self.controller.output_dir),
                                              tr("Recordings (*.h5)"))
        if path:
            self.controller.open_replay(path)

    # --------------------------------------------------------------- dialogs

    def _dialog(self, title: str, rows) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(SPACE.xl, SPACE.lg, SPACE.xl, SPACE.lg)
        for left, right in rows:
            row = QHBoxLayout()
            label = QLabel(left)
            label.setProperty("role", "value" if right else "title")
            row.addWidget(label)
            row.addStretch()
            if right:
                row.addWidget(caption(right))
            layout.addLayout(row)
        close = QPushButton(tr("Close"))
        close.clicked.connect(dialog.accept)
        layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.exec()

    def _show_shortcuts(self) -> None:
        self._dialog(tr("Keyboard shortcuts"), [
            ("Space", tr("Play / pause")), (".", tr("Single step while paused")),
            ("R", tr("Restart scenario")), ("1 … 6", tr("Switch scope")),
            ("+ / −  ·  wheel", tr("Range scale")), ("F", tr("Fit all objects")),
            ("T", tr("Ground truth on/off")), ("Ctrl+J", tr("Event drawer")),
            ("Ctrl+O / Ctrl+S", tr("Open / save scenario")), ("Ctrl+E", tr("Export report")),
            ("Ctrl+R", tr("Record session")), ("Ctrl+Shift+E", tr("Scenario editor")),
            ("Esc", tr("Clear selection")), ("F11", tr("Full screen")),
        ])

    def _show_about(self) -> None:
        self._dialog(tr("About RadarSim"), [
            (f"RadarSim {APP_VERSION}", ""),
            (tr("Physics-based radar simulation and analysis"), ""),
            ("Mehmet Gümüş", tr("Author")),
            ("MIT", tr("Licence")),
            ("IBM Plex", tr("Typeface (SIL OFL 1.1)")),
        ])

    # ---------------------------------------------------------------- window

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._select(None)
        elif event.key() == Qt.Key.Key_F11:
            self.showNormal() if self.isFullScreen() else self.showFullScreen()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "toasts"):
            self.toasts.relayout()

    def _restore_layout(self) -> None:
        for key, widget in (("window/geometry", self), ("window/body", self.body), ("window/vertical", self.vertical)):
            value = self.settings.value(key)
            if isinstance(value, QByteArray):
                (widget.restoreGeometry if widget is self else widget.restoreState)(value)
        if not self.settings.value("window/vertical"):
            self.vertical.setSizes([700, 0])

    def closeEvent(self, event) -> None:
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/body", self.body.saveState())
        self.settings.setValue("window/vertical", self.vertical.saveState())
        self._timer.stop()
        self.controller.shutdown()
        event.accept()


def ensure_application(argv=None) -> QApplication:
    """Create (or reuse) the QApplication with fonts, theme and language."""
    from .i18n import set_language
    from .theme import palette
    from .theme.fonts import load_fonts, ui_font
    from .theme.qss import build_stylesheet

    app = QApplication.instance() or QApplication(argv or [sys.argv[0]])
    app.setApplicationName("RadarSim")
    app.setOrganizationName("RadarSim")
    app.setApplicationVersion(APP_VERSION)
    app.setStyle("Fusion")
    # Numbers use the decimal point everywhere, whatever the system locale.
    QLocale.setDefault(QLocale.c())
    load_fonts()
    app.setFont(ui_font())
    from PySide6.QtGui import QIcon

    from .theme import ASSETS

    app.setWindowIcon(QIcon(str(ASSETS / "radarsim.png")))
    app.setStyleSheet(build_stylesheet(palette()))
    set_language(QSettings("RadarSim", "RadarSim").value("view/language", "en"))
    return app
