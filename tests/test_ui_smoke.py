"""Desktop interface: construction, scenario loading, views, panels and editor."""

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import QSettings  # noqa: E402

from src.ui.main_window import MainWindow, ensure_application  # noqa: E402

SCENARIOS = Path(__file__).parents[1] / "scenarios"


@pytest.fixture(scope="module")
def app():
    QSettings("RadarSim", "RadarSim").clear()
    application = ensure_application()
    yield application
    QSettings("RadarSim", "RadarSim").clear()


def spin(app, condition=lambda: False, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return condition()


@pytest.fixture
def window(app):
    win = MainWindow(str(SCENARIOS / "basic_tracking.json"))
    win.show()
    assert spin(app, lambda: win.controller.latest is not None)
    yield win
    win.close()
    app.processEvents()


def test_window_runs_a_scenario(app, window):
    assert window.controller.config.name == "Basic Tracking"
    first = window.controller.latest.time
    assert spin(app, lambda: window.controller.latest.time > first + 0.5)
    assert window.header.mode.text() == "LIVE"


def test_every_view_renders(app, window):
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index)
        spin(app, timeout=0.3)
    window.tabs.setCurrentIndex(0)


@pytest.mark.parametrize("name", sorted(p.name for p in SCENARIOS.glob("*.*")))
def test_all_scenarios_load_and_keep_their_radar(app, window, name):
    assert window.controller.load_scenario(str(SCENARIOS / name))
    assert spin(app, lambda: window.controller.latest is not None)
    shown = window.settings_panel.frequency.value() * 1e9
    actual = window.controller.call(lambda engine: engine.radar.frequency_hz)
    assert shown == pytest.approx(actual, rel=1e-3)


def test_settings_change_the_engine(app, window):
    window.settings_panel.power.setValue(250.0)  # kW
    window.settings_panel.mti.click()
    assert spin(app, lambda: window.controller.call(lambda e: e.radar.power_watts) == 250e3)
    assert window.controller.call(lambda e: e.mti_enabled) is True


def test_selection_fills_inspector(app, window):
    target_id = window.controller.latest.targets[0].target_id
    window._select(target_id)
    spin(app, timeout=0.5)
    assert window.inspector._name.text().startswith("Target 1")


def test_pause_step_and_restart(app, window):
    window.controller.pause()
    assert spin(app, lambda: window.controller.paused)
    paused_at = window.controller.call(lambda e: e.current_time)
    spin(app, timeout=0.3)
    assert window.controller.call(lambda e: e.current_time) == paused_at
    window.controller.step()
    assert spin(app, lambda: window.controller.call(lambda e: e.current_time) > paused_at)
    window._restart()
    assert spin(app, lambda: window.controller.latest is not None and window.controller.latest.time < 0.5)


def test_editor_adds_edits_saves_and_reloads(app, window, tmp_path):
    window.editor_action.trigger()
    target_id = window.controller.add_object(20_000.0, 30_000.0)
    window.controller.edit_object(target_id, {"name": "Raider", "speed_mps": 300.0, "heading_deg": 90.0})
    spin(app, timeout=0.3)
    path = tmp_path / "edited.yaml"
    assert window.controller.export_scenario(str(path), "Edited")
    window.editor_action.trigger()
    assert window.controller.load_scenario(str(path))
    raider = window.controller.call(lambda e: next(t for t in e.targets if t.name == "Raider"))
    assert raider.velocity[0] == pytest.approx(300.0)
    assert raider.position[0] == pytest.approx(20_000.0)


def test_report_export(app, window, tmp_path):
    spin(app, timeout=1.0)
    path = tmp_path / "report.json"
    assert window.controller.export_report(str(path))
    assert '"objects"' in path.read_text(encoding="utf-8")


def test_bad_scenario_keeps_current_run(app, window, tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("simulation: {pfa: 5}\n", encoding="utf-8")
    engine = window.controller.engine
    assert window.controller.load_scenario(str(bad)) is False
    assert window.controller.engine is engine
    assert window.controller.thread.isRunning()


def test_recording_and_replay(app, window):
    window.controller.start_recording()
    spin(app, timeout=1.5)
    path = window.controller.stop_recording()
    assert path and Path(path).exists()
    assert window.controller.open_replay(path)
    assert spin(app, lambda: window.controller.replaying and window.controller.latest.targets)
    window.controller.close_replay()
    assert not window.controller.replaying
