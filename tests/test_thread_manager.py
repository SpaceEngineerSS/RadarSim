"""Threading-model tests for the GUI simulation bridge (src/ui/thread_manager.py)."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from src.ui.thread_manager import SimulationThread, create_demo_scenario


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def _wait_until(application, condition, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        application.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def running_thread(application):
    thread = SimulationThread(create_demo_scenario(), update_rate_hz=100.0)
    frames = []
    thread.update_data.connect(frames.append)
    thread.start()
    assert _wait_until(application, lambda: len(frames) > 0)
    yield thread, frames
    thread.stop()
    assert thread.wait(5000)


def test_submit_runs_on_worker_between_steps(application, running_thread):
    thread, _frames = running_thread
    thread.submit(lambda engine: setattr(engine.radar, "power_watts", 123_000.0))
    assert _wait_until(application, lambda: thread.engine.radar.power_watts == 123_000.0)


def test_call_sync_returns_result_from_worker(running_thread):
    thread, _frames = running_thread
    assert thread.call_sync(lambda engine: len(engine.targets)) == 4


def test_call_sync_reraises_command_errors(running_thread):
    thread, _frames = running_thread

    def boom(_engine):
        raise RuntimeError("bad command")

    with pytest.raises(RuntimeError, match="bad command"):
        thread.call_sync(boom)


def test_pause_stops_time_and_resume_does_not_jump(application, running_thread):
    thread, frames = running_thread
    thread.pause()
    application.processEvents()
    paused_time = thread.call_sync(lambda engine: engine.simulation_time)
    time.sleep(0.4)
    assert thread.call_sync(lambda engine: engine.simulation_time) == paused_time

    count = len(frames)
    thread.resume()
    assert _wait_until(application, lambda: len(frames) > count)
    # The first frame after a 0.4 s pause must be a normal-sized step, not 0.4 s.
    assert frames[count]["dt"] < 0.3


def test_step_error_pauses_and_reports(application, running_thread):
    thread, _frames = running_thread
    errors, pauses = [], []
    thread.error.connect(errors.append)
    thread.paused_changed.connect(pauses.append)

    def break_engine(engine):
        engine.targets.append(None)  # next step raises AttributeError

    thread.submit(break_engine)
    assert _wait_until(application, lambda: errors and pauses)
    assert pauses[-1] is True
    assert thread.is_paused
    assert thread.isRunning()
    thread.call_sync(lambda engine: engine.targets.pop())


def test_commands_apply_immediately_when_not_running(application):
    thread = SimulationThread(create_demo_scenario())
    thread.submit(lambda engine: setattr(engine.radar, "prf_hz", 2500.0))
    assert thread.engine.radar.prf_hz == 2500.0


def test_main_window_space_toggles_pause_without_crashing(application):
    from src.ui.main_window import MainWindow

    window = MainWindow()
    window.show()
    assert _wait_until(application, lambda: window.sim_thread.isRunning())

    window.setFocus()
    from PySide6.QtTest import QTest

    QTest.keyClick(window, Qt.Key.Key_Space)
    assert _wait_until(application, lambda: window.sim_thread.is_paused)
    assert window.control_panel.play_btn.text().endswith("PLAY")

    QTest.keyClick(window, Qt.Key.Key_Space)
    assert _wait_until(application, lambda: not window.sim_thread.is_paused)
    window.close()
    application.processEvents()


def test_failed_scenario_load_keeps_current_simulation(application, tmp_path, monkeypatch):
    from src.ui.main_window import MainWindow

    monkeypatch.setattr(QMessageBox, "critical", lambda *args, **kwargs: None)
    window = MainWindow()
    window.show()
    engine_before = window.engine
    thread_before = window.sim_thread

    bad = tmp_path / "bad.yaml"
    bad.write_text("simulation: {pfa: 5}\n", encoding="utf-8")
    assert window.load_scenario(str(bad)) is False
    assert window.engine is engine_before
    assert window.sim_thread is thread_before
    assert window.sim_thread.isRunning()
    window.close()
    application.processEvents()


def test_pause_and_speed_survive_scenario_reload(application):
    from pathlib import Path

    from src.ui.main_window import MainWindow

    scenarios = Path(__file__).parents[1] / "scenarios"
    window = MainWindow()
    window.show()
    window.control_panel.speed_slider.setValue(30)  # 3.0x
    window._toggle_pause()
    assert window.load_scenario(str(scenarios / "basic_tracking.json"))
    assert window.sim_thread.is_paused
    assert window.sim_thread.speed_factor == pytest.approx(3.0)
    window.close()
    application.processEvents()
