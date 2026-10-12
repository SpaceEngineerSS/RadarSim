"""Simulation thread: command queue, pause/step, fixed-step timing, errors."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from radarsim.io.scenario_loader import ScenarioLoader  # noqa: E402
from radarsim.simulation.frame import FrameState  # noqa: E402
from radarsim.ui.thread_manager import SimulationThread  # noqa: E402


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def _wait(app, condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def running(application):
    engine = ScenarioLoader("radarsim/scenarios/basic_tracking.json").create_simulation_engine()
    thread = SimulationThread(engine)
    frames = []
    thread.frame_ready.connect(frames.append)
    thread.start()
    assert _wait(application, lambda: frames)
    yield thread, frames
    thread.stop()
    assert thread.wait(5000)


def test_frames_are_immutable_snapshots(running):
    _thread, frames = running
    assert isinstance(frames[-1], FrameState)
    with pytest.raises(Exception):
        frames[-1].time = 0.0


def test_submit_runs_between_steps(application, running):
    thread, _ = running
    thread.submit(lambda engine: setattr(engine.radar, "power_watts", 123e3))
    assert _wait(application, lambda: thread.engine.radar.power_watts == 123e3)


def test_call_sync_returns_and_reraises(running):
    thread, _ = running
    assert thread.call_sync(lambda engine: len(engine.targets)) == 3

    def boom(_engine):
        raise RuntimeError("bad command")

    with pytest.raises(RuntimeError, match="bad command"):
        thread.call_sync(boom)


def test_simulated_time_follows_wall_clock_in_fixed_steps(application, running):
    thread, _ = running
    start = thread.call_sync(lambda e: e.current_time)
    time.sleep(1.0)
    elapsed = thread.call_sync(lambda e: e.current_time) - start
    assert 0.7 < elapsed < 1.3
    dt = thread.engine.dt
    assert thread.call_sync(lambda e: e.current_time) / dt == pytest.approx(
        round(thread.call_sync(lambda e: e.current_time) / dt), abs=1e-6
    )


def test_pause_freezes_and_step_advances_one_step(application, running):
    thread, frames = running
    thread.pause()
    _wait(application, lambda: thread.is_paused)
    time.sleep(0.1)
    frozen = thread.call_sync(lambda e: e.current_time)
    time.sleep(0.3)
    assert thread.call_sync(lambda e: e.current_time) == frozen
    thread.step_once()
    assert _wait(application, lambda: thread.call_sync(lambda e: e.current_time) > frozen)
    assert thread.call_sync(lambda e: e.current_time) == pytest.approx(frozen + thread.engine.dt)


def test_step_error_pauses_and_reports(application, running):
    thread, _ = running
    errors = []
    thread.error.connect(errors.append)
    thread.submit(lambda engine: engine.targets.append(None))
    assert _wait(application, lambda: errors)
    assert thread.is_paused
    thread.call_sync(lambda engine: engine.targets.pop())


def test_commands_apply_immediately_when_not_running(application):
    engine = ScenarioLoader("radarsim/scenarios/basic_tracking.json").create_simulation_engine()
    thread = SimulationThread(engine)
    thread.submit(lambda e: setattr(e.radar, "prf_hz", 2500.0))
    assert engine.radar.prf_hz == 2500.0
