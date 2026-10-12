"""
Simulation thread.

Threading model:
    - While the thread runs, only the worker thread calls ``engine.step()`` or
      reads engine state; each step is published as an immutable ``FrameState``.
    - The GUI never mutates the engine directly. It calls ``submit`` (fire and
      forget) or ``call_sync`` (blocking, with result); both enqueue a callable
      that the worker runs between two steps.
    - When the thread is not running, commands execute immediately on the
      caller's thread because nothing else touches the engine.
    - Simulation time advances in fixed engine steps (``engine.dt``); the speed
      factor changes how many steps run per wall-clock second, so results do
      not depend on how fast the machine is.

Reference: Qt documentation, "Threads and QObjects".
"""

import logging
import queue
import threading
import time
from dataclasses import replace
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal

from radarsim.simulation.engine import SimulationEngine
from radarsim.simulation.frame import build_frame

logger = logging.getLogger(__name__)

EngineCommand = Callable[[SimulationEngine], Any]

MIN_SPEED_FACTOR = 0.1
MAX_SPEED_FACTOR = 20.0

# Wall-clock budget the worker may spend catching up in one go [s]; beyond
# it the simulation slows down instead of freezing the machine.
MAX_CATCH_UP_S = 0.1

# Frames published per second (independent of the engine step rate).
PUBLISH_RATE_HZ = 30.0


class _PendingCall:
    """A queued engine command and, for synchronous calls, its completion state."""

    __slots__ = ("command", "done", "result", "error")

    def __init__(self, command: EngineCommand, synchronous: bool):
        self.command = command
        self.done = threading.Event() if synchronous else None
        self.result: Any = None
        self.error: Optional[BaseException] = None


def _execute(call: _PendingCall, engine: SimulationEngine) -> None:
    try:
        call.result = call.command(engine)
    except Exception as exc:  # reported to the caller or via the error signal
        logger.exception("Engine command failed")
        call.error = exc
    finally:
        if call.done is not None:
            call.done.set()


class SimulationWorker(QObject):
    """Runs the stepping loop; ``run`` executes inside ``SimulationThread``."""

    frame_ready = Signal(object)
    paused_changed = Signal(bool)
    stopped = Signal()
    error = Signal(str)

    def __init__(self, engine: SimulationEngine, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.engine = engine
        self._commands: queue.SimpleQueue[_PendingCall] = queue.SimpleQueue()
        self._running = False
        self._paused = False
        self._speed_factor = 1.0
        self._single_steps = 0
        self.include_rd_map = True

    def run(self) -> None:
        self._running = True
        sim_debt = 0.0  # simulated seconds owed to wall-clock time
        last = time.perf_counter()
        last_publish = 0.0
        events = []

        while self._running:
            self._drain_commands()
            now = time.perf_counter()
            elapsed, last = now - last, now
            dt = self.engine.dt

            if self._paused:
                sim_debt = 0.0
                steps, self._single_steps = self._single_steps, 0
            else:
                sim_debt += min(elapsed, MAX_CATCH_UP_S) * self._speed_factor
                steps = int(sim_debt / dt)
                sim_debt -= steps * dt

            if steps:
                deadline = now + MAX_CATCH_UP_S
                try:
                    for _ in range(steps):
                        self.engine.step()
                        events.extend(self.engine.events)
                        if time.perf_counter() > deadline:
                            sim_debt = 0.0  # machine too slow: drop the backlog
                            break
                except Exception as exc:
                    logger.exception("Simulation step failed")
                    self._set_paused(True)
                    self.error.emit(f"Simulation paused after an error: {exc}")
                    continue

                if now - last_publish >= 1.0 / PUBLISH_RATE_HZ or self._paused:
                    frame = build_frame(self.engine, dt, include_rd_map=self.include_rd_map)
                    frame = replace(frame, events=tuple(events))
                    events = []
                    self.frame_ready.emit(frame)
                    last_publish = now

            time.sleep(0.004)

        self._drain_commands()
        self.stopped.emit()

    def publish_now(self) -> None:
        """Emit a frame of the current state (e.g. after a command while paused)."""
        frame = build_frame(self.engine, self.engine.dt, self.include_rd_map)
        self.frame_ready.emit(replace(frame, events=()))

    def enqueue(self, call: _PendingCall) -> None:
        self._commands.put(call)

    def _drain_commands(self) -> None:
        while True:
            try:
                call = self._commands.get_nowait()
            except queue.Empty:
                return
            _execute(call, self.engine)
            if call.error is not None and call.done is None:
                self.error.emit(str(call.error))

    def stop(self) -> None:
        self._running = False

    def _set_paused(self, paused: bool) -> None:
        if paused != self._paused:
            self._paused = paused
            self.paused_changed.emit(paused)

    def pause(self) -> None:
        self._set_paused(True)

    def resume(self) -> None:
        self._set_paused(False)

    def step_once(self) -> None:
        self._single_steps += 1

    def set_speed(self, factor: float) -> None:
        self._speed_factor = min(max(float(factor), MIN_SPEED_FACTOR), MAX_SPEED_FACTOR)

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def speed_factor(self) -> float:
        return self._speed_factor


class SimulationThread(QThread):
    """Thread that runs a ``SimulationWorker``; the GUI's only door to the engine."""

    frame_ready = Signal(object)
    paused_changed = Signal(bool)
    stopped = Signal()
    error = Signal(str)

    def __init__(self, engine: SimulationEngine, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.engine = engine
        self._worker = SimulationWorker(engine)
        self._worker.frame_ready.connect(self.frame_ready)
        self._worker.paused_changed.connect(self.paused_changed)
        self._worker.stopped.connect(self.stopped)
        self._worker.error.connect(self.error)

    def run(self) -> None:
        self._worker.run()

    def submit(self, command: EngineCommand, republish: bool = True) -> None:
        """Apply ``command(engine)`` between steps (at once if not running)."""

        def wrapped(engine):
            result = command(engine)
            if republish and self.isRunning() and self._worker.is_paused:
                self._worker.publish_now()
            return result

        call = _PendingCall(wrapped, synchronous=False)
        if self.isRunning():
            self._worker.enqueue(call)
            return
        _execute(call, self.engine)
        if call.error is not None:
            self.error.emit(str(call.error))

    def call_sync(self, command: EngineCommand, timeout_s: float = 5.0) -> Any:
        """Run ``command(engine)`` between steps and return its result (blocking)."""
        call = _PendingCall(command, synchronous=True)
        if self.isRunning():
            self._worker.enqueue(call)
            if not call.done.wait(timeout_s):
                raise TimeoutError("simulation thread did not respond")
        else:
            _execute(call, self.engine)
        if call.error is not None:
            raise call.error
        return call.result

    def stop(self) -> None:
        self._worker.stop()

    def pause(self) -> None:
        self._worker.pause()

    def resume(self) -> None:
        self._worker.resume()

    def step_once(self) -> None:
        self._worker.step_once()

    def set_speed(self, factor: float) -> None:
        self._worker.set_speed(factor)

    def set_include_rd_map(self, include: bool) -> None:
        self._worker.include_rd_map = include

    @property
    def is_paused(self) -> bool:
        return self._worker.is_paused

    @property
    def speed_factor(self) -> float:
        return self._worker.speed_factor
