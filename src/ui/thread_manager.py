"""
Simulation Thread Manager

Bridge between the simulation loop (background thread) and the UI (main thread).

Threading model:
    - While the thread runs, only the worker thread calls ``engine.step()`` or reads
      engine state to build frames.
    - The GUI never mutates the engine directly. It calls ``SimulationThread.submit``
      (fire-and-forget) or ``SimulationThread.call_sync`` (blocking, with result);
      both enqueue a callable that the worker executes between two steps.
    - When the thread is not running, commands execute immediately on the caller
      thread because nothing else touches the engine.
    - Frames reach the UI through the queued ``update_data`` signal.

Reference: Qt documentation, "Threads and QObjects".
"""

import logging
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.advanced.eccm import ECCMController
from src.physics.rcs import SwerlingModel
from src.simulation.engine import SimulationEngine
from src.simulation.objects import MotionModel, Radar, Target

logger = logging.getLogger(__name__)

EngineCommand = Callable[[SimulationEngine], Any]

# Longest wall-clock gap fed into one step. Larger gaps (GUI stalls, debugger
# breaks, window drags) would otherwise teleport targets in a single step.
MAX_FRAME_GAP_S = 0.25

MIN_SPEED_FACTOR = 0.1
MAX_SPEED_FACTOR = 10.0


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
    """
    Runs the simulation loop. Created in the GUI thread; ``run`` executes in the
    ``SimulationThread``.

    Signals:
        update_data: Emitted with the state dictionary after each step
        paused_changed: Emitted when the loop pauses or resumes (including on error)
        stopped: Emitted when the loop exits
        error: Emitted with a message when a step or command fails
    """

    update_data = Signal(dict)
    paused_changed = Signal(bool)
    stopped = Signal()
    error = Signal(str)

    def __init__(
        self,
        engine: SimulationEngine,
        update_rate_hz: float = 30.0,
        eccm_controller: Optional[ECCMController] = None,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.update_interval = 1.0 / update_rate_hz
        self.eccm_controller = eccm_controller
        self._commands: "queue.SimpleQueue[_PendingCall]" = queue.SimpleQueue()
        self._running = False
        self._paused = False
        self._speed_factor = 1.0

    # ------------------------------------------------------------------ loop

    def run(self) -> None:
        """Main simulation loop; executes in the worker thread."""
        self._running = True
        last_update = time.perf_counter()

        while self._running:
            self._drain_commands()
            now = time.perf_counter()

            if self._paused:
                # Keep the clock current so resuming does not produce one huge step.
                last_update = now
            elif now - last_update >= self.update_interval:
                dt = min(now - last_update, MAX_FRAME_GAP_S) * self._speed_factor
                last_update = now
                try:
                    results = self.engine.step(dt)
                    state = self._build_state_dict(results, dt)
                except Exception as exc:
                    logger.exception("Simulation step failed")
                    self._set_paused(True)
                    self.error.emit(f"Simulation paused after an error: {exc}")
                    continue
                self.update_data.emit(state)

            time.sleep(0.005)

        self._drain_commands()
        self.stopped.emit()

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

    # --------------------------------------------------------------- control

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

    def set_speed(self, factor: float) -> None:
        self._speed_factor = float(np.clip(factor, MIN_SPEED_FACTOR, MAX_SPEED_FACTOR))

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def speed_factor(self) -> float:
        return self._speed_factor

    # ----------------------------------------------------------------- state

    def _build_state_dict(self, results, dt: float) -> Dict[str, Any]:
        """Build the frame dictionary sent to the UI. Runs in the worker thread."""
        targets_data = []
        for target in self.engine.active_targets:
            geom = self.engine.radar.calculate_target_geometry(
                target.position, target.velocity
            )

            is_detected = self.engine.state.detections.get(target.target_id, False)
            snr = self.engine.state.snr_values.get(target.target_id, 0.0)

            targets_data.append(
                {
                    "id": target.target_id,
                    "name": target.name,
                    "type": target.target_type,
                    "position": target.position.tolist(),
                    "velocity": target.velocity.tolist(),
                    "velocity_mps": target.speed,
                    "heading_rad": target.state.heading,
                    "altitude_m": target.state.altitude,
                    "has_jammer": target.has_jammer,
                    "jammer_active": target.jammer_active,
                    "terrain_masked": self.engine.state.terrain_masked.get(
                        target.target_id, False
                    ),
                    "range_m": geom["range_m"],
                    "range_km": geom["range_m"] / 1000,
                    "azimuth_rad": geom["azimuth_rad"],
                    "azimuth_deg": geom["azimuth_deg"],
                    "elevation_rad": geom["elevation_rad"],
                    "elevation_deg": geom["elevation_deg"],
                    "radial_velocity_mps": geom["radial_velocity_mps"],
                    "rcs_m2": target.rcs_mean,
                    "is_detected": is_detected,
                    "snr_db": snr,
                }
            )

        tracks_data = []
        if self.engine.enable_tracking:
            for track in self.engine.tracks:
                track_dict = {
                    "track_id": track.id,
                    "id": track.id,
                    "status": track.status.value,
                    "position": list(track.position),
                    "velocity": list(track.velocity),
                    "speed_mps": track.speed_mps,
                    "heading_rad": track.heading_rad,
                    "hits": track.hits,
                    "misses": track.misses,
                    "history": track.history[-20:],
                    "classification": track.classification,
                    "confidence": track.confidence,
                }
                # Covariance drives the EKF uncertainty ellipse.
                if getattr(track.state, "P", None) is not None:
                    track_dict["covariance"] = track.state.P.tolist()
                tracks_data.append(track_dict)

        false_targets_data = [
            {
                "target_id": ft.false_id,
                "position": np.asarray(ft.position).tolist(),
                "velocity": np.asarray(ft.velocity).tolist(),
                "rcs_m2": ft.rcs_m2,
                "ecm_type": ft.ecm_type,
                "is_false_target": True,
            }
            for ft in self.engine.false_targets
        ]

        rd_map_data = None
        pd_metadata = None
        rdm = getattr(self.engine, "_rd_map", None)
        if getattr(self.engine, "pulse_doppler_enabled", False) and rdm is not None:
            rd_map_data = rdm.data_db.tolist()
            pd_metadata = {
                "range_axis_m": rdm.range_axis_m.tolist(),
                "velocity_axis_mps": rdm.velocity_axis_mps.tolist(),
                "blind_speeds_mps": rdm.blind_speeds_mps,
                "mti_order": rdm.mti_order,
                "processing_gain_db": rdm.processing_gain_db,
                "n_pulses": rdm.n_pulses,
                "prf_hz": rdm.prf_hz,
                "wavelength_m": rdm.wavelength_m,
                "sample_rate_hz": rdm.sample_rate_hz,
                "range_sample_spacing_m": rdm.range_sample_spacing_m,
                "max_instrumented_range_m": rdm.max_instrumented_range_m,
                "max_unambiguous_range_m": rdm.max_unambiguous_range_m,
                "max_unambiguous_velocity_mps": rdm.max_unambiguous_velocity_mps,
                "coherent_processing_gain_db": rdm.coherent_processing_gain_db,
                "window_enbw_bins": rdm.window_enbw_bins,
            }

        eccm_state = (
            self.eccm_controller.get_status()
            if self.eccm_controller is not None
            else {"jamming_active": False, "effective_jsr_db": -100}
        )

        return {
            "time": self.engine.simulation_time,
            "dt": dt,
            "radar": {
                "position": self.engine.radar.position.tolist(),
                "antenna_azimuth_rad": self.engine.radar.antenna_azimuth,
                "antenna_azimuth_deg": np.degrees(self.engine.radar.antenna_azimuth),
                "frequency_ghz": self.engine.radar.frequency_hz / 1e9,
                "power_kw": self.engine.radar.power_watts / 1e3,
            },
            "targets": targets_data,
            "tracks": tracks_data,
            "detection_count": sum(1 for t in targets_data if t["is_detected"]),
            "total_targets": len(targets_data),
            "total_tracks": len(tracks_data),
            "jamming_active": getattr(self.engine, "ecm_active", False),
            "ecm_type": getattr(self.engine, "ecm_type", "noise"),
            "false_targets": false_targets_data,
            "pulse_doppler_enabled": getattr(self.engine, "pulse_doppler_enabled", False),
            "rd_map": rd_map_data,
            "pd_metadata": pd_metadata,
            "eccm": eccm_state,
            "log": {
                "total_opportunities": self.engine.log.total_opportunities,
                "total_detections": self.engine.log.total_detections,
                "detection_ratio": self.engine.log.detection_ratio,
            },
        }


class SimulationThread(QThread):
    """
    Thread that runs a ``SimulationWorker``; the GUI's only door to the engine.

    Usage:
        thread = SimulationThread(engine, parent=window)
        thread.update_data.connect(window.on_frame)
        thread.submit(lambda engine: setattr(engine.radar, "power_watts", 2e5))
        thread.start()
    """

    update_data = Signal(dict)
    paused_changed = Signal(bool)
    stopped = Signal()
    error = Signal(str)

    def __init__(
        self,
        engine: SimulationEngine,
        update_rate_hz: float = 30.0,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.update_rate_hz = update_rate_hz
        self.eccm_controller = ECCMController(
            center_freq_hz=engine.radar.frequency_hz,
            nominal_prf_hz=engine.radar.prf_hz,
        )
        self._worker = SimulationWorker(
            engine, update_rate_hz, eccm_controller=self.eccm_controller
        )
        self._worker.update_data.connect(self.update_data)
        self._worker.paused_changed.connect(self.paused_changed)
        self._worker.stopped.connect(self.stopped)
        self._worker.error.connect(self.error)

    def run(self) -> None:
        self._worker.run()

    # ------------------------------------------------------ engine commands

    def submit(self, command: EngineCommand) -> None:
        """Apply ``command(engine)`` between steps; runs at once if not running."""
        call = _PendingCall(command, synchronous=False)
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

    # ------------------------------------------------------- loop control

    def stop(self) -> None:
        self._worker.stop()

    def pause(self) -> None:
        self._worker.pause()

    def resume(self) -> None:
        self._worker.resume()

    def set_speed(self, factor: float) -> None:
        self._worker.set_speed(factor)

    @property
    def is_paused(self) -> bool:
        return self._worker.is_paused

    @property
    def speed_factor(self) -> float:
        return self._worker.speed_factor

    # ---------------------------------------------------- feature toggles

    def set_ecm_state(
        self, active: bool, ecm_type: str = "noise", target_id: Optional[int] = None
    ) -> None:
        """Set engine ECM mode and, optionally, one target's jammer state."""

        def apply(engine: SimulationEngine) -> None:
            engine.set_ecm_mode(active, ecm_type)
            if target_id is not None:
                for target in engine.targets:
                    if target.target_id == target_id:
                        target.jammer_active = active
                        break

        self.submit(apply)

    def set_lpi_mode(self, enabled: bool, technique: str = "FHSS") -> None:
        def apply(engine: SimulationEngine) -> None:
            engine.lpi_enabled = enabled
            engine.lpi_technique = technique

        self.submit(apply)

    def set_fusion_mode(
        self, enabled: bool, method: str = "covariance_intersection"
    ) -> None:
        def apply(engine: SimulationEngine) -> None:
            engine.fusion_enabled = enabled
            engine.fusion_method = method

        self.submit(apply)

    def set_clutter_mode(self, enabled: bool, terrain_type: Optional[str] = None) -> None:
        def apply(engine: SimulationEngine) -> None:
            engine.clutter_enabled = enabled
            if terrain_type is not None:
                engine.terrain_type = terrain_type

        self.submit(apply)

    def set_mti_mode(self, enabled: bool, threshold_mps: float = 15.0) -> None:
        def apply(engine: SimulationEngine) -> None:
            engine.mti_enabled = enabled
            engine.mti_threshold_mps = threshold_mps

        self.submit(apply)

    def set_eccm_agility(self, enabled: bool) -> None:
        self.submit(lambda engine: setattr(engine, "frequency_agility_enabled", enabled))

    def set_monopulse_mode(self, enabled: bool) -> None:
        self.submit(lambda engine: setattr(engine, "monopulse_enabled", enabled))

    def set_pulse_doppler_mode(
        self,
        enabled: bool,
        n_pulses: int = 64,
        prf_hz: float = 1000.0,
        mti_order: int = 0,
    ) -> None:
        """Toggle coherent CPI processing (Richards 2005, Ch. 17)."""
        self.submit(
            lambda engine: engine.set_pulse_doppler_mode(
                enabled, n_pulses, prf_hz, mti_order
            )
        )

    def set_ekf_mode(self, enabled: bool) -> None:
        """Switch the tracker between EKF (polar) and linear KF (Bar-Shalom 2001, 5.3)."""

        def apply(engine: SimulationEngine) -> None:
            if engine.track_manager is not None:
                engine.track_manager.set_ekf_mode(enabled)

        self.submit(apply)

    def set_jamming_environment(self, enabled: bool) -> None:
        self.submit(
            lambda _engine: self.eccm_controller.set_jamming_environment(
                enabled, jsr_db=20.0
            )
        )

    def set_freq_agility(self, enabled: bool) -> None:
        if enabled:
            self.submit(lambda _engine: self.eccm_controller.enable_frequency_agility())
        else:
            self.submit(lambda _engine: self.eccm_controller.disable_frequency_agility())

    def set_prf_stagger(self, enabled: bool) -> None:
        if enabled:
            self.submit(lambda _engine: self.eccm_controller.enable_prf_stagger())
        else:
            self.submit(lambda _engine: self.eccm_controller.disable_prf_stagger())


def create_demo_scenario() -> SimulationEngine:
    """
    Create a demo scenario for testing.

    Returns:
        Configured SimulationEngine with radar and targets
    """
    # Create radar at origin - Professional Ground-Based Search Radar specs
    # Reference: AN/TPS-78 class radar parameters
    radar = Radar(
        radar_id="MAIN_RADAR",
        position=np.array([0.0, 0.0, 0.0]),
        frequency_hz=3e9,  # S-band (3 GHz) - better for long range detection
        power_watts=500e3,  # 500 kW peak power
        antenna_gain_db=40.0,  # 40 dB - large aperture antenna
        beamwidth_deg=1.5,
        scan_rate_rpm=6.0,  # 6 RPM = 10 sec per revolution
    )

    # Create multiple targets
    targets = [
        # Aircraft approaching from the North (ENU: x = East, y = North, z = Up)
        Target(
            target_id=1,
            position=np.array([5000.0, 80000.0, 3000.0]),  # 80 km N, 5 km E, 3 km alt
            velocity=np.array([10.0, -150.0, 0.0]),  # Approaching at 150 m/s
            rcs_m2=5.0,
            target_type="aircraft",
            swerling_model=SwerlingModel.SWERLING_1,
            motion_model=MotionModel.CONSTANT_VELOCITY,
        ),
        # Fast mover from the East
        Target(
            target_id=2,
            position=np.array([60000.0, 30000.0, 5000.0]),  # 60 km E, 30 km N
            velocity=np.array([-200.0, -50.0, 0.0]),  # Crossing
            rcs_m2=2.0,
            target_type="aircraft",
            swerling_model=SwerlingModel.SWERLING_1,
            motion_model=MotionModel.CONSTANT_VELOCITY,
        ),
        # Small target from the South
        Target(
            target_id=3,
            position=np.array([20000.0, -40000.0, 2000.0]),
            velocity=np.array([-20.0, 100.0, 0.0]),
            rcs_m2=0.5,
            target_type="missile",
            swerling_model=SwerlingModel.SWERLING_3,
            motion_model=MotionModel.CONSTANT_VELOCITY,
        ),
        # Static target for calibration
        Target(
            target_id=4,
            position=np.array([-25000.0, -25000.0, 0.0]),  # 35 km SW
            velocity=np.array([0.0, 0.0, 0.0]),
            rcs_m2=10.0,
            target_type="ship",
            swerling_model=SwerlingModel.SWERLING_0,
            motion_model=MotionModel.STATIC,
        ),
    ]

    # Create engine
    engine = SimulationEngine(
        radar=radar,
        targets=targets,
        dt=0.033,  # ~30 FPS
        enable_atmospheric=True,
        detection_threshold_db=13.0,
    )

    return engine
