"""
Simulation controller: the single object the interface talks to.

It owns the engine and its thread, exposes typed operations (load, play,
change a radar parameter, record, replay ...) and publishes ``FrameState``
objects. Widgets never touch the engine.
"""

import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from src.io.exporter import export_scenario_to_yaml
from src.io.replay_loader import ReplayLoader
from src.io.scenario_loader import ScenarioLoader
from src.physics.geometry import polar_from_enu
from src.simulation.frame import C_LIGHT, FrameState, RadarFrame, TargetFrame
from src.simulation.recorder import FlightRecorder

from .thread_manager import SimulationThread

logger = logging.getLogger(__name__)

STOP_TIMEOUT_MS = 5000


class SimulationController(QObject):
    frame = Signal(object)  # FrameState
    scenario_loaded = Signal(object)  # ScenarioConfig
    mode_changed = Signal(str)  # "live", "paused", "replay"
    notice = Signal(str, str)  # text, kind ("info", "ok", "warning", "error")
    recording_changed = Signal(bool)
    replay_position = Signal(float, float)  # time, duration

    def __init__(self, output_dir: Path, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.output_dir = Path(output_dir)
        self.engine = None
        self.config = None
        self.scenario_path: Optional[str] = None
        self.thread: Optional[SimulationThread] = None
        self.latest: Optional[FrameState] = None
        self._paused = False
        self._speed = 1.0
        self._recorder: Optional[FlightRecorder] = None
        self._replay: Optional[ReplayLoader] = None
        self._replay_time = 0.0
        self._replay_timer = QTimer(self)
        self._replay_timer.setInterval(33)
        self._replay_timer.timeout.connect(self._advance_replay)

    # ---------------------------------------------------------------- scenario

    def load_scenario(self, path: str) -> bool:
        """Build the new engine first; on failure the current run continues."""
        try:
            loader = ScenarioLoader(path)
            self._install(loader, path)
        except Exception as exc:
            logger.exception("Scenario load failed")
            self.notice.emit(f"Could not load {Path(path).name}: {exc}", "error")
            return False
        for warning in loader.warnings:
            self.notice.emit(warning, "warning")
        return True

    def reload(self) -> None:
        if self.scenario_path:
            self.load_scenario(self.scenario_path)

    def new_scenario(self) -> None:
        """Empty scene around a default S-band surveillance radar."""
        data = {
            "scenario": {"name": "New scenario", "description": "", "duration_seconds": 300},
            "radar": {
                "name": "Surveillance radar",
                "frequency_hz": 3.0e9,
                "power_watts": 200e3,
                "prf_hz": 1000,
                "pulse_width_s": 1e-6,
                "antenna": {"gain_db": 35, "beamwidth_az_deg": 1.5, "beamwidth_el_deg": 3.0, "scan_rate_rpm": 12},
                "position": {"x_m": 0, "y_m": 0, "z_m": 20},
            },
            "targets": [],
        }
        self._install(ScenarioLoader.from_dict(data), None)

    def _install(self, loader: ScenarioLoader, path: Optional[str]) -> None:
        engine = loader.create_simulation_engine()
        self.close_replay(restart=False)
        self.stop_recording()
        self._stop_thread()
        self.engine, self.config, self.scenario_path = engine, loader.get_config(), path
        self.latest = None
        self.scenario_loaded.emit(self.config)
        self._start_thread()

    # ------------------------------------------------------------------ editor

    def begin_editing(self) -> None:
        """Pause and rewind so edits change the scenario's initial state."""
        self.pause()
        self.submit(lambda engine: engine.reset())

    def add_object(self, x_m: float, y_m: float) -> int:
        def apply(engine):
            from src.physics.rcs import SwerlingModel
            from src.simulation.objects import MotionModel, Target

            target_id = max((t.target_id for t in engine.targets), default=0) + 1
            radar = engine.radar.position
            heading = np.arctan2(radar[0] - x_m, radar[1] - y_m)  # toward the radar
            target = Target(
                target_id,
                position=np.array([x_m, y_m, 3000.0]),
                velocity=250.0 * np.array([np.sin(heading), np.cos(heading), 0.0]),
                rcs_m2=2.0,
                swerling_model=SwerlingModel.SWERLING_1,
                motion_model=MotionModel.CONSTANT_VELOCITY,
                name=f"Object {target_id}",
                target_type="aircraft",
                affiliation="hostile",
            )
            engine.add_target(target)
            return target_id

        return self.call(apply)

    def edit_object(self, target_id: int, fields: dict) -> None:
        def apply(engine):
            import copy as _copy

            from src.physics.rcs import SwerlingModel
            from src.simulation.objects import MotionModel

            target = next((t for t in engine.targets if t.target_id == target_id), None)
            if target is None:
                return
            state = target.state
            speed = float(np.hypot(state.velocity[0], state.velocity[1]))
            heading = float(np.arctan2(state.velocity[0], state.velocity[1]))
            if "name" in fields and fields["name"].strip():
                target.name = fields["name"].strip()
            if "affiliation" in fields:
                target.affiliation = fields["affiliation"]
            if "rcs_m2" in fields:
                target.rcs_mean = fields["rcs_m2"]
            if "swerling" in fields:
                target.swerling_model = SwerlingModel(fields["swerling"])
            if "altitude_m" in fields:
                state.position[2] = fields["altitude_m"]
            speed = fields.get("speed_mps", speed)
            heading = np.radians(fields["heading_deg"]) if "heading_deg" in fields else heading
            state.velocity = np.array([speed * np.sin(heading), speed * np.cos(heading), state.velocity[2]])
            if "turn_rate_deg_s" in fields:
                target.turn_rate_rad_s = np.radians(fields["turn_rate_deg_s"])
            if target.turn_rate_rad_s != 0.0:
                target.motion_model = MotionModel.COORDINATED_TURN
            elif speed > 0.0:
                target.motion_model = MotionModel.CONSTANT_VELOCITY
            else:
                target.motion_model = MotionModel.STATIC
            engine._initial_target_states[target_id] = _copy.deepcopy(state)

        self.submit(apply)

    def remove_object(self, target_id: int) -> None:
        def apply(engine):
            engine.targets = [t for t in engine.targets if t.target_id != target_id]
            engine.state.remove_target(target_id)
            engine._initial_target_states.pop(target_id, None)
            engine._last_looks.pop(target_id, None)

        self.submit(apply)

    def export_report(self, path: str) -> bool:
        from src.io.report import build_report, write_report

        name = self.config.name if self.config else ""
        try:
            report = self.call(lambda engine: build_report(engine, name))
            write_report(report, path)
        except Exception as exc:
            self.notice.emit(f"Could not write report: {exc}", "error")
            return False
        self.notice.emit(f"Report saved to {Path(path).name}", "ok")
        return True

    @property
    def terrain(self):
        return self.engine.terrain if self.engine is not None else None

    def export_scenario(self, path: str, name: str) -> bool:
        duration = self.config.duration_s if self.config else None
        try:
            ok = self.call(
                lambda engine: export_scenario_to_yaml(engine, path, name, duration_s=duration)
            )
        except Exception as exc:
            self.notice.emit(f"Could not save scenario: {exc}", "error")
            return False
        self.notice.emit(f"Scenario saved to {Path(path).name}" if ok else "Could not save scenario", "ok" if ok else "error")
        return bool(ok)

    # ----------------------------------------------------------------- running

    def _start_thread(self) -> None:
        self.thread = SimulationThread(self.engine, parent=self)
        self.thread.frame_ready.connect(self._on_frame)
        self.thread.paused_changed.connect(self._on_paused)
        self.thread.error.connect(lambda message: self.notice.emit(message, "error"))
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.set_speed(self._speed)
        if self._paused:
            self.thread.pause()
        self.thread.start()
        self.mode_changed.emit("paused" if self._paused else "live")
        if self._paused:
            self.submit(lambda engine: None)  # publish the initial picture

    def _stop_thread(self) -> None:
        thread, self.thread = self.thread, None
        if thread is None:
            return
        thread.frame_ready.disconnect(self._on_frame)
        thread.paused_changed.disconnect(self._on_paused)
        thread.stop()
        if not thread.wait(STOP_TIMEOUT_MS):
            logger.warning("Simulation thread did not stop in time")

    def shutdown(self) -> None:
        self.stop_recording()
        self.close_replay(restart=False)
        self._stop_thread()

    def _on_frame(self, frame: FrameState) -> None:
        self.latest = frame
        if self._recorder is not None:
            self._recorder.record_state(_record_dict(frame))
        self.frame.emit(frame)

    def _on_paused(self, paused: bool) -> None:
        self._paused = paused
        self.mode_changed.emit("paused" if paused else "live")

    def play(self) -> None:
        if self._replay is not None:
            self._replay_timer.start()
            self.mode_changed.emit("replay")
            return
        self._paused = False
        if self.thread:
            self.thread.resume()

    def pause(self) -> None:
        if self._replay is not None:
            self._replay_timer.stop()
            self.mode_changed.emit("replay")
            return
        self._paused = True
        if self.thread:
            self.thread.pause()

    def toggle(self) -> None:
        if self._replay is not None:
            self.pause() if self._replay_timer.isActive() else self.play()
        elif self._paused:
            self.play()
        else:
            self.pause()

    def step(self) -> None:
        if self.thread and self._paused:
            self.thread.step_once()

    def set_speed(self, factor: float) -> None:
        self._speed = factor
        if self.thread:
            self.thread.set_speed(factor)

    @property
    def paused(self) -> bool:
        return self._paused if self._replay is None else not self._replay_timer.isActive()

    @property
    def speed(self) -> float:
        return self._speed

    def submit(self, command: Callable) -> None:
        if self.thread is not None:
            self.thread.submit(command)
        elif self.engine is not None:
            command(self.engine)

    def call(self, command: Callable):
        if self.thread is not None:
            return self.thread.call_sync(command)
        return command(self.engine)

    def set_radar(self, **values) -> None:
        """Set radar attributes by name (SI units)."""

        def apply(engine):
            radar = engine.radar
            for key, value in values.items():
                if key == "beamwidth_az_rad":
                    radar.beamwidth_rad = value
                elif key == "beamwidth_el_rad":
                    radar.beamwidth_el_rad = value
                else:
                    setattr(radar, key, value)

        self.submit(apply)

    def set_engine(self, **values) -> None:
        """Set engine attributes by name (clutter_enabled, mti_enabled, ...)."""

        def apply(engine):
            for key, value in values.items():
                setattr(engine, key, value)

        self.submit(apply)

    def set_cfar(self, cfar_type: Optional[str], reference_cells: int = 16) -> None:
        from src.simulation.engine import CFARConfig

        config = CFARConfig(cfar_type, reference_cells) if cfar_type else None
        self.set_engine(cfar=config)

    def set_ecm(self, active: bool, technique: str = "per_target") -> None:
        self.submit(lambda engine: engine.set_ecm_mode(active, technique))

    def set_pulse_doppler(self, enabled: bool, mti_order: int = 0) -> None:
        self.submit(lambda engine: engine.set_pulse_doppler_mode(enabled, 64, None, mti_order))

    def set_include_rd_map(self, include: bool) -> None:
        if self.thread:
            self.thread.set_include_rd_map(include)

    # --------------------------------------------------------------- recording

    def start_recording(self) -> None:
        if self._recorder is not None or self.engine is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._recorder = FlightRecorder(output_dir=str(self.output_dir))
        radar = self.engine.radar
        self._recorder.start_recording(
            {
                "radar_frequency_hz": float(radar.frequency_hz),
                "radar_power_watts": float(radar.power_watts),
                "radar_gain_db": float(radar.antenna_gain_db),
                "radar_position_m": json.dumps([float(v) for v in radar.position]),
                "scan_rate_rpm": float(radar.scan_rate_rpm),
                "beamwidth_deg": float(np.degrees(radar.beamwidth_rad)),
                "scenario": self.config.name if self.config else "",
            }
        )
        self.recording_changed.emit(True)
        self.notice.emit("Recording started", "info")

    def stop_recording(self) -> Optional[str]:
        recorder, self._recorder = self._recorder, None
        if recorder is None:
            return None
        path = recorder.stop_recording()
        self.recording_changed.emit(False)
        if path:
            self.notice.emit(f"Recording saved: {Path(path).name}", "ok")
        return path

    @property
    def recording(self) -> bool:
        return self._recorder is not None

    # ------------------------------------------------------------------ replay

    def open_replay(self, path: str) -> bool:
        try:
            replay = ReplayLoader(path)
        except Exception as exc:
            self.notice.emit(f"Could not open recording: {exc}", "error")
            return False
        self.stop_recording()
        if self.thread:
            self.thread.pause()
        self._replay = replay
        self._replay_time = 0.0
        self.mode_changed.emit("replay")
        self._emit_replay_frame()
        self._replay_timer.start()
        return True

    def close_replay(self, restart: bool = True) -> None:
        if self._replay is None:
            return
        self._replay_timer.stop()
        self._replay.close()
        self._replay = None
        if restart and self.thread and not self._paused:
            self.thread.resume()
        self.mode_changed.emit("paused" if self._paused else "live")

    @property
    def replaying(self) -> bool:
        return self._replay is not None

    def seek(self, time_s: float) -> None:
        if self._replay is not None:
            self._replay_time = min(max(time_s, 0.0), self._replay.duration)
            self._emit_replay_frame()

    def _advance_replay(self) -> None:
        self._replay_time += self._replay_timer.interval() / 1000.0 * self._speed
        if self._replay_time >= self._replay.duration:
            self._replay_time = self._replay.duration
            self._replay_timer.stop()
        self._emit_replay_frame()

    def _emit_replay_frame(self) -> None:
        state = self._replay.get_state_at_time(self._replay_time)
        frame = _frame_from_replay(state, self._replay, self.latest)
        self.latest = frame
        self.frame.emit(frame)
        self.replay_position.emit(self._replay_time, self._replay.duration)


def _record_dict(frame: FrameState) -> dict:
    return {
        "time": frame.time,
        "targets": [
            {
                "id": t.target_id,
                "name": t.name,
                "affiliation": t.affiliation,
                "type": t.target_type,
                "position": list(t.position),
                "velocity": list(t.velocity),
                "rcs_m2": t.rcs_m2,
                "has_jammer": t.jammer_active,
                "range_km": t.range_m / 1000.0,
                "azimuth_rad": t.azimuth,
                "snr_db": t.snr_db,
                "is_detected": t.detected,
            }
            for t in frame.targets
        ],
    }


def _frame_from_replay(state, replay: ReplayLoader, template: Optional[FrameState]) -> FrameState:
    radar_position = tuple(float(v) for v in replay._radar_position)
    if template is not None:
        radar = replace(
            template.radar,
            position=radar_position,
            antenna_azimuth=float(state.beam_azimuth_rad),
            swept_legs=(),
        )
    else:
        radar = RadarFrame(
            name="Replay",
            position=radar_position,
            antenna_azimuth=float(state.beam_azimuth_rad),
            swept_legs=(),
            scan_type="circular",
            sector=(-1.0, 1.0),
            beamwidth_az=np.radians(2.0),
            beamwidth_el=np.radians(3.0),
            scan_rate_rpm=6.0,
            frequency_hz=float(state.frequency_hz),
            power_watts=float(state.power_watts),
            prf_hz=1000.0,
            pulse_width_s=1e-6,
            bandwidth_hz=1e6,
            unambiguous_range_m=C_LIGHT / 2000.0,
            range_resolution_m=150.0,
        )
    targets: List[TargetFrame] = []
    for t in state.targets:
        position = np.asarray(t.position, dtype=float)
        rng, az, el = polar_from_enu(position - np.asarray(radar_position))
        targets.append(
            TargetFrame(
                target_id=int(t.target_id) if str(t.target_id).isdigit() else hash(t.target_id) % 100000,
                name=t.name,
                target_type=t.target_type,
                affiliation=getattr(t, "affiliation", "unknown"),
                position=tuple(float(v) for v in position),
                velocity=tuple(float(v) for v in t.velocity),
                speed=float(t.speed_mps),
                heading=float(np.arctan2(t.velocity[0], t.velocity[1])),
                altitude=float(position[2]),
                range_m=rng,
                azimuth=az,
                elevation=el,
                radial_velocity=0.0,
                rcs_m2=float(t.rcs_m2),
                snr_db=float(t.snr_db),
                pd=0.0,
                looked_at=bool(t.is_detected),
                detected=bool(t.is_detected),
                measured_position=tuple(float(v) for v in position) if t.is_detected else None,
                masked=False,
                jammer_active=bool(t.has_jammer),
                ecm_type="",
                jammer_loss_db=0.0,
                clutter_loss_db=0.0,
            )
        )
    return FrameState(
        time=float(state.time),
        dt=0.033,
        radar=radar,
        targets=tuple(targets),
        tracks=(),
        false_targets=(),
        events=(),
    )
