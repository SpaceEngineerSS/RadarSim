"""
Simulation Thread Manager

Bridge between simulation loop (background thread) and UI (main thread).

Uses PySide6 signals/slots for thread-safe communication.

Architecture:
    - SimulationWorker runs engine.step() in a QThread
    - Emits update_data signal with current state
    - UI connects to signal to update visualization

Reference: PySide6 Threading Best Practices
"""

import os

# Import simulation engine
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics.rcs import SwerlingModel
from src.simulation.engine import SimulationEngine
from src.simulation.objects import MotionModel, Radar, Target


class SimulationWorker(QObject):
    """
    Worker object that runs simulation in background thread.

    Emits signals for UI updates without blocking the main thread.

    Signals:
        update_data: Emitted with current simulation state
        finished: Emitted when simulation stops
        error: Emitted on error with message
    """

    # Signals (must be class attributes)
    update_data = Signal(dict)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        engine: SimulationEngine,
        update_rate_hz: float = 30.0,
        parent: Optional[QObject] = None,
    ):
        """
        Initialize simulation worker.

        Args:
            engine: SimulationEngine instance
            update_rate_hz: UI update rate [Hz]
            parent: Parent QObject
        """
        super().__init__(parent)
        self.engine = engine
        self.update_interval = 1.0 / update_rate_hz
        self._running = False
        self._paused = False
        self._speed_factor = 1.0

    def run(self):
        """
        Main simulation loop.

        Runs engine.step() at specified rate and emits update signals.
        """
        self._running = True
        last_update = time.perf_counter()

        while self._running:
            try:
                current_time = time.perf_counter()
                elapsed = current_time - last_update

                if not self._paused and elapsed >= self.update_interval:
                    # Calculate dt based on speed factor
                    dt = elapsed * self._speed_factor

                    # Step simulation
                    results = self.engine.step(dt)

                    # Build state dictionary for UI
                    state = self._build_state_dict(results)

                    # Emit update signal
                    self.update_data.emit(state)

                    last_update = current_time

                # Small sleep to prevent CPU spinning - increased for stability
                time.sleep(0.01)  # 10ms sleep instead of 1ms

            except Exception as e:
                self.error.emit(str(e))
                break

        self.finished.emit()

    def _build_state_dict(self, results) -> Dict[str, Any]:
        """
        Build state dictionary for UI update.

        Returns:
            Dict containing all data needed for visualization
        """
        targets_data = []
        detections_for_tracker = []  # (x, y) positions for tracker

        for target in self.engine.targets:
            geom = self.engine.radar.calculate_target_geometry(
                target.position, target.velocity
            )

            # Get detection status for this target
            is_detected = self.engine.state.detections.get(target.target_id, False)
            snr = self.engine.state.snr_values.get(target.target_id, 0.0)

            target_data = {
                "id": target.target_id,
                "name": target.target_type,  # For MIL-STD-2525 affiliation
                "type": target.target_type,
                "position": target.position.tolist(),
                "velocity": target.velocity.tolist(),
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
            targets_data.append(target_data)

            # Collect detections for tracker
            if is_detected:
                detections_for_tracker.append((target.position[0], target.position[1]))

        # ═══ TRACK-WHILE-SCAN UPDATE ═══
        tracks_data = []
        if (
            hasattr(self.engine, "track_manager")
            and self.engine.track_manager is not None
        ):
            try:
                # Update tracker with detections
                tracks = self.engine.track_manager.update(
                    detections_for_tracker, dt=self.engine.dt
                )

                # Build track data for UI
                for track in tracks:
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
                        "history": track.history[-20:],  # Last 20 positions
                        "classification": track.classification,
                        "confidence": track.confidence,
                    }
                    # Phase 27: Add covariance for EKF uncertainty ellipse
                    if hasattr(track.state, "P") and track.state.P is not None:
                        track_dict["covariance"] = track.state.P.tolist()
                    tracks_data.append(track_dict)
            except Exception:
                pass  # Fail silently

        # Build false targets data for visualization (with safety)
        false_targets_data = []
        try:
            for ft in self.engine.false_targets[:20]:  # Cap at 20 for safety
                false_targets_data.append(
                    {
                        "target_id": ft.false_id,
                        "position": (
                            ft.position.tolist()
                            if hasattr(ft.position, "tolist")
                            else list(ft.position)
                        ),
                        "velocity": (
                            ft.velocity.tolist()
                            if hasattr(ft.velocity, "tolist")
                            else list(ft.velocity)
                        ),
                        "rcs_m2": ft.rcs_m2,
                        "ecm_type": ft.ecm_type,
                        "is_false_target": True,
                    }
                )
        except Exception:
            pass  # Fail silently to prevent crash

        # ═══ PHASE 26: PULSE-DOPPLER R-D MAP ═══
        rd_map_data = None
        pd_metadata = None
        if (
            getattr(self.engine, "pulse_doppler_enabled", False)
            and self.engine._rd_map is not None
        ):
            rdm = self.engine._rd_map
            rd_map_data = rdm.data_db.tolist()  # Serialize for signal
            pd_metadata = {
                "range_axis_m": rdm.range_axis_m.tolist(),
                "velocity_axis_mps": rdm.velocity_axis_mps.tolist(),
                "blind_speeds_mps": rdm.blind_speeds_mps,
                "mti_order": rdm.mti_order,
                "processing_gain_db": rdm.processing_gain_db,
                "n_pulses": rdm.n_pulses,
                "prf_hz": rdm.prf_hz,
                "wavelength_m": rdm.wavelength_m,
            }

        return {
            "time": self.engine.simulation_time,
            "radar": {
                "position": self.engine.radar.position.tolist(),
                "antenna_azimuth_rad": self.engine.radar.antenna_azimuth,
                "antenna_azimuth_deg": np.degrees(self.engine.radar.antenna_azimuth),
                "frequency_ghz": self.engine.radar.frequency_hz / 1e9,
                "power_kw": self.engine.radar.power_watts / 1e3,
            },
            "targets": targets_data,
            "tracks": tracks_data,  # Track-While-Scan data
            "detection_count": sum(1 for t in targets_data if t["is_detected"]),
            "total_targets": len(targets_data),
            "total_tracks": len(tracks_data),
            # ECM state for visualization
            "jamming_active": getattr(self.engine, "ecm_active", False),
            "ecm_type": getattr(self.engine, "ecm_type", "noise"),
            "false_targets": false_targets_data,
            # Phase 26: Pulse-Doppler
            "pulse_doppler_enabled": getattr(
                self.engine, "pulse_doppler_enabled", False
            ),
            "rd_map": rd_map_data,
            "pd_metadata": pd_metadata,
            # Phase 28: ECCM state
            "eccm": self._get_eccm_state(),
            "log": {
                "total_opportunities": self.engine.log.total_opportunities,
                "total_detections": self.engine.log.total_detections,
                "detection_ratio": self.engine.log.detection_ratio,
            },
        }

    def _get_eccm_state(self) -> dict:
        """
        Get ECCM state for UI.

        Returns dict with jamming/ECCM status for the control panel.
        Lazily initializes the ECCM controller on first call.
        """
        # Lazy init
        if not hasattr(self, "_eccm_controller"):
            try:
                from src.advanced.eccm import ECCMController

                self._eccm_controller = ECCMController(
                    center_freq_hz=getattr(self.engine, "frequency_hz", 10e9),
                    nominal_prf_hz=getattr(self.engine, "prf_hz", 1000.0),
                )
            except (ImportError, Exception):
                self._eccm_controller = None

        if self._eccm_controller is not None:
            try:
                return self._eccm_controller.get_status()
            except Exception:
                pass
        return {"jamming_active": False, "effective_jsr_db": -100}

    def stop(self):
        """Stop the simulation loop."""
        self._running = False

    def pause(self):
        """Pause the simulation."""
        self._paused = True

    def resume(self):
        """Resume the simulation."""
        self._paused = False

    def set_speed(self, factor: float):
        """Set simulation speed factor."""
        self._speed_factor = max(0.1, min(10.0, factor))

    @property
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self._running

    @property
    def is_paused(self) -> bool:
        """Check if simulation is paused."""
        return self._paused


class SimulationThread(QThread):
    """
    Thread wrapper for SimulationWorker.

    Provides clean thread lifecycle management.

    Usage:
        thread = SimulationThread(engine)
        thread.update_data.connect(ui.update_display)
        thread.start()
    """

    # Forward signals from worker
    update_data = Signal(dict)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        engine: SimulationEngine,
        update_rate_hz: float = 30.0,
        parent: Optional[QObject] = None,
    ):
        """
        Initialize simulation thread.

        Args:
            engine: SimulationEngine instance
            update_rate_hz: UI update rate [Hz]
            parent: Parent QObject
        """
        super().__init__(parent)
        self.engine = engine
        self.update_rate_hz = update_rate_hz
        self._worker: Optional[SimulationWorker] = None

    def run(self):
        """Thread entry point."""
        self._worker = SimulationWorker(self.engine, self.update_rate_hz)

        # Connect worker signals to thread signals
        self._worker.update_data.connect(self.update_data.emit)
        self._worker.finished.connect(self.finished.emit)
        self._worker.error.connect(self.error.emit)

        # Run worker
        self._worker.run()

    def stop(self):
        """Stop the simulation."""
        if self._worker:
            self._worker.stop()

    def pause(self):
        """Pause the simulation."""
        if self._worker:
            self._worker.pause()

    def resume(self):
        """Resume the simulation."""
        if self._worker:
            self._worker.resume()

    def set_speed(self, factor: float):
        """Set simulation speed factor."""
        if self._worker:
            self._worker.set_speed(factor)

    def set_ecm_state(
        self, active: bool, ecm_type: str = "noise", target_id: Optional[int] = None
    ):
        """
        Set ECM state in the simulation engine (thread-safe).

        Args:
            active: Whether ECM is active
            ecm_type: Type of ECM ('noise_barrage', 'noise_spot', 'drfm', 'chaff', 'decoy')
            target_id: Optional target ID to update jammer state
        """
        try:
            # Update engine ECM mode
            self.engine.set_ecm_mode(active, ecm_type)

            # Update specific target jammer state if provided
            if target_id is not None:
                for target in self.engine.targets:
                    if target.target_id == target_id:
                        target.jammer_active = active
                        break
        except Exception as e:
            print(f"[ECM] Engine update failed: {e}")

    def set_lpi_mode(self, enabled: bool, technique: str = "FHSS"):
        """
        Set LPI (Low Probability of Intercept) mode in the simulation engine.

        Args:
            enabled: Whether LPI mode is active
            technique: LPI technique ('FHSS', 'DSSS', 'Costas')
        """
        try:
            # Store LPI state in engine
            self.engine.lpi_enabled = enabled
            self.engine.lpi_technique = technique

            # LPI mode reduces peak power but spreads energy for lower detectability
            # Scientific basis: LPI radars trade SNR for reduced probability of intercept
            if enabled:
                print(f"[LPI] Mode ACTIVATED | Technique: {technique}")
            else:
                print("[LPI] Mode DEACTIVATED")
        except Exception as e:
            print(f"[LPI] Engine update failed: {e}")

    def set_fusion_mode(self, enabled: bool, method: str = "kalman"):
        """
        Set sensor fusion mode in the simulation engine.

        Args:
            enabled: Whether sensor fusion is active
            method: Fusion method ('kalman', 'particle', 'bayesian')
        """
        try:
            # Store fusion state in engine
            self.engine.fusion_enabled = enabled
            self.engine.fusion_method = method

            if enabled:
                print(f"[FUSION] Mode ACTIVATED | Method: {method}")
            else:
                print("[FUSION] Mode DEACTIVATED")
        except Exception as e:
            print(f"[FUSION] Engine update failed: {e}")

    # ═══ PHASE 19: CLUTTER, MTI & ECCM CONTROLS ═══

    def set_clutter_mode(self, enabled: bool, terrain_type: str = "rural"):
        """
        Set environmental clutter mode.

        Args:
            enabled: Whether clutter is active
            terrain_type: Terrain type for clutter calculation
        """
        try:
            self.engine.clutter_enabled = enabled
            self.engine.terrain_type = terrain_type

            if enabled:
                print(f"[CLUTTER] Mode ACTIVATED | Terrain: {terrain_type}")
            else:
                print("[CLUTTER] Mode DEACTIVATED")
        except Exception as e:
            print(f"[CLUTTER] Engine update failed: {e}")

    def set_mti_mode(self, enabled: bool, threshold_mps: float = 15.0):
        """
        Set MTI (Moving Target Indication) filter mode.

        Args:
            enabled: Whether MTI filter is active
            threshold_mps: Velocity threshold below which targets are rejected [m/s]
        """
        try:
            self.engine.mti_enabled = enabled
            self.engine.mti_threshold_mps = threshold_mps

            if enabled:
                print(f"[MTI] Filter ACTIVATED | Threshold: {threshold_mps} m/s")
            else:
                print("[MTI] Filter DEACTIVATED")
        except Exception as e:
            print(f"[MTI] Engine update failed: {e}")

    def set_eccm_agility(self, enabled: bool):
        """
        Set ECCM Frequency Agility mode.

        Frequency agility hops the radar frequency to defeat spot jammers.

        Args:
            enabled: Whether frequency agility is active
        """
        try:
            self.engine.frequency_agility_enabled = enabled

            if enabled:
                print("[ECCM] Frequency Agility ACTIVATED")
            else:
                print("[ECCM] Frequency Agility DEACTIVATED")
        except Exception as e:
            print(f"[ECCM] Engine update failed: {e}")

    # ═══ PHASE 20: MONOPULSE TRACKING ═══

    def set_monopulse_mode(self, enabled: bool):
        """
        Set Monopulse angle tracking mode.

        Monopulse provides sub-beamwidth angular accuracy using
        Sum/Difference pattern processing.

        Args:
            enabled: Whether monopulse tracking is active
        """
        try:
            self.engine.monopulse_enabled = enabled

            if enabled:
                print("[MONOPULSE] Precision tracking ACTIVATED")
            else:
                print("[MONOPULSE] Standard beam tracking")
        except Exception as e:
            print(f"[MONOPULSE] Engine update failed: {e}")

    # ═══ PHASE 26: PULSE-DOPPLER PROCESSING ═══

    def set_pulse_doppler_mode(
        self,
        enabled: bool,
        n_pulses: int = 64,
        prf_hz: float = 1000.0,
        mti_order: int = 0,
    ):
        """
        Set pulse-Doppler processing mode.

        Enables signal-level coherent processing with true Range-Doppler
        map generation via CPI → Matched Filter → MTI → Doppler FFT.

        Args:
            enabled: Whether pulse-Doppler processing is active
            n_pulses: Pulses per CPI
            prf_hz: Pulse repetition frequency [Hz]
            mti_order: MTI canceller order (0=off, 1=2-pulse, 2=3-pulse)

        Reference: Richards (2005), "Fundamentals of Radar Signal Processing"
        """
        try:
            self.engine.set_pulse_doppler_mode(enabled, n_pulses, prf_hz, mti_order)

            if enabled:
                print(
                    f"[PD] Pulse-Doppler ACTIVATED | N={n_pulses} PRF={prf_hz}Hz MTI={mti_order}"
                )
            else:
                print("[PD] Pulse-Doppler DEACTIVATED → Parametric mode")
        except Exception as e:
            print(f"[PD] Engine update failed: {e}")

    # ═══ PHASE 27: EKF TRACKING ═══

    def set_ekf_mode(self, enabled: bool):
        """
        Set Extended Kalman Filter tracking mode.

        When enabled, the tracker processes [r, θ] measurements directly
        using the nonlinear EKF instead of the linear KF.

        Args:
            enabled: Whether to use EKF

        Reference: Bar-Shalom (2001), Ch. 5.3
        """
        try:
            if (
                hasattr(self.engine, "track_manager")
                and self.engine.track_manager is not None
            ):
                self.engine.track_manager.set_ekf_mode(enabled)
                if enabled:
                    print("[EKF] Extended Kalman Filter ACTIVATED (polar measurements)")
                else:
                    print("[EKF] Linear Kalman Filter (Cartesian measurements)")
            else:
                print("[EKF] Track manager not available")
        except Exception as e:
            print(f"[EKF] Engine update failed: {e}")

    # ═══ PHASE 28: ELECTRONIC WARFARE ═══

    def __init_eccm(self):
        """Lazy-initialize ECCM controller."""
        if not hasattr(self, "_eccm_controller"):
            try:
                from src.advanced.eccm import ECCMController

                self._eccm_controller = ECCMController(
                    center_freq_hz=getattr(self.engine, "frequency_hz", 10e9),
                    nominal_prf_hz=getattr(self.engine, "prf_hz", 1000.0),
                )
            except ImportError:
                self._eccm_controller = None

    def _get_eccm_state(self) -> dict:
        """Get ECCM state for UI."""
        if hasattr(self, "_eccm_controller") and self._eccm_controller is not None:
            return self._eccm_controller.get_status()
        return {"jamming_active": False, "effective_jsr_db": -100}

    def set_jamming_environment(self, enabled: bool) -> None:
        """
        Toggle jamming environment.

        Args:
            enabled: True to activate jamming
        """
        self.__init_eccm()
        if self._eccm_controller:
            self._eccm_controller.set_jamming_environment(enabled, jsr_db=20.0)
            state = "ACTIVE (20 dB J/S)" if enabled else "OFF"
            print(f"[EW] Jamming environment {state}")

    def set_freq_agility(self, enabled: bool) -> None:
        """
        Toggle frequency agility ECCM.

        Args:
            enabled: True to enable
        """
        self.__init_eccm()
        if self._eccm_controller:
            if enabled:
                self._eccm_controller.enable_frequency_agility()
                print(
                    f"[ECCM] Frequency Agility ON — J/S reduced by {self._eccm_controller.freq_agility.js_reduction_db:.0f} dB"
                )
            else:
                self._eccm_controller.disable_frequency_agility()
                print("[ECCM] Frequency Agility OFF")

    def set_prf_stagger(self, enabled: bool) -> None:
        """
        Toggle PRF stagger ECCM.

        Args:
            enabled: True to enable
        """
        self.__init_eccm()
        if self._eccm_controller:
            if enabled:
                self._eccm_controller.enable_prf_stagger()
                print("[ECCM] PRF Stagger ON — ±5% jitter")
            else:
                self._eccm_controller.disable_prf_stagger()
                print("[ECCM] PRF Stagger OFF")


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
        # Aircraft approaching from North
        Target(
            target_id=1,
            position=np.array([80000.0, 5000.0, -3000.0]),  # 80km N, 5km E, 3km alt
            velocity=np.array([-150.0, 10.0, 0.0]),  # Approaching at 150 m/s
            rcs_m2=5.0,
            target_type="aircraft",
            swerling_model=SwerlingModel.SWERLING_1,
            motion_model=MotionModel.CONSTANT_VELOCITY,
        ),
        # Fast mover from East
        Target(
            target_id=2,
            position=np.array([30000.0, 60000.0, -5000.0]),  # 60km E
            velocity=np.array([-50.0, -200.0, 0.0]),  # Cross track
            rcs_m2=2.0,
            target_type="aircraft",
            swerling_model=SwerlingModel.SWERLING_1,
            motion_model=MotionModel.CONSTANT_VELOCITY,
        ),
        # Small target from South
        Target(
            target_id=3,
            position=np.array([-40000.0, 20000.0, -2000.0]),
            velocity=np.array([100.0, -20.0, 0.0]),
            rcs_m2=0.5,
            target_type="missile",
            swerling_model=SwerlingModel.SWERLING_3,
            motion_model=MotionModel.CONSTANT_VELOCITY,
        ),
        # Static target for calibration
        Target(
            target_id=4,
            position=np.array([25000.0, -25000.0, 0.0]),  # 35km SW
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
