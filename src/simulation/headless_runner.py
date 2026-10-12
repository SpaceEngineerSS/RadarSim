"""
Headless Simulation Runner

Monte Carlo detection statistics for one radar and one target, computed with the
same ``SimulationEngine`` physics the GUI uses (radar equation, ITU-R P.676
attenuation, Swerling Pd at the configured Pfa). No GUI dependencies.

Each engine step is one independent look (dwell) at the target; the target flies
radially at ``target_velocity_mps`` (positive = receding).

Usage:
    config = SimulationConfig(target_range_m=40e3, target_rcs_m2=2.0, seed=1)
    result = HeadlessRunner(config).run()
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from src.physics.rcs import SwerlingModel
from src.simulation.engine import SimulationEngine
from src.simulation.objects import MotionModel, Radar, Target


@dataclass
class SimulationConfig:
    """
    Configuration for a headless Monte Carlo run.

    Attributes:
        frequency_hz, power_watts, antenna_gain_db, noise_figure_db: Radar
        target_range_m: Initial slant range [m]
        target_rcs_m2: Mean RCS [m²]
        target_velocity_mps: Radial velocity [m/s] (positive = receding)
        swerling_case: Swerling fluctuation model 0-4
        duration_s: Simulated time [s]
        dt_s: Time between looks [s]
        prf_hz, pulse_width_s: Waveform
        probability_false_alarm: Detector Pfa per cell
        pulses_integrated: Non-coherently integrated pulses per look
        enable_atmospheric: Apply ITU-R P.676 gaseous attenuation
        seed: Random seed for reproducibility
    """

    # Radar
    frequency_hz: float = 10e9
    power_watts: float = 100e3
    antenna_gain_db: float = 30.0
    noise_figure_db: float = 4.0

    # Target
    target_range_m: float = 50000.0
    target_rcs_m2: float = 1.0
    target_velocity_mps: float = 0.0
    swerling_case: int = 1

    # Simulation
    duration_s: float = 10.0
    dt_s: float = 0.033
    prf_hz: float = 1000.0
    pulse_width_s: float = 1e-6
    probability_false_alarm: float = 1e-6
    pulses_integrated: int = 1

    # Options
    enable_atmospheric: bool = True
    seed: Optional[int] = None

    def build_engine(self) -> SimulationEngine:
        """Create the engine for this configuration (radar at origin, target North)."""
        radar = Radar(
            radar_id="headless",
            position=np.zeros(3),
            frequency_hz=self.frequency_hz,
            power_watts=self.power_watts,
            antenna_gain_db=self.antenna_gain_db,
            noise_figure_db=self.noise_figure_db,
            prf_hz=self.prf_hz,
            pulse_width_s=self.pulse_width_s,
            scan_rate_rpm=0.0,
        )
        target = Target(
            target_id=1,
            position=np.array([0.0, self.target_range_m, 0.0]),
            velocity=np.array([0.0, self.target_velocity_mps, 0.0]),
            rcs_m2=self.target_rcs_m2,
            swerling_model=SwerlingModel(self.swerling_case),
            motion_model=MotionModel.CONSTANT_VELOCITY,
        )
        return SimulationEngine(
            radar=radar,
            targets=[target],
            dt=self.dt_s,
            enable_atmospheric=self.enable_atmospheric,
            probability_false_alarm=self.probability_false_alarm,
            pulses_integrated=self.pulses_integrated,
            seed=self.seed,
            scan_gated_detection=False,
        )


@dataclass
class SimulationResult:
    """
    Results from a headless run.

    Attributes:
        config: Original configuration
        n_looks: Detection opportunities (one per step)
        n_detections: Successful detections
        detection_ratio: Empirical Pd = n_detections / n_looks
        mean_pd: Mean theoretical Pd over the looks
        mean_snr_db, min_snr_db, max_snr_db: SNR statistics [dB]
        runtime_s: Wall-clock execution time
    """

    config: SimulationConfig
    n_looks: int = 0
    n_detections: int = 0
    detection_ratio: float = 0.0
    mean_pd: float = 0.0
    mean_snr_db: float = 0.0
    min_snr_db: float = 0.0
    max_snr_db: float = 0.0
    runtime_s: float = 0.0
    snr_history: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        return {
            "range_km": self.config.target_range_m / 1000,
            "rcs_m2": self.config.target_rcs_m2,
            "swerling_case": self.config.swerling_case,
            "frequency_ghz": self.config.frequency_hz / 1e9,
            "power_kw": self.config.power_watts / 1e3,
            "pfa": self.config.probability_false_alarm,
            "n_looks": self.n_looks,
            "n_detections": self.n_detections,
            "detection_ratio": self.detection_ratio,
            "mean_pd": self.mean_pd,
            "mean_snr_db": self.mean_snr_db,
            "min_snr_db": self.min_snr_db,
            "max_snr_db": self.max_snr_db,
            "runtime_s": self.runtime_s,
            "seed": self.config.seed,
        }


class HeadlessRunner:
    """Runs one ``SimulationConfig`` through the engine and collects statistics."""

    def __init__(self, config: SimulationConfig):
        self.config = config

    def run(self) -> SimulationResult:
        start_time = time.perf_counter()
        engine = self.config.build_engine()
        n_steps = int(round(self.config.duration_s / self.config.dt_s))

        snr_values: List[float] = []
        pd_values: List[float] = []
        detections = 0
        for _ in range(n_steps):
            result = engine.step()[0]
            if result.true_range_m <= 0.0:
                continue
            snr_values.append(result.snr_db)
            pd_values.append(result.pd)
            detections += int(result.is_detected)

        snr = np.array(snr_values) if snr_values else np.array([0.0])
        n_looks = len(snr_values)
        return SimulationResult(
            config=self.config,
            n_looks=n_looks,
            n_detections=detections,
            detection_ratio=detections / n_looks if n_looks else 0.0,
            mean_pd=float(np.mean(pd_values)) if pd_values else 0.0,
            mean_snr_db=float(np.mean(snr)),
            min_snr_db=float(np.min(snr)),
            max_snr_db=float(np.max(snr)),
            runtime_s=time.perf_counter() - start_time,
            snr_history=snr_values[:100],
        )


def run_single_simulation(config: SimulationConfig) -> SimulationResult:
    """Convenience function for multiprocessing."""
    return HeadlessRunner(config).run()
