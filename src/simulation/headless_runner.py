"""
Headless Simulation Runner

Runs radar simulation without GUI for batch processing and Monte Carlo analysis.

Features:
    - No GUI dependencies
    - Time-step based execution
    - Results collection (detection logs)
    - Memory-efficient cleanup

Usage:
    config = SimulationConfig(...)
    runner = HeadlessRunner(config)
    result = runner.run()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import time
import gc

# Import physics without GUI dependencies
from src.physics import (
    RadarParameters, calculate_snr, ITU_R_P676
)


@dataclass
class SimulationConfig:
    """
    Configuration for headless simulation.
    
    Attributes:
        radar_params: Radar system parameters
        target_range_m: Target distance from radar [m]
        target_rcs_m2: Target radar cross section [m²]
        target_velocity_mps: Target velocity [m/s] (optional)
        duration_s: Simulation duration [s]
        dt_s: Time step [s]
        detection_threshold_db: SNR threshold for detection [dB]
        enable_atmospheric: Enable ITU-R atmospheric loss
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
    
    # Simulation
    duration_s: float = 10.0
    dt_s: float = 0.033  # ~30 FPS
    prf_hz: float = 1000.0
    detection_threshold_db: float = 13.0
    
    # Options
    enable_atmospheric: bool = True
    seed: Optional[int] = None
    
    def to_radar_params(self) -> RadarParameters:
        """Convert to RadarParameters object."""
        return RadarParameters(
            frequency=self.frequency_hz,
            power_transmitted=self.power_watts,
            antenna_gain_tx=self.antenna_gain_db,
            antenna_gain_rx=self.antenna_gain_db,
            noise_figure=self.noise_figure_db,
            pulse_width=1e-6,
            prf=self.prf_hz,
        )


@dataclass
class SimulationResult:
    """
    Results from a headless simulation run.
    
    Attributes:
        config: Original configuration
        n_pulses: Total pulses transmitted
        n_detections: Number of successful detections
        detection_ratio: Pd = n_detections / n_pulses
        mean_snr_db: Average SNR over simulation
        min_snr_db: Minimum SNR
        max_snr_db: Maximum SNR
        runtime_s: Wall-clock execution time
    """
    config: SimulationConfig
    n_pulses: int = 0
    n_detections: int = 0
    detection_ratio: float = 0.0
    mean_snr_db: float = 0.0
    min_snr_db: float = 0.0
    max_snr_db: float = 0.0
    runtime_s: float = 0.0
    snr_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON export."""
        return {
            'range_km': self.config.target_range_m / 1000,
            'rcs_m2': self.config.target_rcs_m2,
            'frequency_ghz': self.config.frequency_hz / 1e9,
            'power_kw': self.config.power_watts / 1e3,
            'n_pulses': self.n_pulses,
            'n_detections': self.n_detections,
            'detection_ratio': self.detection_ratio,
            'mean_snr_db': self.mean_snr_db,
            'min_snr_db': self.min_snr_db,
            'max_snr_db': self.max_snr_db,
            'runtime_s': self.runtime_s,
            'threshold_db': self.config.detection_threshold_db,
        }


class HeadlessRunner:
    """
    Headless simulation runner.
    
    Executes radar simulation without GUI, collecting detection
    statistics for Monte Carlo analysis.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize headless runner.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.radar_params = config.to_radar_params()
        
        # Set random seed if provided
        if config.seed is not None:
            np.random.seed(config.seed)
        
        # State
        self.current_time = 0.0
        self.target_range = config.target_range_m
        
        # Results accumulators
        self._snr_values: List[float] = []
        self._detections: List[bool] = []
    
    def run(self) -> SimulationResult:
        """
        Execute simulation.
        
        Returns:
            SimulationResult with detection statistics
        """
        start_time = time.perf_counter()
        
        # Reset state
        self.current_time = 0.0
        self.target_range = self.config.target_range_m
        self._snr_values = []
        self._detections = []
        
        # Time loop
        n_steps = int(self.config.duration_s / self.config.dt_s)
        pulses_per_step = int(self.config.prf_hz * self.config.dt_s)
        
        for step in range(n_steps):
            self.current_time = step * self.config.dt_s
            
            # Update target position (if moving)
            if self.config.target_velocity_mps != 0:
                self.target_range += self.config.target_velocity_mps * self.config.dt_s
            
            # Skip if target out of range
            if self.target_range <= 0:
                continue
            
            # Calculate atmospheric loss if enabled
            atm_loss_db = 0.0
            if self.config.enable_atmospheric:
                freq_ghz = self.config.frequency_hz / 1e9
                range_km = self.target_range / 1000
                atm_loss_db = ITU_R_P676.total_attenuation(
                    range_km, freq_ghz, two_way=True
                )
            
            # Calculate SNR
            snr_db = calculate_snr(
                self.radar_params,
                self.config.target_rcs_m2,
                self.target_range,
                atmospheric_loss_db=atm_loss_db
            )
            
            # Add noise fluctuation (Swerling-like)
            snr_db += np.random.normal(0, 1.5)
            
            # Process pulses
            for _ in range(max(1, pulses_per_step)):
                self._snr_values.append(snr_db)
                detected = snr_db > self.config.detection_threshold_db
                self._detections.append(detected)
        
        # Calculate runtime
        runtime = time.perf_counter() - start_time
        
        # Build result
        result = self._build_result(runtime)
        
        # Cleanup
        self._cleanup()
        
        return result
    
    def _build_result(self, runtime: float) -> SimulationResult:
        """Build simulation result from accumulated data."""
        n_pulses = len(self._detections)
        n_detections = sum(self._detections)
        
        snr_array = np.array(self._snr_values) if self._snr_values else np.array([0])
        
        return SimulationResult(
            config=self.config,
            n_pulses=n_pulses,
            n_detections=n_detections,
            detection_ratio=n_detections / n_pulses if n_pulses > 0 else 0.0,
            mean_snr_db=float(np.mean(snr_array)),
            min_snr_db=float(np.min(snr_array)),
            max_snr_db=float(np.max(snr_array)),
            runtime_s=runtime,
            snr_history=self._snr_values[:100]  # Keep first 100 for debugging
        )
    
    def _cleanup(self) -> None:
        """Clean up memory after run."""
        self._snr_values = []
        self._detections = []
        gc.collect()


def run_single_simulation(config: SimulationConfig) -> SimulationResult:
    """
    Convenience function for multiprocessing.
    
    Args:
        config: Simulation configuration
        
    Returns:
        Simulation result
    """
    runner = HeadlessRunner(config)
    return runner.run()
