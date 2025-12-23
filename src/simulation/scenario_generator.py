"""
Scenario Generator

Generates test scenarios from parameter ranges for Monte Carlo analysis.

Features:
    - Cartesian product of parameter values
    - Predefined parameter spaces
    - Configurable ranges

Usage:
    space = ParameterSpace(
        ranges_km=[10, 20, 50, 100],
        rcs_values_m2=[1, 3, 5],
    )
    configs = ScenarioGenerator.generate(space)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Iterator
from itertools import product
import numpy as np

from .headless_runner import SimulationConfig


@dataclass
class ParameterSpace:
    """
    Parameter space definition for Monte Carlo sweep.
    
    Attributes:
        ranges_km: List of target ranges [km]
        rcs_values_m2: List of RCS values [m²]
        frequencies_ghz: List of frequencies [GHz]
        powers_kw: List of transmit powers [kW]
        n_runs_per_config: Number of Monte Carlo runs per configuration
    """
    ranges_km: List[float] = field(default_factory=lambda: [10, 20, 30, 50, 70, 100])
    rcs_values_m2: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])
    frequencies_ghz: List[float] = field(default_factory=lambda: [10.0])
    powers_kw: List[float] = field(default_factory=lambda: [100.0])
    n_runs_per_config: int = 10
    
    # Fixed parameters
    duration_s: float = 10.0
    detection_threshold_db: float = 13.0
    
    @property
    def total_configs(self) -> int:
        """Total number of configurations."""
        return (
            len(self.ranges_km) * 
            len(self.rcs_values_m2) * 
            len(self.frequencies_ghz) * 
            len(self.powers_kw)
        )
    
    @property
    def total_runs(self) -> int:
        """Total number of simulation runs."""
        return self.total_configs * self.n_runs_per_config


class ScenarioGenerator:
    """
    Generates simulation configurations from parameter space.
    """
    
    @staticmethod
    def generate(space: ParameterSpace) -> List[SimulationConfig]:
        """
        Generate all configurations from parameter space.
        
        Args:
            space: Parameter space definition
            
        Returns:
            List of SimulationConfig objects
        """
        configs = []
        
        # Cartesian product of all parameters
        for range_km, rcs, freq_ghz, power_kw in product(
            space.ranges_km,
            space.rcs_values_m2,
            space.frequencies_ghz,
            space.powers_kw
        ):
            # Create multiple runs with different seeds
            for run_idx in range(space.n_runs_per_config):
                config = SimulationConfig(
                    target_range_m=range_km * 1000,
                    target_rcs_m2=rcs,
                    frequency_hz=freq_ghz * 1e9,
                    power_watts=power_kw * 1e3,
                    duration_s=space.duration_s,
                    detection_threshold_db=space.detection_threshold_db,
                    seed=run_idx * 1000 + int(range_km * 10) + int(rcs * 100),
                )
                configs.append(config)
        
        return configs
    
    @staticmethod
    def generate_iterator(space: ParameterSpace) -> Iterator[SimulationConfig]:
        """
        Generate configurations as iterator (memory efficient).
        
        Args:
            space: Parameter space definition
            
        Yields:
            SimulationConfig objects
        """
        for range_km, rcs, freq_ghz, power_kw in product(
            space.ranges_km,
            space.rcs_values_m2,
            space.frequencies_ghz,
            space.powers_kw
        ):
            for run_idx in range(space.n_runs_per_config):
                yield SimulationConfig(
                    target_range_m=range_km * 1000,
                    target_rcs_m2=rcs,
                    frequency_hz=freq_ghz * 1e9,
                    power_watts=power_kw * 1e3,
                    duration_s=space.duration_s,
                    detection_threshold_db=space.detection_threshold_db,
                    seed=run_idx * 1000 + int(range_km * 10) + int(rcs * 100),
                )
    
    @staticmethod
    def quick_sweep(
        range_min_km: float = 10,
        range_max_km: float = 100,
        n_ranges: int = 10,
        rcs_m2: float = 1.0,
        n_runs: int = 5
    ) -> List[SimulationConfig]:
        """
        Quick range sweep for Pd vs Range curve.
        
        Args:
            range_min_km: Minimum range [km]
            range_max_km: Maximum range [km]
            n_ranges: Number of range points
            rcs_m2: Fixed RCS value [m²]
            n_runs: Runs per range point
            
        Returns:
            List of configs
        """
        ranges = np.linspace(range_min_km, range_max_km, n_ranges)
        
        configs = []
        for range_km in ranges:
            for run_idx in range(n_runs):
                configs.append(SimulationConfig(
                    target_range_m=range_km * 1000,
                    target_rcs_m2=rcs_m2,
                    seed=run_idx * 1000 + int(range_km * 10),
                ))
        
        return configs
