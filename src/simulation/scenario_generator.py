"""
Scenario Generator

Generates headless Monte Carlo configurations from parameter ranges.

Every configuration receives an independent, collision-free seed spawned from
one master seed with ``numpy.random.SeedSequence``, so a whole sweep is
reproducible from a single number.

Usage:
    space = ParameterSpace(ranges_km=[10, 20, 50, 100], rcs_values_m2=[1, 3, 5])
    configs = ScenarioGenerator.generate(space)
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Iterator, List

import numpy as np

from .headless_runner import SimulationConfig


@dataclass
class ParameterSpace:
    """
    Parameter space definition for a Monte Carlo sweep.

    Attributes:
        ranges_km: Target ranges [km]
        rcs_values_m2: RCS values [m²]
        frequencies_ghz: Frequencies [GHz]
        powers_kw: Transmit powers [kW]
        n_runs_per_config: Monte Carlo runs per configuration
        duration_s: Simulated time per run [s]
        swerling_case: Target fluctuation model 0-4
        probability_false_alarm: Detector Pfa
        master_seed: Seed from which all run seeds are spawned
    """

    ranges_km: List[float] = field(default_factory=lambda: [10, 20, 30, 50, 70, 100])
    rcs_values_m2: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])
    frequencies_ghz: List[float] = field(default_factory=lambda: [10.0])
    powers_kw: List[float] = field(default_factory=lambda: [100.0])
    n_runs_per_config: int = 10

    duration_s: float = 10.0
    swerling_case: int = 1
    probability_false_alarm: float = 1e-6
    master_seed: int = 2026

    @property
    def total_configs(self) -> int:
        """Total number of configurations."""
        return (
            len(self.ranges_km)
            * len(self.rcs_values_m2)
            * len(self.frequencies_ghz)
            * len(self.powers_kw)
        )

    @property
    def total_runs(self) -> int:
        """Total number of simulation runs."""
        return self.total_configs * self.n_runs_per_config


def _spawn_seeds(master_seed: int, count: int) -> List[int]:
    children = np.random.SeedSequence(master_seed).spawn(count)
    return [int(child.generate_state(1)[0]) for child in children]


class ScenarioGenerator:
    """Generates simulation configurations from a parameter space."""

    @staticmethod
    def generate_iterator(space: ParameterSpace) -> Iterator[SimulationConfig]:
        """Yield every configuration of the Cartesian product, each run seeded."""
        seeds = iter(_spawn_seeds(space.master_seed, space.total_runs))
        for range_km, rcs, freq_ghz, power_kw in product(
            space.ranges_km, space.rcs_values_m2, space.frequencies_ghz, space.powers_kw
        ):
            for _ in range(space.n_runs_per_config):
                yield SimulationConfig(
                    target_range_m=range_km * 1000,
                    target_rcs_m2=rcs,
                    frequency_hz=freq_ghz * 1e9,
                    power_watts=power_kw * 1e3,
                    duration_s=space.duration_s,
                    swerling_case=space.swerling_case,
                    probability_false_alarm=space.probability_false_alarm,
                    seed=next(seeds),
                )

    @staticmethod
    def generate(space: ParameterSpace) -> List[SimulationConfig]:
        """All configurations of the parameter space as a list."""
        return list(ScenarioGenerator.generate_iterator(space))

    @staticmethod
    def quick_sweep(
        range_min_km: float = 10,
        range_max_km: float = 100,
        n_ranges: int = 10,
        rcs_m2: float = 1.0,
        n_runs: int = 5,
        master_seed: int = 2026,
    ) -> List[SimulationConfig]:
        """Range sweep for a Pd-versus-range curve."""
        return ScenarioGenerator.generate(
            ParameterSpace(
                ranges_km=list(np.linspace(range_min_km, range_max_km, n_ranges)),
                rcs_values_m2=[rcs_m2],
                n_runs_per_config=n_runs,
                master_seed=master_seed,
            )
        )
