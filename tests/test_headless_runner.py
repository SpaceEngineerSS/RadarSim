"""Headless Monte Carlo runner and sweep generator."""

import pytest

from src.physics.metrics import calculate_pd_swerling
from src.simulation.headless_runner import HeadlessRunner, SimulationConfig
from src.simulation.scenario_generator import ParameterSpace, ScenarioGenerator


def test_empirical_pd_matches_swerling_theory():
    config = SimulationConfig(
        target_range_m=18e3, target_rcs_m2=1.0, swerling_case=1,
        duration_s=200.0, dt_s=0.05, enable_atmospheric=False, seed=3,
    )
    result = HeadlessRunner(config).run()
    assert result.n_looks == 4000
    expected = calculate_pd_swerling(result.mean_snr_db, pfa=1e-6, swerling_case=1)
    assert 0.05 < expected < 0.95
    assert result.mean_pd == pytest.approx(expected, rel=1e-6)
    # Binomial standard error for 4000 looks is below 0.008
    assert result.detection_ratio == pytest.approx(expected, abs=0.03)


def test_same_seed_same_result_different_seed_different_result():
    def ratio(seed):
        config = SimulationConfig(target_range_m=18e3, duration_s=20.0, seed=seed)
        return HeadlessRunner(config).run().n_detections

    assert ratio(1) == ratio(1)
    assert ratio(1) != ratio(2)


def test_sweep_seeds_are_unique_and_reproducible():
    space = ParameterSpace(ranges_km=[10, 20], rcs_values_m2=[1, 2], n_runs_per_config=5)
    seeds = [c.seed for c in ScenarioGenerator.generate(space)]
    assert len(seeds) == space.total_runs == 20
    assert len(set(seeds)) == len(seeds)
    assert seeds == [c.seed for c in ScenarioGenerator.generate(space)]
