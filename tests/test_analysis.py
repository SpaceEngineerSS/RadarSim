"""Coverage and Monte Carlo analysis use the engine's physics consistently."""

import numpy as np
import pytest

from src.io.scenario_loader import ScenarioLoader
from src.simulation.analysis import coverage_curve, monte_carlo_pd, probe_engine, wilson_interval


@pytest.fixture(scope="module")
def probe():
    engine = ScenarioLoader("scenarios/basic_tracking.json").create_simulation_engine()
    return probe_engine(engine)


def test_coverage_pd_falls_with_range_and_crossings_are_ordered(probe):
    curve = coverage_curve(probe, rcs_m2=1.0, swerling=1, max_range_m=120e3, points=60)
    assert np.all(np.diff(curve.snr_db) < 0)
    assert curve.pd[0] > 0.99 and curve.pd[-1] < 0.01
    assert curve.range_pd90_m < curve.range_pd50_m


def test_coverage_scales_with_rcs_as_fourth_root():
    engine = ScenarioLoader("scenarios/basic_tracking.json").create_simulation_engine()
    probe = probe_engine(engine)
    probe.enable_atmospheric = False
    small = coverage_curve(probe, rcs_m2=1.0, swerling=0, max_range_m=200e3, points=400)
    large = coverage_curve(probe, rcs_m2=16.0, swerling=0, max_range_m=200e3, points=400)
    # Free-space, noise limited: range at fixed SNR grows as sigma^(1/4).
    assert large.range_pd50_m / small.range_pd50_m == pytest.approx(2.0, rel=0.02)


def test_monte_carlo_agrees_with_theory(probe):
    ranges = np.array([40e3, 60e3, 75e3])
    result = monte_carlo_pd(probe, ranges, looks_per_point=400, rcs_m2=1.0, swerling=1, seed=3)
    inside = (result.pd_theory >= result.pd_low - 0.02) & (result.pd_theory <= result.pd_high + 0.02)
    assert inside.all()


def test_wilson_interval_contains_estimate():
    low, high = wilson_interval(np.array([0, 50, 100]), 100)
    assert low[0] == 0.0 and high[2] == pytest.approx(1.0)
    assert low[1] < 0.5 < high[1]
