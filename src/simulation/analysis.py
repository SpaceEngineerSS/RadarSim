"""
Analysis helpers built on the engine's own physics.

``coverage_curve`` evaluates single-look SNR and Pd against range for a probe
target by running the same per-look chain the simulation uses (propagation,
clutter, pulse compression, CFAR loss, receiver limiting, Swerling Pd), so the
curve and the live picture can never disagree.
"""

import copy
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.physics.rcs import SwerlingModel

from .engine import SimulationEngine
from .objects import MotionModel, Target


@dataclass(frozen=True)
class CoverageCurve:
    range_m: np.ndarray
    snr_db: np.ndarray
    pd: np.ndarray
    range_pd50_m: Optional[float]
    range_pd90_m: Optional[float]


def _crossing(ranges: np.ndarray, pd: np.ndarray, level: float) -> Optional[float]:
    """Largest range at which Pd is still at least ``level``."""
    above = np.nonzero(pd >= level)[0]
    if above.size == 0:
        return None
    i = above[-1]
    if i + 1 >= len(ranges):
        return float(ranges[i])
    # Linear interpolation between the last point above and the first below
    r0, r1, p0, p1 = ranges[i], ranges[i + 1], pd[i], pd[i + 1]
    return float(r0 + (level - p0) * (r1 - r0) / (p1 - p0)) if p1 != p0 else float(r0)


def probe_engine(engine: SimulationEngine) -> SimulationEngine:
    """Copy of ``engine``'s radar, environment and processing without targets."""
    probe = SimulationEngine(
        radar=copy.deepcopy(engine.radar),
        targets=[],
        dt=engine.dt,
        enable_atmospheric=engine.enable_atmospheric,
        probability_false_alarm=engine.probability_false_alarm,
        pulses_integrated=engine.pulses_integrated,
        atmospheric_temperature_c=engine.atmospheric_temperature_c,
        atmospheric_pressure_hpa=engine.atmospheric_pressure_hpa,
        water_vapor_density_g_m3=engine.water_vapor_density_g_m3,
        enable_clutter=engine.clutter_enabled,
        terrain_type=engine.terrain_type,
        sea_state=engine.sea_state,
        rain_rate_mm_hr=engine.rain_rate_mm_hr,
        receiver_full_scale_dbm=engine.receiver_full_scale_dbm,
        ground_model=engine.ground_model,
        land_gamma_db=engine.land_gamma_db,
        ground_relative_permittivity=engine.ground_relative_permittivity,
        ground_rms_height_m=engine.ground_rms_height_m,
        scan_gated_detection=False,
        rain_height_m=engine.rain_height_m,
        clutter_velocity_spread_mps=engine.clutter_velocity_spread_mps,
        clutter_sectors=list(engine.clutter_sectors),
        cfar=engine.cfar,
    )
    probe.mti_enabled = engine.mti_enabled
    probe.mti_threshold_mps = 0.0  # report the probe's Pd, not the notch
    probe.enable_tracking = False
    return probe


def coverage_curve(
    probe: SimulationEngine,
    rcs_m2: float = 1.0,
    swerling: int = 1,
    altitude_m: float = 3000.0,
    azimuth_deg: float = 0.0,
    max_range_m: float = 200e3,
    points: int = 160,
) -> CoverageCurve:
    """SNR and Pd of one look against range for a probe target (no jamming)."""
    ranges = np.linspace(max_range_m / points, max_range_m, points)
    origin = probe.radar.position
    az = np.radians(azimuth_deg)
    snr = np.empty(points)
    pd = np.empty(points)
    for index, ground_range in enumerate(ranges):
        target = Target(
            1,
            position=np.array(
                [
                    origin[0] + ground_range * np.sin(az),
                    origin[1] + ground_range * np.cos(az),
                    altitude_m,
                ]
            ),
            velocity=np.array([0.0, -200.0, 0.0]),  # moving, so MTI passes it
            rcs_m2=rcs_m2,
            swerling_model=SwerlingModel(swerling),
            motion_model=MotionModel.STATIC,
        )
        probe.targets = [target]
        probe.state.targets = {1: target}
        probe._last_looks.clear()
        result = probe.step(1e-6)[0]
        snr[index] = result.snr_db
        pd[index] = result.pd
    return CoverageCurve(
        range_m=ranges,
        snr_db=snr,
        pd=pd,
        range_pd50_m=_crossing(ranges, pd, 0.5),
        range_pd90_m=_crossing(ranges, pd, 0.9),
    )


@dataclass(frozen=True)
class MonteCarloResult:
    range_m: np.ndarray
    pd_empirical: np.ndarray
    pd_low: np.ndarray  # 95% Wilson interval
    pd_high: np.ndarray
    pd_theory: np.ndarray
    looks_per_point: int


def wilson_interval(successes: np.ndarray, trials: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion (Wilson 1927)."""
    p = successes / trials
    denominator = 1 + z**2 / trials
    centre = (p + z**2 / (2 * trials)) / denominator
    half = z * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2)) / denominator
    return centre - half, centre + half


def monte_carlo_pd(
    probe: SimulationEngine,
    ranges_m: np.ndarray,
    looks_per_point: int = 200,
    rcs_m2: float = 1.0,
    swerling: int = 1,
    altitude_m: float = 3000.0,
    seed: int = 1,
    progress=None,
) -> MonteCarloResult:
    """Empirical detection probability from repeated independent looks."""
    probe.rng = np.random.default_rng(seed)
    empirical = np.empty(len(ranges_m))
    theory = np.empty(len(ranges_m))
    origin = probe.radar.position
    for index, ground_range in enumerate(ranges_m):
        target = Target(
            1,
            position=np.array([origin[0], origin[1] + ground_range, altitude_m]),
            velocity=np.array([0.0, -200.0, 0.0]),
            rcs_m2=rcs_m2,
            swerling_model=SwerlingModel(swerling),
            motion_model=MotionModel.STATIC,
        )
        probe.targets = [target]
        probe.state.targets = {1: target}
        hits = 0
        for _ in range(looks_per_point):
            probe._last_looks.clear()
            result = probe.step(1e-6)[0]
            hits += int(result.is_detected)
        empirical[index] = hits / looks_per_point
        theory[index] = result.pd
        if progress is not None:
            progress(index + 1, len(ranges_m))
    low, high = wilson_interval(empirical * looks_per_point, looks_per_point)
    return MonteCarloResult(
        range_m=np.asarray(ranges_m, dtype=float),
        pd_empirical=empirical,
        pd_low=low,
        pd_high=high,
        pd_theory=theory,
        looks_per_point=looks_per_point,
    )
