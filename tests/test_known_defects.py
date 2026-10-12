"""
Known-defect regression suite (v3.1 overhaul, Phase 0).

Each test pins down one defect found in the v3.0.1 audit. Tests for defects that are
not fixed yet are marked ``xfail(strict=True)`` with the phase that will fix them, so:

* the suite stays green while the defect exists, and
* the moment a fix lands the test XPASSes, strict mode turns that into a failure,
  and the marker must be removed — the test then guards against regressions.

All defects pinned here were fixed during the overhaul; the tests now guard them.
"""

from pathlib import Path

import numpy as np
import pytest

from src.io.scenario_loader import ScenarioLoader
from src.physics.rcs import SwerlingModel
from src.simulation.engine import SimulationEngine
from src.simulation.objects import MotionModel, Radar, Target

SCENARIOS = Path(__file__).parents[1] / "scenarios"



def _engine(**kwargs) -> SimulationEngine:
    radar = Radar("test", position=np.array([0.0, 0.0, 10.0]), frequency_hz=10e9)
    target = Target(
        1,
        position=np.array([0.0, 20_000.0, 1_000.0]),
        velocity=np.array([0.0, -200.0, 0.0]),
        rcs_m2=5.0,
        swerling_model=SwerlingModel.SWERLING_0,
    )
    return SimulationEngine(radar, [target], dt=0.1, **kwargs)


def _write(tmp_path: Path, text: str) -> str:
    path = tmp_path / "scenario.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


# =============================================================================
# Crashes reachable from normal settings
# =============================================================================


def test_rain_does_not_crash_step():
    """engine.step() referenced the non-existent Radar.beamwidth_deg when raining."""
    engine = _engine(rain_rate_mm_hr=10.0)
    result = engine.step()[0]
    assert result.rain_clutter_loss_db >= 0.0


def test_oh1992_low_grazing_does_not_crash():
    """Low grazing angles are outside Oh-1992's 10-70 deg incidence domain."""
    engine = _engine(enable_clutter=True, ground_model="oh1992")
    result = engine.step()[0]
    assert np.isfinite(result.snr_db)


def test_ecm_target_without_power_does_not_crash(tmp_path):
    """has_ecm: true with no ecm_power_watts produced a zero-power jammer."""
    path = _write(
        tmp_path,
        """
scenario: {name: ECM default power}
radar: {frequency_hz: 3.0e9, power_watts: 100000}
targets:
  - name: Jammer
    initial_position: {x_m: 0, y_m: 30000, z_m: 3000}
    velocity: {vx_mps: 0, vy_mps: -200, vz_mps: 0}
    has_ecm: true
    ecm_type: noise_barrage
""",
    )
    engine = ScenarioLoader(path).create_simulation_engine()
    engine.set_ecm_mode(True, "noise")
    engine.step()


def test_recorder_accepts_target_id_key(tmp_path):
    from src.simulation.recorder import FlightRecorder

    recorder = FlightRecorder(output_dir=str(tmp_path))
    recorder.start_recording({"radar_frequency_hz": 10e9})
    recorder.record_state(
        {"time": 0.0, "targets": [{"target_id": 1001, "position": [1, 2, 3]}]}
    )
    assert recorder.stop_recording() is not None


def test_replay_accepts_non_integer_target_ids(tmp_path):
    from src.io.replay_loader import ReplayLoader
    from src.simulation.recorder import FlightRecorder

    recorder = FlightRecorder(output_dir=str(tmp_path))
    recorder.start_recording({"radar_frequency_hz": 10e9})
    for t in (0.0, 0.1):
        recorder.record_state(
            {"time": t, "targets": [{"id": "target_0", "position": [t, 0, 0]}]}
        )
    path = recorder.stop_recording()
    with ReplayLoader(path) as replay:
        assert len(replay.metadata.target_ids) == 1


# =============================================================================
# Physics and convention defects
# =============================================================================


def test_beam_check_handles_azimuth_wrap():
    """A target 180 deg away from boresight must not be reported in-beam."""
    radar = Radar("test", position=np.zeros(3), beamwidth_deg=2.0)
    radar.antenna_azimuth = 6.2
    target_pos = np.array([np.cos(-3.0), np.sin(-3.0), 0.0]) * 10_000.0
    assert radar.is_target_in_beam(target_pos) is False


def test_sinc_pattern_is_half_power_at_half_beamwidth():
    from src.signal.antenna_pattern import AntennaPattern

    pattern = AntennaPattern(beamwidth_deg=2.0)
    gain = pattern.sinc_pattern(np.radians(1.0))
    assert gain == pytest.approx(0.5, abs=0.01)


def test_doppler_sign_convention_receding_is_negative():
    """Convention: fd = -2 vr / lambda with vr > 0 receding."""
    from src.physics.radar_equation import RadarParameters, calculate_doppler_shift

    radar = RadarParameters(frequency=10e9, power_transmitted=1e3, antenna_gain_tx=30.0)
    fd = calculate_doppler_shift(radar, np.array([10_000.0, 0, 0]), np.array([100.0, 0, 0]))
    assert fd < 0.0
    assert fd == pytest.approx(-2.0 * 100.0 / radar.wavelength, rel=1e-9)


def test_azimuth_is_measured_clockwise_from_north():
    """ENU frame: x = East, y = North. A target due East is at azimuth +90 deg."""
    radar = Radar("test", position=np.zeros(3))
    geom = radar.calculate_target_geometry(np.array([10_000.0, 0.0, 0.0]))
    assert geom["azimuth_deg"] == pytest.approx(90.0)


def test_coordinated_turn_target_moves():
    target = Target(
        1,
        position=np.zeros(3),
        velocity=np.array([0.0, 200.0, 0.0]),
        motion_model=MotionModel.COORDINATED_TURN,
    )
    target.update(1.0)
    assert np.linalg.norm(target.position) == pytest.approx(200.0, rel=0.01)


def test_mountain_peaks_block_line_of_sight():
    """The LOS raycast ignored TerrainConfig.mountain_peaks."""
    from src.physics.terrain import TerrainConfig, TerrainMap

    flat = TerrainConfig(max_height=1.0)
    peaked = TerrainConfig(max_height=1.0, mountain_peaks=[(0.0, 10_000.0, 3_000.0)])
    radar_pos = np.array([0.0, 0.0, 50.0])
    target_pos = np.array([0.0, 20_000.0, 50.0])

    assert TerrainMap(flat).check_line_of_sight(radar_pos, target_pos)[0] is True
    assert TerrainMap(peaked).get_elevation(0.0, 10_000.0) > 2_000.0
    assert TerrainMap(peaked).check_line_of_sight(radar_pos, target_pos)[0] is False


def test_engine_seed_gives_identical_runs():
    def run(seed):
        engine = _engine(seed=seed, scan_gated_detection=False)
        engine.targets[0].swerling_model = SwerlingModel.SWERLING_1
        history = engine.run(2.0).detection_history
        return [(r.is_detected, round(float(np.nan_to_num(r.measured_range_m)), 6)) for r in history]

    first = run(1234)
    assert len(first) == 20
    assert first == run(1234)
    assert first != run(4321)


def test_reset_restores_target_positions():
    engine = _engine()
    start = engine.targets[0].position.copy()
    engine.run(1.0)
    engine.reset()
    np.testing.assert_allclose(engine.targets[0].position, start)


def test_run_step_count_does_not_truncate():
    engine = _engine()
    engine.dt = 0.1
    engine.run(0.3)
    assert engine.current_time == pytest.approx(0.3)


# =============================================================================
# Scenario settings that are silently ignored
# =============================================================================


def test_naval_scenario_uses_sea_clutter():
    engine = ScenarioLoader(str(SCENARIOS / "naval_battlegroup.yaml")).create_simulation_engine()
    assert "sea" in engine.terrain_type.lower()
    assert engine.sea_state == 4


def test_scan_rate_is_read_from_scenario(tmp_path):
    path = _write(
        tmp_path,
        """
scenario: {name: Scan}
radar:
  frequency_hz: 3.0e9
  power_watts: 100000
  antenna: {scan_rate_rpm: 15}
""",
    )
    engine = ScenarioLoader(path).create_simulation_engine()
    assert engine.radar.scan_rate_rpm == pytest.approx(15.0)


def test_unknown_scenario_keys_are_reported(tmp_path):
    path = _write(
        tmp_path,
        """
scenario: {name: Typo}
radar: {frequency_hz: 3.0e9, power_watts: 100000}
environment: {rain_rate_mmhr: 5}
""",
    )
    loader = ScenarioLoader(path)
    assert any("rain_rate_mmhr" in w for w in loader.warnings)


def test_empty_scenario_file_gives_clear_error(tmp_path):
    path = _write(tmp_path, "")
    with pytest.raises(ValueError, match="empty"):
        ScenarioLoader(path)
