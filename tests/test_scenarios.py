"""
Shipped scenarios: schema cleanliness, lossless export round trip, and that each
scenario actually exercises the feature it describes.
"""

from pathlib import Path

import numpy as np
import pytest

from src.io.exporter import export_scenario_to_yaml
from src.io.scenario_loader import ScenarioLoader

SCENARIOS = Path(__file__).parents[1] / "scenarios"
ALL_SCENARIOS = sorted(p.name for p in SCENARIOS.glob("*.*"))


def _engine(name: str, seed: int = 1):
    engine = ScenarioLoader(str(SCENARIOS / name)).create_simulation_engine()
    engine.seed = seed
    engine.rng = np.random.default_rng(seed)
    return engine


def _looks(engine, duration_s: float):
    looks = {}
    for result in engine.run(duration_s).detection_history:
        looks.setdefault(result.target_id, []).append(result)
    return looks


def _by_name(engine, name):
    return next(t for t in engine.targets if t.name == name)


@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_scenario_has_no_unknown_keys(name):
    assert ScenarioLoader(str(SCENARIOS / name)).warnings == []


def _engine_signature(engine):
    radar = engine.radar.to_dict()
    radar["position"] = engine._initial_radar_state.position.tolist()
    targets = []
    for target in engine.targets:
        state = engine._initial_target_states[target.target_id]
        data = target.to_dict()
        data.pop("target_id")
        data["position"] = state.position.tolist()
        data["velocity"] = state.velocity.tolist()
        targets.append(data)
    environment = {
        key: getattr(engine, key)
        for key in (
            "atmospheric_temperature_c", "atmospheric_pressure_hpa",
            "water_vapor_density_g_m3", "rain_rate_mm_hr", "rain_height_m",
            "terrain_type", "sea_state", "ground_model", "land_gamma_db",
            "ground_relative_permittivity", "ground_rms_height_m",
            "clutter_velocity_spread_mps", "clutter_sectors", "cfar",
            "probability_false_alarm", "pulses_integrated", "enable_atmospheric",
            "clutter_enabled", "mti_enabled", "mti_threshold_mps",
            "monopulse_enabled", "frequency_agility_enabled", "dt",
            "receiver_full_scale_dbm", "ecm_active", "scan_gated_detection",
        )
    }
    tm = engine.track_manager
    environment["tracker"] = (tm.kf.process_noise, tm.confirm_hits, tm.max_misses)
    terrain = engine.terrain.config if engine.terrain is not None else None
    return radar, targets, environment, terrain


@pytest.mark.parametrize("name", ALL_SCENARIOS)
def test_export_round_trip_is_lossless(name, tmp_path):
    original = ScenarioLoader(str(SCENARIOS / name)).create_simulation_engine()
    original.run(1.0)  # export must describe the initial state, not the current one
    exported = tmp_path / "round_trip.yaml"
    assert export_scenario_to_yaml(original, str(exported), scenario_name="Round trip")

    loader = ScenarioLoader(str(exported))
    assert loader.warnings == []
    restored = loader.create_simulation_engine()
    first, second = _engine_signature(original), _engine_signature(restored)
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[3] == second[3]


def test_naval_uses_sea_clutter_and_mti_reveals_sea_skimmers():
    engine = _engine("naval_battlegroup.yaml")
    assert engine.terrain_type == "sea" and engine.sea_state == 4
    looks = _looks(engine, 30.0)
    missile = _by_name(engine, "ASM_1_Brahmos")
    results = looks[missile.target_id]
    assert all(r.surface_clutter_model == "nrl_five_parameter" for r in results)
    assert any(r.is_detected for r in results)

    without_mti = _engine("naval_battlegroup.yaml")
    without_mti.mti_enabled = False
    no_mti = _looks(without_mti, 30.0)[missile.target_id]
    assert np.mean([r.surface_clutter_loss_db for r in no_mti]) > np.mean(
        [r.surface_clutter_loss_db for r in results]
    ) + 10.0


def test_f16_releases_decoy_and_chaff_from_alpha():
    engine = _engine("f16_vs_sa6.yaml")
    alpha = _by_name(engine, "F-16 Alpha")
    decoy = _by_name(engine, "ADM-160 MALD")
    chaff = _by_name(engine, "Chaff 1")
    assert decoy.active_from_s == 30.0 and chaff.active_from_s == 60.0
    expected = engine._initial_target_states[alpha.target_id].position + 30.0 * np.array(
        [350.0, -50.0, -10.0]
    )
    np.testing.assert_allclose(decoy.position, expected)
    assert engine.cfar is not None and engine.cfar_loss_db > 1.0

    looks = _looks(engine, 120.0)
    assert min(r.time for r in looks[decoy.target_id]) >= 30.0
    assert min(r.time for r in looks[chaff.target_id]) >= 60.0
    assert any(r.is_detected for r in looks[chaff.target_id])
    # Alpha's self-screening barrage jammer hides it
    assert all(r.jammer_loss_db > 20.0 for r in looks[alpha.target_id])


def test_close_air_combat_sector_scan_loses_bandit_leaving_the_sector():
    engine = _engine("close_air_combat.yaml")
    assert engine.radar.scan_type.value == "sector"
    looks = _looks(engine, 90.0)
    # Bandit 1 turns at 9 deg/s inside the sector and is looked at throughout.
    bandit_1 = looks[_by_name(engine, "Bandit 1").target_id]
    assert bandit_1[0].time < 5.0 and bandit_1[-1].time > 80.0
    # Bandit 2 drifts West out of the +/-60 deg raster and is no longer looked at.
    bandit_2 = looks[_by_name(engine, "Bandit 2").target_id]
    assert bandit_2[-1].time < 60.0
    for result in bandit_1 + bandit_2:
        assert abs(np.degrees(result.true_azimuth_rad)) <= 60.5
    # The tuned tracker follows the break turn with a single, unbroken track.
    lost = [e for e in engine.event_log if e.kind.value == "track_lost"]
    assert not lost


def test_mountain_ambush_target_emerges_from_terrain_mask():
    engine = _engine("mountain_ambush.yaml")
    assert engine.terrain is not None
    ground = engine.terrain.get_elevation(0.0, 0.0)
    assert engine.radar.position[2] == pytest.approx(ground + 30.0)
    low = _looks(engine, 180.0)[_by_name(engine, "Low Flyer").target_id]
    assert low[0].terrain_masked
    assert not low[-1].terrain_masked
    assert any(r.is_detected for r in low if not r.terrain_masked)


def test_ground_clutter_sectors_change_the_surface_class():
    engine = _engine("ground_clutter_filtering.yaml")
    assert len(engine.clutter_sectors) == 3
    urban = engine._surface_class(np.radians(60.0), 15_000.0)
    forest = engine._surface_class(np.radians(-60.0), 20_000.0)
    elsewhere = engine._surface_class(np.radians(100.0), 15_000.0)
    assert urban == ("urban", 0.0)
    assert forest == ("forest", 3.0)
    assert elsewhere == ("rural", 0.0)


def test_hypersonic_seeker_stares_and_detects_late():
    engine = _engine("hypersonic_interception.yaml")
    assert engine.radar.scan_type.value == "stare"
    looks = _looks(engine, 20.0)[1]
    # A look every step until the HGV overflies the seeker (~18 s) and leaves the beam
    assert 500 < len(looks) < 600
    assert next(r.time for r in looks if r.is_detected) > 5.0
