from pathlib import Path

import pytest

from src.io.exporter import export_scenario_to_yaml
from src.io.scenario_loader import ScenarioLoader


SCENARIOS = Path(__file__).parents[1] / "scenarios"


def test_f16_scenario_parameters_reach_engine():
    loader = ScenarioLoader(str(SCENARIOS / "f16_vs_sa6.yaml"))
    engine = loader.create_simulation_engine()

    assert engine.radar.frequency_hz == pytest.approx(3.0e9)
    assert engine.radar.prf_hz == pytest.approx(2000.0)
    assert engine.radar.pulse_width_s == pytest.approx(2.0e-6)
    assert engine.radar.receiver_bandwidth_hz == pytest.approx(100.0e6)
    assert engine.radar.noise_figure_db == pytest.approx(3.0)
    assert engine.receiver_full_scale_dbm == pytest.approx(-20.0)
    assert engine._radar_params.noise_bandwidth == pytest.approx(100.0e6)
    assert engine._radar_params.prf == pytest.approx(2000.0)
    assert engine._radar_params.pulse_width == pytest.approx(2.0e-6)
    assert engine.probability_false_alarm == pytest.approx(1.0e-6)
    assert engine.clutter_enabled is True
    assert engine.terrain_type == "rural"
    assert engine.sea_state == 0
    assert engine.atmospheric_temperature_c == pytest.approx(15.0)
    assert engine.atmospheric_pressure_hpa == pytest.approx(1013.25)
    assert engine.water_vapor_density_g_m3 == pytest.approx(7.5)
    assert engine.targets[0].has_jammer is True
    assert engine.targets[0].jammer_power_watts == pytest.approx(200.0)


def test_hypersonic_scenario_preserves_high_prf_and_atmosphere():
    loader = ScenarioLoader(str(SCENARIOS / "hypersonic_interception.yaml"))
    engine = loader.create_simulation_engine()

    assert engine.radar.frequency_hz == pytest.approx(35.0e9)
    assert engine.radar.prf_hz == pytest.approx(100_000.0)
    assert engine.radar.receiver_bandwidth_hz == pytest.approx(20.0e6)
    assert engine.atmospheric_temperature_c == pytest.approx(-60.0)
    assert engine.atmospheric_pressure_hpa == pytest.approx(5.0)
    assert engine.water_vapor_density_g_m3 == 0.0
    assert engine.probability_false_alarm == pytest.approx(1.0e-5)


def test_invalid_false_alarm_probability_is_rejected(tmp_path):
    scenario = tmp_path / "invalid.yaml"
    scenario.write_text(
        """
scenario:
  name: Invalid
radar:
  frequency_hz: 1.0e9
  power_watts: 1000
simulation:
  pfa: 1.5
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="probability_false_alarm"):
        ScenarioLoader(str(scenario))


def test_exported_scenario_round_trips_runtime_parameters(tmp_path):
    original = ScenarioLoader(
        str(SCENARIOS / "f16_vs_sa6.yaml")
    ).create_simulation_engine()
    exported = tmp_path / "round_trip.yaml"

    assert export_scenario_to_yaml(original, str(exported), scenario_name="Round trip")

    restored = ScenarioLoader(str(exported)).create_simulation_engine()
    assert restored.radar.frequency_hz == original.radar.frequency_hz
    assert restored.radar.prf_hz == original.radar.prf_hz
    assert restored.radar.pulse_width_s == original.radar.pulse_width_s
    assert restored.radar.receiver_bandwidth_hz == original.radar.receiver_bandwidth_hz
    assert restored.radar.noise_figure_db == original.radar.noise_figure_db
    assert restored.receiver_full_scale_dbm == original.receiver_full_scale_dbm
    assert restored.radar.polarization_tilt_deg == original.radar.polarization_tilt_deg
    assert restored.probability_false_alarm == original.probability_false_alarm
    assert restored.clutter_enabled == original.clutter_enabled
    assert restored.terrain_type == original.terrain_type
    assert restored.atmospheric_pressure_hpa == original.atmospheric_pressure_hpa
    assert restored.targets[0].rcs_mean == original.targets[0].rcs_mean
    assert restored.targets[0].has_jammer == original.targets[0].has_jammer


def test_ground_surface_parameters_reach_engine_results_and_round_trip(tmp_path):
    scenario = tmp_path / "ground.yaml"
    scenario.write_text(
        """
scenario:
  name: Ground model
  update_rate_hz: 10
radar:
  frequency_hz: 5.0e9
  power_watts: 10000
  position: {x_m: 0, y_m: 0, z_m: 1000}
targets:
  - name: Surface target
    rcs_m2: 2
    initial_position: {x_m: 1000, y_m: 0, z_m: 0}
    has_ecm: true
    ecm_type: drfm
    ecm_power_watts: 250
    drfm:
      gain_over_skin_db: 13
      capture_dwell_s: 0.7
      pull_rate_mps: 85
      max_pull_m: 1600
      mode: rgpo
      inherent_delay_s: 4.0e-7
environment:
  terrain_type: rural
  ground_surface:
    model: oh1992
    gamma_db: -17.5
    relative_permittivity_real: 10.0
    relative_permittivity_loss: 1.2
    rms_height_m: 0.012
simulation:
  enable_atmospheric_loss: false
  enable_clutter: true
""",
        encoding="utf-8",
    )

    engine = ScenarioLoader(str(scenario)).create_simulation_engine()
    assert engine.ground_model == "oh1992"
    assert engine.land_gamma_db == pytest.approx(-17.5)
    assert engine.ground_relative_permittivity == complex(10.0, -1.2)
    assert engine.ground_rms_height_m == pytest.approx(0.012)
    assert engine.targets[0].drfm_gain_over_skin_db == pytest.approx(13.0)
    assert engine.targets[0].drfm_capture_dwell_s == pytest.approx(0.7)
    assert engine.targets[0].drfm_pull_rate_mps == pytest.approx(85.0)
    assert engine.targets[0].drfm_max_pull_m == pytest.approx(1600.0)
    assert engine.targets[0].drfm_inherent_delay_s == pytest.approx(4.0e-7)

    result = engine.step()[0]
    assert result.surface_clutter_model == "oh1992_bare_soil"
    assert result.surface_sigma0_db is not None
    assert result.surface_cell_area_m2 > 0.0
    assert result.surface_clutter_rcs_m2 > 0.0
    assert result.to_dict()["surface_clutter_model"] == "oh1992_bare_soil"

    exported = tmp_path / "ground_round_trip.yaml"
    assert export_scenario_to_yaml(engine, str(exported))
    restored = ScenarioLoader(str(exported)).create_simulation_engine()
    assert restored.ground_model == engine.ground_model
    assert restored.land_gamma_db == engine.land_gamma_db
    assert restored.ground_relative_permittivity == engine.ground_relative_permittivity
    assert restored.ground_rms_height_m == engine.ground_rms_height_m
    assert restored.targets[0].drfm_gain_over_skin_db == pytest.approx(13.0)
    assert restored.targets[0].drfm_capture_dwell_s == pytest.approx(0.7)
    assert restored.targets[0].drfm_pull_rate_mps == pytest.approx(85.0)
    assert restored.targets[0].drfm_max_pull_m == pytest.approx(1600.0)
    assert restored.targets[0].drfm_inherent_delay_s == pytest.approx(4.0e-7)
