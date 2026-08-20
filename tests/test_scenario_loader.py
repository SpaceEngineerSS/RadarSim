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
    assert restored.radar.polarization_tilt_deg == original.radar.polarization_tilt_deg
    assert restored.probability_false_alarm == original.probability_false_alarm
    assert restored.clutter_enabled == original.clutter_enabled
    assert restored.terrain_type == original.terrain_type
    assert restored.atmospheric_pressure_hpa == original.atmospheric_pressure_hpa
    assert restored.targets[0].rcs_mean == original.targets[0].rcs_mean
    assert restored.targets[0].has_jammer == original.targets[0].has_jammer
