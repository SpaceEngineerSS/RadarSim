"""
Scenario Exporter

Serializes the current simulation state to YAML format,
allowing users to save and share scenarios.
"""

from datetime import datetime
from typing import Any, Dict

import yaml


def export_scenario_to_yaml(
    engine, filepath: str, scenario_name: str = "Custom Scenario", description: str = ""
) -> bool:
    """
    Export current simulation state to YAML file.

    Collects all relevant simulation parameters and serializes
    them to the RadarSim YAML format.

    Args:
        engine: SimulationEngine instance
        filepath: Output file path
        scenario_name: Human-readable scenario name
        description: Scenario description

    Returns:
        True if export successful, False otherwise
    """
    try:
        scenario_data = {
            "scenario": {
                "name": scenario_name,
                "description": description
                or f"Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "version": "1.0",
                "duration_seconds": max(float(engine.current_time), float(engine.dt)),
                "update_rate_hz": 1.0 / float(engine.dt),
            },
            "radar": _extract_radar_config(engine),
            "targets": _extract_targets(engine),
            "environment": _extract_environment(engine),
            "simulation": _extract_simulation_params(engine),
        }

        # Write to file
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                scenario_data,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

        print(f"[EXPORT] Scenario saved to: {filepath}")
        return True

    except Exception as e:
        print(f"[EXPORT] Failed to export scenario: {e}")
        return False


def _extract_radar_config(engine) -> Dict[str, Any]:
    """Extract radar configuration from engine."""
    radar = engine.radar

    return {
        "name": str(radar.radar_id),
        "type": "pulse_doppler",
        "frequency_hz": float(radar.frequency_hz),
        "power_watts": float(radar.power_watts),
        "antenna": {
            "gain_db": float(radar.antenna_gain_db),
            "beamwidth_az_deg": float(radar.to_dict()["beamwidth_deg"]),
            "beamwidth_el_deg": float(radar.to_dict()["beamwidth_el_deg"]),
            "polarization_tilt_deg": float(radar.polarization_tilt_deg),
        },
        "receiver": {
            "noise_figure_db": float(radar.noise_figure_db),
            "bandwidth_hz": float(radar.receiver_bandwidth_hz),
            "system_temperature_k": float(radar.system_temperature_k),
        },
        "prf_hz": float(radar.prf_hz),
        "pulse_width_s": float(radar.pulse_width_s),
        "system_losses_db": float(radar.system_losses_db),
        "scan_rate_rpm": float(radar.scan_rate_rpm),
        "position": {
            "x_m": float(radar.position[0]),
            "y_m": float(radar.position[1]),
            "z_m": float(radar.position[2]) if len(radar.position) > 2 else 0.0,
        },
    }


def _extract_targets(engine) -> list:
    """Extract target configurations from engine."""
    targets = []

    for t in engine.targets:
        target_data = {
            "name": f"Target_{int(t.target_id)}",
            "type": str(getattr(t, "target_type", "aircraft")),
            "initial_position": {
                "x_m": float(t.position[0]),
                "y_m": float(t.position[1]),
                "z_m": float(t.position[2]) if len(t.position) > 2 else 0.0,
            },
            "velocity": {
                "vx_mps": float(t.velocity[0]),
                "vy_mps": float(t.velocity[1]),
                "vz_mps": float(t.velocity[2]) if len(t.velocity) > 2 else 0.0,
            },
            "rcs_m2": float(t.rcs_mean),
            "swerling_model": (
                int(t.swerling_model.value)
                if hasattr(t.swerling_model, "value")
                else t.swerling_model
            ),
        }

        # Optional jammer settings
        if hasattr(t, "jammer_active") and t.jammer_active:
            target_data["has_ecm"] = True
            target_data["ecm_type"] = str(getattr(t, "ecm_type", "noise_barrage"))
            target_data["ecm_power_watts"] = float(
                getattr(t, "jammer_power_watts", 1000)
            )
            target_data["ecm_bandwidth_hz"] = float(
                getattr(t, "jammer_bandwidth_hz", 100e6)
            )

        targets.append(target_data)

    return targets


def _extract_environment(engine) -> Dict[str, Any]:
    """Extract environment settings from engine."""
    env = {
        "enable_atmospheric": bool(engine.enable_atmospheric),
        "temperature_c": float(engine.atmospheric_temperature_c),
        "pressure_hpa": float(engine.atmospheric_pressure_hpa),
        "water_vapor_gpm3": float(engine.water_vapor_density_g_m3),
        "rain_rate_mm_hr": float(engine.rain_rate_mm_hr),
        "clutter": {
            "enabled": bool(getattr(engine, "clutter_enabled", False)),
            "terrain_type": str(getattr(engine, "terrain_type", "rural")),
            "sea_state": int(getattr(engine, "sea_state", 0)),
        },
        "ground_surface": {
            "model": str(getattr(engine, "ground_model", "gamma")),
            "gamma_db": getattr(engine, "land_gamma_db", None),
            "relative_permittivity_real": float(
                getattr(engine, "ground_relative_permittivity", complex(8.0, -0.8)).real
            ),
            "relative_permittivity_loss": float(
                -getattr(engine, "ground_relative_permittivity", complex(8.0, -0.8)).imag
            ),
            "rms_height_m": float(getattr(engine, "ground_rms_height_m", 0.01)),
        },
    }

    # ECM settings
    if hasattr(engine, "ecm_active"):
        env["ecm"] = {
            "active": bool(engine.ecm_active),
            "type": str(getattr(engine, "ecm_type", "noise")),
        }

    return env


def _extract_simulation_params(engine) -> Dict[str, Any]:
    """Extract simulation parameters from engine."""
    params = {
        "enable_atmospheric_loss": bool(engine.enable_atmospheric),
        "enable_clutter": bool(engine.clutter_enabled),
        "pfa": float(engine.probability_false_alarm),
        "pulses_integrated": int(engine.pulses_integrated),
    }

    # Advanced features
    if getattr(engine, "mti_enabled", False):
        params["mti"] = {
            "enabled": True,
            "threshold_mps": float(engine.mti_threshold_mps),
        }

    if getattr(engine, "frequency_agility_enabled", False):
        params["eccm"] = {"frequency_agility": True}

    if getattr(engine, "monopulse_enabled", False):
        params["monopulse"] = True

    return params


def get_default_filename() -> str:
    """Generate default filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"scenario_{timestamp}.yaml"
