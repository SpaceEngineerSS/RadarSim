"""
Scenario Exporter

Writes a ``SimulationEngine`` back to the canonical scenario format read by
``ScenarioLoader`` (see ``scenario_schema``). The exported file describes the
scenario's *initial* state, so loading it reproduces the same run (with the
same seed).
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

logger = logging.getLogger(__name__)


def _xyz(vector, keys=("x_m", "y_m", "z_m")) -> Dict[str, float]:
    return {key: float(value) for key, value in zip(keys, vector)}


def _velocity(vector) -> Dict[str, float]:
    return _xyz(vector, ("vx_mps", "vy_mps", "vz_mps"))


def scenario_dict(
    engine,
    scenario_name: str = "Custom Scenario",
    description: str = "",
    duration_s: Optional[float] = None,
) -> Dict[str, Any]:
    """Canonical scenario mapping for ``engine``'s initial state."""
    if duration_s is None:
        duration_s = max(float(engine.current_time), float(engine.dt))
    data: Dict[str, Any] = {
        "scenario": {
            "name": scenario_name,
            "description": description
            or f"Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "duration_seconds": float(duration_s),
            "update_rate_hz": 1.0 / float(engine.dt),
        },
        "radar": _radar(engine),
        "targets": [],
        "environment": _environment(engine),
        "simulation": _simulation(engine),
    }
    chaff, decoys = [], []
    for target in engine.targets:
        state = engine._initial_target_states.get(target.target_id, target.state)
        if target.target_type == "chaff":
            chaff.append(
                {
                    "name": target.name,
                    "deploy_time_s": target.active_from_s,
                    "position": _xyz(state.position),
                    "velocity": _velocity(state.velocity),
                    "rcs_m2": float(target.rcs_mean),
                    "bloom_time_s": target.rcs_bloom_time_s,
                    "decay_time_s": target.rcs_decay_time_s,
                }
            )
        elif target.target_type == "decoy":
            decoys.append(
                {
                    "name": target.name,
                    "deploy_time_s": target.active_from_s,
                    "initial_position": _xyz(state.position),
                    "velocity": _velocity(state.velocity),
                    "rcs_m2": float(target.rcs_mean),
                    "swerling_model": target.swerling_model.value,
                }
            )
        else:
            data["targets"].append(_target(target, state))
    if chaff or decoys:
        data["ecm"] = {}
        if decoys:
            data["ecm"]["decoys"] = decoys
        if chaff:
            data["ecm"]["chaff_clouds"] = chaff
    return data


def export_scenario_to_yaml(
    engine,
    filepath: str,
    scenario_name: str = "Custom Scenario",
    description: str = "",
    duration_s: Optional[float] = None,
) -> bool:
    """
    Write ``engine`` to ``filepath`` as a canonical YAML scenario.

    Returns:
        True on success, False if the file could not be written
    """
    data = scenario_dict(engine, scenario_name, description, duration_s)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    except OSError:
        logger.exception("Could not write scenario to %s", filepath)
        return False
    return True


def _radar(engine) -> Dict[str, Any]:
    radar = engine.radar
    initial = engine._initial_radar_state
    antenna: Dict[str, Any] = {
        "gain_db": float(radar.antenna_gain_db),
        "beamwidth_az_deg": float(np.degrees(radar.beamwidth_rad)),
        "beamwidth_el_deg": float(np.degrees(radar.beamwidth_el_rad)),
        "polarization_tilt_deg": float(radar.polarization_tilt_deg),
        "sidelobe_db": float(radar.sidelobe_level_db),
        "scan_type": radar.scan_type.value,
        "scan_rate_rpm": float(radar.scan_rate_rpm),
    }
    if radar.scan_type.value == "sector":
        antenna["sector_az_min_deg"] = float(np.degrees(radar.sector_az_min_rad))
        antenna["sector_az_max_deg"] = float(np.degrees(radar.sector_az_max_rad))
        antenna["scan_speed_deg_s"] = float(radar.scan_speed_deg_s)
    elif radar.scan_type.value == "stare":
        antenna["stare_azimuth_deg"] = float(np.degrees(radar.stare_azimuth_rad))
    return {
        "name": str(radar.radar_id),
        "frequency_hz": float(radar.frequency_hz),
        "power_watts": float(radar.power_watts),
        "prf_hz": float(radar.prf_hz),
        "pulse_width_s": float(radar.pulse_width_s),
        "system_losses_db": float(radar.system_losses_db),
        "antenna": antenna,
        "receiver": {
            "noise_figure_db": float(radar.noise_figure_db),
            "bandwidth_hz": float(radar.receiver_bandwidth_hz),
            "system_temperature_k": float(radar.system_temperature_k),
            "full_scale_dbm": float(engine.receiver_full_scale_dbm),
        },
        "position": _xyz(initial.position),
        "velocity": _velocity(initial.velocity),
    }


def _target(target, state) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "name": target.name,
        "type": target.target_type,
        "rcs_m2": float(target.rcs_mean),
        "swerling_model": target.swerling_model.value,
        "initial_position": _xyz(state.position),
        "velocity": _velocity(state.velocity),
    }
    if np.any(state.acceleration != 0.0):
        data["acceleration"] = _xyz(state.acceleration, ("ax_mps2", "ay_mps2", "az_mps2"))
    if target.turn_rate_rad_s != 0.0:
        data["turn_rate_deg_s"] = float(np.degrees(target.turn_rate_rad_s))
    if target.active_from_s > 0.0:
        data["deploy_time_s"] = target.active_from_s
    data["has_ecm"] = bool(target.has_jammer)
    if target.has_jammer:
        data.update(
            {
                "ecm_type": target.ecm_type,
                "ecm_power_watts": float(target.jammer_power_watts),
                "ecm_bandwidth_hz": float(target.jammer_bandwidth_hz),
                "drfm": {
                    "gain_over_skin_db": target.drfm_gain_over_skin_db,
                    "capture_dwell_s": target.drfm_capture_dwell_s,
                    "pull_rate_mps": target.drfm_pull_rate_mps,
                    "max_pull_m": target.drfm_max_pull_m,
                    "mode": target.drfm_mode,
                    "vgpo_rate_hz_per_s": target.drfm_vgpo_rate_hz_per_s,
                    "max_doppler_pull_hz": target.drfm_max_doppler_pull_hz,
                    "inherent_delay_s": target.drfm_inherent_delay_s,
                },
            }
        )
    return data


def _environment(engine) -> Dict[str, Any]:
    ground: Dict[str, Any] = {
        "model": engine.ground_model,
        "relative_permittivity_real": float(engine.ground_relative_permittivity.real),
        "relative_permittivity_loss": float(-engine.ground_relative_permittivity.imag),
        "rms_height_m": float(engine.ground_rms_height_m),
    }
    if engine.land_gamma_db is not None:
        ground["gamma_db"] = float(engine.land_gamma_db)
    env: Dict[str, Any] = {
        "temperature_c": float(engine.atmospheric_temperature_c),
        "pressure_hpa": float(engine.atmospheric_pressure_hpa),
        "water_vapor_gpm3": float(engine.water_vapor_density_g_m3),
        "rain_rate_mm_hr": float(engine.rain_rate_mm_hr),
        "rain_height_m": float(engine.rain_height_m),
        "terrain_type": str(engine.terrain_type),
        "sea_state": int(engine.sea_state),
        "ground_surface": ground,
    }
    clutter: Dict[str, Any] = {}
    if engine.clutter_velocity_spread_mps is not None:
        clutter["velocity_spread_mps"] = float(engine.clutter_velocity_spread_mps)
    if engine.clutter_sectors:
        clutter["sectors"] = [
            {
                "azimuth_start_deg": s.azimuth_start_deg,
                "azimuth_end_deg": s.azimuth_end_deg,
                "range_start_km": s.range_start_m / 1000.0,
                "range_end_km": s.range_end_m / 1000.0,
                "terrain_type": s.terrain_type,
                "sigma0_adjustment_db": s.sigma0_adjustment_db,
            }
            for s in engine.clutter_sectors
        ]
    if clutter:
        env["clutter"] = clutter
    if engine.terrain is not None:
        config = engine.terrain.config
        env["terrain"] = {
            "seed": int(config.seed),
            "scale_m": float(config.scale),
            "max_height_m": float(config.max_height),
            "octaves": int(config.octaves),
            "persistence": float(config.persistence),
            "lacunarity": float(config.lacunarity),
            "peaks": [
                {"x_m": float(x), "y_m": float(y), "height_m": float(h)}
                for x, y, h in config.mountain_peaks
            ],
        }
    return env


def _simulation(engine) -> Dict[str, Any]:
    sim: Dict[str, Any] = {
        "enable_atmospheric_loss": bool(engine.enable_atmospheric),
        "enable_clutter": bool(engine.clutter_enabled),
        "pfa": float(engine.probability_false_alarm),
        "pulses_integrated": int(engine.pulses_integrated),
        "range_noise_std_m": float(engine.range_noise_std),
        "angle_noise_std_rad": float(engine.angle_noise_std),
        "scan_gated_detection": bool(engine.scan_gated_detection),
        "mti": {"enabled": bool(engine.mti_enabled), "threshold_mps": float(engine.mti_threshold_mps)},
        "monopulse": {"enabled": bool(engine.monopulse_enabled)},
        "eccm": {"frequency_agility": bool(engine.frequency_agility_enabled)},
    }
    if engine.seed is not None:
        sim["seed"] = int(engine.seed)
    if engine.cfar is not None:
        sim["cfar"] = {
            "type": engine.cfar.cfar_type,
            "guard_cells": int(engine.cfar.guard_cells),
            "reference_cells": int(engine.cfar.reference_cells),
        }
    return sim


__all__: List[str] = ["export_scenario_to_yaml", "scenario_dict"]
