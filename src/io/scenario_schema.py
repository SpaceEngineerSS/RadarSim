"""
Scenario file schema.

Describes every key a RadarSim scenario (YAML or JSON) may contain. The loader
reports keys that are not in the schema, with a "did you mean" suggestion, so
typos such as ``rain_rate_mmhr`` no longer disappear silently.

Schema notation:
    dict          -> nested mapping of allowed keys
    [dict]        -> list of mappings with those keys
    None          -> scalar leaf
"""

import difflib
from typing import Any, Dict, List, Optional

_POSITION = {"x_m": None, "y_m": None, "z_m": None}
_VELOCITY = {"vx_mps": None, "vy_mps": None, "vz_mps": None}
_ACCELERATION = {"ax_mps2": None, "ay_mps2": None, "az_mps2": None}

SCHEMA: Dict[str, Any] = {
    "scenario": {
        "name": None,
        "description": None,
        "duration_seconds": None,
        "update_rate_hz": None,
        "required_radar_preset": None,
        "seed": None,
    },
    "radar": {
        "name": None,
        "type": None,  # informational (e.g. "phased_array")
        "frequency_hz": None,
        "power_watts": None,
        "prf_hz": None,
        "pulse_width_s": None,
        "system_losses_db": None,
        "antenna": {
            "gain_db": None,
            "beamwidth_az_deg": None,
            "beamwidth_el_deg": None,
            "polarization_tilt_deg": None,
            "sidelobe_db": None,
            "scan_type": None,
            "scan_rate_rpm": None,
            "scan_rate_deg_s": None,
            "sector_az_min_deg": None,
            "sector_az_max_deg": None,
            "scan_speed_deg_s": None,
            "stare_azimuth_deg": None,
        },
        "receiver": {
            "noise_figure_db": None,
            "bandwidth_hz": None,
            "system_temperature_k": None,
            "full_scale_dbm": None,
            "dynamic_range_db": None,
        },
        "position": _POSITION,
        "height_above_terrain_m": None,
        "velocity": _VELOCITY,
    },
    "targets": [
        {
            "name": None,
            "type": None,
            "affiliation": None,
            "rcs_m2": None,
            "swerling_model": None,
            "initial_position": _POSITION,
            "velocity": _VELOCITY,
            "acceleration": _ACCELERATION,
            "turn_rate_deg_s": None,
            "deploy_time_s": None,
            "has_ecm": None,
            "ecm_type": None,
            "ecm_power_watts": None,
            "ecm_bandwidth_hz": None,
            "drfm": {
                "gain_over_skin_db": None,
                "capture_dwell_s": None,
                "pull_rate_mps": None,
                "max_pull_m": None,
                "mode": None,
                "vgpo_rate_hz_per_s": None,
                "max_doppler_pull_hz": None,
                "inherent_delay_s": None,
            },
        }
    ],
    "environment": {
        "temperature_c": None,
        "pressure_hpa": None,
        "water_vapor_gpm3": None,
        "rain_rate_mm_hr": None,
        "rain_height_m": None,
        "terrain_type": None,
        "sea_state": None,
        "ground_surface": {
            "model": None,
            "gamma_db": None,
            "relative_permittivity_real": None,
            "relative_permittivity_loss": None,
            "rms_height_m": None,
        },
        "clutter": {
            "terrain_type": None,
            "sea_state": None,
            "velocity_spread_mps": None,
            "sectors": [
                {
                    "name": None,
                    "azimuth_start_deg": None,
                    "azimuth_end_deg": None,
                    "range_start_km": None,
                    "range_end_km": None,
                    "terrain_type": None,
                    "sigma0_adjustment_db": None,
                }
            ],
        },
        "terrain": {
            "preset": None,
            "seed": None,
            "scale_m": None,
            "max_height_m": None,
            "octaves": None,
            "persistence": None,
            "lacunarity": None,
            "peaks": [{"x_m": None, "y_m": None, "height_m": None}],
        },
    },
    "ecm": {
        "chaff_clouds": [
            {
                "name": None,
                "deploy_time_s": None,
                "deployed_by": None,
                "position": _POSITION,
                "velocity": _VELOCITY,
                "rcs_m2": None,
                "bloom_time_s": None,
                "decay_time_s": None,
            }
        ],
        "decoys": [
            {
                "name": None,
                "deploy_time_s": None,
                "deployed_by": None,
                "initial_position": _POSITION,
                "velocity": _VELOCITY,
                "rcs_m2": None,
                "swerling_model": None,
            }
        ],
    },
    "simulation": {
        "enable_atmospheric_loss": None,
        "enable_clutter": None,
        "pfa": None,
        "pulses_integrated": None,
        "seed": None,
        "range_noise_std_m": None,
        "angle_noise_std_rad": None,
        "scan_gated_detection": None,
        "cfar": {"type": None, "guard_cells": None, "reference_cells": None},
        "mti": {"enabled": None, "threshold_mps": None},
        "monopulse": {"enabled": None},
        "tracker": {"process_noise_mps2": None, "confirm_hits": None, "max_misses": None},
        "eccm": {"frequency_agility": None},
    },
}


def validate(data: Any) -> List[str]:
    """
    Return a warning for every key in ``data`` that the schema does not know.

    Raises:
        ValueError: if a section that must be a mapping or list has another type
    """
    return _validate(data, SCHEMA, "")


def _validate(data: Any, schema: Any, path: str) -> List[str]:
    warnings: List[str] = []
    if isinstance(schema, list):
        if data is None:
            return warnings
        if not isinstance(data, list):
            raise ValueError(f"'{path}' must be a list")
        for index, item in enumerate(data):
            warnings.extend(_validate(item, schema[0], f"{path}[{index}]"))
        return warnings
    if isinstance(schema, dict):
        if data is None:
            return warnings
        if not isinstance(data, dict):
            raise ValueError(f"'{path or 'scenario file'}' must be a mapping")
        for key, value in data.items():
            child = f"{path}.{key}" if path else str(key)
            if key not in schema:
                warnings.append(_unknown_key_message(child, key, schema))
                continue
            warnings.extend(_validate(value, schema[key], child))
    return warnings


def _unknown_key_message(path: str, key: Any, schema: Dict[str, Any]) -> str:
    suggestion: Optional[List[str]] = difflib.get_close_matches(
        str(key), list(schema.keys()), n=1, cutoff=0.75
    )
    hint = f" (did you mean '{suggestion[0]}'?)" if suggestion else ""
    return f"Unknown key '{path}' is ignored{hint}"
