"""
Scenario Exporter

Serializes the current simulation state to YAML format,
allowing users to save and share scenarios.
"""

import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np


def export_scenario_to_yaml(
    engine,
    filepath: str,
    scenario_name: str = "Custom Scenario",
    description: str = ""
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
            'scenario': {
                'name': scenario_name,
                'description': description or f"Exported on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                'version': '1.0'
            },
            'radar': _extract_radar_config(engine),
            'targets': _extract_targets(engine),
            'environment': _extract_environment(engine),
            'simulation': _extract_simulation_params(engine)
        }
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(scenario_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"[EXPORT] Scenario saved to: {filepath}")
        return True
        
    except Exception as e:
        print(f"[EXPORT] Failed to export scenario: {e}")
        return False


def _extract_radar_config(engine) -> Dict[str, Any]:
    """Extract radar configuration from engine."""
    radar = engine.radar
    
    return {
        'frequency_hz': float(radar.frequency_hz),
        'power_watts': float(radar.power_watts),
        'antenna_gain_db': float(radar.antenna_gain_db),
        'beamwidth_deg': float(radar.beamwidth_deg),
        'scan_rate_rpm': float(radar.scan_rate_rpm),
        'position': {
            'x_m': float(radar.position[0]),
            'y_m': float(radar.position[1]),
            'z_m': float(radar.position[2]) if len(radar.position) > 2 else 0.0
        }
    }


def _extract_targets(engine) -> list:
    """Extract target configurations from engine."""
    targets = []
    
    for t in engine.targets:
        target_data = {
            'id': int(t.target_id),
            'type': str(getattr(t, 'target_type', 'aircraft')),
            'position_m': {
                'x': float(t.position[0]),
                'y': float(t.position[1]),
                'z': float(t.position[2]) if len(t.position) > 2 else 0.0
            },
            'velocity_mps': {
                'vx': float(t.velocity[0]),
                'vy': float(t.velocity[1]),
                'vz': float(t.velocity[2]) if len(t.velocity) > 2 else 0.0
            },
            'rcs_m2': float(t.rcs_m2),
            'swerling_model': int(t.swerling_model.value) if hasattr(t.swerling_model, 'value') else t.swerling_model
        }
        
        # Optional jammer settings
        if hasattr(t, 'jammer_active') and t.jammer_active:
            target_data['jammer'] = {
                'active': True,
                'power_watts': float(getattr(t, 'jammer_power', 1000))
            }
        
        targets.append(target_data)
    
    return targets


def _extract_environment(engine) -> Dict[str, Any]:
    """Extract environment settings from engine."""
    env = {
        'enable_atmospheric': bool(engine.enable_atmospheric),
        'clutter': {
            'enabled': bool(getattr(engine, 'clutter_enabled', False)),
            'terrain_type': str(getattr(engine, 'terrain_type', 'rural'))
        }
    }
    
    # ECM settings
    if hasattr(engine, 'ecm_active'):
        env['ecm'] = {
            'active': bool(engine.ecm_active),
            'type': str(getattr(engine, 'ecm_type', 'noise'))
        }
    
    return env


def _extract_simulation_params(engine) -> Dict[str, Any]:
    """Extract simulation parameters from engine."""
    params = {
        'dt': float(engine.dt),
        'detection_threshold_db': float(engine.detection_threshold_db),
    }
    
    # Advanced features
    if getattr(engine, 'mti_enabled', False):
        params['mti'] = {
            'enabled': True,
            'threshold_mps': float(engine.mti_threshold_mps)
        }
    
    if getattr(engine, 'frequency_agility_enabled', False):
        params['eccm'] = {
            'frequency_agility': True
        }
    
    if getattr(engine, 'monopulse_enabled', False):
        params['monopulse'] = True
    
    return params


def get_default_filename() -> str:
    """Generate default filename with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"scenario_{timestamp}.yaml"
