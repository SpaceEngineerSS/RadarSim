"""
RadarSim Physics Package

Modular physics calculations for radar simulation.
IEEE Std 686-2008 compliant with scientific references.

Modules:
    - constants: Physical constants (SI units)
    - radar_equation: Core radar equation calculations
    - atmospheric: ITU-R P.676-12 atmospheric attenuation
    - rcs: RCS and Swerling fluctuation models
    - clutter: Ground, sea, and weather clutter models
    - ecm: Electronic countermeasures simulation
    - terrain: Terrain generation and line-of-sight physics
"""

from .constants import (
    SPEED_OF_LIGHT,
    BOLTZMANN_CONSTANT,
    STANDARD_TEMPERATURE,
    STANDARD_PRESSURE,
    EARTH_RADIUS,
)

from .radar_equation import (
    RadarParameters,
    calculate_received_power,
    calculate_snr,
    calculate_detection_range,
    calculate_doppler_shift,
)

from .atmospheric import ITU_R_P676

from .rcs import (
    SwerlingModel,
    SwerlingRCS,
    TargetType,
    calculate_aspect_dependent_rcs,
)

from .ecm import ECMSimulator, ECMType

__all__ = [
    # Constants
    'SPEED_OF_LIGHT',
    'BOLTZMANN_CONSTANT', 
    'STANDARD_TEMPERATURE',
    'STANDARD_PRESSURE',
    'EARTH_RADIUS',
    # Radar Equation
    'RadarParameters',
    'calculate_received_power',
    'calculate_snr',
    'calculate_detection_range',
    'calculate_doppler_shift',
    # Atmospheric
    'ITU_R_P676',
    # RCS
    'SwerlingModel',
    'SwerlingRCS',
    'TargetType',
    'calculate_aspect_dependent_rcs',
    # ECM
    'ECMSimulator',
    'ECMType',
]

