"""
RadarSim Source Package

Professional radar simulation with:
- 3D radar physics (ITU-R P.676, Swerling RCS)
- Target tracking and simulation
- ECM/ECCM simulation
- Modern PyQt6 visualization
"""

# Re-export from modern subpackages
from src.physics import (
    RadarParameters,
    calculate_snr,
    calculate_received_power,
    calculate_doppler_shift,
    ITU_R_P676,
    SwerlingRCS,
    SwerlingModel,
    SPEED_OF_LIGHT,
    BOLTZMANN_CONSTANT,
    ECMSimulator,
    ECMType,
)

from src.simulation.objects import (
    Target,
    Radar,
    MotionModel,
)

from src.simulation.engine import SimulationEngine

__version__ = "1.0.0"
__author__ = "RadarSim Contributors"

__all__ = [
    # Physics
    'RadarParameters', 'calculate_snr', 'calculate_received_power',
    'calculate_doppler_shift', 'ITU_R_P676', 'SwerlingRCS', 
    'SwerlingModel', 'SPEED_OF_LIGHT', 'BOLTZMANN_CONSTANT',
    # ECM
    'ECMSimulator', 'ECMType',
    # Simulation
    'Target', 'Radar', 'MotionModel', 'SimulationEngine',
]
