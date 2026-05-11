"""
RadarSim Source Package

Professional radar simulation with:
- 3D radar physics (ITU-R P.676, Swerling RCS)
- Target tracking and simulation
- ECM/ECCM simulation
- Modern PySide6 visualization
"""

# Re-export from modern subpackages
from src.physics import (
    BOLTZMANN_CONSTANT,
    ITU_R_P676,
    SPEED_OF_LIGHT,
    ECMSimulator,
    ECMType,
    RadarParameters,
    SwerlingModel,
    SwerlingRCS,
    calculate_doppler_shift,
    calculate_received_power,
    calculate_snr,
)
from src.simulation.engine import SimulationEngine
from src.simulation.objects import (
    MotionModel,
    Radar,
    Target,
)

__version__ = "1.0.0"
__author__ = "RadarSim Contributors"

__all__ = [
    # Physics
    "RadarParameters",
    "calculate_snr",
    "calculate_received_power",
    "calculate_doppler_shift",
    "ITU_R_P676",
    "SwerlingRCS",
    "SwerlingModel",
    "SPEED_OF_LIGHT",
    "BOLTZMANN_CONSTANT",
    # ECM
    "ECMSimulator",
    "ECMType",
    # Simulation
    "Target",
    "Radar",
    "MotionModel",
    "SimulationEngine",
]
