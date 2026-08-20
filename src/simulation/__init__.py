"""
RadarSim Simulation Package

Headless simulation runners and batch processing tools.
"""

from .headless_runner import HeadlessRunner, SimulationConfig, SimulationResult
from .scenario_generator import ParameterSpace, ScenarioGenerator

__all__ = [
    "HeadlessRunner",
    "SimulationConfig",
    "SimulationResult",
    "ScenarioGenerator",
    "ParameterSpace",
    "NetworkManager",
    "CovarianceIntersection",
    "StrobeTriangulator",
]

try:
    from .network_manager import (
        CovarianceIntersection,
        NetworkManager,
        StrobeTriangulator,
    )
except ImportError:
    pass
