"""
RadarSim UI Module

Professional scientific GUI for radar simulation visualization.

Components:
    - thread_manager: Simulation-UI bridge using QThread
    - ppi_scope: Plan Position Indicator (polar radar display)
    - a_scope: Amplitude vs Range diagnostic view
    - range_doppler: Range-Doppler Map (pulse-Doppler visualization)
    - main_window: Application shell
"""

from .thread_manager import SimulationWorker, SimulationThread
from .ppi_scope import PPIScope
from .a_scope import AScope
from .range_doppler import RangeDopplerScope
from .main_window import MainWindow

__all__ = [
    'SimulationWorker',
    'SimulationThread',
    'PPIScope',
    'AScope',
    'RangeDopplerScope',
    'MainWindow',
]

