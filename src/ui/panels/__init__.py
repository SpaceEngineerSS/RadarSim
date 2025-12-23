"""
UI Panels Package

Modular UI components extracted from main_window.py for better maintainability.

Components:
    - ControlPanel: Radar parameter adjustments (frequency, power, range)
    - TargetInspector: Selected target information and ECM controls
"""

from .radar_controls import ControlPanel
from .target_inspector import TargetInspector

__all__ = [
    'ControlPanel',
    'TargetInspector',
]
