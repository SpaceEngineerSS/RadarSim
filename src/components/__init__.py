"""
Components Package

High-level radar component models (antenna, receiver, transmitter).
"""

from .antenna import PhasedArrayAntenna, AntennaParameters

__all__ = [
    'PhasedArrayAntenna',
    'AntennaParameters',
]
