"""
RadarSim: physics-based radar simulation.

Subpackages:
    physics     radar equation, propagation, clutter, RCS, ECM, terrain
    signal      waveforms, CFAR, pulse-Doppler processing, SAR/ISAR
    tracking    Kalman/EKF filters, track-while-scan manager, monopulse
    simulation  time-stepping engine, frames, events, analysis, recording
    io          scenario files, export, replay, reports
    ui          desktop interface (optional ``gui`` extra)
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("radarsim")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "4.0.0"

__all__ = ["__version__"]
