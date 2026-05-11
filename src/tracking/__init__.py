# Developed by Mehmet Gümüş (@SpaceEngineerSS) - RadarSim v2.x
"""
Tracking Module

Multi-target tracking system for radar simulation.

Components:
    - LinearKalmanFilter: Constant Velocity Kalman Filter (Cartesian)
    - ExtendedKalmanFilter: EKF with polar [r, θ] measurement model
    - TrackManager: Multi-target track management with data association
    - Track: Individual target track container
    - TrackStatus: Track lifecycle states

Example:
    >>> from src.tracking import TrackManager
    >>> manager = TrackManager(gate_distance=500)
    >>> manager.set_ekf_mode(True)  # Enable EKF for polar measurements
    >>> tracks = manager.update_polar([(10000.0, 0.5)], dt=0.1)

Reference:
    - Bar-Shalom, Y. "Estimation with Applications to Tracking", 2001
"""

from .kalman import KalmanState, LinearKalmanFilter
from .tracker import Track, TrackManager, TrackStatus

# Extended Kalman Filter (Phase 27)
try:
    from .ekf import ExtendedKalmanFilter

    __all__ = [
        "LinearKalmanFilter",
        "ExtendedKalmanFilter",
        "KalmanState",
        "TrackManager",
        "Track",
        "TrackStatus",
    ]
except ImportError:
    __all__ = ["LinearKalmanFilter", "KalmanState", "TrackManager", "Track", "TrackStatus"]
