"""
Tracking Module

Multi-target tracking system for radar simulation.

Components:
    - LinearKalmanFilter: Constant Velocity Kalman Filter
    - TrackManager: Multi-target track management with data association
    - Track: Individual target track container
    - TrackStatus: Track lifecycle states

Example:
    >>> from src.tracking import TrackManager
    >>> manager = TrackManager(gate_distance=500)
    >>> tracks = manager.update([(1000, 2000), (3000, 4000)], dt=0.1)
"""

from .kalman import LinearKalmanFilter, KalmanState
from .tracker import TrackManager, Track, TrackStatus

__all__ = [
    'LinearKalmanFilter',
    'KalmanState',
    'TrackManager',
    'Track',
    'TrackStatus'
]
