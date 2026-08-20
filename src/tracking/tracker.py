# Developed by Mehmet Gümüş (@SpaceEngineerSS) - RadarSim v2.x
"""
Track Manager for Multi-Target Tracking

Manages multiple radar tracks using Kalman Filters and Nearest-Neighbor
data association. Handles track initiation, maintenance, and deletion.

Track Lifecycle:
    TENTATIVE -> CONFIRMED -> COASTING -> DELETED

Reference:
    - Blackman, S. "Multiple-Target Tracking with Radar Applications", 1986
    - Bar-Shalom, Y. "Multitarget-Multisensor Tracking", 1990
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from .kalman import KalmanState, LinearKalmanFilter

# Extended Kalman filter
try:
    from .ekf import ExtendedKalmanFilter

    EKF_AVAILABLE = True
except ImportError:
    EKF_AVAILABLE = False
    ExtendedKalmanFilter = None


class TrackStatus(Enum):
    """Track lifecycle states."""

    TENTATIVE = "tentative"  # New track, needs confirmation
    CONFIRMED = "confirmed"  # Established track
    COASTING = "coasting"  # No measurements, predicting only
    DELETED = "deleted"  # Marked for removal


@dataclass
class Track:
    """
    Single target track.

    Attributes:
        id: Unique track identifier
        state: Kalman filter state [x, y, vx, vy]
        status: Track lifecycle status
        hits: Number of successful associations
        misses: Consecutive missed associations
        age: Time since track creation (seconds)
        last_update: Last measurement time
        history: Position history for trail display
    """

    id: int
    state: KalmanState
    status: TrackStatus = TrackStatus.TENTATIVE
    hits: int = 1
    consecutive_hits: int = 1
    misses: int = 0
    creation_time: float = 0.0
    last_update: float = 0.0
    current_time: float = 0.0
    history: List[Tuple[float, float]] = field(default_factory=list)

    # Optional classification metadata supplied by a caller.
    classification: str = "Unknown"
    confidence: float = 0.0

    @property
    def position(self) -> Tuple[float, float]:
        """Get current position (x, y) in meters."""
        return (self.state.x[0], self.state.x[1])

    @property
    def velocity(self) -> Tuple[float, float]:
        """Get current velocity (vx, vy) in m/s."""
        return (self.state.x[2], self.state.x[3])

    @property
    def speed_mps(self) -> float:
        """Get speed in m/s."""
        return np.sqrt(self.state.x[2] ** 2 + self.state.x[3] ** 2)

    @property
    def heading_rad(self) -> float:
        """Get heading in radians (0 = North, CW positive)."""
        return np.arctan2(self.state.x[3], self.state.x[2])

    @property
    def age_seconds(self) -> float:
        """Get track age in seconds."""
        return self.current_time - self.creation_time


class TrackManager:
    """
    Multi-target track manager with Nearest-Neighbor association.

    Features:
        - Automatic track initiation from unassigned detections
        - Nearest-neighbor data association with gating
        - Track coasting (prediction-only when no measurement)
        - Track deletion after max misses
        - Track history for trail visualization

    Example:
        >>> manager = TrackManager(gate_distance=1000, max_misses=5)
        >>> detections = [(1000, 2000), (3000, 4000)]
        >>> tracks = manager.update(detections, dt=0.1)
        >>> for track in tracks:
        ...     print(f"Track {track.id}: {track.position}")
    """

    def __init__(
        self,
        gate_distance: float = 500.0,
        max_misses: int = 5,
        confirm_hits: int = 3,
        max_history: int = 50,
        process_noise: float = 5.0,
        measurement_noise: float = 50.0,
        gate_probability: float = 0.9973,
    ) -> None:
        """
        Initialize Track Manager.

        Args:
            gate_distance: Maximum distance for association (meters)
            max_misses: Delete track after this many missed updates
            confirm_hits: Hits needed to confirm tentative track
            max_history: Maximum track history length
            process_noise: Kalman filter process noise
            measurement_noise: Kalman filter measurement noise
        """
        if gate_distance <= 0.0 or measurement_noise <= 0.0:
            raise ValueError("gate distance and measurement noise must be positive")
        if process_noise < 0.0:
            raise ValueError("process noise cannot be negative")
        if max_misses < 1 or confirm_hits < 1 or max_history < 1:
            raise ValueError("track count parameters must be at least one")
        self.gate_distance = gate_distance
        self.max_misses = max_misses
        self.confirm_hits = confirm_hits
        self.max_history = max_history
        if not 0.0 < gate_probability < 1.0:
            raise ValueError("gate_probability must be between zero and one")
        self.gate_probability = gate_probability
        self.gate_threshold = float(chi2.ppf(gate_probability, df=2))

        # Kalman filter for all tracks
        self.kf = LinearKalmanFilter(
            process_noise=process_noise, measurement_noise=measurement_noise
        )

        self.use_ekf = False
        self._ekf: Optional["ExtendedKalmanFilter"] = None

        # Track storage
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
        self.current_time = 0.0

    def update(
        self,
        detections: List[Tuple[float, float]],
        dt: float,
        detection_data: Optional[List[Dict]] = None,
    ) -> List[Track]:
        """
        Process new detections and update tracks.

        Steps:
            1. Predict all existing tracks
            2. Associate detections to tracks (nearest-neighbor)
            3. Update associated tracks with measurements
            4. Coast unassigned tracks (predict only)
            5. Initiate new tracks from unassigned detections
            6. Delete stale tracks

        Args:
            detections: List of (x, y) position measurements
            dt: Time since last update (seconds)
            detection_data: Optional metadata for each detection

        Returns:
            List of active tracks
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.current_time += dt
        current_time = self.current_time

        # 1. Predict all tracks
        for track in self.tracks.values():
            if track.status != TrackStatus.DELETED:
                if self.use_ekf and self._ekf is not None:
                    track.state = self._ekf.predict(track.state, dt)
                else:
                    track.state = self.kf.predict(track.state, dt)

        # 2. Data association (Nearest-Neighbor with gating)
        associations, unassigned_detections, unassigned_tracks = self._associate(
            detections
        )

        # 3. Update associated tracks
        for track_id, det_idx in associations.items():
            track = self.tracks[track_id]
            measurement = detections[det_idx]

            # Kalman update (EKF or Linear)
            if self.use_ekf and self._ekf is not None:
                track.state = self._ekf.update_cartesian(track.state, measurement)
            else:
                track.state = self.kf.update(track.state, measurement)
            track.last_update = current_time
            track.hits += 1
            track.consecutive_hits += 1
            track.misses = 0
            track.current_time = current_time

            # Promote tentative -> confirmed
            if (
                track.status == TrackStatus.TENTATIVE
                and track.consecutive_hits >= self.confirm_hits
            ):
                track.status = TrackStatus.CONFIRMED
            elif track.status == TrackStatus.COASTING:
                track.status = TrackStatus.CONFIRMED

            # Update history
            track.history.append(track.position)
            if len(track.history) > self.max_history:
                track.history.pop(0)

            # Copy detection metadata if available
            if detection_data and det_idx < len(detection_data):
                data = detection_data[det_idx]
                if "classification" in data:
                    track.classification = data["classification"]
                if "confidence" in data:
                    track.confidence = data["confidence"]

        # 4. Coast unassigned tracks
        for track_id in unassigned_tracks:
            track = self.tracks[track_id]
            track.misses += 1
            track.consecutive_hits = 0
            track.current_time = current_time

            if track.status == TrackStatus.CONFIRMED:
                track.status = TrackStatus.COASTING

            # Delete if too many misses
            if track.misses >= self.max_misses:
                track.status = TrackStatus.DELETED

            # Still update history with predicted position
            track.history.append(track.position)
            if len(track.history) > self.max_history:
                track.history.pop(0)

        # 5. Initiate new tracks from unassigned detections
        for det_idx in unassigned_detections:
            measurement = detections[det_idx]
            self._create_track(measurement, detection_data, det_idx)

        # 6. Remove deleted tracks
        self.tracks = {
            tid: track
            for tid, track in self.tracks.items()
            if track.status != TrackStatus.DELETED
        }

        return list(self.tracks.values())

    def _associate(
        self, detections: List[Tuple[float, float]]
    ) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Nearest-Neighbor data association with gating.

        Returns:
            - associations: {track_id: detection_index}
            - unassigned_detections: [detection indices]
            - unassigned_tracks: [track ids]
        """
        if not detections or not self.tracks:
            return (
                {},
                list(range(len(detections))),
                list(self.tracks.keys()),
            )

        track_ids = [
            tid for tid, t in self.tracks.items() if t.status != TrackStatus.DELETED
        ]
        cost = np.full((len(track_ids), len(detections)), np.inf)
        for row, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            innovation_covariance = (
                self.kf.H @ track.state.P @ self.kf.H.T + self.kf.R
            )
            for col, detection in enumerate(detections):
                innovation = np.asarray(detection) - np.asarray(track.position)
                euclidean_distance = float(np.linalg.norm(innovation))
                if euclidean_distance > self.gate_distance:
                    continue
                nis = float(
                    innovation
                    @ np.linalg.solve(innovation_covariance, innovation)
                )
                if nis <= self.gate_threshold:
                    cost[row, col] = nis

        associations: Dict[int, int] = {}
        assigned_detections = set()
        assigned_tracks = set()
        if np.isfinite(cost).any():
            assignment_cost = np.where(np.isfinite(cost), cost, 1e12)
            rows, columns = linear_sum_assignment(assignment_cost)
            for row, col in zip(rows, columns):
                if not np.isfinite(cost[row, col]):
                    continue
                track_id = track_ids[row]
                associations[track_id] = int(col)
                assigned_tracks.add(track_id)
                assigned_detections.add(int(col))

        unassigned_detections = [
            i for i in range(len(detections)) if i not in assigned_detections
        ]
        unassigned_tracks = [tid for tid in track_ids if tid not in assigned_tracks]

        return associations, unassigned_detections, unassigned_tracks

    def _create_track(
        self,
        measurement: Tuple[float, float],
        detection_data: Optional[List[Dict]],
        det_idx: int,
    ) -> Track:
        """Create a new track from unassigned detection."""
        state = self.kf.initialize(measurement)

        track = Track(
            id=self._next_id,
            state=state,
            status=TrackStatus.TENTATIVE,
            creation_time=self.current_time,
            last_update=self.current_time,
            current_time=self.current_time,
        )
        track.history.append(measurement)

        # Copy classification if available
        if detection_data and det_idx < len(detection_data):
            data = detection_data[det_idx]
            if "classification" in data:
                track.classification = data["classification"]
            if "confidence" in data:
                track.confidence = data["confidence"]

        self.tracks[self._next_id] = track
        self._next_id += 1

        return track

    def get_confirmed_tracks(self) -> List[Track]:
        """Get established tracks, including tracks currently coasting."""
        established = {TrackStatus.CONFIRMED, TrackStatus.COASTING}
        return [t for t in self.tracks.values() if t.status in established]

    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)

    def clear(self) -> None:
        """Clear all tracks."""
        self.tracks.clear()
        self._next_id = 1
        self.current_time = 0.0


    def set_ekf_mode(self, enabled: bool) -> None:
        """
        Enable/disable Extended Kalman Filter for polar measurements.

        When enabled, the EKF processes [r, θ] measurements directly
        without pre-converting to Cartesian coordinates.

        Args:
            enabled: True to use EKF, False for linear KF

        Reference: Bar-Shalom (2001), Ch. 5.3
        """
        self.use_ekf = enabled and EKF_AVAILABLE
        if self.use_ekf and self._ekf is None:
            self._ekf = ExtendedKalmanFilter(
                process_noise=self.kf.process_noise,
                range_std=self.kf.measurement_noise,
                angle_std=0.02,  # ~1.15°
                snr_adapt=True,
            )

    def update_polar(
        self,
        polar_detections: List[Tuple[float, float]],
        dt: float,
        snr_values: Optional[List[float]] = None,
    ) -> List[Track]:
        """
        Update tracks with polar [r, θ] detections from Pulse-Doppler engine.

        Feeds raw polar measurements directly to EKF without
        Cartesian pre-conversion.

        Args:
            polar_detections: List of (range_m, azimuth_rad)
            dt: Time step [s]
            snr_values: Optional SNR [dB] for each detection

        Returns:
            List of active tracks

        Reference: Richards (2005), Bar-Shalom (2001)
        """
        if not self.use_ekf or self._ekf is None:
            # Fallback: convert to Cartesian and use standard update
            cartesian = [
                (r * np.cos(theta), r * np.sin(theta)) for r, theta in polar_detections
            ]
            return self.update(cartesian, dt)

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.current_time += dt
        current_time = self.current_time

        # 1. Predict all tracks
        for track in self.tracks.values():
            if track.status != TrackStatus.DELETED:
                track.state = self._ekf.predict(track.state, dt)

        # 2. Convert polar to Cartesian for association only
        cartesian_dets = [
            (r * np.cos(theta), r * np.sin(theta)) for r, theta in polar_detections
        ]

        # 3. Data association (in Cartesian space)
        associations, unassigned_dets, unassigned_tracks = self._associate(
            cartesian_dets
        )

        # 4. Update associated tracks with polar measurements
        for track_id, det_idx in associations.items():
            track = self.tracks[track_id]
            z_polar = polar_detections[det_idx]
            snr = (
                snr_values[det_idx]
                if snr_values and det_idx < len(snr_values)
                else 20.0
            )

            track.state = self._ekf.update(track.state, z_polar, snr_db=snr)
            track.last_update = current_time
            track.hits += 1
            track.consecutive_hits += 1
            track.misses = 0
            track.current_time = current_time

            if (
                track.status == TrackStatus.TENTATIVE
                and track.consecutive_hits >= self.confirm_hits
            ):
                track.status = TrackStatus.CONFIRMED
            elif track.status == TrackStatus.COASTING:
                track.status = TrackStatus.CONFIRMED

            track.history.append(track.position)
            if len(track.history) > self.max_history:
                track.history.pop(0)

        # 5. Coast unassigned tracks
        for track_id in unassigned_tracks:
            track = self.tracks[track_id]
            track.misses += 1
            track.consecutive_hits = 0
            track.current_time = current_time
            if track.status == TrackStatus.CONFIRMED:
                track.status = TrackStatus.COASTING
            if track.misses >= self.max_misses:
                track.status = TrackStatus.DELETED
            track.history.append(track.position)
            if len(track.history) > self.max_history:
                track.history.pop(0)

        # 6. Initiate new tracks from unassigned detections
        for det_idx in unassigned_dets:
            r_m, theta_rad = polar_detections[det_idx]
            state = self._ekf.initialize_from_polar(r_m, theta_rad)
            track = Track(
                id=self._next_id,
                state=state,
                status=TrackStatus.TENTATIVE,
                creation_time=self.current_time,
                last_update=self.current_time,
                current_time=self.current_time,
            )
            cart_pos = (r_m * np.cos(theta_rad), r_m * np.sin(theta_rad))
            track.history.append(cart_pos)
            self.tracks[self._next_id] = track
            self._next_id += 1

        # 7. Remove deleted tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items() if t.status != TrackStatus.DELETED
        }

        return list(self.tracks.values())
