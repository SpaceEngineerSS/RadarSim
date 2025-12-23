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

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import time

from .kalman import LinearKalmanFilter, KalmanState


class TrackStatus(Enum):
    """Track lifecycle states."""
    TENTATIVE = "tentative"    # New track, needs confirmation
    CONFIRMED = "confirmed"    # Established track
    COASTING = "coasting"      # No measurements, predicting only
    DELETED = "deleted"        # Marked for removal


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
    misses: int = 0
    creation_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    history: List[Tuple[float, float]] = field(default_factory=list)
    
    # Track classification (from ML or manual)
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
        return np.sqrt(self.state.x[2]**2 + self.state.x[3]**2)
    
    @property
    def heading_rad(self) -> float:
        """Get heading in radians (0 = North, CW positive)."""
        return np.arctan2(self.state.x[2], self.state.x[3])
    
    @property
    def age_seconds(self) -> float:
        """Get track age in seconds."""
        return time.time() - self.creation_time


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
        measurement_noise: float = 50.0
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
        self.gate_distance = gate_distance
        self.max_misses = max_misses
        self.confirm_hits = confirm_hits
        self.max_history = max_history
        
        # Kalman filter for all tracks
        self.kf = LinearKalmanFilter(
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )
        
        # Track storage
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1
    
    def update(
        self,
        detections: List[Tuple[float, float]],
        dt: float,
        detection_data: Optional[List[Dict]] = None
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
        current_time = time.time()
        
        # 1. Predict all tracks
        for track in self.tracks.values():
            if track.status != TrackStatus.DELETED:
                track.state = self.kf.predict(track.state, dt)
        
        # 2. Data association (Nearest-Neighbor with gating)
        associations, unassigned_detections, unassigned_tracks = \
            self._associate(detections)
        
        # 3. Update associated tracks
        for track_id, det_idx in associations.items():
            track = self.tracks[track_id]
            measurement = detections[det_idx]
            
            # Kalman update
            track.state = self.kf.update(track.state, measurement)
            track.last_update = current_time
            track.hits += 1
            track.misses = 0
            
            # Promote tentative -> confirmed
            if track.status == TrackStatus.TENTATIVE and track.hits >= self.confirm_hits:
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
                if 'classification' in data:
                    track.classification = data['classification']
                if 'confidence' in data:
                    track.confidence = data['confidence']
        
        # 4. Coast unassigned tracks
        for track_id in unassigned_tracks:
            track = self.tracks[track_id]
            track.misses += 1
            
            if track.status == TrackStatus.CONFIRMED:
                track.status = TrackStatus.COASTING
            
            # Delete if too many misses
            if track.misses > self.max_misses:
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
            tid: track for tid, track in self.tracks.items()
            if track.status != TrackStatus.DELETED
        }
        
        return list(self.tracks.values())
    
    def _associate(
        self,
        detections: List[Tuple[float, float]]
    ) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Nearest-Neighbor data association with gating.
        
        Returns:
            - associations: {track_id: detection_index}
            - unassigned_detections: [detection indices]
            - unassigned_tracks: [track ids]
        """
        associations = {}
        assigned_detections = set()
        assigned_tracks = set()
        
        if not detections or not self.tracks:
            return (
                associations,
                list(range(len(detections))),
                list(self.tracks.keys())
            )
        
        # Build distance matrix
        track_ids = [
            tid for tid, t in self.tracks.items()
            if t.status != TrackStatus.DELETED
        ]
        
        for track_id in track_ids:
            track = self.tracks[track_id]
            track_pos = track.position
            
            min_dist = float('inf')
            best_det = -1
            
            for det_idx, det in enumerate(detections):
                if det_idx in assigned_detections:
                    continue
                
                dist = np.sqrt(
                    (det[0] - track_pos[0])**2 +
                    (det[1] - track_pos[1])**2
                )
                
                # Gating
                if dist < self.gate_distance and dist < min_dist:
                    min_dist = dist
                    best_det = det_idx
            
            if best_det >= 0:
                associations[track_id] = best_det
                assigned_detections.add(best_det)
                assigned_tracks.add(track_id)
        
        unassigned_detections = [
            i for i in range(len(detections))
            if i not in assigned_detections
        ]
        unassigned_tracks = [
            tid for tid in track_ids
            if tid not in assigned_tracks
        ]
        
        return associations, unassigned_detections, unassigned_tracks
    
    def _create_track(
        self,
        measurement: Tuple[float, float],
        detection_data: Optional[List[Dict]],
        det_idx: int
    ) -> Track:
        """Create a new track from unassigned detection."""
        state = self.kf.initialize(measurement)
        
        track = Track(
            id=self._next_id,
            state=state,
            status=TrackStatus.TENTATIVE
        )
        track.history.append(measurement)
        
        # Copy classification if available
        if detection_data and det_idx < len(detection_data):
            data = detection_data[det_idx]
            if 'classification' in data:
                track.classification = data['classification']
            if 'confidence' in data:
                track.confidence = data['confidence']
        
        self.tracks[self._next_id] = track
        self._next_id += 1
        
        return track
    
    def get_confirmed_tracks(self) -> List[Track]:
        """Get only confirmed tracks."""
        return [
            t for t in self.tracks.values()
            if t.status == TrackStatus.CONFIRMED
        ]
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)
    
    def clear(self) -> None:
        """Clear all tracks."""
        self.tracks.clear()
        self._next_id = 1
