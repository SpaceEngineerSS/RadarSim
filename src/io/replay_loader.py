"""
HDF5 Replay Loader

Lazy-loading HDF5 session reader with interpolation for smooth playback.

Matches the HDF5 schema from src/simulation/recorder.py:
    /config - radar configuration attributes
    /targets/{id}/positions - (Nx4: t, x, y, z)
    /targets/{id}/velocities - (Nx4: t, vx, vy, vz)
    /targets/{id} attrs['metadata'] - JSON with type, rcs, jammer info
    /measurements - time, target_id, range_m, azimuth_rad, snr_db, detected
    /timestamps - simulation time array

Features:
    - Lazy load (file stays open, data read on demand)
    - Linear interpolation for smooth timeline scrubbing
    - Thread-safe read operations

Usage:
    loader = ReplayLoader('output/session_20231222_120000.h5')
    state = loader.get_state_at_time(15.5)
    loader.close()
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import h5py
import json


@dataclass
class TargetState:
    """State of a single target at a specific time."""
    target_id: int
    name: str
    target_type: str
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    range_km: float
    azimuth_deg: float
    speed_mps: float
    rcs_m2: float = 1.0
    has_jammer: bool = False
    snr_db: float = 0.0
    is_detected: bool = False


@dataclass
class SimulationState:
    """
    Complete simulation state at a specific time.
    
    Universal data structure for both live and replay modes.
    All UI components consume this format.
    """
    time: float
    targets: List[TargetState] = field(default_factory=list)
    detection_count: int = 0
    total_targets: int = 0
    beam_azimuth_rad: float = 0.0
    
    # Radar config (from file)
    frequency_hz: float = 10e9
    power_watts: float = 100e3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by PPI/A-scope."""
        return {
            'time': self.time,
            'targets': [
                {
                    'id': t.target_id,
                    'type': t.target_type,
                    'range_km': t.range_km,
                    'azimuth_rad': np.radians(t.azimuth_deg),
                    'azimuth_deg': t.azimuth_deg,
                    'snr_db': t.snr_db,
                    'is_detected': t.is_detected,
                    'velocity_mps': t.speed_mps,
                    'rcs_m2': t.rcs_m2,
                    'has_jammer': t.has_jammer,
                    'position': t.position.tolist(),
                    'velocity': t.velocity.tolist(),
                }
                for t in self.targets
            ],
            'detection_count': self.detection_count,
            'total_targets': self.total_targets,
            'radar': {
                'antenna_azimuth_rad': self.beam_azimuth_rad,
            }
        }


@dataclass
class ReplayMetadata:
    """Metadata from HDF5 recording file."""
    duration: float
    frequency_hz: float
    power_watts: float
    max_range_km: float
    target_ids: List[int]
    num_measurements: int
    version: str = "1.0"
    created: str = ""


class ReplayLoader:
    """
    Lazy-loading HDF5 session replay.
    
    Reads HDF5 files created by FlightRecorder and provides
    interpolated state access for smooth timeline scrubbing.
    
    HDF5 Schema (from recorder.py):
        /config
            attrs: radar_frequency_hz, radar_power_watts, max_range_km
        /targets/{id}
            positions: (Nx4 array: t, x, y, z)
            velocities: (Nx4 array: t, vx, vy, vz)
            attrs['metadata']: JSON with type, rcs_m2, has_jammer
        /measurements
            time, target_id, range_m, azimuth_rad, snr_db, detected
        /timestamps
            array of simulation times
    """
    
    def __init__(self, filepath: str):
        """
        Open HDF5 file for replay.
        
        Args:
            filepath: Path to .h5 session file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        self.filepath = filepath
        self._file: Optional[h5py.File] = None
        self._timestamps: Optional[np.ndarray] = None
        self._target_data: Dict[int, Dict] = {}  # id -> {positions, velocities, metadata}
        self._measurements: Optional[Dict[str, np.ndarray]] = None
        self.metadata: Optional[ReplayMetadata] = None
        
        self._open_file()
    
    def _open_file(self) -> None:
        """Open and validate HDF5 file."""
        try:
            self._file = h5py.File(self.filepath, 'r')
        except OSError as e:
            raise FileNotFoundError(f"Cannot open file: {self.filepath}") from e
        
        # Validate structure
        if 'timestamps' not in self._file:
            raise ValueError(f"Invalid file format: missing 'timestamps' dataset")
        
        # Load timestamps (small, keep in memory)
        self._timestamps = np.array(self._file['timestamps'])
        
        # Load config
        config = self._file.get('config', None)
        frequency_hz = 10e9
        power_watts = 100e3
        max_range_km = 150.0
        
        if config is not None:
            frequency_hz = config.attrs.get('radar_frequency_hz', 
                           config.attrs.get('frequency_hz', 10e9))
            power_watts = config.attrs.get('radar_power_watts',
                          config.attrs.get('power_watts', 100e3))
            max_range_km = config.attrs.get('max_range_km', 150.0)
        
        # Load target data
        targets_group = self._file.get('targets', None)
        if targets_group:
            for target_id_str in targets_group.keys():
                target_id = int(target_id_str)
                target_group = targets_group[target_id_str]
                
                # Load positions (Nx4: t, x, y, z)
                positions = None
                if 'positions' in target_group:
                    positions = np.array(target_group['positions'])
                
                # Load velocities (Nx4: t, vx, vy, vz)
                velocities = None
                if 'velocities' in target_group:
                    velocities = np.array(target_group['velocities'])
                
                # Load metadata
                metadata = {'type': 'unknown', 'rcs_m2': 1.0, 'has_jammer': False}
                if 'metadata' in target_group.attrs:
                    try:
                        metadata = json.loads(target_group.attrs['metadata'])
                    except:
                        pass
                
                self._target_data[target_id] = {
                    'positions': positions,
                    'velocities': velocities,
                    'metadata': metadata
                }
        
        # Load measurements
        meas_group = self._file.get('measurements', None)
        if meas_group:
            self._measurements = {
                'time': np.array(meas_group['time']) if 'time' in meas_group else np.array([]),
                'target_id': np.array(meas_group['target_id']) if 'target_id' in meas_group else np.array([]),
                'range_m': np.array(meas_group['range_m']) if 'range_m' in meas_group else np.array([]),
                'azimuth_rad': np.array(meas_group['azimuth_rad']) if 'azimuth_rad' in meas_group else np.array([]),
                'snr_db': np.array(meas_group['snr_db']) if 'snr_db' in meas_group else np.array([]),
                'detected': np.array(meas_group['detected']) if 'detected' in meas_group else np.array([]),
            }
        
        # Create metadata
        self.metadata = ReplayMetadata(
            duration=float(self._timestamps[-1]) if len(self._timestamps) > 0 else 0.0,
            frequency_hz=float(frequency_hz),
            power_watts=float(power_watts),
            max_range_km=float(max_range_km),
            target_ids=list(self._target_data.keys()),
            num_measurements=len(self._measurements['time']) if self._measurements else 0,
            version=self._file.attrs.get('version', '1.0') if self._file else '1.0',
            created=self._file.attrs.get('created', '') if self._file else ''
        )
    
    @property
    def duration(self) -> float:
        """Total recording duration in seconds."""
        return self.metadata.duration if self.metadata else 0.0
    
    @property
    def is_open(self) -> bool:
        """Check if file is still open."""
        return self._file is not None
    
    def close(self) -> None:
        """Close HDF5 file."""
        if self._file:
            self._file.close()
            self._file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def _find_time_indices(self, t: float) -> Tuple[int, int, float]:
        """
        Find indices surrounding time t for interpolation.
        
        Returns:
            (i0, i1, alpha) where state = lerp(state[i0], state[i1], alpha)
        """
        if self._timestamps is None or len(self._timestamps) == 0:
            return 0, 0, 0.0
        
        # Clamp to valid range
        t = max(0.0, min(t, self.duration))
        
        # Binary search for efficiency
        idx = np.searchsorted(self._timestamps, t)
        
        if idx == 0:
            return 0, 0, 0.0
        if idx >= len(self._timestamps):
            idx = len(self._timestamps) - 1
            return idx, idx, 0.0
        
        i0 = idx - 1
        i1 = idx
        
        # Calculate interpolation factor
        t0 = self._timestamps[i0]
        t1 = self._timestamps[i1]
        
        if t1 - t0 > 0:
            alpha = (t - t0) / (t1 - t0)
        else:
            alpha = 0.0
        
        return i0, i1, alpha
    
    def get_state_at_time(self, t: float) -> SimulationState:
        """
        Get interpolated simulation state at time t.
        
        Uses linear interpolation between recorded frames
        for smooth timeline scrubbing at >30 FPS.
        
        Args:
            t: Time in seconds from start
            
        Returns:
            SimulationState with all targets at time t
        """
        i0, i1, alpha = self._find_time_indices(t)
        
        # Find measurements near this time
        measurement_lookup = {}
        if self._measurements and len(self._measurements['time']) > 0:
            time_arr = self._measurements['time']
            mask = np.abs(time_arr - t) < 0.5  # Within 0.5s
            
            for idx in np.where(mask)[0]:
                tid = int(self._measurements['target_id'][idx])
                measurement_lookup[tid] = {
                    'snr_db': float(self._measurements['snr_db'][idx]),
                    'detected': bool(self._measurements['detected'][idx]),
                    'range_m': float(self._measurements['range_m'][idx]),
                    'azimuth_rad': float(self._measurements['azimuth_rad'][idx]),
                }
        
        # Build target states with interpolation
        targets = []
        detection_count = 0
        
        for target_id, data in self._target_data.items():
            positions = data['positions']
            velocities = data['velocities']
            metadata = data['metadata']
            
            if positions is None or len(positions) == 0:
                continue
            
            # Find indices for this target's data
            n_pos = len(positions)
            i0_clamped = min(i0, n_pos - 1)
            i1_clamped = min(i1, n_pos - 1)
            
            # Interpolate position (columns: t, x, y, z)
            pos0 = positions[i0_clamped, 1:4]
            pos1 = positions[i1_clamped, 1:4]
            position = pos0 + alpha * (pos1 - pos0)
            
            # Get velocity
            if velocities is not None and len(velocities) > 0:
                vel0 = velocities[min(i0_clamped, len(velocities)-1), 1:4]
                vel1 = velocities[min(i1_clamped, len(velocities)-1), 1:4]
                velocity = vel0 + alpha * (vel1 - vel0)
            else:
                velocity = np.zeros(3)
            
            # Calculate range and azimuth
            range_m = np.linalg.norm(position)
            azimuth_rad = np.arctan2(position[1], position[0])  # East from North
            azimuth_deg = np.degrees(azimuth_rad)
            if azimuth_deg < 0:
                azimuth_deg += 360
            
            # Get measurement data if available
            meas = measurement_lookup.get(target_id, {})
            snr_db = meas.get('snr_db', 0.0)
            is_detected = meas.get('detected', False)
            
            if is_detected:
                detection_count += 1
            
            targets.append(TargetState(
                target_id=target_id,
                name=f"Target {target_id}",
                target_type=metadata.get('type', 'unknown'),
                position=position,
                velocity=velocity,
                range_km=range_m / 1000,
                azimuth_deg=azimuth_deg,
                speed_mps=np.linalg.norm(velocity),
                rcs_m2=metadata.get('rcs_m2', 1.0),
                has_jammer=metadata.get('has_jammer', False),
                snr_db=snr_db,
                is_detected=is_detected
            ))
        
        # Calculate beam azimuth from time (assuming 6 RPM = 36 deg/s)
        beam_scan_rate = 36.0  # deg/s
        beam_azimuth_rad = np.radians((t * beam_scan_rate) % 360)
        
        return SimulationState(
            time=t,
            targets=targets,
            detection_count=detection_count,
            total_targets=len(targets),
            beam_azimuth_rad=beam_azimuth_rad,
            frequency_hz=self.metadata.frequency_hz if self.metadata else 10e9,
            power_watts=self.metadata.power_watts if self.metadata else 100e3
        )
    
    def get_snr_history(self, target_id: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get SNR values over time.
        
        Args:
            target_id: Optional target ID filter
            
        Returns:
            (times, snr_values) arrays
        """
        if self._measurements is None or len(self._measurements['time']) == 0:
            return np.array([]), np.array([])
        
        times = self._measurements['time']
        snr_values = self._measurements['snr_db']
        
        if target_id is not None:
            mask = self._measurements['target_id'] == target_id
            times = times[mask]
            snr_values = snr_values[mask]
        
        return times, snr_values
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get detection statistics for analysis.
        
        Returns:
            Dictionary with detection metrics
        """
        if self._measurements is None or len(self._measurements['time']) == 0:
            return {
                'total_measurements': 0,
                'total_detections': 0,
                'detection_rate': 0.0,
                'mean_snr': 0.0,
                'max_snr': 0.0,
                'min_snr': 0.0,
                'duration': 0.0,
            }
        
        detected = self._measurements['detected']
        snr_values = self._measurements['snr_db']
        
        total = len(detected)
        n_detected = np.sum(detected)
        
        return {
            'total_measurements': total,
            'total_detections': int(n_detected),
            'detection_rate': float(n_detected / total) if total > 0 else 0.0,
            'mean_snr': float(np.mean(snr_values)),
            'max_snr': float(np.max(snr_values)),
            'min_snr': float(np.min(snr_values)),
            'duration': self.duration,
        }
