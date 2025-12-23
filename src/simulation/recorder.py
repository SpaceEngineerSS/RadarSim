"""
HDF5 Flight Recorder

Saves simulation session data to HDF5 files for post-analysis.

Features:
    - Radar configuration storage
    - Target trajectories with timestamps
    - Detection measurements
    - Automatic filename with timestamp

Reference: HDF5 Best Practices for Scientific Data

Author: RadarSim Scientific Team
"""

import h5py
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import threading
import json


@dataclass
class RecordingSession:
    """
    Session data accumulator for HDF5 recording.
    
    Collects simulation data during runtime and writes to HDF5 on save.
    
    Attributes:
        config: Radar configuration dictionary
        target_tracks: Dict of target_id -> list of (time, x, y, z)
        measurements: List of detection measurements
        timestamps: List of simulation timestamps
    """
    config: Dict[str, Any] = field(default_factory=dict)
    target_tracks: Dict[int, List] = field(default_factory=dict)
    measurements: List[Dict] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    
    def clear(self):
        """Clear all recorded data."""
        self.config = {}
        self.target_tracks = {}
        self.measurements = []
        self.timestamps = []


class FlightRecorder:
    """
    HDF5 Flight Data Recorder.
    
    Records simulation data in HDF5 format:
    
    File Structure:
        /config
            - radar_frequency_hz
            - radar_power_watts
            - radar_gain_db
            - max_range_km
        /targets/{target_id}
            - positions (Nx4 array: time, x, y, z)
            - velocities (Nx4 array: time, vx, vy, vz)
            - metadata (JSON string)
        /measurements
            - time (array)
            - target_id (array)
            - range_m (array)
            - azimuth_rad (array)
            - snr_db (array)
            - detected (array)
    
    Reference: HDF5 for Scientific Data, NSCA
    """
    
    def __init__(self, output_dir: str = "output"):
        """
        Initialize flight recorder.
        
        Args:
            output_dir: Directory for HDF5 output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = RecordingSession()
        self.is_recording = False
        self._lock = threading.Lock()
    
    def start_recording(self, config: Dict[str, Any]):
        """
        Start a new recording session.
        
        Args:
            config: Radar configuration dictionary
        """
        with self._lock:
            self.session.clear()
            self.session.config = config.copy()
            self.is_recording = True
    
    def stop_recording(self) -> Optional[str]:
        """
        Stop recording and save to HDF5.
        
        Returns:
            Path to saved file, or None if no data
        """
        with self._lock:
            self.is_recording = False
            
            if not self.session.timestamps:
                return None
            
            return self._save_to_hdf5()
    
    def record_state(self, state: Dict[str, Any]):
        """
        Record a simulation state snapshot.
        
        Args:
            state: Simulation state dictionary with:
                - time: Current simulation time
                - targets: List of target data dicts
                - radar: Radar state dict
        """
        if not self.is_recording:
            return
        
        with self._lock:
            time = state.get('time', 0.0)
            self.session.timestamps.append(time)
            
            # Record target positions
            for target in state.get('targets', []):
                target_id = target['id']
                
                if target_id not in self.session.target_tracks:
                    self.session.target_tracks[target_id] = {
                        'positions': [],
                        'velocities': [],
                        'metadata': {
                            'type': target.get('type', 'unknown'),
                            'rcs_m2': target.get('rcs_m2', 1.0),
                            'has_jammer': target.get('has_jammer', False)
                        }
                    }
                
                track = self.session.target_tracks[target_id]
                
                # Position: (t, x, y, z) - convert from km to m if needed
                pos = target.get('position', [0, 0, 0])
                track['positions'].append([time, pos[0], pos[1], pos[2]])
                
                # Velocity: (t, vx, vy, vz)
                vel = target.get('velocity', [0, 0, 0])
                track['velocities'].append([time, vel[0], vel[1], vel[2]])
                
                # Record measurement
                self.session.measurements.append({
                    'time': time,
                    'target_id': target_id,
                    'range_m': target.get('range_km', 0) * 1000,
                    'azimuth_rad': target.get('azimuth_rad', 0),
                    'snr_db': target.get('snr_db', 0),
                    'detected': target.get('is_detected', False)
                })
    
    def _save_to_hdf5(self) -> str:
        """
        Save session data to HDF5 file.
        
        Returns:
            Path to saved file
        """
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"session_{timestamp}.h5"
        filepath = self.output_dir / filename
        
        with h5py.File(filepath, 'w') as f:
            # Write config
            config_group = f.create_group('config')
            for key, value in self.session.config.items():
                if isinstance(value, (int, float, str, bool)):
                    config_group.attrs[key] = value
                else:
                    config_group.attrs[key] = json.dumps(value)
            
            # Write target tracks
            targets_group = f.create_group('targets')
            for target_id, track in self.session.target_tracks.items():
                target_group = targets_group.create_group(str(target_id))
                
                # Position array (Nx4)
                if track['positions']:
                    positions = np.array(track['positions'], dtype=np.float64)
                    target_group.create_dataset('positions', data=positions)
                
                # Velocity array (Nx4)
                if track['velocities']:
                    velocities = np.array(track['velocities'], dtype=np.float64)
                    target_group.create_dataset('velocities', data=velocities)
                
                # Metadata as JSON attribute
                target_group.attrs['metadata'] = json.dumps(track['metadata'])
            
            # Write measurements
            if self.session.measurements:
                meas_group = f.create_group('measurements')
                
                times = [m['time'] for m in self.session.measurements]
                target_ids = [m['target_id'] for m in self.session.measurements]
                ranges = [m['range_m'] for m in self.session.measurements]
                azimuths = [m['azimuth_rad'] for m in self.session.measurements]
                snrs = [m['snr_db'] for m in self.session.measurements]
                detected = [m['detected'] for m in self.session.measurements]
                
                meas_group.create_dataset('time', data=np.array(times))
                meas_group.create_dataset('target_id', data=np.array(target_ids))
                meas_group.create_dataset('range_m', data=np.array(ranges))
                meas_group.create_dataset('azimuth_rad', data=np.array(azimuths))
                meas_group.create_dataset('snr_db', data=np.array(snrs))
                meas_group.create_dataset('detected', data=np.array(detected))
            
            # Write simulation timestamps
            if self.session.timestamps:
                f.create_dataset('timestamps', data=np.array(self.session.timestamps))
            
            # Add file attributes
            f.attrs['version'] = '1.0'
            f.attrs['created'] = timestamp
            f.attrs['software'] = 'RadarSim'
        
        return str(filepath)
    
    def get_recording_stats(self) -> Dict[str, Any]:
        """
        Get current recording statistics.
        
        Returns:
            Dict with recording stats
        """
        with self._lock:
            return {
                'is_recording': self.is_recording,
                'duration_s': self.session.timestamps[-1] if self.session.timestamps else 0,
                'num_snapshots': len(self.session.timestamps),
                'num_targets': len(self.session.target_tracks),
                'num_measurements': len(self.session.measurements)
            }


def validate_hdf5_structure(filepath: str) -> Dict[str, Any]:
    """
    Validate HDF5 file structure.
    
    Args:
        filepath: Path to HDF5 file
        
    Returns:
        Validation result dictionary
    """
    result = {
        'valid': True,
        'groups': [],
        'datasets': [],
        'errors': []
    }
    
    try:
        with h5py.File(filepath, 'r') as f:
            # Check required groups
            for group_name in ['config', 'targets', 'measurements']:
                if group_name in f:
                    result['groups'].append(group_name)
                else:
                    result['errors'].append(f"Missing group: {group_name}")
                    result['valid'] = False
            
            # List datasets
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    result['datasets'].append(name)
            
            f.visititems(visitor)
            
    except Exception as e:
        result['valid'] = False
        result['errors'].append(str(e))
    
    return result


# Module test
if __name__ == "__main__":
    # Test recording
    recorder = FlightRecorder(output_dir="output")
    
    # Start recording
    recorder.start_recording({
        'radar_frequency_hz': 10e9,
        'radar_power_watts': 100e3,
        'max_range_km': 150
    })
    
    # Simulate some data
    for i in range(10):
        recorder.record_state({
            'time': i * 0.1,
            'targets': [
                {
                    'id': 1,
                    'type': 'aircraft',
                    'position': [50000 + i*100, 30000, 5000],
                    'velocity': [100, 0, 0],
                    'range_km': 58.3,
                    'azimuth_rad': 0.52,
                    'snr_db': 15.0,
                    'is_detected': True
                }
            ]
        })
    
    # Save
    filepath = recorder.stop_recording()
    print(f"Saved to: {filepath}")
    
    # Validate
    result = validate_hdf5_structure(filepath)
    print(f"Validation: {'PASS' if result['valid'] else 'FAIL'}")
    print(f"Groups: {result['groups']}")
    print(f"Datasets: {result['datasets']}")
