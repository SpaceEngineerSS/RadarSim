"""
Simulation Objects

Core kinematic objects for radar simulation: Target and Radar.

References:
    - Bar-Shalom, Y. (2001). "Estimation with Applications to Tracking and Navigation"
    - Zarchan, P. (2012). "Tactical and Strategic Missile Guidance"

Features:
    - 3D position, velocity, acceleration kinematics
    - Constant Velocity (CV) and Constant Acceleration (CA) motion models
    - RCS model integration with aspect angle and Swerling models
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from enum import Enum
import numba

# Import physics modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.physics.rcs import (
    SwerlingModel, SwerlingRCS, calculate_aspect_angle,
    calculate_aspect_dependent_rcs, get_target_type_rcs
)


class MotionModel(Enum):
    """Target motion models."""
    STATIC = "static"                    # No motion
    CONSTANT_VELOCITY = "cv"             # Constant velocity
    CONSTANT_ACCELERATION = "ca"         # Constant acceleration
    COORDINATED_TURN = "ct"              # Coordinated turn (2D)


@dataclass
class KinematicState:
    """
    3D Kinematic state vector.
    
    State: [x, y, z, vx, vy, vz, ax, ay, az]
    
    Coordinate system: North-East-Down (NED)
    - x: North [m]
    - y: East [m]  
    - z: Down [m] (altitude is negative z)
    
    Attributes:
        position: [x, y, z] position [m]
        velocity: [vx, vy, vz] velocity [m/s]
        acceleration: [ax, ay, az] acceleration [m/s²]
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    def __post_init__(self):
        """Ensure arrays are numpy float64."""
        self.position = np.asarray(self.position, dtype=np.float64)
        self.velocity = np.asarray(self.velocity, dtype=np.float64)
        self.acceleration = np.asarray(self.acceleration, dtype=np.float64)
        
        # Pad to 3D if 2D
        if len(self.position) == 2:
            self.position = np.array([self.position[0], self.position[1], 0.0])
        if len(self.velocity) == 2:
            self.velocity = np.array([self.velocity[0], self.velocity[1], 0.0])
        if len(self.acceleration) == 2:
            self.acceleration = np.array([self.acceleration[0], self.acceleration[1], 0.0])
    
    def to_state_vector(self) -> np.ndarray:
        """Return full 9-element state vector."""
        return np.concatenate([self.position, self.velocity, self.acceleration])
    
    @classmethod
    def from_state_vector(cls, state: np.ndarray) -> 'KinematicState':
        """Create from 9-element state vector."""
        if len(state) == 6:
            return cls(
                position=state[0:3],
                velocity=state[3:6],
                acceleration=np.zeros(3)
            )
        elif len(state) == 9:
            return cls(
                position=state[0:3],
                velocity=state[3:6],
                acceleration=state[6:9]
            )
        else:
            raise ValueError(f"State vector must be 6 or 9 elements, got {len(state)}")
    
    @property
    def speed(self) -> float:
        """Total speed [m/s]."""
        return np.linalg.norm(self.velocity)
    
    @property
    def heading(self) -> float:
        """Heading angle [rad] (0 = North, π/2 = East)."""
        return np.arctan2(self.velocity[1], self.velocity[0])
    
    @property
    def altitude(self) -> float:
        """Altitude [m] (positive up, opposite of z)."""
        return -self.position[2]


@numba.jit(nopython=True, cache=True)
def _update_kinematics_cv(
    pos: np.ndarray,
    vel: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled Constant Velocity (CV) motion model.
    
    x_new = x + v * dt
    v_new = v (unchanged)
    
    Args:
        pos: Current position [x, y, z]
        vel: Current velocity [vx, vy, vz]
        dt: Time step [s]
        
    Returns:
        Tuple of (new_position, new_velocity)
    """
    new_pos = pos + vel * dt
    new_vel = vel.copy()
    return new_pos, new_vel


@numba.jit(nopython=True, cache=True)
def _update_kinematics_ca(
    pos: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    dt: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    JIT-compiled Constant Acceleration (CA) motion model.
    
    x_new = x + v*dt + 0.5*a*dt²
    v_new = v + a*dt
    a_new = a (unchanged)
    
    Reference: Bar-Shalom, Eq. 6.2.1-3
    
    Args:
        pos: Current position [x, y, z]
        vel: Current velocity [vx, vy, vz]
        acc: Current acceleration [ax, ay, az]
        dt: Time step [s]
        
    Returns:
        Tuple of (new_position, new_velocity, new_acceleration)
    """
    dt2 = dt * dt
    new_pos = pos + vel * dt + 0.5 * acc * dt2
    new_vel = vel + acc * dt
    new_acc = acc.copy()
    return new_pos, new_vel, new_acc


class Target:
    """
    Radar target with 3D kinematics and RCS model.
    
    Supports Constant Velocity (CV) and Constant Acceleration (CA) models.
    
    Reference: Bar-Shalom, Y. (2001). "Estimation with Applications to Tracking"
    """
    
    def __init__(
        self,
        target_id: int,
        position: np.ndarray,
        velocity: np.ndarray = None,
        acceleration: np.ndarray = None,
        rcs_m2: float = 1.0,
        target_type: str = "aircraft",
        swerling_model: SwerlingModel = SwerlingModel.SWERLING_1,
        motion_model: MotionModel = MotionModel.CONSTANT_VELOCITY,
        has_jammer: bool = False,
        jammer_power_watts: float = 1000.0,
        jammer_bandwidth_hz: float = 100e6
    ):
        """
        Initialize target.
        
        Args:
            target_id: Unique target identifier
            position: Initial position [x, y, z] [m]
            velocity: Initial velocity [vx, vy, vz] [m/s]
            acceleration: Initial acceleration [ax, ay, az] [m/s²]
            rcs_m2: Mean RCS value [m²]
            target_type: Target classification (aircraft, missile, etc.)
            swerling_model: RCS fluctuation model
            motion_model: Kinematic motion model
            has_jammer: Whether target carries an active jammer
            jammer_power_watts: Jammer ERP if has_jammer is True [W]
            jammer_bandwidth_hz: Jammer bandwidth [Hz]
        """
        self.target_id = target_id
        self.target_type = target_type
        self.rcs_mean = rcs_m2
        self.swerling_model = swerling_model
        self.motion_model = motion_model
        
        # ECM capability
        self.has_jammer = has_jammer
        self.jammer_power_watts = jammer_power_watts
        self.jammer_bandwidth_hz = jammer_bandwidth_hz
        self.jammer_active = has_jammer  # Active by default if equipped
        
        # Initialize kinematic state
        if velocity is None:
            velocity = np.zeros(3)
        if acceleration is None:
            acceleration = np.zeros(3)
            
        self.state = KinematicState(
            position=np.asarray(position, dtype=np.float64),
            velocity=np.asarray(velocity, dtype=np.float64),
            acceleration=np.asarray(acceleration, dtype=np.float64)
        )
        
        # Track state history for debugging
        self._position_history = []
    
    def update(self, dt: float) -> None:
        """
        Update target kinematics by one time step.
        
        Uses equations of motion:
        - CV: x = x + v*dt
        - CA: x = x + v*dt + 0.5*a*dt²
        
        Args:
            dt: Time step [s]
        """
        if self.motion_model == MotionModel.STATIC:
            # No motion
            pass
        
        elif self.motion_model == MotionModel.CONSTANT_VELOCITY:
            new_pos, new_vel = _update_kinematics_cv(
                self.state.position,
                self.state.velocity,
                dt
            )
            self.state.position = new_pos
            self.state.velocity = new_vel
        
        elif self.motion_model == MotionModel.CONSTANT_ACCELERATION:
            new_pos, new_vel, new_acc = _update_kinematics_ca(
                self.state.position,
                self.state.velocity,
                self.state.acceleration,
                dt
            )
            self.state.position = new_pos
            self.state.velocity = new_vel
            self.state.acceleration = new_acc
        
        # Store history (limit to last 1000 points)
        self._position_history.append(self.state.position.copy())
        if len(self._position_history) > 1000:
            self._position_history.pop(0)
    
    def get_rcs(self, radar_position: np.ndarray = None) -> float:
        """
        Get current RCS with Swerling fluctuation.
        
        Args:
            radar_position: Radar position for aspect angle calculation
            
        Returns:
            Fluctuated RCS [m²]
        """
        mean_rcs = self.rcs_mean
        
        # Apply aspect angle factor if radar position provided
        if radar_position is not None:
            aspect_angle = calculate_aspect_angle(
                self.state.position,
                self.state.heading,
                radar_position
            )
            # Simple aspect factor (reduced at nose-on/tail-on)
            aspect_factor = 0.5 + 0.5 * abs(np.sin(aspect_angle))
            mean_rcs *= aspect_factor
        
        # Apply Swerling fluctuation
        return SwerlingRCS.generate_rcs(mean_rcs, self.swerling_model)
    
    @property
    def position(self) -> np.ndarray:
        """Current position [m]."""
        return self.state.position
    
    @property
    def velocity(self) -> np.ndarray:
        """Current velocity [m/s]."""
        return self.state.velocity
    
    @property
    def speed(self) -> float:
        """Current speed [m/s]."""
        return self.state.speed
    
    def range_to(self, other_position: np.ndarray) -> float:
        """
        Calculate range to another position.
        
        Args:
            other_position: Target position [x, y, z]
            
        Returns:
            Slant range [m]
        """
        return np.linalg.norm(self.state.position - other_position)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'target_id': self.target_id,
            'target_type': self.target_type,
            'position': self.state.position.tolist(),
            'velocity': self.state.velocity.tolist(),
            'acceleration': self.state.acceleration.tolist(),
            'rcs_mean': self.rcs_mean,
            'swerling_model': self.swerling_model.name,
            'motion_model': self.motion_model.value,
            'has_jammer': self.has_jammer,
            'jammer_active': self.jammer_active,
            'jammer_power_watts': self.jammer_power_watts,
            'jammer_bandwidth_hz': self.jammer_bandwidth_hz,
        }


class Radar:
    """
    Radar system with position and orientation.
    
    Can be stationary or mobile (e.g., airborne radar).
    """
    
    def __init__(
        self,
        radar_id: str,
        position: np.ndarray,
        velocity: np.ndarray = None,
        frequency_hz: float = 10e9,
        power_watts: float = 100e3,
        antenna_gain_db: float = 30.0,
        beamwidth_deg: float = 2.0,
        scan_rate_rpm: float = 6.0
    ):
        """
        Initialize radar.
        
        Args:
            radar_id: Radar identifier
            position: Radar position [x, y, z] [m]
            velocity: Radar velocity [vx, vy, vz] [m/s] (for mobile radars)
            frequency_hz: Operating frequency [Hz]
            power_watts: Transmit power [W]
            antenna_gain_db: Antenna gain [dB]
            beamwidth_deg: 3dB beamwidth [degrees]
            scan_rate_rpm: Antenna rotation rate [RPM]
        """
        self.radar_id = radar_id
        self.frequency_hz = frequency_hz
        self.power_watts = power_watts
        self.antenna_gain_db = antenna_gain_db
        self.beamwidth_rad = np.radians(beamwidth_deg)
        self.scan_rate_rpm = scan_rate_rpm
        
        # Kinematic state
        if velocity is None:
            velocity = np.zeros(3)
            
        self.state = KinematicState(
            position=np.asarray(position, dtype=np.float64),
            velocity=np.asarray(velocity, dtype=np.float64)
        )
        
        # Antenna pointing
        self.antenna_azimuth = 0.0  # Current azimuth [rad]
        self.antenna_elevation = 0.0  # Current elevation [rad]
    
    def update(self, dt: float) -> None:
        """
        Update radar state (position and antenna rotation).
        
        Args:
            dt: Time step [s]
        """
        # Update position if moving
        if np.any(self.state.velocity != 0):
            new_pos, _ = _update_kinematics_cv(
                self.state.position,
                self.state.velocity,
                dt
            )
            self.state.position = new_pos
        
        # Update antenna azimuth (scanning)
        omega = self.scan_rate_rpm * 2 * np.pi / 60  # rad/s
        self.antenna_azimuth = (self.antenna_azimuth + omega * dt) % (2 * np.pi)
    
    def calculate_target_geometry(
        self,
        target_position: np.ndarray,
        target_velocity: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate relative geometry to a target.
        
        Returns:
            Dict with range, azimuth, elevation, radial_velocity
        """
        # Range vector
        delta = target_position - self.state.position
        range_m = np.linalg.norm(delta)
        
        # Azimuth and elevation
        azimuth = np.arctan2(delta[1], delta[0])  # East from North
        elevation = np.arctan2(-delta[2], np.sqrt(delta[0]**2 + delta[1]**2))
        
        # Radial velocity (if target has velocity)
        radial_velocity = 0.0
        if target_velocity is not None and range_m > 1e-6:
            relative_vel = target_velocity - self.state.velocity
            range_unit = delta / range_m
            radial_velocity = np.dot(relative_vel, range_unit)
        
        return {
            'range_m': range_m,
            'azimuth_rad': azimuth,
            'azimuth_deg': np.degrees(azimuth),
            'elevation_rad': elevation,
            'elevation_deg': np.degrees(elevation),
            'radial_velocity_mps': radial_velocity,
        }
    
    def is_target_in_beam(
        self,
        target_position: np.ndarray
    ) -> bool:
        """
        Check if target is within antenna beam.
        
        Args:
            target_position: Target position [x, y, z]
            
        Returns:
            True if target is in beam
        """
        geom = self.calculate_target_geometry(target_position)
        
        # Angular difference from current beam pointing
        az_diff = abs(geom['azimuth_rad'] - self.antenna_azimuth)
        if az_diff > np.pi:
            az_diff = 2 * np.pi - az_diff
        
        el_diff = abs(geom['elevation_rad'] - self.antenna_elevation)
        
        # Check if within beamwidth
        return (az_diff <= self.beamwidth_rad / 2 and 
                el_diff <= self.beamwidth_rad / 2)
    
    @property
    def position(self) -> np.ndarray:
        """Current radar position [m]."""
        return self.state.position
    
    @property
    def wavelength(self) -> float:
        """Wavelength [m]."""
        c = 299792458.0
        return c / self.frequency_hz
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'radar_id': self.radar_id,
            'position': self.state.position.tolist(),
            'velocity': self.state.velocity.tolist(),
            'frequency_hz': self.frequency_hz,
            'power_watts': self.power_watts,
            'antenna_gain_db': self.antenna_gain_db,
            'beamwidth_deg': np.degrees(self.beamwidth_rad),
            'scan_rate_rpm': self.scan_rate_rpm,
        }


@dataclass
class SimulationState:
    """
    Complete simulation state at a point in time.
    
    Contains radar, all targets, and simulation metadata.
    """
    time: float = 0.0
    radar: Optional[Radar] = None
    targets: Dict[int, Target] = field(default_factory=dict)
    detections: Dict[int, bool] = field(default_factory=dict)
    snr_values: Dict[int, float] = field(default_factory=dict)
    
    def add_target(self, target: Target) -> None:
        """Add a target to the simulation."""
        self.targets[target.target_id] = target
    
    def remove_target(self, target_id: int) -> None:
        """Remove a target from the simulation."""
        if target_id in self.targets:
            del self.targets[target_id]
    
    def update_all(self, dt: float) -> None:
        """Update all objects by one time step."""
        self.time += dt
        
        if self.radar:
            self.radar.update(dt)
        
        for target in self.targets.values():
            target.update(dt)
