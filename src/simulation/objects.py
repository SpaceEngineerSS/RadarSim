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

import os

# Import physics modules
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numba
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.physics.geometry import (
    heading_from_velocity,
    polar_from_enu,
    wrap_to_pi,
)
from src.physics.rcs import (
    SwerlingModel,
    SwerlingRCS,
    calculate_aspect_angle,
)


AFFILIATIONS = ("friend", "hostile", "neutral", "unknown")


class ScanType(Enum):
    """Antenna scan patterns."""

    CIRCULAR = "circular"  # Continuous 360 deg mechanical rotation
    SECTOR = "sector"  # Back-and-forth raster between two azimuth limits
    STARE = "stare"  # Fixed pointing


class MotionModel(Enum):
    """Target motion models."""

    STATIC = "static"  # No motion
    CONSTANT_VELOCITY = "cv"  # Constant velocity
    CONSTANT_ACCELERATION = "ca"  # Constant acceleration
    COORDINATED_TURN = "ct"  # Constant-rate horizontal turn


@dataclass
class KinematicState:
    """
    3D Kinematic state vector.

    State: [x, y, z, vx, vy, vz, ax, ay, az]

    Coordinate system: East-North-Up (ENU), see ``src.physics.geometry``
    - x: East [m]
    - y: North [m]
    - z: Up [m] (altitude above the local ground plane)

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
            self.acceleration = np.array(
                [self.acceleration[0], self.acceleration[1], 0.0]
            )

    def to_state_vector(self) -> np.ndarray:
        """Return full 9-element state vector."""
        return np.concatenate([self.position, self.velocity, self.acceleration])

    @classmethod
    def from_state_vector(cls, state: np.ndarray) -> "KinematicState":
        """Create from 9-element state vector."""
        if len(state) == 6:
            return cls(
                position=state[0:3], velocity=state[3:6], acceleration=np.zeros(3)
            )
        elif len(state) == 9:
            return cls(
                position=state[0:3], velocity=state[3:6], acceleration=state[6:9]
            )
        else:
            raise ValueError(f"State vector must be 6 or 9 elements, got {len(state)}")

    @property
    def speed(self) -> float:
        """Total speed [m/s]."""
        return np.linalg.norm(self.velocity)

    @property
    def heading(self) -> float:
        """Heading angle [rad], clockwise from North (0 = North, π/2 = East)."""
        return heading_from_velocity(self.velocity)

    @property
    def altitude(self) -> float:
        """Altitude above the local ground plane [m]."""
        return float(self.position[2])


@numba.jit(nopython=True, cache=True)
def _update_kinematics_cv(
    pos: np.ndarray, vel: np.ndarray, dt: float
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
    pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, dt: float
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


def _update_kinematics_ct(
    pos: np.ndarray, vel: np.ndarray, turn_rate_rad_s: float, dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact constant-rate coordinated-turn propagation in the horizontal plane.

    ``turn_rate_rad_s`` is the heading rate, positive clockwise (right turn), matching
    the azimuth convention. Vertical motion is constant velocity.

    Reference: Bar-Shalom, Li & Kirubarajan (2001), Sec. 11.7.1, Eq. 11.7.1-4
    """
    # Bar-Shalom's Omega is counter-clockwise in the x-y plane.
    omega = -turn_rate_rad_s
    if abs(omega) < 1e-12:
        return _update_kinematics_cv(pos, vel, dt)
    s, c = np.sin(omega * dt), np.cos(omega * dt)
    vx, vy = vel[0], vel[1]
    new_pos = pos.copy()
    new_pos[0] += (s * vx - (1.0 - c) * vy) / omega
    new_pos[1] += ((1.0 - c) * vx + s * vy) / omega
    new_pos[2] += vel[2] * dt
    new_vel = vel.copy()
    new_vel[0] = c * vx - s * vy
    new_vel[1] = s * vx + c * vy
    return new_pos, new_vel


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
        jammer_bandwidth_hz: float = 100e6,
        ecm_type: str = "noise_barrage",
        drfm_gain_over_skin_db: float = 10.0,
        drfm_capture_dwell_s: float = 2.0,
        drfm_pull_rate_mps: float = 50.0,
        drfm_max_pull_m: float = 2000.0,
        drfm_mode: str = "rgpo",
        drfm_vgpo_rate_hz_per_s: float = 50.0,
        drfm_max_doppler_pull_hz: float = 500.0,
        drfm_inherent_delay_s: float = 0.0,
        turn_rate_rad_s: float = 0.0,
        name: Optional[str] = None,
        active_from_s: float = 0.0,
        rcs_bloom_time_s: float = 0.0,
        rcs_decay_time_s: float = 0.0,
        affiliation: str = "unknown",
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
            ecm_type: Jammer or deception technique identifier
            turn_rate_rad_s: Heading rate for COORDINATED_TURN, positive clockwise
            name: Display name (defaults to "<type> <id>")
            active_from_s: Simulation time the object appears (deployment) [s]
            rcs_bloom_time_s: RCS rise time constant after deployment (chaff) [s]
            rcs_decay_time_s: RCS decay time constant after deployment (chaff) [s]
            affiliation: "friend", "hostile", "neutral" or "unknown" (display only)
        """
        self.target_id = target_id
        self.target_type = target_type
        self.name = name if name else f"{target_type} {target_id}"
        self.turn_rate_rad_s = float(turn_rate_rad_s)
        if active_from_s < 0.0 or rcs_bloom_time_s < 0.0 or rcs_decay_time_s < 0.0:
            raise ValueError("deployment and RCS time constants cannot be negative")
        self.active_from_s = float(active_from_s)
        self.rcs_bloom_time_s = float(rcs_bloom_time_s)
        self.rcs_decay_time_s = float(rcs_decay_time_s)
        self.age_s = 0.0  # Time since deployment
        if affiliation not in AFFILIATIONS:
            raise ValueError(f"affiliation must be one of {AFFILIATIONS}")
        self.affiliation = affiliation
        self.rcs_mean = rcs_m2
        self.swerling_model = swerling_model
        self.motion_model = motion_model

        # ECM capability
        self.has_jammer = has_jammer
        if jammer_power_watts < 0.0 or jammer_bandwidth_hz <= 0.0:
            raise ValueError(
                "jammer power cannot be negative and bandwidth must be positive"
            )
        self.jammer_power_watts = jammer_power_watts
        self.jammer_bandwidth_hz = jammer_bandwidth_hz
        self.ecm_type = ecm_type
        self.jammer_active = has_jammer  # Active by default if equipped
        drfm_mode = drfm_mode.lower()
        if drfm_mode not in {"rgpo", "vgpo"}:
            raise ValueError("drfm_mode must be 'rgpo' or 'vgpo'")
        if drfm_capture_dwell_s < 0.0 or drfm_inherent_delay_s < 0.0:
            raise ValueError("DRFM timing parameters cannot be negative")
        if drfm_pull_rate_mps < 0.0 or drfm_vgpo_rate_hz_per_s < 0.0:
            raise ValueError("DRFM pull rates cannot be negative")
        if drfm_max_pull_m <= 0.0 or drfm_max_doppler_pull_hz <= 0.0:
            raise ValueError("DRFM maximum offsets must be positive")
        self.drfm_gain_over_skin_db = drfm_gain_over_skin_db
        self.drfm_capture_dwell_s = drfm_capture_dwell_s
        self.drfm_pull_rate_mps = drfm_pull_rate_mps
        self.drfm_max_pull_m = drfm_max_pull_m
        self.drfm_mode = drfm_mode
        self.drfm_vgpo_rate_hz_per_s = drfm_vgpo_rate_hz_per_s
        self.drfm_max_doppler_pull_hz = drfm_max_doppler_pull_hz
        self.drfm_inherent_delay_s = drfm_inherent_delay_s

        # Initialize kinematic state
        if velocity is None:
            velocity = np.zeros(3)
        if acceleration is None:
            acceleration = np.zeros(3)

        self.state = KinematicState(
            position=np.asarray(position, dtype=np.float64),
            velocity=np.asarray(velocity, dtype=np.float64),
            acceleration=np.asarray(acceleration, dtype=np.float64),
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
        self.age_s += dt
        if self.motion_model == MotionModel.STATIC:
            # No motion
            pass

        elif self.motion_model == MotionModel.CONSTANT_VELOCITY:
            new_pos, new_vel = _update_kinematics_cv(
                self.state.position, self.state.velocity, dt
            )
            self.state.position = new_pos
            self.state.velocity = new_vel

        elif self.motion_model == MotionModel.CONSTANT_ACCELERATION:
            new_pos, new_vel, new_acc = _update_kinematics_ca(
                self.state.position, self.state.velocity, self.state.acceleration, dt
            )
            self.state.position = new_pos
            self.state.velocity = new_vel
            self.state.acceleration = new_acc

        elif self.motion_model == MotionModel.COORDINATED_TURN:
            new_pos, new_vel = _update_kinematics_ct(
                self.state.position, self.state.velocity, self.turn_rate_rad_s, dt
            )
            self.state.position = new_pos
            self.state.velocity = new_vel

        # Store history (limit to last 1000 points)
        self._position_history.append(self.state.position.copy())
        if len(self._position_history) > 1000:
            self._position_history.pop(0)

    def is_active(self, time_s: float) -> bool:
        """Whether the object has been deployed at ``time_s``."""
        return time_s >= self.active_from_s

    def rcs_time_factor(self) -> float:
        """
        Temporal RCS profile after deployment (1 when no profile is set).

        Chaff: (1 - exp(-t/tau_bloom)) * exp(-t/tau_decay), the cloud blooming
        as dipoles separate and decaying as it disperses and falls out.
        Reference: Schleher (1999), Sec. 6.4.
        """
        factor = 1.0
        if self.rcs_bloom_time_s > 0.0:
            factor *= 1.0 - np.exp(-self.age_s / self.rcs_bloom_time_s)
        if self.rcs_decay_time_s > 0.0:
            factor *= np.exp(-self.age_s / self.rcs_decay_time_s)
        return float(factor)

    def mean_rcs(self, radar_position: np.ndarray = None) -> float:
        """
        Aspect-dependent mean RCS seen from ``radar_position`` [m²].

        The Swerling fluctuation is *not* applied here: the detection model integrates
        over the fluctuation statistics, so sampling it as well would count it twice.
        """
        mean_rcs = self.rcs_mean * self.rcs_time_factor()
        if radar_position is not None and self.speed > 1e-3:
            aspect_angle = calculate_aspect_angle(
                self.state.position, self.state.heading, radar_position
            )
            # Broadside-dominant aspect weighting (nose/tail at half the beam RCS).
            mean_rcs *= 0.5 + 0.5 * abs(np.sin(aspect_angle))
        return mean_rcs

    def get_rcs(
        self,
        radar_position: np.ndarray = None,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        """One Swerling-fluctuated RCS realization [m²] (for signal-level simulation)."""
        return SwerlingRCS.generate_rcs(
            self.mean_rcs(radar_position), self.swerling_model, rng=rng
        )

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
            "target_id": self.target_id,
            "name": self.name,
            "affiliation": self.affiliation,
            "target_type": self.target_type,
            "position": self.state.position.tolist(),
            "velocity": self.state.velocity.tolist(),
            "acceleration": self.state.acceleration.tolist(),
            "rcs_mean": self.rcs_mean,
            "swerling_model": self.swerling_model.name,
            "motion_model": self.motion_model.value,
            "turn_rate_rad_s": self.turn_rate_rad_s,
            "active_from_s": self.active_from_s,
            "rcs_bloom_time_s": self.rcs_bloom_time_s,
            "rcs_decay_time_s": self.rcs_decay_time_s,
            "has_jammer": self.has_jammer,
            "jammer_active": self.jammer_active,
            "jammer_power_watts": self.jammer_power_watts,
            "jammer_bandwidth_hz": self.jammer_bandwidth_hz,
            "ecm_type": self.ecm_type,
            "drfm_gain_over_skin_db": self.drfm_gain_over_skin_db,
            "drfm_capture_dwell_s": self.drfm_capture_dwell_s,
            "drfm_pull_rate_mps": self.drfm_pull_rate_mps,
            "drfm_max_pull_m": self.drfm_max_pull_m,
            "drfm_mode": self.drfm_mode,
            "drfm_vgpo_rate_hz_per_s": self.drfm_vgpo_rate_hz_per_s,
            "drfm_max_doppler_pull_hz": self.drfm_max_doppler_pull_hz,
            "drfm_inherent_delay_s": self.drfm_inherent_delay_s,
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
        scan_rate_rpm: float = 6.0,
        beamwidth_el_deg: float = None,
        prf_hz: float = 1000.0,
        pulse_width_s: float = 1e-6,
        receiver_bandwidth_hz: float = None,
        noise_figure_db: float = 4.0,
        system_temperature_k: float = 290.0,
        system_losses_db: float = 4.0,
        polarization_tilt_deg: float = 0.0,
        sidelobe_level_db: float = -30.0,
        scan_type: ScanType = ScanType.CIRCULAR,
        sector_az_min_deg: float = -60.0,
        sector_az_max_deg: float = 60.0,
        scan_speed_deg_s: Optional[float] = None,
        stare_azimuth_deg: float = 0.0,
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
            beamwidth_el_deg: Elevation 3dB beamwidth [degrees]
            prf_hz: Pulse repetition frequency [Hz]
            pulse_width_s: Transmitted pulse width [s]
            receiver_bandwidth_hz: Equivalent receiver noise bandwidth [Hz]
            noise_figure_db: Receiver noise figure [dB]
            system_temperature_k: Receiver system temperature [K]
            system_losses_db: Combined transmit and receive losses [dB]
            polarization_tilt_deg: Polarization tilt relative to horizontal [degrees]
            sidelobe_level_db: One-way peak sidelobe level relative to the main beam [dB]
            scan_type: Circular rotation, sector raster or stare
            sector_az_min_deg, sector_az_max_deg: Sector limits, clockwise from North
            scan_speed_deg_s: Sector scan speed (default: scan_rate_rpm * 6)
            stare_azimuth_deg: Pointing direction in stare mode
        """
        positive_parameters = {
            "frequency_hz": frequency_hz,
            "power_watts": power_watts,
            "beamwidth_deg": beamwidth_deg,
            "prf_hz": prf_hz,
            "pulse_width_s": pulse_width_s,
            "system_temperature_k": system_temperature_k,
        }
        for name, value in positive_parameters.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")

        if beamwidth_el_deg is None:
            beamwidth_el_deg = beamwidth_deg
        if receiver_bandwidth_hz is None:
            receiver_bandwidth_hz = 1.0 / pulse_width_s
        if beamwidth_el_deg <= 0.0 or receiver_bandwidth_hz <= 0.0:
            raise ValueError(
                "beamwidth_el_deg and receiver_bandwidth_hz must be positive"
            )
        if noise_figure_db < 0.0 or system_losses_db < 0.0:
            raise ValueError("noise_figure_db and system_losses_db cannot be negative")
        if sidelobe_level_db >= 0.0:
            raise ValueError("sidelobe_level_db must be negative")
        scan_type = ScanType(scan_type)
        if scan_rate_rpm < 0.0:
            raise ValueError("scan_rate_rpm cannot be negative")
        if scan_type == ScanType.SECTOR:
            if not -180.0 <= sector_az_min_deg < sector_az_max_deg <= 180.0:
                raise ValueError("sector limits must satisfy -180 <= min < max <= 180 deg")
            if scan_speed_deg_s is None:
                scan_speed_deg_s = scan_rate_rpm * 6.0
            if scan_speed_deg_s <= 0.0:
                raise ValueError("sector scan speed must be positive")

        self.radar_id = radar_id
        self.frequency_hz = frequency_hz
        self.power_watts = power_watts
        self.antenna_gain_db = antenna_gain_db
        self.beamwidth_rad = np.radians(beamwidth_deg)
        self.beamwidth_el_rad = np.radians(beamwidth_el_deg)
        self.scan_rate_rpm = scan_rate_rpm
        self.prf_hz = prf_hz
        self.pulse_width_s = pulse_width_s
        self.receiver_bandwidth_hz = receiver_bandwidth_hz
        self.noise_figure_db = noise_figure_db
        self.system_temperature_k = system_temperature_k
        self.system_losses_db = system_losses_db
        self.polarization_tilt_deg = polarization_tilt_deg
        self.sidelobe_level_db = sidelobe_level_db
        self.scan_type = scan_type
        self.sector_az_min_rad = np.radians(sector_az_min_deg)
        self.sector_az_max_rad = np.radians(sector_az_max_deg)
        self.scan_speed_deg_s = scan_speed_deg_s
        self.stare_azimuth_rad = np.radians(stare_azimuth_deg)

        # Kinematic state
        if velocity is None:
            velocity = np.zeros(3)

        self.state = KinematicState(
            position=np.asarray(position, dtype=np.float64),
            velocity=np.asarray(velocity, dtype=np.float64),
        )

        # Antenna pointing, azimuth clockwise from North
        self.antenna_elevation = 0.0  # Current elevation [rad]
        self._sector_direction = 1.0
        if scan_type == ScanType.SECTOR:
            self.antenna_azimuth = self.sector_az_min_rad
        elif scan_type == ScanType.STARE:
            self.antenna_azimuth = self.stare_azimuth_rad
        else:
            self.antenna_azimuth = 0.0
        self.previous_antenna_azimuth = self.antenna_azimuth
        # Beam sweeps of the last update: (start azimuth, signed sweep angle) [rad]
        self.swept_legs: List[Tuple[float, float]] = []

    def update(self, dt: float) -> None:
        """
        Update radar state (position and antenna rotation).

        Args:
            dt: Time step [s]
        """
        # Update position if moving
        if np.any(self.state.velocity != 0):
            new_pos, _ = _update_kinematics_cv(
                self.state.position, self.state.velocity, dt
            )
            self.state.position = new_pos

        self.previous_antenna_azimuth = self.antenna_azimuth
        if self.scan_type == ScanType.CIRCULAR:
            sweep = self.scan_rate_rpm * 2 * np.pi / 60 * dt  # clockwise
            self.swept_legs = [(self.antenna_azimuth, sweep)] if sweep > 0.0 else []
            self.antenna_azimuth = (self.antenna_azimuth + sweep) % (2 * np.pi)
        elif self.scan_type == ScanType.SECTOR:
            self.swept_legs = self._advance_sector(np.radians(self.scan_speed_deg_s) * dt)
        else:
            self.swept_legs = []

    def _advance_sector(self, sweep: float) -> List[Tuple[float, float]]:
        """Move the beam along the sector raster, reversing at the limits."""
        legs = []
        angle = wrap_to_pi(self.antenna_azimuth)
        while sweep > 1e-12:
            limit = self.sector_az_max_rad if self._sector_direction > 0 else self.sector_az_min_rad
            room = (limit - angle) * self._sector_direction
            if room <= 1e-12:
                self._sector_direction = -self._sector_direction
                continue
            step = min(room, sweep)
            legs.append((angle, self._sector_direction * step))
            angle += self._sector_direction * step
            sweep -= step
        self.antenna_azimuth = angle
        return legs

    @property
    def revisit_period_s(self) -> float:
        """Time between consecutive looks at a fixed azimuth (nominal) [s]."""
        if self.scan_type == ScanType.CIRCULAR and self.scan_rate_rpm > 0.0:
            return 60.0 / self.scan_rate_rpm
        if self.scan_type == ScanType.SECTOR:
            width_deg = np.degrees(self.sector_az_max_rad - self.sector_az_min_rad)
            return 2.0 * width_deg / self.scan_speed_deg_s
        return 0.0

    def beam_swept(self, azimuth_rad: float) -> bool:
        """
        True if the beam centre crossed ``azimuth_rad`` during the last update.

        Each azimuth is crossed once per circular revolution (twice per sector
        cycle), giving one detection opportunity per pass. A staring or
        non-rotating antenna looks at whatever lies inside its main lobe.
        """
        if not self.swept_legs:
            return abs(wrap_to_pi(azimuth_rad - self.antenna_azimuth)) <= 0.5 * self.beamwidth_rad
        for start, sweep in self.swept_legs:
            if abs(sweep) >= 2.0 * np.pi:
                return True
            if sweep > 0.0:
                offset = (azimuth_rad - start) % (2.0 * np.pi)
            else:
                offset = (start - azimuth_rad) % (2.0 * np.pi)
            if 0.0 < offset <= abs(sweep):
                return True
        return False

    def calculate_target_geometry(
        self, target_position: np.ndarray, target_velocity: np.ndarray = None
    ) -> Dict[str, float]:
        """
        Calculate relative geometry to a target (ENU frame).

        Returns:
            Dict with slant range, azimuth (clockwise from North, (-π, π]),
            elevation and radial velocity (positive = receding)
        """
        delta = np.asarray(target_position, dtype=np.float64) - self.state.position
        range_m, azimuth, elevation = polar_from_enu(delta)

        # Radial velocity (if target has velocity)
        radial_velocity = 0.0
        if target_velocity is not None and range_m > 1e-6:
            relative_vel = target_velocity - self.state.velocity
            range_unit = delta / range_m
            radial_velocity = np.dot(relative_vel, range_unit)

        return {
            "range_m": range_m,
            "azimuth_rad": azimuth,
            "azimuth_deg": np.degrees(azimuth),
            "elevation_rad": elevation,
            "elevation_deg": np.degrees(elevation),
            "radial_velocity_mps": radial_velocity,
        }

    def is_target_in_beam(self, target_position: np.ndarray) -> bool:
        """
        Check if target is within antenna beam.

        Args:
            target_position: Target position [x, y, z]

        Returns:
            True if target is in beam
        """
        geom = self.calculate_target_geometry(target_position)

        az_diff = abs(wrap_to_pi(geom["azimuth_rad"] - self.antenna_azimuth))
        el_diff = abs(geom["elevation_rad"] - self.antenna_elevation)

        # Check if within beamwidth
        return bool(
            az_diff <= self.beamwidth_rad / 2 and el_diff <= self.beamwidth_el_rad / 2
        )

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
            "radar_id": self.radar_id,
            "position": self.state.position.tolist(),
            "velocity": self.state.velocity.tolist(),
            "frequency_hz": self.frequency_hz,
            "power_watts": self.power_watts,
            "antenna_gain_db": self.antenna_gain_db,
            "beamwidth_deg": np.degrees(self.beamwidth_rad),
            "beamwidth_el_deg": np.degrees(self.beamwidth_el_rad),
            "scan_rate_rpm": self.scan_rate_rpm,
            "prf_hz": self.prf_hz,
            "pulse_width_s": self.pulse_width_s,
            "receiver_bandwidth_hz": self.receiver_bandwidth_hz,
            "noise_figure_db": self.noise_figure_db,
            "system_temperature_k": self.system_temperature_k,
            "system_losses_db": self.system_losses_db,
            "polarization_tilt_deg": self.polarization_tilt_deg,
            "sidelobe_level_db": self.sidelobe_level_db,
            "scan_type": self.scan_type.value,
            "sector_az_min_deg": float(np.degrees(self.sector_az_min_rad)),
            "sector_az_max_deg": float(np.degrees(self.sector_az_max_rad)),
            "scan_speed_deg_s": self.scan_speed_deg_s,
            "stare_azimuth_deg": float(np.degrees(self.stare_azimuth_rad)),
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
    terrain_masked: Dict[int, bool] = field(default_factory=dict)

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
            if target.is_active(self.time):
                # Only the time since deployment counts in the deployment step.
                target.update(min(dt, self.time - target.active_from_s))
