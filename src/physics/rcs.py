"""
Radar Cross Section (RCS) and Swerling Fluctuation Models

Implements RCS calculations with aspect angle dependency and Swerling
statistical fluctuation models for realistic target simulation.

References:
    - Swerling, P., "Probability of Detection for Fluctuating Targets",
      IRE Transactions on Information Theory, Vol. IT-6, pp. 269-308, April 1960
    - Skolnik, "Radar Handbook", 3rd Ed., Chapter 14
    - IEEE Std 686-2008, "IEEE Standard Radar Definitions"
"""

import numpy as np
from enum import Enum
from typing import Union, Optional, Dict
from dataclasses import dataclass, field
import numba


class SwerlingModel(Enum):
    """
    Swerling RCS Fluctuation Models
    
    Reference: Swerling, 1960, IRE Transactions
    
    Model 0 (Marcum): Non-fluctuating target (deterministic RCS)
    Model 1: Scan-to-scan fluctuation, Rayleigh distribution (many small scatterers)
    Model 2: Pulse-to-pulse fluctuation, Rayleigh distribution
    Model 3: Scan-to-scan fluctuation, Chi-squared 4 DoF (one dominant + small)
    Model 4: Pulse-to-pulse fluctuation, Chi-squared 4 DoF
    """
    SWERLING_0 = 0  # Non-fluctuating (Marcum)
    SWERLING_1 = 1  # Slow fluctuation, Rayleigh
    SWERLING_2 = 2  # Fast fluctuation, Rayleigh
    SWERLING_3 = 3  # Slow fluctuation, Chi-squared
    SWERLING_4 = 4  # Fast fluctuation, Chi-squared


class TargetType(Enum):
    """
    Target classification types with typical RCS values
    
    Values based on open literature (DTIC, NATO publications)
    """
    AIRCRAFT = "aircraft"
    FIGHTER = "fighter"
    BOMBER = "bomber"
    GROUND_VEHICLE = "ground_vehicle"
    TANK = "tank"
    MISSILE = "missile"
    CRUISE_MISSILE = "cruise_missile"
    BALLISTIC_MISSILE = "ballistic_missile"
    SHIP = "ship"
    SMALL_BOAT = "small_boat"
    DRONE = "drone"
    HELICOPTER = "helicopter"
    BIRD = "bird"
    CHAFF = "chaff"
    DECOY = "decoy"


# Median RCS values [m²] - from open literature
RCS_DATABASE: Dict[str, float] = {
    "aircraft": 10.0,
    "fighter": 3.0,  # Modern 4th gen (F-16, Su-27)
    "bomber": 25.0,  # Large bomber (B-52)
    "ground_vehicle": 50.0,
    "tank": 15.0,
    "missile": 0.1,  # Tactical missile
    "cruise_missile": 0.5,
    "ballistic_missile": 1.0,
    "ship": 1000.0,  # Destroyer-class
    "small_boat": 10.0,
    "drone": 0.01,  # Small UAV
    "helicopter": 5.0,
    "bird": 0.001,
    "chaff": 0.01,  # Per chaff cloud
    "decoy": 5.0,  # Active decoy
}


@dataclass
class TargetKinematics:
    """
    Target kinematic state with attitude (6-DOF compatible)
    
    Coordinate system: North-East-Down (NED)
    
    Attributes:
        position: [x, y, z] position [m]
        velocity: [vx, vy, vz] velocity [m/s]
        acceleration: [ax, ay, az] acceleration [m/s²]
        roll: Roll angle [rad]
        pitch: Pitch angle [rad]
        yaw: Yaw/heading angle [rad]
    """
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    def __post_init__(self) -> None:
        """Ensure arrays are 3D numpy arrays."""
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
    
    def update_attitude_from_velocity(self) -> None:
        """
        Update attitude angles based on velocity vector (coordinated flight assumption).
        
        This implements a simple kinematic attitude update where:
        - Yaw aligns with horizontal velocity direction
        - Pitch aligns with velocity elevation
        - Roll is derived from turn rate (coordinated turn)
        """
        vel_horizontal = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        
        if vel_horizontal > 1e-3:
            # Yaw from horizontal velocity
            self.yaw = np.arctan2(self.velocity[1], self.velocity[0])
            
            # Pitch from velocity vector
            self.pitch = np.arctan2(-self.velocity[2], vel_horizontal)
        
        # Roll from lateral acceleration (coordinated turn)
        if vel_horizontal > 10.0:  # Only if moving significantly
            lateral_accel = (self.acceleration[0] * np.sin(self.yaw) - 
                           self.acceleration[1] * np.cos(self.yaw))
            # φ = arctan(a_lateral / g)
            self.roll = np.arctan2(lateral_accel, 9.81)
            self.roll = np.clip(self.roll, -np.pi/3, np.pi/3)  # Limit to ±60°


class SwerlingRCS:
    """
    Swerling RCS Fluctuation Models
    
    Generates fluctuating RCS values based on statistical models.
    
    Reference: Swerling, P., "Probability of Detection for Fluctuating Targets"
               IRE Transactions on Information Theory, Vol. IT-6, April 1960
    """
    
    @staticmethod
    def generate_rcs(
        mean_rcs: float,
        model: SwerlingModel,
        n_pulses: int = 1,
        correlation: float = 1.0
    ) -> float:
        """
        Generate fluctuating RCS based on Swerling model.
        
        Args:
            mean_rcs: Mean RCS value [m²]
            model: Swerling model type (0-4)
            n_pulses: Number of pulses for integration
            correlation: Scan-to-scan correlation (0-1)
            
        Returns:
            Fluctuated RCS value [m²]
            
        Reference: Swerling, 1960, Eqs. 3-8
        """
        if mean_rcs <= 0:
            return 0.0
        
        if model == SwerlingModel.SWERLING_0:
            # Non-fluctuating (Marcum case)
            return mean_rcs
        
        elif model == SwerlingModel.SWERLING_1:
            # Slow fluctuation, exponential (Rayleigh amplitude)
            # PDF: p(σ) = (1/σ_avg) * exp(-σ/σ_avg)
            return np.random.exponential(mean_rcs)
        
        elif model == SwerlingModel.SWERLING_2:
            # Fast fluctuation, exponential
            # Pulse-to-pulse decorrelation
            if n_pulses > 1:
                samples = np.random.exponential(mean_rcs, n_pulses)
                return np.mean(samples)
            return np.random.exponential(mean_rcs)
        
        elif model == SwerlingModel.SWERLING_3:
            # Slow fluctuation, Chi-squared 4 DoF
            # One dominant scatterer + many small
            # PDF: p(σ) = (4σ/σ_avg²) * exp(-2σ/σ_avg)
            return np.random.gamma(2, mean_rcs / 2)
        
        elif model == SwerlingModel.SWERLING_4:
            # Fast fluctuation, Chi-squared 4 DoF
            if n_pulses > 1:
                samples = np.random.gamma(2, mean_rcs / 2, n_pulses)
                return np.mean(samples)
            return np.random.gamma(2, mean_rcs / 2)
        
        return mean_rcs
    
    @staticmethod
    def get_pdf(
        rcs_values: np.ndarray,
        mean_rcs: float,
        model: SwerlingModel
    ) -> np.ndarray:
        """
        Get probability density function for RCS values.
        
        Args:
            rcs_values: Array of RCS values to evaluate [m²]
            mean_rcs: Mean RCS value [m²]
            model: Swerling model type
            
        Returns:
            PDF values for each RCS value
        """
        sigma = np.asarray(rcs_values)
        sigma = np.maximum(sigma, 1e-10)  # Avoid division by zero
        sigma_avg = max(mean_rcs, 1e-10)
        
        if model in [SwerlingModel.SWERLING_1, SwerlingModel.SWERLING_2]:
            # Exponential (Rayleigh power)
            pdf = (1 / sigma_avg) * np.exp(-sigma / sigma_avg)
        
        elif model in [SwerlingModel.SWERLING_3, SwerlingModel.SWERLING_4]:
            # Chi-squared with 4 degrees of freedom
            pdf = (4 * sigma / (sigma_avg ** 2)) * np.exp(-2 * sigma / sigma_avg)
        
        else:
            # Non-fluctuating (delta function approximation)
            pdf = np.zeros_like(sigma)
            idx = np.argmin(np.abs(sigma - sigma_avg))
            if len(pdf) > 0:
                pdf[idx] = 1.0
        
        return pdf
    
    @staticmethod
    def get_fluctuation_loss(
        model: SwerlingModel,
        pd: float = 0.9,
        pfa: float = 1e-6
    ) -> float:
        """
        Get fluctuation loss compared to Swerling 0 (non-fluctuating).
        
        Args:
            model: Swerling model type
            pd: Probability of detection
            pfa: Probability of false alarm
            
        Returns:
            Fluctuation loss [dB] (additional SNR required)
            
        Reference: Skolnik, Table 2.2
        """
        # Approximate fluctuation losses (Pd=0.9, Pfa=1e-6)
        fluctuation_loss = {
            SwerlingModel.SWERLING_0: 0.0,
            SwerlingModel.SWERLING_1: 8.4,
            SwerlingModel.SWERLING_2: 7.5,
            SwerlingModel.SWERLING_3: 5.8,
            SwerlingModel.SWERLING_4: 5.5,
        }
        
        return fluctuation_loss.get(model, 0.0)


@numba.jit(nopython=True, cache=True)
def _aspect_angle_factor_jit(
    aspect_angle_rad: float,
    nose_factor: float = 0.3,
    beam_factor: float = 1.0,
    tail_factor: float = 0.5
) -> float:
    """
    JIT-compiled aspect angle RCS factor.
    
    Simplified model: RCS varies with aspect angle based on typical
    aircraft shape (lower RCS from nose, higher from beam).
    
    Args:
        aspect_angle_rad: Aspect angle from nose (0 = head-on) [rad]
        nose_factor: RCS factor at head-on aspect
        beam_factor: RCS factor at broadside (90°)
        tail_factor: RCS factor at tail-on (180°)
        
    Returns:
        RCS multiplication factor
    """
    # Normalize angle to [0, π]
    angle = abs(aspect_angle_rad) % (2 * np.pi)
    if angle > np.pi:
        angle = 2 * np.pi - angle
    
    # Interpolate based on aspect regions
    if angle < np.pi / 4:  # 0-45° (nose region)
        t = angle / (np.pi / 4)
        factor = nose_factor + t * (beam_factor - nose_factor)
    elif angle < 3 * np.pi / 4:  # 45-135° (beam region)
        factor = beam_factor
    else:  # 135-180° (tail region)
        t = (angle - 3 * np.pi / 4) / (np.pi / 4)
        factor = beam_factor + t * (tail_factor - beam_factor)
    
    return max(0.01, factor)


def calculate_aspect_angle(
    target_position: np.ndarray,
    target_heading: float,
    radar_position: np.ndarray
) -> float:
    """
    Calculate aspect angle between radar and target.
    
    Aspect angle = 0° when radar is looking at target nose
    Aspect angle = 90° when radar is looking at target beam (side)
    Aspect angle = 180° when radar is looking at target tail
    
    Args:
        target_position: Target position [x, y, z] [m]
        target_heading: Target heading angle [rad] (0 = North)
        radar_position: Radar position [x, y, z] [m]
        
    Returns:
        Aspect angle [rad] (0 to π)
    """
    # Vector from target to radar
    delta = np.asarray(radar_position) - np.asarray(target_position)
    
    # Bearing from target to radar
    bearing_to_radar = np.arctan2(delta[1], delta[0])
    
    # Aspect angle is difference between target heading and bearing to radar
    aspect = bearing_to_radar - target_heading
    
    # Normalize to [0, π]
    aspect = abs(aspect) % (2 * np.pi)
    if aspect > np.pi:
        aspect = 2 * np.pi - aspect
    
    return aspect


def calculate_aspect_dependent_rcs(
    target_type: Union[str, TargetType],
    aspect_angle: float,
    swerling_model: SwerlingModel = SwerlingModel.SWERLING_1,
    frequency_ghz: float = 10.0
) -> float:
    """
    Calculate RCS with aspect angle dependency.
    
    Args:
        target_type: Target classification
        aspect_angle: Aspect angle [rad] (0 = nose-on)
        swerling_model: Swerling fluctuation model
        frequency_ghz: Radar frequency [GHz] (for resonance effects)
        
    Returns:
        RCS value [m²]
        
    Reference: Skolnik, "Radar Handbook", 3rd Ed., Chapter 14
    """
    # Get base RCS from database
    if isinstance(target_type, TargetType):
        type_str = target_type.value
    else:
        type_str = target_type
    
    base_rcs = RCS_DATABASE.get(type_str, 1.0)
    
    # Apply aspect angle factor
    aspect_factor = _aspect_angle_factor_jit(aspect_angle)
    mean_rcs = base_rcs * aspect_factor
    
    # Apply Swerling fluctuation
    rcs = SwerlingRCS.generate_rcs(mean_rcs, swerling_model)
    
    return max(0.001, rcs)  # Minimum 0.001 m²


def get_target_type_rcs(target_type: Union[str, TargetType]) -> float:
    """
    Get median RCS for a target type.
    
    Args:
        target_type: Target classification
        
    Returns:
        Median RCS [m²]
    """
    if isinstance(target_type, TargetType):
        type_str = target_type.value
    else:
        type_str = target_type
    
    return RCS_DATABASE.get(type_str, 1.0)


# =============================================================================
# VALIDATION FUNCTIONS (Reference: Swerling, 1960)
# =============================================================================

def validate_swerling_distribution(
    model: SwerlingModel,
    mean_rcs: float = 10.0,
    n_samples: int = 10000,
    significance_level: float = 0.05
) -> dict:
    """
    Monte Carlo validation of Swerling RCS distributions
    
    Reference: Swerling, P. (1960). "Probability of Detection for Fluctuating Targets"
               IRE Transactions on Information Theory, Vol. IT-6, pp. 269-308
    
    Uses Kolmogorov-Smirnov test to verify:
        - Swerling 0: Constant (deterministic)
        - Swerling I/II: Exponential distribution (Chi-squared k=2)
        - Swerling III/IV: Chi-squared k=4 (Gamma with shape=2)
    
    Args:
        model: Swerling model type to validate
        mean_rcs: Mean RCS value for testing [m²]
        n_samples: Number of Monte Carlo samples
        significance_level: KS test significance level (p > this to pass)
        
    Returns:
        Dict containing validation results and statistics
    """
    from scipy import stats
    
    # Generate samples
    samples = np.array([
        SwerlingRCS.generate_rcs(mean_rcs, model) 
        for _ in range(n_samples)
    ])
    
    if model == SwerlingModel.SWERLING_0:
        # Non-fluctuating: all samples should equal mean_rcs
        is_constant = np.allclose(samples, mean_rcs)
        std_dev = np.std(samples)
        
        return {
            'model': 'SWERLING_0',
            'n_samples': n_samples,
            'mean_rcs_input': mean_rcs,
            'statistics': {
                'sample_mean': np.mean(samples),
                'sample_std': std_dev,
                'all_equal_to_mean': is_constant,
            },
            'validation': {
                'is_valid': is_constant,
                'test_type': 'constant_check',
                'reference': 'Swerling (1960) - Non-fluctuating (Marcum) case',
            }
        }
    
    elif model in [SwerlingModel.SWERLING_1, SwerlingModel.SWERLING_2]:
        # Exponential distribution: p(σ) = (1/σ_avg) * exp(-σ/σ_avg)
        # Equivalent to scipy.stats.expon with scale=mean_rcs
        
        ks_stat, p_value = stats.kstest(
            samples, 
            lambda x: stats.expon.cdf(x, scale=mean_rcs)
        )
        
        is_valid = p_value > significance_level
        
        return {
            'model': model.name,
            'distribution': 'Exponential (Rayleigh power)',
            'pdf': 'p(σ) = (1/σ_avg) × exp(-σ/σ_avg)',
            'n_samples': n_samples,
            'mean_rcs_input': mean_rcs,
            'statistics': {
                'sample_mean': np.mean(samples),
                'sample_std': np.std(samples),
                'theoretical_mean': mean_rcs,
                'theoretical_std': mean_rcs,  # For exponential, std = mean
            },
            'ks_test': {
                'statistic': ks_stat,
                'p_value': p_value,
                'significance_level': significance_level,
            },
            'validation': {
                'is_valid': is_valid,
                'test_type': 'Kolmogorov-Smirnov',
                'reference': 'Swerling (1960), Eq. 4 - Exponential PDF',
            }
        }
    
    elif model in [SwerlingModel.SWERLING_3, SwerlingModel.SWERLING_4]:
        # Chi-squared k=4 distribution: p(σ) = (4σ/σ_avg²) × exp(-2σ/σ_avg)
        # Equivalent to Gamma(shape=2, scale=σ_avg/2)
        
        ks_stat, p_value = stats.kstest(
            samples,
            lambda x: stats.gamma.cdf(x, 2, scale=mean_rcs/2)
        )
        
        is_valid = p_value > significance_level
        
        return {
            'model': model.name,
            'distribution': 'Chi-squared (k=4) / Gamma(2, σ_avg/2)',
            'pdf': 'p(σ) = (4σ/σ_avg²) × exp(-2σ/σ_avg)',
            'n_samples': n_samples,
            'mean_rcs_input': mean_rcs,
            'statistics': {
                'sample_mean': np.mean(samples),
                'sample_std': np.std(samples),
                'theoretical_mean': mean_rcs,
                'theoretical_std': mean_rcs / np.sqrt(2),  # For Gamma(2), std = mean/sqrt(2)
            },
            'ks_test': {
                'statistic': ks_stat,
                'p_value': p_value,
                'significance_level': significance_level,
            },
            'validation': {
                'is_valid': is_valid,
                'test_type': 'Kolmogorov-Smirnov',
                'reference': 'Swerling (1960), Eq. 5 - Chi-squared PDF',
            }
        }
    
    else:
        return {
            'error': f'Unknown Swerling model: {model}',
            'validation': {'is_valid': False}
        }

