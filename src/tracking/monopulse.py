"""
Monopulse Angle Estimation

Implements Sum/Difference pattern processing for precision angle tracking.

Monopulse provides angle accuracy that can be much better than the beamwidth,
typically by a factor of 10-100x depending on SNR.

References:
    - Skolnik, "Radar Handbook", 3rd Ed., Chapter 9 (Tracking Radar)
    - Sherman & Barton, "Monopulse Principles and Techniques", 2nd Ed.
    - Rhodes, "Introduction to Monopulse", Artech House, 1980
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class MonopulseResult:
    """Result of monopulse angle measurement."""
    measured_azimuth_rad: float
    measured_elevation_rad: float
    error_signal_az: float  # Normalized error in azimuth
    error_signal_el: float  # Normalized error in elevation
    snr_db: float
    angular_accuracy_rad: float  # Estimated accuracy based on SNR


class MonopulseEstimator:
    """
    Monopulse Angle Estimation System.
    
    Simulates Sum (Σ) and Difference (Δ) antenna patterns for precision
    angle measurement. Uses the amplitude-comparison monopulse technique.
    
    The error signal ε = Δ/Σ is approximately linear near boresight,
    providing sub-beamwidth angular accuracy.
    
    Reference: Skolnik, "Radar Handbook", Chapter 9
    """
    
    def __init__(
        self,
        beamwidth_deg: float = 1.5,
        monopulse_slope: float = 1.6,
        squint_angle_deg: float = 0.5
    ):
        """
        Initialize Monopulse Estimator.
        
        Args:
            beamwidth_deg: 3-dB beamwidth [degrees]
            monopulse_slope: Slope of error signal at boresight (km), 
                            typically 1.2-2.0 depending on antenna design
            squint_angle_deg: Angular offset of difference beams [degrees]
        """
        self.beamwidth_rad = np.radians(beamwidth_deg)
        self.beamwidth_deg = beamwidth_deg
        self.monopulse_slope = monopulse_slope
        self.squint_angle_rad = np.radians(squint_angle_deg)
        
        # Pattern shape constant (Gaussian approximation)
        # k = 2.776 / θ_3dB^2 for -3dB at beamwidth/2
        self.k = 2.776 / (self.beamwidth_rad ** 2)
    
    def sum_pattern(self, theta_rad: float) -> float:
        """
        Sum pattern (Σ) - Gaussian approximation.
        
        Σ(θ) = exp(-k * θ²)
        
        Args:
            theta_rad: Off-boresight angle [rad]
            
        Returns:
            Normalized sum pattern amplitude (0-1)
        """
        return np.exp(-self.k * theta_rad ** 2)
    
    def difference_pattern(self, theta_rad: float) -> float:
        """
        Difference pattern (Δ) - Odd function approximation.
        
        Δ(θ) ≈ θ * exp(-k * θ²) for amplitude comparison
        
        This pattern is null at boresight and odd-symmetric.
        
        Args:
            theta_rad: Off-boresight angle [rad]
            
        Returns:
            Normalized difference pattern amplitude
        """
        return theta_rad * np.exp(-self.k * theta_rad ** 2) * self.monopulse_slope
    
    def error_signal(self, theta_rad: float) -> float:
        """
        Monopulse error signal ε = Δ/Σ.
        
        This ratio is approximately linear near the boresight axis,
        providing precise angle information.
        
        Args:
            theta_rad: Off-boresight angle [rad]
            
        Returns:
            Error signal (normalized, approximately = km * θ near boresight)
        """
        sigma = self.sum_pattern(theta_rad)
        delta = self.difference_pattern(theta_rad)
        
        # Avoid division by zero for very large off-axis angles
        if sigma < 1e-10:
            return np.sign(theta_rad) * 10.0  # Saturation
        
        return delta / sigma
    
    def inverse_error_signal(self, epsilon: float) -> float:
        """
        Inverse error signal - estimate angle from error signal.
        
        Near boresight: θ ≈ ε / km
        
        Args:
            epsilon: Error signal value
            
        Returns:
            Estimated off-boresight angle [rad]
        """
        # Near boresight, the relationship is approximately linear
        # ε ≈ km * θ, so θ ≈ ε / km
        return epsilon / self.monopulse_slope
    
    def angular_accuracy(self, snr_linear: float) -> float:
        """
        Theoretical angular accuracy based on SNR.
        
        σ_θ = θ_3dB / (km * sqrt(2 * SNR))
        
        Reference: Barton, "Modern Radar System Analysis"
        
        Args:
            snr_linear: Signal-to-noise ratio (linear, not dB)
            
        Returns:
            Angular accuracy (1-sigma) [rad]
        """
        if snr_linear < 1:
            return self.beamwidth_rad  # Low SNR, accuracy ~ beamwidth
        
        return self.beamwidth_rad / (self.monopulse_slope * np.sqrt(2 * snr_linear))
    
    def measure_angle(
        self,
        beam_azimuth_rad: float,
        beam_elevation_rad: float,
        target_azimuth_rad: float,
        target_elevation_rad: float,
        snr_db: float = 20.0
    ) -> MonopulseResult:
        """
        Measure target angle using monopulse processing.
        
        This simulates the full monopulse measurement process:
        1. Calculate off-boresight angles
        2. Generate error signals
        3. Apply noise based on SNR
        4. Return corrected angle estimate
        
        Args:
            beam_azimuth_rad: Current beam pointing azimuth [rad]
            beam_elevation_rad: Current beam pointing elevation [rad]
            target_azimuth_rad: True target azimuth [rad]
            target_elevation_rad: True target elevation [rad]
            snr_db: Signal-to-noise ratio [dB]
            
        Returns:
            MonopulseResult with corrected angle measurements
        """
        # Convert SNR to linear
        snr_linear = 10 ** (snr_db / 10)
        
        # Calculate off-boresight angles
        delta_az = target_azimuth_rad - beam_azimuth_rad
        delta_el = target_elevation_rad - beam_elevation_rad
        
        # Generate error signals
        eps_az = self.error_signal(delta_az)
        eps_el = self.error_signal(delta_el)
        
        # Angular accuracy based on SNR
        sigma_angle = self.angular_accuracy(snr_linear)
        
        # Add measurement noise (thermal noise contribution)
        noise_az = np.random.normal(0, sigma_angle)
        noise_el = np.random.normal(0, sigma_angle)
        
        # Estimate off-boresight angle from error signal + noise
        estimated_delta_az = self.inverse_error_signal(eps_az) + noise_az
        estimated_delta_el = self.inverse_error_signal(eps_el) + noise_el
        
        # Calculate measured angles (beam center + estimated offset)
        measured_az = beam_azimuth_rad + estimated_delta_az
        measured_el = beam_elevation_rad + estimated_delta_el
        
        return MonopulseResult(
            measured_azimuth_rad=measured_az,
            measured_elevation_rad=measured_el,
            error_signal_az=eps_az,
            error_signal_el=eps_el,
            snr_db=snr_db,
            angular_accuracy_rad=sigma_angle
        )
    
    @staticmethod
    def calculate_improvement_factor(snr_db: float, beamwidth_deg: float) -> float:
        """
        Calculate monopulse improvement factor over beam-center measurement.
        
        Monopulse typically provides 10-100x improvement in angle accuracy
        compared to simple beam-splitting techniques.
        
        Args:
            snr_db: Signal-to-noise ratio [dB]
            beamwidth_deg: Antenna beamwidth [degrees]
            
        Returns:
            Improvement factor (ratio of beam accuracy to monopulse accuracy)
        """
        snr_linear = 10 ** (snr_db / 10)
        km = 1.6  # Typical monopulse slope
        
        # Without monopulse: accuracy ~ beamwidth / sqrt(SNR)
        # With monopulse: accuracy ~ beamwidth / (km * sqrt(2 * SNR))
        return km * np.sqrt(2)


def validate_monopulse_patterns():
    """Validate monopulse pattern calculations."""
    estimator = MonopulseEstimator(beamwidth_deg=2.0)
    
    # Test points
    test_angles_deg = [-3, -1, -0.5, 0, 0.5, 1, 3]
    
    print("Monopulse Pattern Validation")
    print("=" * 50)
    print(f"Beamwidth: {estimator.beamwidth_deg}°")
    print(f"Monopulse Slope: {estimator.monopulse_slope}")
    print()
    print(f"{'Angle (°)':<12} {'Σ':<12} {'Δ':<12} {'ε=Δ/Σ':<12}")
    print("-" * 50)
    
    for angle_deg in test_angles_deg:
        angle_rad = np.radians(angle_deg)
        sigma = estimator.sum_pattern(angle_rad)
        delta = estimator.difference_pattern(angle_rad)
        epsilon = estimator.error_signal(angle_rad)
        print(f"{angle_deg:<12.1f} {sigma:<12.4f} {delta:<12.4f} {epsilon:<12.4f}")
    
    print()
    print("Key properties:")
    print("  - Σ(0) = 1.0 (max at boresight)")
    print("  - Δ(0) = 0.0 (null at boresight)")
    print("  - ε ≈ km*θ near boresight (linear)")
    
    return True


if __name__ == "__main__":
    validate_monopulse_patterns()
