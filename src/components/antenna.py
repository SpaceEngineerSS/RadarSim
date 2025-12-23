"""
Antenna Pattern and Phased Array Models

Implements antenna pattern calculations including beamwidth,
sidelobe levels, and array factor for phased arrays.

References:
    - Mailloux, "Phased Array Antenna Handbook", Artech House, 2005
    - Taylor, "Design of Line-Source Antennas for Narrow Beamwidth and Low
      Sidelobes", IRE Transactions on Antennas and Propagation, 1955
    - Skolnik, "Radar Handbook", 3rd Ed., Chapter 4
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import numba

from src.physics.constants import SPEED_OF_LIGHT


@dataclass
class AntennaParameters:
    """
    Antenna system parameters.
    
    Attributes:
        num_elements_az: Number of elements in azimuth
        num_elements_el: Number of elements in elevation
        element_spacing_az: Azimuth element spacing [m]
        element_spacing_el: Elevation element spacing [m]
        frequency: Operating frequency [Hz]
        weighting: Aperture weighting ('uniform', 'taylor', 'chebyshev')
        sidelobe_target_db: Target sidelobe level [dB] for weighted designs
    """
    num_elements_az: int
    num_elements_el: int
    element_spacing_az: float
    element_spacing_el: float
    frequency: float
    weighting: str = "taylor"
    sidelobe_target_db: float = -40.0
    
    @property
    def wavelength(self) -> float:
        """Operating wavelength [m]."""
        return SPEED_OF_LIGHT / self.frequency
    
    @property
    def element_spacing_az_lambda(self) -> float:
        """Azimuth element spacing in wavelengths."""
        return self.element_spacing_az / self.wavelength
    
    @property
    def element_spacing_el_lambda(self) -> float:
        """Elevation element spacing in wavelengths."""
        return self.element_spacing_el / self.wavelength


class PhasedArrayAntenna:
    """
    Phased Array Antenna Model
    
    Implements:
    - Beamwidth calculation
    - Sidelobe level estimation
    - Array factor pattern
    - Beam steering
    
    Reference: Mailloux, "Phased Array Antenna Handbook"
    """
    
    def __init__(self, params: AntennaParameters):
        """
        Initialize phased array antenna.
        
        Args:
            params: Antenna configuration parameters
        """
        self.params = params
        self._weights_az = self._compute_weights(
            params.num_elements_az,
            params.weighting,
            params.sidelobe_target_db
        )
        self._weights_el = self._compute_weights(
            params.num_elements_el,
            params.weighting,
            params.sidelobe_target_db
        )
    
    @staticmethod
    def _compute_weights(
        n_elements: int,
        weighting: str,
        sidelobe_db: float
    ) -> np.ndarray:
        """
        Compute aperture weights for sidelobe control.
        
        Args:
            n_elements: Number of array elements
            weighting: Weighting type
            sidelobe_db: Target sidelobe level [dB]
            
        Returns:
            Array of element weights
        """
        if weighting == "uniform":
            return np.ones(n_elements)
        
        elif weighting == "taylor":
            # Taylor one-parameter distribution
            # Reference: Taylor, 1955
            n = np.arange(n_elements)
            n_centered = n - (n_elements - 1) / 2
            
            # Sidelobe ratio
            r = 10 ** (-sidelobe_db / 20)
            
            # Taylor parameter A
            A = np.arccosh(r) / np.pi
            
            # Number of side-lobes at design level (n-bar)
            n_bar = int(2 * A**2 + 0.5)
            n_bar = max(2, min(n_bar, n_elements // 2))
            
            # Compute coefficients
            weights = np.ones(n_elements)
            sigma_m = n_bar / np.sqrt(A**2 + (n_bar - 0.5)**2)
            
            for m in range(1, n_bar):
                numerator = 1.0
                for n_idx in range(1, n_bar):
                    if n_idx != m:
                        numerator *= 1 - (m / n_idx)**2
                
                denominator = 1.0
                for n_idx in range(1, n_bar):
                    if n_idx != m:
                        x = A**2 + (n_idx - 0.5)**2
                        y = A**2 + (m - 0.5)**2
                        denominator *= 1 - x / y if y != 0 else 1
                
                F_m = ((-1)**(m+1)) * numerator / (2 * denominator) if denominator != 0 else 0
                
                for i, nc in enumerate(n_centered):
                    weights[i] *= 1 + 2 * F_m * np.cos(2 * np.pi * m * nc / n_elements)
            
            return weights / np.max(weights)
        
        elif weighting == "chebyshev":
            # Dolph-Chebyshev weights
            # Reference: Dolph, 1946
            n = n_elements
            
            # Sidelobe ratio
            r = 10 ** (-sidelobe_db / 20)
            
            # x0 parameter
            x0 = np.cosh(np.arccosh(r) / (n - 1))
            
            # Compute weights using Chebyshev polynomials
            weights = np.zeros(n)
            for i in range(n):
                sum_val = 0.0
                for m in range(n):
                    x = x0 * np.cos(np.pi * m / n)
                    # Chebyshev polynomial T_{n-1}(x)
                    if abs(x) <= 1:
                        T_n = np.cos((n - 1) * np.arccos(x))
                    else:
                        T_n = np.cosh((n - 1) * np.arccosh(abs(x)))
                        if x < 0:
                            T_n *= (-1) ** (n - 1)
                    sum_val += T_n * np.cos(2 * np.pi * i * m / n)
                weights[i] = sum_val / n
            
            return weights / np.max(weights)
        
        else:
            # Default to uniform
            return np.ones(n_elements)
    
    def calculate_beamwidth(self) -> Tuple[float, float]:
        """
        Calculate 3dB beamwidth in azimuth and elevation.
        
        θ_3dB ≈ 0.886 * λ / (N * d * cos(θ_0))
        
        Returns:
            (azimuth_beamwidth, elevation_beamwidth) in radians
            
        Reference: Skolnik, Eq. 4.30
        """
        # Beamwidth factor depends on weighting
        # Uniform: ~0.886, Taylor: ~1.0-1.2
        k_az = 0.886 if self.params.weighting == "uniform" else 1.0
        k_el = 0.886 if self.params.weighting == "uniform" else 1.0
        
        # Aperture sizes
        aperture_az = self.params.num_elements_az * self.params.element_spacing_az
        aperture_el = self.params.num_elements_el * self.params.element_spacing_el
        
        # 3dB beamwidth (broadside)
        beamwidth_az = k_az * self.params.wavelength / aperture_az
        beamwidth_el = k_el * self.params.wavelength / aperture_el
        
        return beamwidth_az, beamwidth_el
    
    def calculate_sidelobe_level(self) -> float:
        """
        Calculate peak sidelobe level.
        
        Returns:
            Peak sidelobe level [dB]
            
        Reference: Mailloux, Chapter 2
        """
        if self.params.weighting == "uniform":
            # Uniform weighting: -13.26 dB first sidelobe
            return -13.26
        elif self.params.weighting in ["taylor", "chebyshev"]:
            # Designed sidelobe level
            return self.params.sidelobe_target_db
        else:
            return -13.26
    
    def calculate_gain(self) -> float:
        """
        Calculate antenna gain.
        
        G = 4π * A_e / λ² ≈ 4π * η * N_az * N_el * d_az * d_el / λ²
        
        Returns:
            Antenna gain [dB]
            
        Reference: Skolnik, Eq. 4.16
        """
        # Aperture efficiency (~0.6-0.7 for weighted arrays)
        if self.params.weighting == "uniform":
            efficiency = 1.0
        else:
            efficiency = 0.7  # Weighted designs have lower efficiency
        
        # Physical aperture area
        aperture = (self.params.num_elements_az * self.params.element_spacing_az *
                   self.params.num_elements_el * self.params.element_spacing_el)
        
        # Gain
        gain_linear = 4 * np.pi * efficiency * aperture / (self.params.wavelength ** 2)
        
        return 10 * np.log10(gain_linear)
    
    def get_array_factor_1d(
        self,
        theta: np.ndarray,
        n_elements: int,
        element_spacing: float,
        weights: np.ndarray,
        steer_angle: float = 0.0
    ) -> np.ndarray:
        """
        Calculate 1D array factor.
        
        AF(θ) = Σ w_n * exp(j*k*d*n*(sin(θ) - sin(θ_0)))
        
        Args:
            theta: Observation angles [rad]
            n_elements: Number of elements
            element_spacing: Element spacing [m]
            weights: Element weights
            steer_angle: Beam steering angle [rad]
            
        Returns:
            Normalized array factor magnitude
        """
        k = 2 * np.pi / self.params.wavelength
        d = element_spacing
        
        # Element positions centered at origin
        n = np.arange(n_elements) - (n_elements - 1) / 2
        
        # Phase difference
        psi = k * d * (np.sin(theta)[:, np.newaxis] - np.sin(steer_angle))
        
        # Array factor
        af = np.sum(weights * np.exp(1j * n * psi), axis=1)
        
        # Normalize
        af_mag = np.abs(af)
        af_mag = af_mag / np.max(af_mag)
        
        return af_mag
    
    def get_pattern_2d(
        self,
        az_angles: np.ndarray,
        el_angles: np.ndarray,
        steer_az: float = 0.0,
        steer_el: float = 0.0
    ) -> np.ndarray:
        """
        Calculate 2D antenna pattern.
        
        Args:
            az_angles: Azimuth angles [rad]
            el_angles: Elevation angles [rad]
            steer_az: Azimuth steering angle [rad]
            steer_el: Elevation steering angle [rad]
            
        Returns:
            2D pattern magnitude [az x el]
        """
        # 1D patterns (separable array assumption)
        af_az = self.get_array_factor_1d(
            az_angles,
            self.params.num_elements_az,
            self.params.element_spacing_az,
            self._weights_az,
            steer_az
        )
        
        af_el = self.get_array_factor_1d(
            el_angles,
            self.params.num_elements_el,
            self.params.element_spacing_el,
            self._weights_el,
            steer_el
        )
        
        # 2D pattern (outer product)
        pattern_2d = np.outer(af_az, af_el)
        
        return pattern_2d
    
    def get_pattern_db(
        self,
        az_angles: np.ndarray,
        el_angles: np.ndarray,
        steer_az: float = 0.0,
        steer_el: float = 0.0,
        min_db: float = -60.0
    ) -> np.ndarray:
        """
        Calculate 2D antenna pattern in dB.
        
        Args:
            Same as get_pattern_2d()
            min_db: Minimum value clipping [dB]
            
        Returns:
            2D pattern [dB]
        """
        pattern = self.get_pattern_2d(az_angles, el_angles, steer_az, steer_el)
        
        # Avoid log of zero
        pattern = np.maximum(pattern, 10 ** (min_db / 20))
        
        return 20 * np.log10(pattern)
