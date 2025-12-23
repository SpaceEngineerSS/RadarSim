"""
Surface and Volume Clutter Models

Implements ground, sea, and weather clutter models for radar simulation.
Uses statistical distributions (Weibull, K-distribution, Log-Normal) for
realistic clutter generation.

References:
    - Sekine & Mao, "Weibull Radar Clutter", Peter Peregrinus, 1990
    - Ward, "Compound Representation of High Resolution Sea Clutter", 
      Electronics Letters, Vol. 17, 1981
    - Marshall & Palmer, "The Distribution of Raindrops with Size",
      Journal of Meteorology, Vol. 5, 1948
    - Skolnik, "Radar Handbook", 3rd Ed., Chapter 5
"""

import numpy as np
from typing import Tuple, Optional, Dict
from enum import Enum
import numba

from .constants import SPEED_OF_LIGHT


class TerrainType(Enum):
    """Terrain classification for ground clutter."""
    URBAN = "urban"
    SUBURBAN = "suburban"
    RURAL = "rural"
    FOREST = "forest"
    DESERT = "desert"
    MOUNTAINS = "mountains"


class SeaState(Enum):
    """Douglas Sea State scale."""
    CALM = 0      # Mirror-like
    SMOOTH = 1    # Ripples
    SLIGHT = 2    # Small wavelets
    MODERATE = 3  # Large wavelets
    ROUGH = 4     # Moderate waves
    VERY_ROUGH = 5  # Large waves
    HIGH = 6      # Very large waves


# Terrain parameters: (A, B) for σ0 = A + B*sin(ψ) [dB]
# Reference: Nathanson, "Radar Design Principles", Table 7.1
TERRAIN_PARAMETERS: Dict[str, Tuple[float, float]] = {
    "urban": (-15, 15),
    "suburban": (-20, 12),
    "rural": (-25, 10),
    "forest": (-20, 12),
    "desert": (-30, 8),
    "mountains": (-18, 14),
    "sea_calm": (-40, 5),
    "sea_rough": (-25, 12),
}


class ClutterModel:
    """
    Surface and Volume Clutter Models
    
    Provides realistic clutter RCS generation for radar simulation.
    
    Reference: Skolnik, "Radar Handbook", 3rd Ed., Chapter 5
    """
    
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _weibull_samples_jit(
        shape: float,
        scale: float,
        size: int
    ) -> np.ndarray:
        """
        JIT-compiled Weibull distributed samples.
        
        Weibull PDF: p(x) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k)
        
        Args:
            shape: Shape parameter k (Weibull shape)
            scale: Scale parameter λ
            size: Number of samples
            
        Returns:
            Array of Weibull-distributed values
        """
        # Generate uniform samples and transform to Weibull
        u = np.random.random(size)
        # Inverse CDF: x = λ * (-ln(1-u))^(1/k)
        samples = scale * ((-np.log(1 - u + 1e-10)) ** (1 / shape))
        return samples
    
    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _k_distribution_samples_jit(
        mean: float,
        shape_nu: float,
        size: int
    ) -> np.ndarray:
        """
        JIT-compiled K-distribution samples (compound model).
        
        K-distribution models sea clutter as product of:
        - Rayleigh (thermal noise/speckle)
        - Gamma (texture due to sea surface modulation)
        
        Args:
            mean: Mean value
            shape_nu: Shape parameter ν (controls spikiness)
            size: Number of samples
            
        Returns:
            Array of K-distributed values
            
        Reference: Ward, 1981
        """
        # Gamma-distributed texture component
        gamma_samples = np.random.gamma(shape_nu, mean / shape_nu, size)
        
        # Rayleigh speckle component (exponential for power)
        rayleigh_power = np.random.exponential(1.0, size)
        
        # K-distribution is product
        k_samples = gamma_samples * rayleigh_power
        
        return k_samples
    
    @staticmethod
    def ground_clutter_sigma0(
        grazing_angle_rad: float,
        terrain_type: str = "rural",
        frequency_ghz: float = 10.0,
        polarization: str = "HH"
    ) -> float:
        """
        Ground clutter backscatter coefficient (σ0).
        
        Uses empirical model: σ0 = A + B*sin(ψ) [dB]
        
        Args:
            grazing_angle_rad: Grazing angle [rad]
            terrain_type: Terrain classification
            frequency_ghz: Radar frequency [GHz]
            polarization: 'HH' or 'VV'
            
        Returns:
            σ0 in dB (dB relative to 1 m²/m²)
            
        Reference: Nathanson, "Radar Design Principles", Table 7.1
        """
        sin_psi = np.sin(grazing_angle_rad)
        
        # Get terrain parameters
        A, B = TERRAIN_PARAMETERS.get(terrain_type, (-25, 10))
        
        sigma0_db = A + B * sin_psi
        
        # Frequency adjustment (σ0 increases ~3 dB per octave above X-band)
        if frequency_ghz > 10:
            sigma0_db += 3 * np.log2(frequency_ghz / 10)
        
        # VV polarization typically 2-4 dB higher than HH
        if polarization == "VV":
            sigma0_db += 2.5
        
        return sigma0_db
    
    @staticmethod
    def sea_clutter_sigma0(
        grazing_angle_rad: float,
        sea_state: int = 3,
        frequency_ghz: float = 10.0,
        polarization: str = "HH"
    ) -> float:
        """
        Sea clutter backscatter coefficient (σ0).
        
        Uses simplified GIT (Georgia Tech) model.
        
        Args:
            grazing_angle_rad: Grazing angle [rad]
            sea_state: Douglas sea state (0-6)
            frequency_ghz: Radar frequency [GHz]
            polarization: 'HH' or 'VV'
            
        Returns:
            σ0 in dB
            
        Reference: GIT model, Nathanson Table 7.2
        """
        sea_state = max(0, min(6, sea_state))
        
        if polarization == "HH":
            sigma0_db = -50 + 10 * np.log10(sea_state + 0.1) + 25 * np.sin(grazing_angle_rad)
        else:  # VV
            sigma0_db = -45 + 10 * np.log10(sea_state + 0.1) + 20 * np.sin(grazing_angle_rad)
        
        # Frequency adjustment
        sigma0_db += 3 * np.log10(frequency_ghz / 10)
        
        return sigma0_db
    
    @staticmethod
    def ground_clutter_weibull(
        sigma0_db: float,
        cell_area_m2: float,
        shape: float = 2.0,
        size: int = 1
    ) -> np.ndarray:
        """
        Generate Weibull-distributed ground clutter RCS values.
        
        Args:
            sigma0_db: Backscatter coefficient [dB]
            cell_area_m2: Radar resolution cell area [m²]
            shape: Weibull shape parameter (1.5-3.0 typical)
            size: Number of samples
            
        Returns:
            Array of clutter RCS values [m²]
            
        Reference: Sekine & Mao, "Weibull Radar Clutter"
        """
        # Mean clutter RCS
        sigma0_linear = 10 ** (sigma0_db / 10)
        mean_rcs = sigma0_linear * cell_area_m2
        
        # Weibull scale from mean and shape
        # E[X] = scale * Γ(1 + 1/shape)
        from scipy.special import gamma as gamma_func
        scale = mean_rcs / gamma_func(1 + 1/shape)
        
        return ClutterModel._weibull_samples_jit(shape, scale, size)
    
    @staticmethod
    def sea_clutter_k_distribution(
        sigma0_db: float,
        cell_area_m2: float,
        sea_state: int = 3,
        size: int = 1
    ) -> np.ndarray:
        """
        Generate K-distributed sea clutter RCS values.
        
        Args:
            sigma0_db: Backscatter coefficient [dB]
            cell_area_m2: Radar resolution cell area [m²]
            sea_state: Douglas sea state (determines shape)
            size: Number of samples
            
        Returns:
            Array of clutter RCS values [m²]
            
        Reference: Ward, 1981
        """
        # Mean clutter RCS
        sigma0_linear = 10 ** (sigma0_db / 10)
        mean_rcs = sigma0_linear * cell_area_m2
        
        # Shape parameter depends on sea state (higher state = spikier clutter)
        shape_nu = max(0.5, 10.0 - sea_state)
        
        return ClutterModel._k_distribution_samples_jit(mean_rcs, shape_nu, size)
    
    @staticmethod
    def rain_reflectivity_marshall_palmer(
        rain_rate_mm_hr: float,
        frequency_ghz: float
    ) -> float:
        """
        Rain radar reflectivity using Marshall-Palmer Z-R relationship.
        
        Z = 200 * R^1.6 (mm^6/m^3)
        η = π^5 * |K|^2 * Z / λ^4 (m^-1)
        
        Args:
            rain_rate_mm_hr: Rain rate [mm/hr]
            frequency_ghz: Radar frequency [GHz]
            
        Returns:
            Volume reflectivity η [m²/m³]
            
        Reference: Marshall & Palmer, 1948
        """
        if rain_rate_mm_hr <= 0:
            return 0.0
        
        # Marshall-Palmer Z-R relationship
        Z = 200 * (rain_rate_mm_hr ** 1.6)  # mm^6/m^3
        
        # Convert to radar reflectivity
        wavelength_m = SPEED_OF_LIGHT / (frequency_ghz * 1e9)
        K_squared = 0.93  # for water at radar frequencies
        
        # η = π^5 * |K|^2 * Z / λ^4
        eta = (np.pi ** 5) * K_squared * Z * 1e-18 / (wavelength_m ** 4)
        
        return eta
    
    @staticmethod
    def volume_clutter_rcs(
        eta: float,
        range_m: float,
        beamwidth_rad: float,
        pulse_width_s: float
    ) -> float:
        """
        Calculate volume clutter RCS from reflectivity.
        
        σ_c = η * V_cell
        V_cell = (π/4) * R² * θ² * (c*τ/2)
        
        Args:
            eta: Volume reflectivity [m²/m³]
            range_m: Range to clutter cell [m]
            beamwidth_rad: Radar beamwidth [rad]
            pulse_width_s: Pulse width [s]
            
        Returns:
            Volume clutter RCS [m²]
        """
        range_resolution = SPEED_OF_LIGHT * pulse_width_s / 2
        
        # Resolution cell volume
        volume = (np.pi / 4) * (range_m ** 2) * (beamwidth_rad ** 2) * range_resolution
        
        return eta * volume
    
    @staticmethod
    def generate_clutter_map(
        range_bins: int,
        azimuth_bins: int,
        max_range_m: float,
        terrain_type: str = "rural",
        frequency_ghz: float = 10.0,
        radar_altitude_m: float = 0.0
    ) -> np.ndarray:
        """
        Generate 2D clutter map for PPI display.
        
        Args:
            range_bins: Number of range bins
            azimuth_bins: Number of azimuth bins
            max_range_m: Maximum range [m]
            terrain_type: Terrain classification
            frequency_ghz: Radar frequency [GHz]
            radar_altitude_m: Radar altitude [m]
            
        Returns:
            2D array of clutter power [linear]
        """
        # Create range and azimuth arrays
        ranges = np.linspace(100, max_range_m, range_bins)
        azimuths = np.linspace(0, 2*np.pi, azimuth_bins)
        
        # Calculate grazing angles
        clutter_map = np.zeros((range_bins, azimuth_bins))
        
        for i, r in enumerate(ranges):
            # Grazing angle (simplified flat earth)
            grazing = np.arctan2(radar_altitude_m, r) if radar_altitude_m > 0 else 0.05
            grazing = max(0.01, grazing)  # Minimum 0.5°
            
            # Get σ0
            sigma0_db = ClutterModel.ground_clutter_sigma0(
                grazing, terrain_type, frequency_ghz
            )
            
            # Resolution cell area (approximate)
            cell_area = 100 * 10  # 100m range x 10m cross-range
            
            # Generate clutter samples for all azimuths
            clutter_rcs = ClutterModel.ground_clutter_weibull(
                sigma0_db, cell_area, shape=2.0, size=azimuth_bins
            )
            
            clutter_map[i, :] = clutter_rcs
        
        return clutter_map
