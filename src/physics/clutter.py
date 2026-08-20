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

from enum import Enum
from typing import Dict, Optional

import numba
import numpy as np

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

    CALM = 0  # Mirror-like
    SMOOTH = 1  # Ripples
    SLIGHT = 2  # Small wavelets
    MODERATE = 3  # Large wavelets
    ROUGH = 4  # Moderate waves
    VERY_ROUGH = 5  # Large waves
    HIGH = 6  # Very large waves


LAND_GAMMA_PRIORS_DB: Dict[str, float] = {
    "urban": -5.0,
    "suburban": -12.0,
    "rural": -20.0,
    "forest": -15.0,
    "desert": -30.0,
    "mountains": -8.0,
}


class ClutterModel:
    """
    Surface and Volume Clutter Models

    Provides realistic clutter RCS generation for radar simulation.

    Reference: Skolnik, "Radar Handbook", 3rd Ed., Chapter 5
    """

    @staticmethod
    @numba.jit(nopython=True, cache=True)
    def _weibull_samples_jit(shape: float, scale: float, size: int) -> np.ndarray:
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
        mean: float, shape_nu: float, size: int
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
        polarization: str = "HH",
        gamma_db: Optional[float] = None,
    ) -> float:
        """
        Ground clutter backscatter coefficient (σ0).

        Uses the constant-gamma engineering model σ0 = γ sin(ψ).

        Args:
            grazing_angle_rad: Grazing angle [rad]
            terrain_type: Terrain classification
            frequency_ghz: Radar frequency [GHz]
            polarization: 'HH' or 'VV'

        Returns:
            σ0 in dB (dB relative to 1 m²/m²)

        Terrain-name values are nominal priors, not site calibration. Pass gamma_db from
        measured clutter whenever quantitative accuracy is required.
        """
        if not 0.0 < grazing_angle_rad <= np.pi / 2.0:
            raise ValueError("grazing_angle_rad must be between 0 and pi/2")
        if frequency_ghz <= 0.0:
            raise ValueError("frequency_ghz must be positive")
        if polarization.upper() not in {"HH", "VV"}:
            raise ValueError("polarization must be 'HH' or 'VV'")
        if gamma_db is None:
            try:
                gamma_db = LAND_GAMMA_PRIORS_DB[terrain_type.lower()]
            except KeyError as error:
                raise ValueError(f"unknown terrain type: {terrain_type}") from error
        return float(gamma_db + 10.0 * np.log10(np.sin(grazing_angle_rad)))

    @staticmethod
    def bare_soil_oh1992_sigma0(
        grazing_angle_rad: float,
        frequency_ghz: float,
        relative_permittivity: complex,
        rms_height_m: float,
        polarization: str = "HH",
    ) -> float:
        """Oh-Sarabandi-Ulaby (1992) bare-soil normalized backscatter."""
        incidence = np.pi / 2.0 - grazing_angle_rad
        incidence_deg = float(np.degrees(incidence))
        if not 10.0 <= incidence_deg <= 70.0:
            raise ValueError("Oh-1992 model requires 10-70 degree incidence")
        if not 1.0 <= frequency_ghz <= 10.0:
            raise ValueError("Oh-1992 measurement domain is L-, C-, and X-band")
        if relative_permittivity.real <= 1.0 or relative_permittivity.imag > 0.0:
            raise ValueError("permittivity must use the passive convention eps'-j eps''")
        if rms_height_m <= 0.0:
            raise ValueError("rms_height_m must be positive")

        wavelength = SPEED_OF_LIGHT / (frequency_ghz * 1e9)
        ks = 2.0 * np.pi / wavelength * rms_height_m
        if not 0.1 <= ks <= 6.0:
            raise ValueError("Oh-1992 model requires 0.1 <= k*s <= 6")

        root_eps = np.sqrt(relative_permittivity)
        gamma_0 = abs((1.0 - root_eps) / (1.0 + root_eps)) ** 2
        root_term = np.sqrt(relative_permittivity - np.sin(incidence) ** 2)
        r_h = (np.cos(incidence) - root_term) / (np.cos(incidence) + root_term)
        r_v = (
            relative_permittivity * np.cos(incidence) - root_term
        ) / (relative_permittivity * np.cos(incidence) + root_term)
        gamma_h = abs(r_h) ** 2
        gamma_v = abs(r_v) ** 2

        sqrt_p = 1.0 - (2.0 * incidence / np.pi) ** (1.0 / (3.0 * gamma_0)) * np.exp(
            -ks
        )
        if sqrt_p <= 0.0:
            raise ValueError("Oh-1992 co-polarization ratio is outside its physical domain")
        p = sqrt_p**2
        q = 0.23 * np.sqrt(gamma_0) * (1.0 - np.exp(-ks))
        g = 0.7 * (1.0 - np.exp(-0.65 * ks**1.8))
        sigma_vv = g * np.cos(incidence) ** 3 * (gamma_v + gamma_h) / sqrt_p
        sigma = {"VV": sigma_vv, "HH": p * sigma_vv, "HV": q * sigma_vv}
        try:
            return float(10.0 * np.log10(sigma[polarization.upper()]))
        except KeyError as error:
            raise ValueError("polarization must be HH, VV, or HV") from error

    @staticmethod
    def sea_clutter_sigma0(
        grazing_angle_rad: float,
        sea_state: int = 3,
        frequency_ghz: float = 10.0,
        polarization: str = "HH",
    ) -> float:
        """
        Sea clutter backscatter coefficient using the NRL five-parameter model.

        Args:
            grazing_angle_rad: Grazing angle [rad]
            sea_state: Douglas sea state (0-6)
            frequency_ghz: Radar frequency [GHz]
            polarization: 'HH' or 'VV'

        Returns:
            σ0 in dB

        Valid for 0.5-35 GHz, 0.1-60 degree grazing, and sea states 0-6.

        Reference: Gregers-Hansen and Mital, IEEE TAES, 2012.
        """
        grazing_angle_deg = float(np.degrees(grazing_angle_rad))
        if not 0.1 <= grazing_angle_deg <= 60.0:
            raise ValueError("NRL sea clutter model requires 0.1-60 degree grazing")
        if not 0.5 <= frequency_ghz <= 35.0:
            raise ValueError("NRL sea clutter model requires 0.5-35 GHz")
        if not 0 <= sea_state <= 6:
            raise ValueError("sea_state must be between 0 and 6")

        coefficients = {
            "HH": (-73.00, 20.781, 7.351, 25.65, 0.00540),
            "VV": (-50.796, 25.93, 0.7093, 21.588, 0.00211),
        }
        try:
            c1, c2, c3, c4, c5 = coefficients[polarization.upper()]
        except KeyError as error:
            raise ValueError("polarization must be 'HH' or 'VV'") from error

        frequency_term = (
            (27.5 + c3 * grazing_angle_deg)
            * np.log10(frequency_ghz)
            / (1.0 + 0.95 * grazing_angle_deg)
        )
        sea_state_term = c4 * (1.0 + sea_state) ** (
            1.0 / (2.0 + 0.085 * grazing_angle_deg + 0.033 * sea_state)
        )
        return float(
            c1
            + c2 * np.log10(np.sin(grazing_angle_rad))
            + frequency_term
            + sea_state_term
            + c5 * grazing_angle_deg**2
        )

    @staticmethod
    def surface_resolution_cell_area(
        range_m: float,
        pulse_width_s: float,
        azimuth_beamwidth_rad: float,
        elevation_beamwidth_rad: float,
        grazing_angle_rad: float,
    ) -> float:
        """Approximate -3 dB surface area shared by range gate and antenna beam."""
        positive = (
            range_m,
            pulse_width_s,
            azimuth_beamwidth_rad,
            elevation_beamwidth_rad,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("range, pulse width, and beamwidths must be positive")
        if not 0.0 < grazing_angle_rad < np.pi / 2.0:
            raise ValueError("grazing_angle_rad must be between 0 and pi/2")

        range_gate_m = SPEED_OF_LIGHT * pulse_width_s / 2.0
        ground_range_extent = range_gate_m / np.cos(grazing_angle_rad)
        elevation_limited_extent = (
            2.0
            * range_m
            * np.tan(elevation_beamwidth_rad / 2.0)
            / np.sin(grazing_angle_rad)
        )
        cross_range_extent = 2.0 * range_m * np.tan(azimuth_beamwidth_rad / 2.0)
        return float(
            np.pi
            / 4.0
            * cross_range_extent
            * min(ground_range_extent, elevation_limited_extent)
        )

    @staticmethod
    def signal_to_noise_plus_clutter_db(
        snr_db: float, target_rcs_m2: float, clutter_rcs_m2: float
    ) -> float:
        """Combine thermal-noise SNR with co-range clutter using echo-power ratios."""
        if target_rcs_m2 <= 0.0:
            raise ValueError("target_rcs_m2 must be positive")
        if clutter_rcs_m2 < 0.0:
            raise ValueError("clutter_rcs_m2 cannot be negative")
        snr_linear = 10.0 ** (snr_db / 10.0)
        inverse_sinr = 1.0 / snr_linear + clutter_rcs_m2 / target_rcs_m2
        return float(-10.0 * np.log10(inverse_sinr))

    @staticmethod
    def ground_clutter_weibull(
        sigma0_db: float, cell_area_m2: float, shape: float = 2.0, size: int = 1
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

        scale = mean_rcs / gamma_func(1 + 1 / shape)

        return ClutterModel._weibull_samples_jit(shape, scale, size)

    @staticmethod
    def sea_clutter_k_distribution(
        sigma0_db: float, cell_area_m2: float, sea_state: int = 3, size: int = 1
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
        rain_rate_mm_hr: float, frequency_ghz: float
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
        Z = 200 * (rain_rate_mm_hr**1.6)  # mm^6/m^3

        # Convert to radar reflectivity
        wavelength_m = SPEED_OF_LIGHT / (frequency_ghz * 1e9)
        K_squared = 0.93  # for water at radar frequencies

        # η = π^5 * |K|^2 * Z / λ^4
        eta = (np.pi**5) * K_squared * Z * 1e-18 / (wavelength_m**4)

        return eta

    @staticmethod
    def volume_clutter_rcs(
        eta: float, range_m: float, beamwidth_rad: float, pulse_width_s: float
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
        volume = (np.pi / 4) * (range_m**2) * (beamwidth_rad**2) * range_resolution

        return eta * volume

    @staticmethod
    def generate_clutter_map(
        range_bins: int,
        azimuth_bins: int,
        max_range_m: float,
        terrain_type: str = "rural",
        frequency_ghz: float = 10.0,
        radar_altitude_m: float = 0.0,
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
        # Create range array
        ranges = np.linspace(100, max_range_m, range_bins)

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
