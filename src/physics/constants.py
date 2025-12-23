"""
Physical Constants for Radar Simulation

All constants are in SI units as per IEEE Std 686-2008 and CODATA 2018.

References:
    - CODATA 2018: Fundamental Physical Constants
    - ITU-R P.676-12: Attenuation by atmospheric gases
    - IEEE Std 686-2008: Standard Radar Definitions
"""

from typing import Final

# =============================================================================
# FUNDAMENTAL CONSTANTS (CODATA 2018 - Exact definitions)
# =============================================================================

SPEED_OF_LIGHT: Final[float] = 299_792_458.0
"""Speed of light in vacuum [m/s] - Exact SI definition"""

BOLTZMANN_CONSTANT: Final[float] = 1.380649e-23
"""Boltzmann constant [J/K] - Exact SI definition (2019 redefinition)"""

PLANCK_CONSTANT: Final[float] = 6.62607015e-34
"""Planck constant [J·s] - Exact SI definition"""

# =============================================================================
# ATMOSPHERIC/ENVIRONMENTAL CONSTANTS (ITU-R P.676-12)
# =============================================================================

STANDARD_TEMPERATURE: Final[float] = 290.0
"""Standard reference temperature [K] - IEEE noise figure reference"""

STANDARD_PRESSURE: Final[float] = 1013.25
"""Standard sea-level atmospheric pressure [hPa]"""

STANDARD_WATER_VAPOR_DENSITY: Final[float] = 7.5
"""Standard water vapor density [g/m³] - ITU-R P.676"""

# =============================================================================
# EARTH PARAMETERS
# =============================================================================

EARTH_RADIUS: Final[float] = 6_371_000.0
"""Mean Earth radius [m] - WGS84 mean radius"""

EARTH_RADIUS_EFFECTIVE: Final[float] = 6_371_000.0 * (4.0 / 3.0)
"""Effective Earth radius for 4/3 Earth model [m] - Standard radar refraction"""

# =============================================================================
# RADAR-SPECIFIC CONSTANTS
# =============================================================================

RADAR_CONSTANT_4PI_CUBED: Final[float] = (4.0 * 3.141592653589793) ** 3
"""(4π)³ constant used in radar equation denominator"""

DB_TO_LINEAR_FACTOR: Final[float] = 10.0
"""Factor for dB to linear power conversion: P_linear = 10^(P_dB/10)"""

# =============================================================================
# NATO SYMBOLOGY COLORS (MIL-STD-2525D)
# =============================================================================

class NATOColors:
    """NATO APP-6 / MIL-STD-2525D standard symbology colors
    
    Reference: NATO STANAG 2019 Edition
    """
    FRIENDLY: Final[str] = "#00BFFF"  # Cyan/Light Blue
    HOSTILE: Final[str] = "#FF4444"   # Red
    NEUTRAL: Final[str] = "#00FF00"   # Green
    UNKNOWN: Final[str] = "#FFFF00"   # Yellow
    
    # UI Theme Colors (Dark Aerospace)
    BACKGROUND: Final[str] = "#111111"
    GRID: Final[str] = "#333333"
    TEXT_PRIMARY: Final[str] = "#E0E0E0"
    TEXT_SECONDARY: Final[str] = "#888899"
    ACCENT_CYAN: Final[str] = "#00D4FF"
    ACCENT_AMBER: Final[str] = "#FFCC00"
    ACCENT_LIME: Final[str] = "#00FF88"
