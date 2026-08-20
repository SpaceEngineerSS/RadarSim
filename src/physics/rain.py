"""Rain-specific attenuation from Recommendation ITU-R P.838-3."""

from typing import Tuple

import numpy as np


_KH = np.array(
    [
        (-5.33980, -0.10008, 1.13098),
        (-0.35351, 1.26970, 0.45400),
        (-0.23789, 0.86036, 0.15354),
        (-0.94158, 0.64552, 0.16817),
    ]
)
_KV = np.array(
    [
        (-3.80595, 0.56934, 0.81061),
        (-3.44965, -0.22911, 0.51059),
        (-0.39902, 0.73042, 0.11899),
        (0.50167, 1.07319, 0.27195),
    ]
)
_ALPHA_H = np.array(
    [
        (-0.14318, 1.82442, -0.55187),
        (0.29591, 0.77564, 0.19822),
        (0.32177, 0.63773, 0.13164),
        (-5.37610, -0.96230, 1.47828),
        (16.1721, -3.29980, 3.43990),
    ]
)
_ALPHA_V = np.array(
    [
        (-0.07771, 2.33840, -0.76284),
        (0.56727, 0.95545, 0.54039),
        (-0.20238, 1.14520, 0.26809),
        (-48.2991, 0.791669, 0.116226),
        (48.5833, 0.791459, 0.116479),
    ]
)


def _coefficient(
    frequency_ghz: float,
    terms: np.ndarray,
    slope: float,
    intercept: float,
    logarithmic_output: bool,
) -> float:
    log_frequency = np.log10(frequency_ghz)
    a, b, c = terms.T
    value = np.sum(a * np.exp(-(((log_frequency - b) / c) ** 2)))
    value += slope * log_frequency + intercept
    return float(10.0**value if logarithmic_output else value)


class ITU_R_P838:
    """P.838-3 power-law coefficients and specific rain attenuation."""

    @staticmethod
    def polarization_coefficients(
        frequency_ghz: float,
        elevation_angle_deg: float = 0.0,
        polarization_tilt_deg: float = 0.0,
    ) -> Tuple[float, float]:
        if not 1.0 <= frequency_ghz <= 1000.0:
            raise ValueError("P.838-3 is valid from 1 to 1000 GHz")
        if not -90.0 <= elevation_angle_deg <= 90.0:
            raise ValueError("elevation_angle_deg must be between -90 and 90")

        k_h = _coefficient(frequency_ghz, _KH, -0.18961, 0.71147, True)
        k_v = _coefficient(frequency_ghz, _KV, -0.16398, 0.63297, True)
        alpha_h = _coefficient(frequency_ghz, _ALPHA_H, 0.67849, -1.95537, False)
        alpha_v = _coefficient(frequency_ghz, _ALPHA_V, -0.053739, 0.83433, False)

        elevation = np.radians(elevation_angle_deg)
        tilt = np.radians(polarization_tilt_deg)
        geometry = np.cos(elevation) ** 2 * np.cos(2.0 * tilt)
        k = 0.5 * (k_h + k_v + (k_h - k_v) * geometry)
        alpha = (
            0.5
            * (
                k_h * alpha_h
                + k_v * alpha_v
                + (k_h * alpha_h - k_v * alpha_v) * geometry
            )
            / k
        )
        return float(k), float(alpha)

    @classmethod
    def specific_attenuation(
        cls,
        frequency_ghz: float,
        rain_rate_mm_hr: float,
        elevation_angle_deg: float = 0.0,
        polarization_tilt_deg: float = 0.0,
    ) -> float:
        if rain_rate_mm_hr < 0.0:
            raise ValueError("rain_rate_mm_hr cannot be negative")
        if rain_rate_mm_hr == 0.0:
            return 0.0
        k, alpha = cls.polarization_coefficients(
            frequency_ghz, elevation_angle_deg, polarization_tilt_deg
        )
        return float(k * rain_rate_mm_hr**alpha)

    @classmethod
    def path_attenuation(
        cls,
        path_length_km: float,
        frequency_ghz: float,
        rain_rate_mm_hr: float,
        elevation_angle_deg: float = 0.0,
        polarization_tilt_deg: float = 0.0,
        two_way: bool = True,
    ) -> float:
        if path_length_km < 0.0:
            raise ValueError("path_length_km cannot be negative")
        specific = cls.specific_attenuation(
            frequency_ghz,
            rain_rate_mm_hr,
            elevation_angle_deg,
            polarization_tilt_deg,
        )
        return specific * path_length_km * (2.0 if two_way else 1.0)
