"""ITU-R P.676-13 line-by-line gaseous attenuation."""

from typing import Tuple

import numpy as np

from .constants import STANDARD_PRESSURE, STANDARD_WATER_VAPOR_DENSITY


_OXYGEN_LINES = np.array(
    [
        [50.474214, 0.975, 9.651, 6.690, 0.0, 2.566, 6.850],
        [50.987745, 2.529, 8.653, 7.170, 0.0, 2.246, 6.800],
        [51.503360, 6.193, 7.709, 7.640, 0.0, 1.947, 6.729],
        [52.021429, 14.320, 6.819, 8.110, 0.0, 1.667, 6.640],
        [52.542418, 31.240, 5.983, 8.580, 0.0, 1.388, 6.526],
        [53.066934, 64.290, 5.201, 9.060, 0.0, 1.349, 6.206],
        [53.595775, 124.600, 4.474, 9.550, 0.0, 2.227, 5.085],
        [54.130025, 227.300, 3.800, 9.960, 0.0, 3.170, 3.750],
        [54.671180, 389.700, 3.182, 10.370, 0.0, 3.558, 2.654],
        [55.221384, 627.100, 2.618, 10.890, 0.0, 2.560, 2.952],
        [55.783815, 945.300, 2.109, 11.340, 0.0, -1.172, 6.135],
        [56.264774, 543.400, 0.014, 17.030, 0.0, 3.525, -0.978],
        [56.363399, 1331.800, 1.654, 11.890, 0.0, -2.378, 6.547],
        [56.968211, 1746.600, 1.255, 12.230, 0.0, -3.545, 6.451],
        [57.612486, 2120.100, 0.910, 12.620, 0.0, -5.416, 6.056],
        [58.323877, 2363.700, 0.621, 12.950, 0.0, -1.932, 0.436],
        [58.446588, 1442.100, 0.083, 14.910, 0.0, 6.768, -1.273],
        [59.164204, 2379.900, 0.387, 13.530, 0.0, -6.561, 2.309],
        [59.590983, 2090.700, 0.207, 14.080, 0.0, 6.957, -0.776],
        [60.306056, 2103.400, 0.207, 14.150, 0.0, -6.395, 0.699],
        [60.434778, 2438.000, 0.386, 13.390, 0.0, 6.342, -2.825],
        [61.150562, 2479.500, 0.621, 12.920, 0.0, 1.014, -0.584],
        [61.800158, 2275.900, 0.910, 12.630, 0.0, 5.014, -6.619],
        [62.411220, 1915.400, 1.255, 12.170, 0.0, 3.029, -6.759],
        [62.486253, 1503.000, 0.083, 15.130, 0.0, -4.499, 0.844],
        [62.997984, 1490.200, 1.654, 11.740, 0.0, 1.856, -6.675],
        [63.568526, 1078.000, 2.108, 11.340, 0.0, 0.658, -6.139],
        [64.127775, 728.700, 2.617, 10.880, 0.0, -3.036, -2.895],
        [64.678910, 461.300, 3.181, 10.380, 0.0, -3.968, -2.590],
        [65.224078, 274.000, 3.800, 9.960, 0.0, -3.528, -3.680],
        [65.764779, 153.000, 4.473, 9.550, 0.0, -2.548, -5.002],
        [66.302096, 80.400, 5.200, 9.060, 0.0, -1.660, -6.091],
        [66.836834, 39.800, 5.982, 8.580, 0.0, -1.680, -6.393],
        [67.369601, 18.560, 6.818, 8.110, 0.0, -1.956, -6.475],
        [67.900868, 8.172, 7.708, 7.640, 0.0, -2.216, -6.545],
        [68.431006, 3.397, 8.652, 7.170, 0.0, -2.492, -6.600],
        [68.960312, 1.334, 9.650, 6.690, 0.0, -2.773, -6.650],
        [118.750334, 940.300, 0.010, 16.640, 0.0, -0.439, 0.079],
        [368.498246, 67.400, 0.048, 16.400, 0.0, 0.000, 0.000],
        [424.763020, 637.700, 0.044, 16.400, 0.0, 0.000, 0.000],
        [487.249273, 237.400, 0.049, 16.000, 0.0, 0.000, 0.000],
        [715.392902, 98.100, 0.145, 16.000, 0.0, 0.000, 0.000],
        [773.839490, 572.300, 0.141, 16.200, 0.0, 0.000, 0.000],
        [834.145546, 183.100, 0.145, 14.700, 0.0, 0.000, 0.000],
    ],
    dtype=np.float64,
)


_WATER_VAPOR_LINES = np.array(
    [
        [22.235080, 0.1079, 2.144, 26.38, 0.76, 5.087, 1.00],
        [67.803960, 0.0011, 8.732, 28.58, 0.69, 4.930, 0.82],
        [119.995940, 0.0007, 8.353, 29.48, 0.70, 4.780, 0.79],
        [183.310087, 2.2730, 0.668, 29.06, 0.77, 5.022, 0.85],
        [321.225630, 0.0470, 6.179, 24.04, 0.67, 4.398, 0.54],
        [325.152888, 1.5140, 1.541, 28.23, 0.64, 4.893, 0.74],
        [336.227764, 0.0010, 9.825, 26.93, 0.69, 4.740, 0.61],
        [380.197353, 11.670, 1.048, 28.11, 0.54, 5.063, 0.89],
        [390.134508, 0.0045, 7.347, 21.52, 0.63, 4.810, 0.55],
        [437.346667, 0.0632, 5.048, 18.45, 0.60, 4.230, 0.48],
        [439.150807, 0.9098, 3.595, 20.07, 0.63, 4.483, 0.52],
        [443.018343, 0.1920, 5.048, 15.55, 0.60, 5.083, 0.50],
        [448.001085, 10.410, 1.405, 25.64, 0.66, 5.028, 0.67],
        [470.888999, 0.3254, 3.597, 21.34, 0.66, 4.506, 0.65],
        [474.689092, 1.2600, 2.379, 23.20, 0.65, 4.804, 0.64],
        [488.490108, 0.2529, 2.852, 25.86, 0.69, 5.201, 0.72],
        [503.568532, 0.0372, 6.731, 16.12, 0.61, 3.980, 0.43],
        [504.482692, 0.0124, 6.731, 16.12, 0.61, 4.010, 0.45],
        [547.676440, 0.9785, 0.158, 26.00, 0.70, 4.500, 1.00],
        [552.020960, 0.1840, 0.158, 26.00, 0.70, 4.500, 1.00],
        [556.935985, 497.00, 0.159, 30.86, 0.69, 4.552, 1.00],
        [620.700807, 5.0150, 2.391, 24.38, 0.71, 4.856, 0.68],
        [645.766085, 0.0067, 8.633, 18.00, 0.60, 4.000, 0.50],
        [658.005280, 0.2732, 7.816, 32.10, 0.69, 4.140, 1.00],
        [752.033113, 243.40, 0.396, 30.86, 0.68, 4.352, 0.84],
        [841.051732, 0.0134, 8.177, 15.90, 0.33, 5.760, 0.45],
        [859.965698, 0.1325, 8.055, 30.60, 0.68, 4.090, 0.84],
        [899.303175, 0.0547, 7.914, 29.85, 0.68, 4.530, 0.90],
        [902.611085, 0.0386, 8.429, 28.65, 0.70, 5.100, 0.95],
        [906.205957, 0.1836, 5.110, 24.08, 0.70, 4.700, 0.53],
        [916.171582, 8.4000, 1.441, 26.73, 0.70, 5.150, 0.78],
        [923.112692, 0.0079, 10.293, 29.00, 0.70, 5.000, 0.80],
        [970.315022, 9.0090, 1.919, 25.50, 0.64, 4.940, 0.67],
        [987.926764, 134.60, 0.257, 29.85, 0.68, 4.550, 0.90],
        [1780.00000, 17506.0, 0.952, 196.30, 2.00, 24.150, 5.00],
    ],
    dtype=np.float64,
)


def _line_shape(
    frequency_ghz: float, line_frequency: np.ndarray, width: np.ndarray, delta
):
    lower = (width - delta * (line_frequency - frequency_ghz)) / (
        (line_frequency - frequency_ghz) ** 2 + width**2
    )
    upper = (width - delta * (line_frequency + frequency_ghz)) / (
        (line_frequency + frequency_ghz) ** 2 + width**2
    )
    return frequency_ghz / line_frequency * (lower + upper)


class ITU_R_P676:
    """Specific gaseous attenuation from ITU-R P.676-13 Annex 1."""

    @staticmethod
    def _atmospheric_state(
        temperature_c: float,
        pressure_hpa: float,
        water_vapor_density: float,
    ) -> Tuple[float, float, float]:
        temperature_k = temperature_c + 273.15
        if temperature_k <= 0.0:
            raise ValueError("temperature must be above absolute zero")
        if pressure_hpa <= 0.0:
            raise ValueError("pressure_hpa must be greater than zero")
        if water_vapor_density < 0.0:
            raise ValueError("water_vapor_density cannot be negative")

        vapor_pressure = water_vapor_density * temperature_k / 216.7
        return temperature_k, pressure_hpa, vapor_pressure

    @classmethod
    def specific_attenuation_oxygen(
        cls,
        frequency_ghz: float,
        temperature_c: float = 15.0,
        pressure_hpa: float = STANDARD_PRESSURE,
        water_vapor_density: float = STANDARD_WATER_VAPOR_DENSITY,
    ) -> float:
        if not 1.0 <= frequency_ghz <= 1000.0:
            raise ValueError("P.676-13 line-by-line model is valid from 1 to 1000 GHz")

        temperature_k, dry_pressure, vapor_pressure = cls._atmospheric_state(
            temperature_c, pressure_hpa, water_vapor_density
        )
        theta = 300.0 / temperature_k
        f0, a1, a2, a3, a4, a5, a6 = _OXYGEN_LINES.T
        strength = a1 * 1e-7 * dry_pressure * theta**3 * np.exp(a2 * (1.0 - theta))
        width = (
            a3
            * 1e-4
            * (dry_pressure * theta ** (0.8 - a4) + 1.1 * vapor_pressure * theta)
        )
        width = np.sqrt(width**2 + 2.25e-6)
        delta = (a5 + a6 * theta) * 1e-4 * (dry_pressure + vapor_pressure) * theta**0.8
        line_refractivity = np.sum(
            strength * _line_shape(frequency_ghz, f0, width, delta)
        )

        debye_width = 5.6e-4 * (dry_pressure + vapor_pressure) * theta**0.8
        continuum = (
            frequency_ghz
            * dry_pressure
            * theta**2
            * (
                6.14e-5 / (debye_width * (1.0 + (frequency_ghz / debye_width) ** 2))
                + 1.4e-12
                * dry_pressure
                * theta**1.5
                / (1.0 + 1.9e-5 * frequency_ghz**1.5)
            )
        )
        return float(0.1820 * frequency_ghz * (line_refractivity + continuum))

    @classmethod
    def specific_attenuation_water_vapor(
        cls,
        frequency_ghz: float,
        temperature_c: float = 15.0,
        water_vapor_density: float = STANDARD_WATER_VAPOR_DENSITY,
        pressure_hpa: float = STANDARD_PRESSURE,
    ) -> float:
        if not 1.0 <= frequency_ghz <= 1000.0:
            raise ValueError("P.676-13 line-by-line model is valid from 1 to 1000 GHz")

        temperature_k, dry_pressure, vapor_pressure = cls._atmospheric_state(
            temperature_c, pressure_hpa, water_vapor_density
        )
        if vapor_pressure == 0.0:
            return 0.0

        theta = 300.0 / temperature_k
        f0, b1, b2, b3, b4, b5, b6 = _WATER_VAPOR_LINES.T
        strength = b1 * 1e-1 * vapor_pressure * theta**3.5 * np.exp(b2 * (1.0 - theta))
        pressure_width = (
            b3 * 1e-4 * (dry_pressure * theta**b4 + b5 * vapor_pressure * theta**b6)
        )
        width = 0.535 * pressure_width + np.sqrt(
            0.217 * pressure_width**2 + 2.1316e-12 * f0**2 / theta
        )
        line_refractivity = np.sum(
            strength * _line_shape(frequency_ghz, f0, width, 0.0)
        )
        return float(0.1820 * frequency_ghz * line_refractivity)

    @classmethod
    def total_attenuation(
        cls,
        range_km: float,
        frequency_ghz: float,
        temperature_c: float = 15.0,
        pressure_hpa: float = STANDARD_PRESSURE,
        water_vapor_density: float = STANDARD_WATER_VAPOR_DENSITY,
        two_way: bool = True,
    ) -> float:
        if range_km < 0.0:
            raise ValueError("range_km cannot be negative")
        _, _, total = cls.get_attenuation_components(
            range_km,
            frequency_ghz,
            temperature_c,
            pressure_hpa,
            water_vapor_density,
            two_way,
        )
        return total

    @classmethod
    def get_attenuation_components(
        cls,
        range_km: float,
        frequency_ghz: float,
        temperature_c: float = 15.0,
        pressure_hpa: float = STANDARD_PRESSURE,
        water_vapor_density: float = STANDARD_WATER_VAPOR_DENSITY,
        two_way: bool = True,
    ) -> Tuple[float, float, float]:
        if range_km < 0.0:
            raise ValueError("range_km cannot be negative")
        oxygen_specific = cls.specific_attenuation_oxygen(
            frequency_ghz,
            temperature_c,
            pressure_hpa,
            water_vapor_density,
        )
        water_specific = cls.specific_attenuation_water_vapor(
            frequency_ghz,
            temperature_c,
            water_vapor_density,
            pressure_hpa,
        )
        path_multiplier = 2.0 if two_way else 1.0
        oxygen = oxygen_specific * range_km * path_multiplier
        water = water_specific * range_km * path_multiplier
        return oxygen, water, oxygen + water


def validate_itu_60ghz() -> dict:
    gamma_o = ITU_R_P676.specific_attenuation_oxygen(60.0)
    expected_gamma = 14.6234747964861
    tolerance = 1e-9
    return {
        "test_parameters": {
            "frequency_ghz": 60.0,
            "temperature_c": 15.0,
            "pressure_hpa": STANDARD_PRESSURE,
        },
        "computed_values": {"gamma_oxygen_dB_per_km": gamma_o},
        "expected_values": {
            "gamma_oxygen_dB_per_km": expected_gamma,
            "tolerance_dB_per_km": tolerance,
        },
        "validation": {
            "is_valid": abs(gamma_o - expected_gamma) <= tolerance,
            "error_dB_per_km": abs(gamma_o - expected_gamma),
            "reference": "ITU-R P.676-13 Annex 1",
        },
    }


def validate_itu_xband() -> dict:
    gamma_o = ITU_R_P676.specific_attenuation_oxygen(10.0)
    gamma_w = ITU_R_P676.specific_attenuation_water_vapor(10.0)
    gamma_total = gamma_o + gamma_w
    max_oxygen = 0.02
    return {
        "test_parameters": {
            "frequency_ghz": 10.0,
            "temperature_c": 15.0,
            "pressure_hpa": STANDARD_PRESSURE,
            "water_vapor_gpm3": STANDARD_WATER_VAPOR_DENSITY,
        },
        "computed_values": {
            "gamma_oxygen_dB_per_km": gamma_o,
            "gamma_water_dB_per_km": gamma_w,
            "gamma_total_dB_per_km": gamma_total,
        },
        "expected_values": {
            "max_oxygen_dB_per_km": max_oxygen,
        },
        "validation": {
            "is_valid": gamma_o < max_oxygen,
            "oxygen_below_limit": gamma_o < max_oxygen,
            "reference": "ITU-R P.676-13 Annex 1",
        },
    }
