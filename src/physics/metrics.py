"""
Radar Performance Metrics

Provides statistical calculations for radar detection performance analysis.

Includes:
    - Albersheim's equation for Pd calculation
    - Shnidman's approximation for Swerling targets
    - Maximum range calculations
    - ROC curve generation

References:
    - Albersheim, W.J., "A Closed-Form Approximation to Robertson's
      Detection Characteristics", IEEE Trans. AES, 1981
    - Shnidman, D.A., "Determination of Required SNR Values",
      IEEE Trans. AES, 2002
    - Skolnik, "Radar Handbook", 3rd Ed., Chapter 2
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy import special, stats


@dataclass
class DetectionMetrics:
    """Container for detection performance metrics."""

    pd: float  # Probability of detection
    pfa: float  # Probability of false alarm
    snr_db: float  # Required/actual SNR
    max_range_km: float  # Maximum detection range
    rcs_m2: float  # Target RCS


def albersheim_snr(pd: float, pfa: float, n_pulses: int = 1) -> float:
    """
    Albersheim's equation: SNR required for given Pd and Pfa.

    This closed-form approximation is valid for:
        - 0.1 < Pd < 0.9999
        - 1e-10 < Pfa < 1e-3
        - 1 ≤ n_pulses ≤ 8096

    Args:
        pd: Probability of detection (0-1)
        pfa: Probability of false alarm (0-1)
        n_pulses: Number of pulses integrated

    Returns:
        Required SNR in dB

    Reference: Albersheim, IEEE Trans. AES, 1981
    """
    if not 0.0 < pd < 1.0:
        raise ValueError("pd must be between 0 and 1")
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be between 0 and 1")
    if isinstance(n_pulses, bool) or not isinstance(n_pulses, (int, np.integer)):
        raise TypeError("n_pulses must be an integer")
    if n_pulses < 1:
        raise ValueError("n_pulses must be at least 1")

    a = np.log(0.62 / pfa)
    b = np.log(pd / (1.0 - pd))
    argument = a + 0.12 * a * b + 1.7 * b
    if argument <= 0.0:
        raise ValueError("pd and pfa are outside the usable Albersheim domain")

    integration_factor = 6.2 + 4.54 / np.sqrt(n_pulses + 0.44)
    return float(-5.0 * np.log10(n_pulses) + integration_factor * np.log10(argument))


def _fluctuating_target_pd(
    snr_linear: float,
    threshold: float,
    n_pulses: int,
    gamma_shape: float,
    quadrature_order: int = 48,
) -> float:
    if gamma_shape >= 64.0:
        return float(
            stats.ncx2.sf(
                2.0 * threshold,
                2 * n_pulses,
                2.0 * n_pulses * snr_linear,
            )
        )

    nodes, weights = special.roots_genlaguerre(quadrature_order, gamma_shape - 1.0)
    rcs_scale = nodes / gamma_shape
    conditional_pd = stats.ncx2.sf(
        2.0 * threshold,
        2 * n_pulses,
        2.0 * n_pulses * snr_linear * rcs_scale,
    )
    return float(np.dot(weights, conditional_pd) / special.gamma(gamma_shape))


def calculate_pd_swerling(
    snr_db: float,
    pfa: float = 1e-6,
    swerling_case: int = 1,
    n_pulses: int = 1,
) -> float:
    """Return square-law detection probability for Swerling cases 0 through 4.

    ``snr_db`` is the mean single-pulse signal-to-noise ratio. Integrated
    noise power is gamma distributed. Conditional target power is evaluated
    with the non-central chi-square survival function; gamma-distributed RCS
    is averaged by generalized Gauss-Laguerre quadrature.
    """
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be between 0 and 1")
    if not np.isfinite(snr_db):
        return float(pfa) if snr_db < 0.0 else 1.0
    if isinstance(n_pulses, bool) or not isinstance(n_pulses, (int, np.integer)):
        raise TypeError("n_pulses must be an integer")
    if n_pulses < 1:
        raise ValueError("n_pulses must be at least 1")
    if swerling_case not in (0, 1, 2, 3, 4):
        raise ValueError("swerling_case must be one of 0, 1, 2, 3, or 4")

    snr_linear = 10.0 ** (snr_db / 10.0)
    threshold = float(special.gammainccinv(n_pulses, pfa))

    if swerling_case == 0:
        pd = stats.ncx2.sf(
            2.0 * threshold,
            2 * n_pulses,
            2.0 * n_pulses * snr_linear,
        )
    else:
        gamma_shape = 1.0 if swerling_case in (1, 2) else 2.0
        if swerling_case in (2, 4):
            gamma_shape *= n_pulses
        pd = _fluctuating_target_pd(
            snr_linear,
            threshold,
            n_pulses,
            gamma_shape,
        )

    return float(np.clip(pd, 0.0, 1.0))


def calculate_pd_vs_range(
    ranges_km: np.ndarray,
    radar_power_w: float,
    antenna_gain_db: float,
    wavelength_m: float,
    rcs_m2: float,
    noise_figure_db: float = 5.0,
    bandwidth_hz: float = 1e6,
    pfa: float = 1e-6,
    swerling_case: int = 1,
    losses_db: float = 10.0,
) -> np.ndarray:
    """
    Calculate Pd vs Range curve.

    Uses the radar equation to compute SNR at each range,
    then converts to Pd using Swerling model.

    Args:
        ranges_km: Array of ranges [km]
        radar_power_w: Transmit power [W]
        antenna_gain_db: Antenna gain [dB]
        wavelength_m: Wavelength [m]
        rcs_m2: Target RCS [m²]
        noise_figure_db: Receiver noise figure [dB]
        bandwidth_hz: Receiver bandwidth [Hz]
        pfa: False alarm probability
        swerling_case: Swerling fluctuation model
        losses_db: System losses [dB]

    Returns:
        Array of Pd values
    """
    ranges_m = ranges_km * 1000

    # Constants
    k_boltzmann = 1.38e-23  # J/K
    T0 = 290  # K (standard temperature)

    # Convert to linear
    antenna_gain = 10 ** (antenna_gain_db / 10)
    noise_figure = 10 ** (noise_figure_db / 10)
    losses = 10 ** (losses_db / 10)

    # Noise power
    noise_power = k_boltzmann * T0 * bandwidth_hz * noise_figure

    # Received power (radar equation)
    numerator = radar_power_w * (antenna_gain**2) * (wavelength_m**2) * rcs_m2
    denominator = ((4 * np.pi) ** 3) * (ranges_m**4) * losses

    received_power = numerator / denominator

    # SNR
    snr_linear = received_power / noise_power
    snr_db = 10 * np.log10(np.maximum(snr_linear, 1e-10))

    # Calculate Pd for each range
    pd_values = np.array(
        [calculate_pd_swerling(snr, pfa, swerling_case) for snr in snr_db]
    )

    return pd_values


def generate_roc_curves(
    snr_values_db: List[float] = [5, 10, 13, 15, 20],
    pfa_range: Tuple[float, float] = (1e-10, 1e-2),
    n_points: int = 100,
    swerling_case: int = 1,
) -> dict:
    """
    Generate ROC curves for multiple SNR values.

    Args:
        snr_values_db: List of SNR values to plot
        pfa_range: (min, max) Pfa range
        n_points: Number of points per curve
        swerling_case: Swerling fluctuation model

    Returns:
        Dict with 'pfa' array and 'pd' dict (keyed by SNR)
    """
    pfa_values = np.logspace(np.log10(pfa_range[0]), np.log10(pfa_range[1]), n_points)

    result = {"pfa": pfa_values, "pd": {}}

    for snr in snr_values_db:
        pd_values = np.array(
            [calculate_pd_swerling(snr, pfa, swerling_case) for pfa in pfa_values]
        )
        result["pd"][snr] = pd_values

    return result


def calculate_max_range(
    pd: float,
    pfa: float,
    rcs_m2: float,
    radar_power_w: float,
    antenna_gain_db: float,
    frequency_hz: float,
    noise_figure_db: float = 5.0,
    bandwidth_hz: float = 1e6,
    losses_db: float = 10.0,
    swerling_case: int = 1,
) -> float:
    """
    Calculate maximum detection range for given Pd requirement.

    Inverts the radar equation using binary search.

    Args:
        pd: Required probability of detection
        pfa: Probability of false alarm
        rcs_m2: Target RCS [m²]
        radar_power_w: Transmit power [W]
        antenna_gain_db: Antenna gain [dB]
        frequency_hz: Radar frequency [Hz]
        noise_figure_db: Noise figure [dB]
        bandwidth_hz: Receiver bandwidth [Hz]
        losses_db: System losses [dB]
        swerling_case: Swerling model

    Returns:
        Maximum range in km
    """
    wavelength_m = 3e8 / frequency_hz

    # Binary search for max range
    r_min, r_max = 1.0, 1000.0  # km

    for _ in range(50):  # 50 iterations for convergence
        r_mid = (r_min + r_max) / 2

        pd_at_range = calculate_pd_vs_range(
            np.array([r_mid]),
            radar_power_w,
            antenna_gain_db,
            wavelength_m,
            rcs_m2,
            noise_figure_db,
            bandwidth_hz,
            pfa,
            swerling_case,
            losses_db,
        )[0]

        if pd_at_range > pd:
            r_min = r_mid
        else:
            r_max = r_mid

        if r_max - r_min < 0.1:
            break

    return r_mid


if __name__ == "__main__":
    # Quick test
    print("Radar Metrics Test")
    print("=" * 50)

    # Test Albersheim
    snr = albersheim_snr(pd=0.9, pfa=1e-6)
    print(f"Required SNR for Pd=0.9, Pfa=1e-6: {snr:.1f} dB")

    # Test Pd calculation
    pd = calculate_pd_swerling(snr_db=15, pfa=1e-6, swerling_case=1)
    print(f"Pd at SNR=15dB, Pfa=1e-6, Swerling I: {pd:.3f}")

    # Generate ROC
    roc = generate_roc_curves()
    print(f"\nROC curves generated for SNRs: {list(roc['pd'].keys())} dB")
