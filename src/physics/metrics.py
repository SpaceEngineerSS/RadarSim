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

import numpy as np
from typing import Tuple, List, Optional
from scipy import special
from dataclasses import dataclass


@dataclass
class DetectionMetrics:
    """Container for detection performance metrics."""
    pd: float           # Probability of detection
    pfa: float          # Probability of false alarm
    snr_db: float       # Required/actual SNR
    max_range_km: float # Maximum detection range
    rcs_m2: float       # Target RCS


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
    # Clamp inputs to valid range
    pd = np.clip(pd, 0.01, 0.9999)
    pfa = np.clip(pfa, 1e-12, 0.1)
    
    # Albersheim's approximation
    A = np.log(0.62 / pfa)
    B = np.log(pd / (1 - pd))
    
    # Single pulse SNR
    snr_1 = A + 0.12 * A * B + 1.7 * B
    
    # Integration gain (non-coherent)
    if n_pulses > 1:
        snr_n = snr_1 - 6.2 + 4.54 * np.sqrt(n_pulses + 0.44)
        snr_db = snr_n / n_pulses
    else:
        snr_db = snr_1
    
    return snr_db


def calculate_pd_swerling(
    snr_db: float,
    pfa: float = 1e-6,
    swerling_case: int = 1,
    n_pulses: int = 1
) -> float:
    """
    Calculate Probability of Detection for Swerling target models.
    
    Uses Shnidman's approximation for fast calculation.
    
    Swerling Cases:
        0: Non-fluctuating (Marcum's case)
        1: Slow fluctuation, many scatterers (exponential)
        2: Fast fluctuation, many scatterers
        3: Slow fluctuation, dominant + small scatterers
        4: Fast fluctuation, dominant + small scatterers
    
    Args:
        snr_db: Signal-to-noise ratio [dB]
        pfa: Probability of false alarm
        swerling_case: Swerling model (0-4)
        n_pulses: Number of pulses integrated
        
    Returns:
        Probability of detection (0-1)
        
    Reference: Shnidman, IEEE Trans. AES, 2002
    """
    snr_linear = 10 ** (snr_db / 10)
    
    if snr_linear <= 0:
        return 0.0
    
    # Detection threshold from Pfa
    # Threshold = -ln(Pfa) for single pulse
    threshold = -np.log(max(pfa, 1e-15))
    
    # Swerling case adjustments
    if swerling_case == 0:
        # Non-fluctuating (Marcum)
        # Pd ≈ Q(x, sqrt(2*SNR)) where Q is Marcum-Q function
        # Simplified: use chi-square approximation
        x = threshold
        lambda_param = 2 * snr_linear * n_pulses
        
        # Marcum Q-function approximation
        if snr_db > 15:
            pd = 0.999
        elif snr_db < -5:
            pd = 0.001
        else:
            # Use normal approximation for moderate SNR
            mu = lambda_param
            sigma = np.sqrt(2 * lambda_param)
            pd = 0.5 * special.erfc((x - mu) / (sigma * np.sqrt(2)))
    
    elif swerling_case == 1 or swerling_case == 2:
        # Exponential RCS fluctuation (many scatterers)
        # Pd = (1 + 1/SNR)^(n-1) * exp(-threshold/(1 + SNR))
        snr_eff = snr_linear * n_pulses
        
        if swerling_case == 1:
            # Slow fluctuation - correlated across pulses
            pd = np.exp(-threshold / (1 + snr_eff))
        else:
            # Fast fluctuation - decorrelated
            pd = 1 - (1 - np.exp(-threshold / (1 + snr_linear))) ** n_pulses
            
    elif swerling_case == 3 or swerling_case == 4:
        # Chi-square 4 DOF (dominant scatterer)
        snr_eff = snr_linear * n_pulses
        x = threshold
        
        if swerling_case == 3:
            pd = (1 + x / (1 + snr_eff)) * np.exp(-x / (1 + snr_eff))
        else:
            # Approximate for Swerling 4
            pd = 1 - (1 - (1 + x) * np.exp(-x / (1 + snr_linear))) ** n_pulses
    
    else:
        # Default to Swerling 1
        pd = np.exp(-threshold / (1 + snr_linear * n_pulses))
    
    return np.clip(pd, 0.0, 1.0)


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
    losses_db: float = 10.0
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
    numerator = radar_power_w * (antenna_gain ** 2) * (wavelength_m ** 2) * rcs_m2
    denominator = ((4 * np.pi) ** 3) * (ranges_m ** 4) * losses
    
    received_power = numerator / denominator
    
    # SNR
    snr_linear = received_power / noise_power
    snr_db = 10 * np.log10(np.maximum(snr_linear, 1e-10))
    
    # Calculate Pd for each range
    pd_values = np.array([
        calculate_pd_swerling(snr, pfa, swerling_case)
        for snr in snr_db
    ])
    
    return pd_values


def generate_roc_curves(
    snr_values_db: List[float] = [5, 10, 13, 15, 20],
    pfa_range: Tuple[float, float] = (1e-10, 1e-2),
    n_points: int = 100,
    swerling_case: int = 1
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
    pfa_values = np.logspace(
        np.log10(pfa_range[0]),
        np.log10(pfa_range[1]),
        n_points
    )
    
    result = {
        'pfa': pfa_values,
        'pd': {}
    }
    
    for snr in snr_values_db:
        pd_values = np.array([
            calculate_pd_swerling(snr, pfa, swerling_case)
            for pfa in pfa_values
        ])
        result['pd'][snr] = pd_values
    
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
    swerling_case: int = 1
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
            losses_db
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
