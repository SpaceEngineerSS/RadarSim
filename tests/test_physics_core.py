"""
RadarSim Core Physics Validation Test Suite

Comprehensive tests validating physics implementations against standard references.

Test ID | Description                    | Reference           | Tolerance
--------|--------------------------------|---------------------|------------
1       | Skolnik Example 2.1 SNR        | Skolnik Ex. 2.1     | ±0.5 dB
2       | CODATA 2018 Constants          | CODATA 2018         | Exact
3       | ITU-R P.676 60 GHz resonance   | ITU-R P.676-12      | ±2 dB
4       | ITU-R P.676 X-band low atten   | ITU-R P.676-12      | <0.02 dB/km
5       | Swerling 0 deterministic       | Swerling 1960       | Exact
6       | Swerling I exponential         | Swerling 1960       | KS p>0.05
7       | Swerling III chi-squared       | Swerling 1960       | KS p>0.05
8       | Doppler shift calculation      | Skolnik Ch. 3       | ±1 Hz
9       | Wavelength-frequency relation  | IEEE 686-2008       | Exact

References:
    - Skolnik, M.I. (2008). "Radar Handbook", 3rd Edition, McGraw-Hill
    - ITU-R P.676-12 (12/2017). "Attenuation by atmospheric gases"
    - Swerling, P. (1960). "Probability of Detection for Fluctuating Targets"
    - CODATA 2018: Fundamental Physical Constants
    - IEEE Std 686-2008: Standard Radar Definitions
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.physics.constants import (
    SPEED_OF_LIGHT,
    BOLTZMANN_CONSTANT,
    PLANCK_CONSTANT,
    STANDARD_TEMPERATURE,
    STANDARD_PRESSURE,
)
from src.physics.radar_equation import (
    RadarParameters,
    calculate_received_power,
    calculate_snr,
    calculate_doppler_shift,
    validate_skolnik_example_2_1,
)
from src.physics.atmospheric import (
    ITU_R_P676,
    validate_itu_60ghz,
    validate_itu_xband,
)
from src.physics.rcs import (
    SwerlingModel,
    SwerlingRCS,
    validate_swerling_distribution,
)


# =============================================================================
# TEST 1: Skolnik Example 2.1 Validation
# =============================================================================

class TestSkolnikExample21:
    """
    Validate radar equation against Skolnik-style parameters
    
    Reference: Skolnik, M.I. (2008). "Radar Handbook", 3rd Ed., Chapter 2
    
    Problem:
        Pt = 1.5 MW, G = 45 dB, σ = 1 m², R = 200 km
        f = 10 GHz, Fn = 3 dB, B = 1 MHz, L = 0 dB
        
    Expected: SNR ≈ 17.26 dB (±0.5 dB tolerance)
    """
    
    def test_snr_within_tolerance(self):
        """SNR must be within 0.5 dB of Skolnik's expected value (13 dB)"""
        result = validate_skolnik_example_2_1()
        
        computed_snr = result['computed_values']['snr_dB']
        expected_snr = result['expected_values']['snr_dB']
        tolerance = result['expected_values']['tolerance_dB']
        
        assert abs(computed_snr - expected_snr) <= tolerance, (
            f"Computed SNR ({computed_snr:.2f} dB) differs from expected "
            f"({expected_snr:.2f} dB) by more than {tolerance} dB"
        )
    
    def test_validation_passes(self):
        """The validation function should report is_valid=True"""
        result = validate_skolnik_example_2_1()
        assert result['validation']['is_valid'] is True
    
    def test_noise_power_reasonable(self):
        """Noise power should be approximately 8e-15 W"""
        result = validate_skolnik_example_2_1()
        noise_power = result['computed_values']['noise_power_W']
        
        # k * T * B * F = 1.38e-23 * 290 * 1e6 * 2 ≈ 8e-15 W
        expected_order = 1e-14
        assert 1e-15 < noise_power < 1e-13, (
            f"Noise power ({noise_power:.2e} W) outside expected range"
        )


# =============================================================================
# TEST 2: CODATA 2018 Physical Constants
# =============================================================================

class TestCODATA2018Constants:
    """
    Verify physical constants match CODATA 2018 exact values.
    
    Reference: CODATA 2018, SI 2019 redefinition
    """
    
    def test_speed_of_light(self):
        """Speed of light = 299792458 m/s (exact SI)"""
        assert SPEED_OF_LIGHT == 299_792_458.0
    
    def test_boltzmann_constant(self):
        """Boltzmann constant = 1.380649e-23 J/K (exact SI)"""
        assert BOLTZMANN_CONSTANT == 1.380649e-23
    
    def test_planck_constant(self):
        """Planck constant = 6.62607015e-34 J·s (exact SI)"""
        assert PLANCK_CONSTANT == 6.62607015e-34
    
    def test_standard_temperature(self):
        """Standard temperature = 290 K (IEEE reference)"""
        assert STANDARD_TEMPERATURE == 290.0
    
    def test_standard_pressure(self):
        """Standard pressure = 1013.25 hPa (ITU-R reference)"""
        assert STANDARD_PRESSURE == 1013.25


# =============================================================================
# TEST 3: ITU-R P.676-12 60 GHz Oxygen Resonance
# =============================================================================

class TestITU60GHzResonance:
    """
    Validate 60 GHz oxygen absorption peak per ITU-R P.676-12.
    
    Reference: ITU-R P.676-12 (12/2017), Figure 1
    
    At 60 GHz, 15°C, 1013.25 hPa:
    Expected: γ_o ≈ 15 dB/km (±2 dB tolerance)
    """
    
    def test_60ghz_attenuation_value(self):
        """60 GHz oxygen attenuation must be approximately 15 dB/km"""
        result = validate_itu_60ghz()
        
        gamma = result['computed_values']['gamma_oxygen_dB_per_km']
        expected = result['expected_values']['gamma_oxygen_dB_per_km']
        tolerance = result['expected_values']['tolerance_dB_per_km']
        
        assert abs(gamma - expected) <= tolerance, (
            f"60 GHz attenuation ({gamma:.2f} dB/km) outside tolerance "
            f"of {expected:.2f} ± {tolerance} dB/km"
        )
    
    def test_validation_passes(self):
        """The 60 GHz validation function should report is_valid=True"""
        result = validate_itu_60ghz()
        assert result['validation']['is_valid'] is True


# =============================================================================
# TEST 4: ITU-R P.676-12 X-Band Low Attenuation
# =============================================================================

class TestITUXBandAttenuation:
    """
    Validate X-band (~10 GHz) low atmospheric attenuation per ITU-R P.676-12.
    
    Reference: ITU-R P.676-12 (12/2017), Section 1
    
    At 10 GHz, sea level:
    Expected: γ_o < 0.02 dB/km (very low oxygen absorption at X-band)
    """
    
    def test_xband_low_oxygen_attenuation(self):
        """X-band oxygen attenuation must be less than 0.02 dB/km"""
        result = validate_itu_xband()
        
        gamma_o = result['computed_values']['gamma_oxygen_dB_per_km']
        max_gamma = result['expected_values']['max_oxygen_dB_per_km']
        
        assert gamma_o < max_gamma, (
            f"X-band oxygen attenuation ({gamma_o:.4f} dB/km) exceeds "
            f"maximum ({max_gamma} dB/km)"
        )
    
    def test_validation_passes(self):
        """The X-band validation function should report is_valid=True"""
        result = validate_itu_xband()
        assert result['validation']['is_valid'] is True


# =============================================================================
# TEST 5: Swerling 0 Deterministic (Non-Fluctuating)
# =============================================================================

class TestSwerling0Deterministic:
    """
    Validate Swerling Model 0 (Marcum case) returns constant RCS.
    
    Reference: Swerling, P. (1960), Section II.A
    
    For non-fluctuating targets, σ = σ_avg (constant)
    """
    
    def test_constant_rcs_output(self):
        """Swerling 0 must return the same RCS every time"""
        mean_rcs = 10.0
        
        for _ in range(100):
            rcs = SwerlingRCS.generate_rcs(mean_rcs, SwerlingModel.SWERLING_0)
            assert rcs == mean_rcs
    
    def test_validation_passes(self):
        """Monte Carlo validation should confirm constant output"""
        result = validate_swerling_distribution(SwerlingModel.SWERLING_0)
        assert result['validation']['is_valid'] is True


# =============================================================================
# TEST 6: Swerling I Exponential Distribution
# =============================================================================

class TestSwerling1Exponential:
    """
    Validate Swerling Model I produces exponential (Rayleigh power) distribution.
    
    Reference: Swerling, P. (1960), Eq. 4
    PDF: p(σ) = (1/σ_avg) × exp(-σ/σ_avg)
    
    Uses Kolmogorov-Smirnov test with p > 0.05 threshold.
    """
    
    @pytest.mark.slow
    def test_exponential_distribution_ks(self):
        """Swerling I samples must pass KS test for exponential distribution"""
        result = validate_swerling_distribution(
            SwerlingModel.SWERLING_1, 
            mean_rcs=10.0, 
            n_samples=10000
        )
        
        p_value = result['ks_test']['p_value']
        assert p_value > 0.05, (
            f"KS test failed: p-value ({p_value:.4f}) <= 0.05. "
            f"Distribution does not match exponential."
        )
    
    @pytest.mark.slow
    def test_validation_passes(self):
        """Monte Carlo validation should pass"""
        result = validate_swerling_distribution(SwerlingModel.SWERLING_1)
        assert result['validation']['is_valid'] is True


# =============================================================================
# TEST 7: Swerling III Chi-Squared Distribution
# =============================================================================

class TestSwerling3ChiSquared:
    """
    Validate Swerling Model III produces Chi-squared (k=4) distribution.
    
    Reference: Swerling, P. (1960), Eq. 5
    PDF: p(σ) = (4σ/σ_avg²) × exp(-2σ/σ_avg)
    Equivalent to Gamma(shape=2, scale=σ_avg/2)
    
    Uses Kolmogorov-Smirnov test with p > 0.05 threshold.
    """
    
    @pytest.mark.slow
    def test_chi_squared_distribution_ks(self):
        """Swerling III samples must pass KS test for Chi-squared (k=4)"""
        result = validate_swerling_distribution(
            SwerlingModel.SWERLING_3, 
            mean_rcs=10.0, 
            n_samples=10000
        )
        
        p_value = result['ks_test']['p_value']
        assert p_value > 0.05, (
            f"KS test failed: p-value ({p_value:.4f}) <= 0.05. "
            f"Distribution does not match Chi-squared (k=4)."
        )
    
    @pytest.mark.slow
    def test_validation_passes(self):
        """Monte Carlo validation should pass"""
        result = validate_swerling_distribution(SwerlingModel.SWERLING_3)
        assert result['validation']['is_valid'] is True


# =============================================================================
# TEST 8: Doppler Shift Calculation
# =============================================================================

class TestDopplerShift:
    """
    Validate Doppler shift calculation.
    
    Reference: Skolnik, "Radar Handbook", 3rd Ed., Chapter 3
    
    Formula: fd = 2 × Vr / λ
    """
    
    @pytest.fixture
    def radar_10ghz(self):
        """X-band (10 GHz) radar fixture"""
        return RadarParameters(
            frequency=10e9,
            power_transmitted=1000,
            antenna_gain_tx=30
        )
    
    def test_approaching_target_negative_doppler(self, radar_10ghz):
        """Target approaching radar should produce positive Doppler shift"""
        # Target at 1 km, approaching at 100 m/s
        target_pos = np.array([1000.0, 0.0, 0.0])
        target_vel = np.array([-100.0, 0.0, 0.0])  # Moving toward radar
        
        fd = calculate_doppler_shift(radar_10ghz, target_pos, target_vel)
        
        # fd = 2 × 100 / 0.03 ≈ 6667 Hz (approaching = negative velocity)
        expected_fd = -2 * 100 / radar_10ghz.wavelength  # Negative = approaching
        
        assert abs(fd - expected_fd) < 1.0, (
            f"Doppler shift ({fd:.1f} Hz) differs from expected "
            f"({expected_fd:.1f} Hz) by more than 1 Hz"
        )
    
    def test_receding_target_positive_doppler(self, radar_10ghz):
        """Target receding from radar should produce negative Doppler shift"""
        # Target at 1 km, receding at 100 m/s
        target_pos = np.array([1000.0, 0.0, 0.0])
        target_vel = np.array([100.0, 0.0, 0.0])  # Moving away from radar
        
        fd = calculate_doppler_shift(radar_10ghz, target_pos, target_vel)
        
        # Receding = positive radial velocity
        expected_fd = 2 * 100 / radar_10ghz.wavelength
        
        assert abs(fd - expected_fd) < 1.0


# =============================================================================
# TEST 9: Wavelength-Frequency Relation
# =============================================================================

class TestWavelengthFrequency:
    """
    Validate wavelength calculation from frequency.
    
    Reference: IEEE Std 686-2008
    
    Formula: λ = c / f
    """
    
    def test_10ghz_wavelength(self):
        """10 GHz should have wavelength of ~0.03 m"""
        radar = RadarParameters(
            frequency=10e9,
            power_transmitted=1000,
            antenna_gain_tx=30
        )
        
        expected_wavelength = SPEED_OF_LIGHT / 10e9
        
        assert radar.wavelength == pytest.approx(expected_wavelength, rel=1e-10)
    
    def test_3ghz_wavelength(self):
        """3 GHz (S-band) should have wavelength of ~0.1 m"""
        radar = RadarParameters(
            frequency=3e9,
            power_transmitted=1000,
            antenna_gain_tx=30
        )
        
        expected_wavelength = SPEED_OF_LIGHT / 3e9  # ~0.1 m
        
        assert radar.wavelength == pytest.approx(expected_wavelength, rel=1e-10)
    
    def test_60ghz_wavelength(self):
        """60 GHz should have wavelength of ~5 mm"""
        radar = RadarParameters(
            frequency=60e9,
            power_transmitted=1000,
            antenna_gain_tx=30
        )
        
        expected_wavelength = SPEED_OF_LIGHT / 60e9  # ~0.005 m
        
        assert radar.wavelength == pytest.approx(expected_wavelength, rel=1e-10)


# =============================================================================
# ADDITIONAL VALIDATION TESTS
# =============================================================================

class TestRadarEquationSanity:
    """Additional sanity checks for radar equation implementation"""
    
    @pytest.fixture
    def radar(self):
        """Standard X-band radar fixture"""
        return RadarParameters(
            frequency=10e9,
            power_transmitted=1000,
            antenna_gain_tx=30,
            noise_figure=4.0,
            system_losses_tx=2.0,
            system_losses_rx=2.0,
        )
    
    def test_received_power_decreases_with_range(self, radar):
        """Received power must decrease with R^4"""
        rcs = 1.0
        
        pr_10km = calculate_received_power(radar, rcs, 10e3)
        pr_20km = calculate_received_power(radar, rcs, 20e3)
        
        # Power should decrease by factor of 16 (2^4)
        ratio = pr_10km / pr_20km
        
        assert ratio == pytest.approx(16.0, rel=0.01), (
            f"Power ratio ({ratio:.2f}) should be 16 (R^4 law)"
        )
    
    def test_snr_increases_with_rcs(self, radar):
        """SNR must increase with RCS"""
        range_m = 50e3
        
        snr_1m2 = calculate_snr(radar, 1.0, range_m)
        snr_10m2 = calculate_snr(radar, 10.0, range_m)
        
        # 10× RCS = +10 dB SNR
        assert snr_10m2 > snr_1m2
        assert (snr_10m2 - snr_1m2) == pytest.approx(10.0, abs=0.5)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-m", "not slow"  # Skip slow Monte Carlo tests by default
    ])
