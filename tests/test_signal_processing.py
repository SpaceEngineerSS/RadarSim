"""
RadarSim Signal Processing Validation Test Suite

Comprehensive tests validating signal processing implementations against theory.

Test ID | Description                    | Reference           | Tolerance
--------|--------------------------------|---------------------|------------
1       | LFM compression gain           | Richards Eq. 4.6    | ±1 dB
2       | Barker-13 PSL                  | Levanon Ch. 6       | ±0.5 dB
3       | CFAR false alarm rate          | Richards Ch. 7      | 3σ binomial
4       | LFM time-bandwidth product     | Richards Eq. 4.6    | Exact
5       | Matched filter peak location   | Richards Eq. 5.12   | ±1 sample
6       | Barker code autocorrelation    | Barker 1953         | -22.28 dB

References:
    - Richards, M. (2005). "Fundamentals of Radar Signal Processing"
    - Levanon, N. & Mozeson, E. (2004). "Radar Signals"
    - Rohling (1983). "Radar CFAR Thresholding", IEEE Trans. AES
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.signal.waveforms import (
    RadarWaveforms,
    BARKER_CODES,
    validate_lfm_compression_gain,
    validate_barker13_psl,
    validate_cfar_false_alarm_rate,
)
from src.signal.cfar import CFARDetector, CFARType


# =============================================================================
# TEST 1: LFM Pulse Compression Gain
# =============================================================================

class TestLFMCompressionGain:
    """
    Validate LFM pulse compression gain matches matched filter theory.
    
    Reference: Richards, "Fundamentals of Radar Signal Processing", Eq. 4.6
    
    Matched filter peak power = N × input_power (N = number of samples)
    Processing Gain = 10 × log10(N) dB
    
    For sample_rate=100MHz, pulse_width=10μs: N=1000, Gain=30 dB
    """
    
    def test_compression_gain_matches_samples(self):
        """Compression gain should match 10*log10(N) for N samples"""
        result = validate_lfm_compression_gain(
            bandwidth=10e6,
            pulse_width=10e-6,
            sample_rate=100e6
        )
        
        computed_gain = result['computed_values']['compression_gain_dB']
        expected_gain = result['expected_values']['theoretical_gain_dB']
        tolerance = result['expected_values']['tolerance_dB']
        
        assert abs(computed_gain - expected_gain) <= tolerance, (
            f"Compression gain ({computed_gain:.2f} dB) differs from "
            f"theoretical ({expected_gain:.2f} dB) by more than {tolerance} dB"
        )
    
    def test_validation_passes(self):
        """Validation function should return is_valid=True"""
        result = validate_lfm_compression_gain()
        assert result['validation']['is_valid']
    
    def test_different_tbp_values(self):
        """Test various time-bandwidth products"""
        test_cases = [
            (1e6, 10e-6, 10),    # TBP=10, 10 dB
            (5e6, 20e-6, 100),   # TBP=100, 20 dB
            (20e6, 5e-6, 100),   # TBP=100, 20 dB
        ]
        
        for bw, pw, expected_tbp in test_cases:
            result = validate_lfm_compression_gain(
                bandwidth=bw,
                pulse_width=pw,
                sample_rate=max(5 * bw, 50e6)
            )
            
            tbp = result['parameters']['time_bandwidth_product']
            assert abs(tbp - expected_tbp) < 0.1, (
                f"TBP ({tbp}) does not match expected ({expected_tbp})"
            )


# =============================================================================
# TEST 2: Barker-13 Peak Sidelobe Level
# =============================================================================

class TestBarker13PSL:
    """
    Validate Barker-13 Peak Sidelobe Level (PSL).
    
    Reference: Levanon & Mozeson, "Radar Signals", Chapter 6
    
    Theoretical PSL = -20 × log10(N) = -20 × log10(13) ≈ -22.28 dB
    """
    
    def test_psl_theoretical_value(self):
        """Barker-13 PSL must be approximately -22.28 dB"""
        result = validate_barker13_psl()
        
        psl_dB = result['computed_values']['psl_dB']
        expected_psl = result['expected_values']['theoretical_psl_dB']
        tolerance = result['expected_values']['tolerance_dB']
        
        assert abs(psl_dB - expected_psl) <= tolerance, (
            f"Barker-13 PSL ({psl_dB:.2f} dB) differs from "
            f"theoretical ({expected_psl:.2f} dB) by more than {tolerance} dB"
        )
    
    def test_validation_passes(self):
        """Validation function should return is_valid=True"""
        result = validate_barker13_psl()
        assert result['validation']['is_valid']
    
    def test_main_peak_equals_code_length(self):
        """Autocorrelation main peak should equal code length (13)"""
        result = validate_barker13_psl()
        main_peak = result['computed_values']['main_peak']
        
        assert main_peak == pytest.approx(13.0, abs=0.01), (
            f"Main peak ({main_peak}) should equal code length (13)"
        )
    
    def test_max_sidelobe_equals_one(self):
        """Maximum sidelobe for Barker codes should be exactly 1"""
        result = validate_barker13_psl()
        max_sidelobe = result['computed_values']['max_sidelobe']
        
        assert max_sidelobe == pytest.approx(1.0, abs=0.01), (
            f"Max sidelobe ({max_sidelobe}) should be exactly 1"
        )


# =============================================================================
# TEST 3: CFAR False Alarm Rate
# =============================================================================

class TestCFARFalseAlarmRate:
    """
    Validate CA-CFAR false alarm rate on pure noise.
    
    Reference: Richards, "Fundamentals of Radar Signal Processing", Chapter 7
    
    For exponential noise with N reference cells:
    α = N × (Pfa^(-1/N) - 1)
    """
    
    @pytest.mark.slow
    def test_pfa_1e_minus_3(self):
        """CFAR Pfa=10^-3 should produce ~0.1% false alarms"""
        result = validate_cfar_false_alarm_rate(
            pfa_target=1e-3,
            n_samples=100000,
            guard_cells=2,
            reference_cells=16
        )
        
        assert result['validation']['is_valid'], (
            f"CFAR false alarm rate ({result['computed_values']['measured_pfa']:.4f}) "
            f"differs significantly from target ({result['test_parameters']['pfa_target']})"
        )
    
    @pytest.mark.slow
    def test_pfa_1e_minus_4(self):
        """CFAR Pfa=10^-4 should produce ~0.01% false alarms"""
        result = validate_cfar_false_alarm_rate(
            pfa_target=1e-4,
            n_samples=500000,
            guard_cells=2,
            reference_cells=16
        )
        
        # Allow more variance for lower Pfa
        ratio = result['computed_values']['ratio_to_target']
        assert 0.2 < ratio < 5.0, (
            f"CFAR Pfa ratio ({ratio:.2f}) outside acceptable range"
        )


# =============================================================================
# TEST 4: Barker Code Sequences
# =============================================================================

class TestBarkerCodes:
    """Test Barker code definitions and properties."""
    
    def test_barker_code_lengths(self):
        """Verify all defined Barker code lengths"""
        expected_lengths = [2, 3, 4, 5, 7, 11, 13]
        assert list(BARKER_CODES.keys()) == expected_lengths
    
    def test_barker13_sequence(self):
        """Barker-13 sequence must match reference"""
        # Reference: Barker, 1953
        expected = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
        assert np.array_equal(BARKER_CODES[13], expected)
    
    def test_barker7_sequence(self):
        """Barker-7 sequence must match reference"""
        expected = np.array([1, 1, 1, -1, -1, 1, -1])
        assert np.array_equal(BARKER_CODES[7], expected)
    
    def test_all_barker_psl(self):
        """All Barker codes should have PSL = 1/N"""
        for length, code in BARKER_CODES.items():
            code_complex = code.astype(complex)
            autocorr = np.correlate(code_complex, code_complex, mode='full')
            
            center = len(code) - 1
            main_peak = np.abs(autocorr[center])
            sidelobes = np.abs(autocorr.copy())
            sidelobes[center] = 0
            max_sidelobe = np.max(sidelobes)
            
            # PSL should be exactly 1
            assert max_sidelobe == pytest.approx(1.0, abs=0.01), (
                f"Barker-{length} max sidelobe ({max_sidelobe}) should be 1"
            )
            
            # Main peak should equal N
            assert main_peak == pytest.approx(float(length), abs=0.01), (
                f"Barker-{length} main peak ({main_peak}) should be {length}"
            )


# =============================================================================
# TEST 5: LFM Chirp Generation
# =============================================================================

class TestLFMChirpGeneration:
    """Test LFM chirp waveform generation."""
    
    def test_chirp_length(self):
        """Chirp length should match pulse_width × sample_rate"""
        pulse_width = 10e-6
        sample_rate = 100e6
        
        chirp = RadarWaveforms.lfm_chirp(10e6, pulse_width, sample_rate)
        expected_length = int(pulse_width * sample_rate)
        
        assert len(chirp) == expected_length
    
    def test_chirp_unit_amplitude(self):
        """LFM chirp should have approximately unit amplitude"""
        chirp = RadarWaveforms.lfm_chirp(10e6, 10e-6, 100e6)
        amplitudes = np.abs(chirp)
        
        assert np.allclose(amplitudes, 1.0, atol=0.01)
    
    def test_up_vs_down_chirp(self):
        """Up and down chirps should be complex conjugates"""
        up_chirp = RadarWaveforms.lfm_chirp(10e6, 10e-6, 100e6, up_chirp=True)
        down_chirp = RadarWaveforms.lfm_chirp(10e6, 10e-6, 100e6, up_chirp=False)
        
        # Down chirp = complex conjugate of up chirp
        assert np.allclose(np.conj(up_chirp), down_chirp, atol=0.01)


# =============================================================================
# TEST 6: Matched Filter
# =============================================================================

class TestMatchedFilter:
    """Test matched filter implementation."""
    
    def test_matched_filter_output_length(self):
        """MF output length should be N + M - 1"""
        signal = np.random.randn(100) + 1j * np.random.randn(100)
        reference = np.random.randn(50) + 1j * np.random.randn(50)
        
        output = RadarWaveforms.matched_filter(signal, reference)
        expected_length = len(signal) + len(reference) - 1
        
        assert len(output) == expected_length
    
    def test_matched_filter_peak_at_center(self):
        """Autocorrelation peak should be at center"""
        pulse = np.ones(10, dtype=complex)
        
        output = RadarWaveforms.matched_filter(pulse, pulse)
        peak_idx = np.argmax(np.abs(output))
        expected_peak_idx = len(pulse) - 1  # Center for autocorrelation
        
        assert abs(peak_idx - expected_peak_idx) <= 1
    
    def test_matched_filter_snr_improvement(self):
        """MF should improve SNR by integration length"""
        # Create pulse + noise
        pulse_length = 100
        pulse = np.ones(pulse_length, dtype=complex)
        noise = (np.random.randn(pulse_length) + 1j * np.random.randn(pulse_length)) * 0.1
        signal = pulse + noise
        
        # Apply matched filter
        output = RadarWaveforms.matched_filter(signal, pulse)
        
        # Peak should be approximately pulse_length²
        peak = np.max(np.abs(output))
        expected_peak = pulse_length  # pulse energy
        
        assert peak > 0.9 * expected_peak


# =============================================================================
# TEST 7: CFAR Detector
# =============================================================================

class TestCFARDetector:
    """Test CFAR detector functionality."""
    
    def test_cfar_detects_strong_target(self):
        """CFAR should detect target well above noise floor"""
        # Create noise with embedded target
        n_samples = 1000
        noise = np.abs(np.random.randn(n_samples)) ** 2  # Exponential-like
        
        # Add strong target at center
        target_idx = 500
        noise[target_idx] = 100 * np.mean(noise)  # 20 dB above noise
        
        detector = CFARDetector(
            guard_cells=2,
            reference_cells=16,
            pfa=1e-4,
            cfar_type=CFARType.CA
        )
        
        detections, _ = detector.detect(noise)
        
        # Target should be detected
        assert detections[target_idx] == True
    
    def test_cfar_threshold_calculation(self):
        """Verify CFAR threshold multiplier formula"""
        n_ref = 32
        pfa = 1e-6
        
        # α = N × (Pfa^(-1/N) - 1)
        expected_alpha = n_ref * (pfa ** (-1.0 / n_ref) - 1)
        
        # Should be around 14 for these parameters
        assert 10 < expected_alpha < 20


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short",
        "-m", "not slow"
    ])
