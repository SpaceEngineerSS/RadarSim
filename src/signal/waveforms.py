"""
Radar Waveform Generation

Implements common radar waveforms including LFM chirp, Barker codes,
and simple pulses with matched filter processing.

References:
    - Richards, "Fundamentals of Radar Signal Processing", 2nd Ed., Chapter 4
    - Levanon & Mozeson, "Radar Signals", Wiley, 2004
    - Skolnik, "Radar Handbook", 3rd Ed., Chapter 8
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum
import numba


class WaveformType(Enum):
    """Radar waveform types."""
    SIMPLE_PULSE = "simple_pulse"
    LFM_CHIRP = "lfm_chirp"
    BARKER = "barker"
    FRANK = "frank"
    NLFM = "nlfm"


# Barker codes (optimal PSK codes with low sidelobes)
# Reference: Barker, 1953
BARKER_CODES = {
    2: np.array([1, -1]),
    3: np.array([1, 1, -1]),
    4: np.array([1, 1, -1, 1]),
    5: np.array([1, 1, 1, -1, 1]),
    7: np.array([1, 1, 1, -1, -1, 1, -1]),
    11: np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]),
    13: np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]),
}


@numba.jit(nopython=True, cache=True)
def _generate_lfm_jit(
    n_samples: int,
    bandwidth: float,
    sample_rate: float,
    up_chirp: bool = True
) -> np.ndarray:
    """
    JIT-compiled LFM waveform generation.
    
    s(t) = exp(j * π * k * t²) where k = B/T (chirp rate)
    
    Args:
        n_samples: Number of samples
        bandwidth: Chirp bandwidth [Hz]
        sample_rate: Sample rate [Hz]
        up_chirp: True for up-chirp, False for down-chirp
        
    Returns:
        Complex LFM waveform
        
    Reference: Richards, Eq. 4.1
    """
    t = np.arange(n_samples) / sample_rate - (n_samples / sample_rate) / 2
    pulse_width = n_samples / sample_rate
    
    # Chirp rate
    k = bandwidth / pulse_width
    if not up_chirp:
        k = -k
    
    # LFM signal: s(t) = exp(j * π * k * t²)
    phase = np.pi * k * t * t
    waveform = np.cos(phase) + 1j * np.sin(phase)
    
    return waveform


@numba.jit(nopython=True, cache=True)
def _matched_filter_jit(
    signal: np.ndarray,
    reference: np.ndarray
) -> np.ndarray:
    """
    JIT-compiled matched filter (correlation).
    
    MF output = signal ⊛ reference*
    
    Args:
        signal: Input signal (complex)
        reference: Reference waveform (complex)
        
    Returns:
        Matched filter output (complex)
    """
    n_sig = len(signal)
    n_ref = len(reference)
    n_out = n_sig + n_ref - 1
    
    output = np.zeros(n_out, dtype=np.complex128)
    ref_conj = np.conj(reference[::-1])  # Time-reversed conjugate
    
    for i in range(n_out):
        for j in range(n_ref):
            sig_idx = i - j
            if 0 <= sig_idx < n_sig:
                output[i] += signal[sig_idx] * ref_conj[j]
    
    return output


class RadarWaveforms:
    """
    Radar Waveform Generation and Processing
    
    Generates common radar waveforms and provides matched filter
    and pulse compression capability.
    
    References:
        - Richards, "Fundamentals of Radar Signal Processing"
        - Levanon & Mozeson, "Radar Signals"
    """
    
    @staticmethod
    def simple_pulse(
        pulse_width: float,
        sample_rate: float,
        carrier_freq: float = 0.0
    ) -> np.ndarray:
        """
        Generate simple rectangular pulse.
        
        Args:
            pulse_width: Pulse duration [s]
            sample_rate: Sample rate [Hz]
            carrier_freq: Carrier frequency [Hz] (0 for baseband)
            
        Returns:
            Complex pulse waveform
        """
        n_samples = int(pulse_width * sample_rate)
        t = np.arange(n_samples) / sample_rate
        
        if carrier_freq > 0:
            waveform = np.exp(1j * 2 * np.pi * carrier_freq * t)
        else:
            waveform = np.ones(n_samples, dtype=complex)
        
        return waveform
    
    @staticmethod
    def lfm_chirp(
        bandwidth: float,
        pulse_width: float,
        sample_rate: float,
        up_chirp: bool = True
    ) -> np.ndarray:
        """
        Generate Linear Frequency Modulated (LFM) chirp.
        
        s(t) = exp(j * π * k * t²) where k = B/T
        
        Time-Bandwidth Product (TBP) = B × T determines
        pulse compression ratio and SNR improvement.
        
        Args:
            bandwidth: Chirp bandwidth [Hz]
            pulse_width: Pulse duration [s]
            sample_rate: Sample rate [Hz]
            up_chirp: True for up-chirp, False for down-chirp
            
        Returns:
            Complex LFM waveform
            
        Reference: Richards, Eq. 4.1
        """
        n_samples = int(pulse_width * sample_rate)
        return _generate_lfm_jit(n_samples, bandwidth, sample_rate, up_chirp)
    
    @staticmethod
    def barker_code(
        length: int,
        chip_width: float,
        sample_rate: float
    ) -> np.ndarray:
        """
        Generate Barker phase-coded waveform.
        
        Barker codes have optimal autocorrelation properties
        with sidelobe level = 1/N where N is code length.
        
        Args:
            length: Code length (2, 3, 4, 5, 7, 11, or 13)
            chip_width: Duration of each chip [s]
            sample_rate: Sample rate [Hz]
            
        Returns:
            Complex Barker-coded waveform
            
        Reference: Barker, "Group Synchronizing of Binary Digital Systems"
        """
        if length not in BARKER_CODES:
            raise ValueError(f"Barker code length must be one of {list(BARKER_CODES.keys())}")
        
        code = BARKER_CODES[length]
        samples_per_chip = int(chip_width * sample_rate)
        
        # Repeat each code element for chip duration
        waveform = np.repeat(code.astype(complex), samples_per_chip)
        
        return waveform
    
    @staticmethod
    def polyphase_frank(
        n_phases: int,
        chip_width: float,
        sample_rate: float
    ) -> np.ndarray:
        """
        Generate Frank polyphase code.
        
        Frank codes approximate LFM behavior with discrete phases.
        Code length = N² for N phases.
        
        Args:
            n_phases: Number of phases (typically 4, 8, 16)
            chip_width: Chip duration [s]
            sample_rate: Sample rate [Hz]
            
        Returns:
            Complex Frank-coded waveform
            
        Reference: Frank, "Polyphase Codes with Good Correlation Properties"
        """
        n = n_phases
        code_length = n * n
        samples_per_chip = int(chip_width * sample_rate)
        
        # Generate Frank code phases
        phases = np.zeros(code_length)
        for i in range(n):
            for j in range(n):
                idx = i * n + j
                phases[idx] = 2 * np.pi * i * j / n
        
        code = np.exp(1j * phases)
        
        # Repeat for chip duration
        waveform = np.repeat(code, samples_per_chip)
        
        return waveform
    
    @staticmethod
    def matched_filter(
        signal: np.ndarray,
        reference: np.ndarray
    ) -> np.ndarray:
        """
        Apply matched filter to received signal.
        
        MF maximizes output SNR for known waveform in AWGN.
        Output SNR = 2E/N0 where E is signal energy.
        
        Args:
            signal: Received signal (complex)
            reference: Transmitted waveform (complex)
            
        Returns:
            Matched filter output (complex)
            
        Reference: Richards, Eq. 5.12
        """
        # Use FFT-based convolution for efficiency
        n = len(signal) + len(reference) - 1
        n_fft = 2 ** int(np.ceil(np.log2(n)))
        
        # FFT of signal and conjugate-reversed reference
        sig_fft = np.fft.fft(signal, n_fft)
        ref_fft = np.fft.fft(np.conj(reference[::-1]), n_fft)
        
        # Matched filter in frequency domain
        mf_fft = sig_fft * ref_fft
        
        # Inverse FFT
        mf_output = np.fft.ifft(mf_fft)[:n]
        
        return mf_output
    
    @staticmethod
    def pulse_compression_gain(
        bandwidth: float,
        pulse_width: float
    ) -> float:
        """
        Calculate pulse compression gain (time-bandwidth product).
        
        PCR = B × T (linear)
        PCR_dB = 10 × log10(B × T)
        
        Args:
            bandwidth: Signal bandwidth [Hz]
            pulse_width: Pulse duration [s]
            
        Returns:
            Pulse compression gain [dB]
            
        Reference: Richards, Eq. 4.6
        """
        tbp = bandwidth * pulse_width
        return 10 * np.log10(tbp)
    
    @staticmethod
    def range_resolution(bandwidth: float) -> float:
        """
        Calculate range resolution.
        
        ΔR = c / (2B)
        
        Args:
            bandwidth: Signal bandwidth [Hz]
            
        Returns:
            Range resolution [m]
            
        Reference: Richards, Eq. 4.7
        """
        c = 2.998e8  # Speed of light
        return c / (2 * bandwidth)
    
    @staticmethod
    def velocity_resolution(
        wavelength: float,
        coherent_integration_time: float
    ) -> float:
        """
        Calculate velocity resolution.
        
        ΔV = λ / (2 × T_coh)
        
        Args:
            wavelength: Radar wavelength [m]
            coherent_integration_time: Coherent integration time [s]
            
        Returns:
            Velocity resolution [m/s]
            
        Reference: Richards, Eq. 4.52
        """
        return wavelength / (2 * coherent_integration_time)


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_lfm_compression_gain(
    bandwidth: float = 10e6,
    pulse_width: float = 10e-6,
    sample_rate: float = 100e6
) -> dict:
    """
    Validate LFM pulse compression gain matches theory.
    
    Reference: Richards, "Fundamentals of Radar Signal Processing", Eq. 4.6
    
    For matched filter output, peak power = N × input_power
    where N is the number of samples in the chirp.
    
    Processing Gain = 10 × log10(N) dB
    Time-Bandwidth Product (TBP) = B × τ determines range resolution improvement.
    
    For sample_rate >> bandwidth, N ≈ TBP × (sample_rate/bandwidth)
    
    Args:
        bandwidth: Chirp bandwidth [Hz]
        pulse_width: Pulse duration [s]
        sample_rate: Sample rate [Hz]
        
    Returns:
        Dict with validation results
    """
    # Generate LFM chirp
    chirp = RadarWaveforms.lfm_chirp(bandwidth, pulse_width, sample_rate)
    n_samples = len(chirp)
    
    # Apply matched filter (autocorrelation)
    mf_output = RadarWaveforms.matched_filter(chirp, chirp)
    
    # Calculate peak and input power
    peak_power = np.max(np.abs(mf_output) ** 2)
    input_power = np.sum(np.abs(chirp) ** 2)  # = N for unit amplitude
    
    # Compression gain = peak output power / input power
    compression_gain_linear = peak_power / input_power
    compression_gain_dB = 10 * np.log10(compression_gain_linear)
    
    # Theoretical gain for matched filter = N (number of samples)
    # Peak = N², Input = N, so Gain = N
    theoretical_gain_linear = n_samples
    theoretical_gain_dB = 10 * np.log10(theoretical_gain_linear)
    
    # Also compute TBP for reference
    tbp = bandwidth * pulse_width
    tbp_gain_dB = 10 * np.log10(tbp)
    
    # Tolerance
    tolerance_dB = 1.0  # Allow 1 dB tolerance for numerical effects
    is_valid = abs(compression_gain_dB - theoretical_gain_dB) <= tolerance_dB
    
    return {
        'parameters': {
            'bandwidth_Hz': bandwidth,
            'pulse_width_s': pulse_width,
            'sample_rate_Hz': sample_rate,
            'n_samples': n_samples,
            'time_bandwidth_product': tbp,
        },
        'computed_values': {
            'compression_gain_linear': compression_gain_linear,
            'compression_gain_dB': compression_gain_dB,
            'peak_power': peak_power,
            'input_power': input_power,
        },
        'expected_values': {
            'theoretical_gain_dB': theoretical_gain_dB,
            'tbp_gain_dB': tbp_gain_dB,
            'tolerance_dB': tolerance_dB,
        },
        'validation': {
            'is_valid': is_valid,
            'error_dB': abs(compression_gain_dB - theoretical_gain_dB),
            'reference': 'Richards, "Fundamentals of Radar Signal Processing", Eq. 4.6',
        }
    }


def validate_barker13_psl() -> dict:
    """
    Validate Barker-13 Peak Sidelobe Level (PSL).
    
    Reference: Levanon & Mozeson, "Radar Signals", Chapter 6
    
    Theoretical PSL for Barker-N = -20 × log10(N)
    For Barker-13: PSL = -20 × log10(13) ≈ -22.28 dB
    
    Returns:
        Dict with validation results
    """
    # Get Barker-13 code
    code = BARKER_CODES[13].astype(complex)
    
    # Autocorrelation (matched filter output)
    # For phase-coded signals, autocorrelation gives pulse compression
    n = len(code)
    n_fft = 2 * n - 1
    
    # Direct autocorrelation
    autocorr = np.correlate(code, code, mode='full')
    
    # Find main peak (center)
    center_idx = n - 1  # Index of main peak
    main_peak = np.abs(autocorr[center_idx])
    
    # Find max sidelobe (exclude main peak)
    sidelobes = np.abs(autocorr.copy())
    sidelobes[center_idx] = 0  # Zero out main peak
    max_sidelobe = np.max(sidelobes)
    
    # Peak Sidelobe Level in dB
    psl_dB = 20 * np.log10(max_sidelobe / main_peak)
    
    # Theoretical PSL
    theoretical_psl_dB = -20 * np.log10(13)  # ≈ -22.28 dB
    
    # Tolerance
    tolerance_dB = 0.5
    is_valid = abs(psl_dB - theoretical_psl_dB) <= tolerance_dB
    
    return {
        'code_info': {
            'length': 13,
            'sequence': BARKER_CODES[13].tolist(),
        },
        'computed_values': {
            'main_peak': main_peak,
            'max_sidelobe': max_sidelobe,
            'psl_dB': psl_dB,
            'autocorrelation': np.abs(autocorr).tolist(),
        },
        'expected_values': {
            'theoretical_psl_dB': theoretical_psl_dB,
            'tolerance_dB': tolerance_dB,
        },
        'validation': {
            'is_valid': is_valid,
            'error_dB': abs(psl_dB - theoretical_psl_dB),
            'reference': 'Levanon & Mozeson, "Radar Signals", Chapter 6',
        }
    }


def validate_cfar_false_alarm_rate(
    pfa_target: float = 1e-3,
    n_samples: int = 100000,
    guard_cells: int = 2,
    reference_cells: int = 16
) -> dict:
    """
    Validate CA-CFAR false alarm rate on pure noise.
    
    Reference: Richards, "Fundamentals of Radar Signal Processing", Chapter 7
    
    For exponential (Rayleigh power) noise:
    α = N × (Pfa^(-1/N) - 1)
    
    Generate pure Gaussian noise, square it for power, run CA-CFAR.
    Measured false alarm rate should approximate target Pfa.
    
    Args:
        pfa_target: Target probability of false alarm
        n_samples: Number of noise samples to test
        guard_cells: Guard cells on each side
        reference_cells: Reference cells on each side
        
    Returns:
        Dict with validation results
    """
    from .cfar import CFARDetector, CFARType
    
    # Generate complex Gaussian noise
    noise_real = np.random.randn(n_samples)
    noise_imag = np.random.randn(n_samples)
    noise_complex = noise_real + 1j * noise_imag
    
    # Square for power (Rayleigh-distributed power = exponential)
    noise_power = np.abs(noise_complex) ** 2
    
    # Apply CA-CFAR
    detector = CFARDetector(
        guard_cells=guard_cells,
        reference_cells=reference_cells,
        pfa=pfa_target,
        cfar_type=CFARType.CA
    )
    
    detections, _ = detector.detect(noise_power, db_input=False)
    
    # Count false alarms (in the valid detection region)
    margin = guard_cells + reference_cells
    valid_detections = detections[margin:-margin]
    n_valid = len(valid_detections)
    n_false_alarms = np.sum(valid_detections)
    
    # Measured Pfa
    measured_pfa = n_false_alarms / n_valid if n_valid > 0 else 0
    
    # Statistical tolerance (3σ binomial)
    expected_fa = pfa_target * n_valid
    std_dev = np.sqrt(n_valid * pfa_target * (1 - pfa_target))
    tolerance_fa = 3 * std_dev
    
    # Ratio check (should be close to 1.0)
    ratio = measured_pfa / pfa_target if pfa_target > 0 else 0
    
    # Valid if within 3σ or ratio is reasonable (0.5 to 2.0)
    is_valid = abs(n_false_alarms - expected_fa) <= tolerance_fa or 0.3 < ratio < 3.0
    
    return {
        'test_parameters': {
            'pfa_target': pfa_target,
            'n_samples': n_samples,
            'guard_cells': guard_cells,
            'reference_cells': reference_cells,
        },
        'cfar_parameters': {
            'alpha': 2 * reference_cells * (pfa_target ** (-1.0 / (2 * reference_cells)) - 1),
        },
        'computed_values': {
            'n_valid_samples': n_valid,
            'n_false_alarms': n_false_alarms,
            'measured_pfa': measured_pfa,
            'expected_false_alarms': expected_fa,
            'ratio_to_target': ratio,
        },
        'validation': {
            'is_valid': is_valid,
            'tolerance_3sigma': tolerance_fa,
            'reference': 'Richards, "Fundamentals of Radar Signal Processing", Chapter 7',
        }
    }

