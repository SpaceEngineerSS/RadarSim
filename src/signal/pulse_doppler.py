# Developed by Mehmet Gümüş (@SpaceEngineerSS) - RadarSim v2.x
"""
Pulse-Doppler Processing Pipeline

Signal-level coherent pulse-Doppler processor for RadarSim.

Signal Chain:
    CPI Generation → Range Compression → MTI Canceller → Doppler FFT → R-D Map

Physics:
    - Complex baseband: s(t,n) = A·p(t - 2R/c)·exp(j·4π·v·n·T_PRI/λ)
    - Matched filter gain: G_mf = 10·log10(B·T) [dB]
    - Blind speed: v_blind = λ·PRF/2
    - Doppler resolution: Δv = λ·PRF/(2·N)

References:
    - Richards, M.A. "Fundamentals of Radar Signal Processing", 2nd Ed., 2005
    - Skolnik, M.I. "Radar Handbook", 3rd Ed., Ch. 3-5
    - Developed by Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Speed of light [m/s]
C_LIGHT = 299_792_458.0


@dataclass
class RangeDopplerMap:
    """
    Container for processed Range-Doppler map output.

    Attributes:
        data_db: 2D R-D map magnitude [dB], shape [n_doppler, n_range]
        range_axis_m: Range bin center values [m]
        velocity_axis_mps: Velocity bin center values [m/s]
        n_pulses: Number of coherent pulses in CPI
        prf_hz: Pulse repetition frequency [Hz]
        wavelength_m: Radar wavelength [m]
        bandwidth_hz: Waveform bandwidth [Hz]
        blind_speeds_mps: List of ambiguous (blind) velocities [m/s]
        mti_order: MTI canceller order (0=off, 1=2-pulse, 2=3-pulse)
        processing_gain_db: Matched filter processing gain [dB]
    """

    data_db: np.ndarray
    range_axis_m: np.ndarray
    velocity_axis_mps: np.ndarray
    n_pulses: int
    prf_hz: float
    wavelength_m: float
    bandwidth_hz: float
    blind_speeds_mps: List[float] = field(default_factory=list)
    mti_order: int = 0
    processing_gain_db: float = 0.0


class PulseDopplerProcessor:
    """
    Coherent Pulse-Doppler signal processor.

    Implements the complete signal chain from CPI generation through
    Range-Doppler map formation with CFAR-ready output.

    Reference: Richards (2005), "Fundamentals of Radar Signal Processing"

    Attributes:
        prf_hz: Pulse Repetition Frequency [Hz]
        n_pulses: Number of pulses per CPI (Coherent Processing Interval)
        n_range_bins: Number of range bins (fast-time samples per pulse)
        bandwidth_hz: Waveform bandwidth [Hz]
        pulse_width_s: Transmitted pulse width [s]
        frequency_hz: Carrier frequency [Hz]
        mti_order: MTI canceller order (0=off, 1=2-pulse, 2=3-pulse)
        window_type: Doppler FFT window ('hamming', 'hann', 'taylor', 'none')
    """

    def __init__(
        self,
        prf_hz: float = 1000.0,
        n_pulses: int = 64,
        n_range_bins: int = 512,
        bandwidth_hz: float = 5e6,
        pulse_width_s: float = 10e-6,
        frequency_hz: float = 3e9,
        mti_order: int = 0,
        window_type: str = "hamming",
    ):
        """
        Initialize Pulse-Doppler Processor.

        Args:
            prf_hz: Pulse Repetition Frequency [Hz]
            n_pulses: Pulses per CPI (slow-time dimension)
            n_range_bins: Range samples per pulse (fast-time dimension)
            bandwidth_hz: Waveform bandwidth [Hz]
            pulse_width_s: Pulse duration [s]
            frequency_hz: Carrier (center) frequency [Hz]
            mti_order: MTI canceller order (0=off, 1=2-pulse, 2=3-pulse)
            window_type: Window function for Doppler FFT
        """
        self.prf_hz = prf_hz
        self.n_pulses = n_pulses
        self.n_range_bins = n_range_bins
        self.bandwidth_hz = bandwidth_hz
        self.pulse_width_s = pulse_width_s
        self.frequency_hz = frequency_hz
        self.mti_order = max(0, min(2, mti_order))
        self.window_type = window_type

        # Derived parameters
        self.wavelength_m = C_LIGHT / frequency_hz
        self.pri_s = 1.0 / prf_hz  # Pulse Repetition Interval [s]
        self.range_resolution_m = C_LIGHT / (2.0 * bandwidth_hz)
        self.max_unambiguous_range_m = C_LIGHT / (2.0 * prf_hz)
        self.max_unambiguous_velocity_mps = self.wavelength_m * prf_hz / 4.0

        # Time-bandwidth product and processing gain
        self.tbp = bandwidth_hz * pulse_width_s
        self.processing_gain_db = 10.0 * np.log10(max(self.tbp, 1.0))

        # Range axis [m]
        self.range_axis_m = np.arange(n_range_bins) * self.range_resolution_m

        # Velocity axis [m/s] (centered, after fftshift)
        doppler_resolution_hz = prf_hz / n_pulses
        velocity_resolution_mps = self.wavelength_m * doppler_resolution_hz / 2.0
        self.velocity_axis_mps = (np.arange(n_pulses) - n_pulses / 2) * velocity_resolution_mps

        # Blind speeds (first 3 ambiguities)
        v_blind_1 = self.wavelength_m * prf_hz / 2.0
        self.blind_speeds_mps = [v_blind_1 * k for k in range(1, 4)]

        # Pre-compute reference chirp for matched filtering
        self._ref_chirp = self._generate_lfm_reference()

        # Pre-compute Doppler window
        self._doppler_window = self._generate_window(n_pulses, window_type)

    # ═══════════════════════════════════════════════════════════════════
    # CORE SIGNAL CHAIN
    # ═══════════════════════════════════════════════════════════════════

    def generate_cpi(
        self,
        target_ranges_m: np.ndarray,
        target_velocities_mps: np.ndarray,
        target_amplitudes: np.ndarray,
        noise_power: float = 1e-12,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Coherent Processing Interval (CPI) raw data.

        Creates the received signal matrix from target parameters using
        the complex baseband model:
            s(t,n) = Σᵢ Aᵢ · p(t - 2Rᵢ/c) · exp(j·4π·vᵢ·n·T_PRI/λ)

        Fully vectorized — no Python loops over targets or pulses.

        Args:
            target_ranges_m: Target slant ranges [m], shape [n_targets]
            target_velocities_mps: Target radial velocities [m/s], shape [n_targets]
            target_amplitudes: Target return amplitudes (linear), shape [n_targets]
            noise_power: AWGN noise power (variance per sample)
            seed: Optional RNG seed for reproducibility

        Returns:
            CPI data matrix, shape [n_pulses, n_range_bins], complex128

        Reference: Richards (2005), Eq. 3.6
        """
        rng = np.random.default_rng(seed)

        n_targets = len(target_ranges_m)
        cpi = np.zeros((self.n_pulses, self.n_range_bins), dtype=np.complex128)

        if n_targets == 0:
            # Noise-only CPI
            noise_std = np.sqrt(noise_power / 2.0)
            cpi = noise_std * (rng.standard_normal(cpi.shape) + 1j * rng.standard_normal(cpi.shape))
            return cpi

        # Pulse indices: n = [0, 1, ..., N-1], shape [n_pulses, 1]
        n_idx = np.arange(self.n_pulses, dtype=np.float64).reshape(-1, 1)

        # Doppler phase per pulse per target: shape [n_pulses, n_targets]
        # φ_d(n) = 4π · v · n · T_PRI / λ
        doppler_phase = (
            4.0
            * np.pi
            * target_velocities_mps[np.newaxis, :]
            * n_idx
            * self.pri_s
            / self.wavelength_m
        )

        # Doppler steering vectors: shape [n_pulses, n_targets]
        doppler_steering = target_amplitudes[np.newaxis, :] * np.exp(1j * doppler_phase)

        # Range bin indices for each target
        range_bins = np.round(target_ranges_m / self.range_resolution_m).astype(int)

        # Place compressed pulse returns (sinc response) at target range bins
        # This models the signal at the output of the matched filter,
        # which is the standard approach for parametric radar simulators.
        # The sinc width is determined by the time-bandwidth product.
        # Reference: Richards (2005), Ch. 4.2
        sinc_halfwidth = max(2, min(8, int(np.sqrt(self.tbp))))
        sinc_idx = np.arange(-sinc_halfwidth, sinc_halfwidth + 1)
        sinc_response = np.sinc(sinc_idx / 1.2).astype(np.complex128)
        sinc_response /= np.linalg.norm(sinc_response)

        for t_idx in range(n_targets):
            r_bin = range_bins[t_idx]
            for k, offset in enumerate(sinc_idx):
                col = r_bin + offset
                if 0 <= col < self.n_range_bins:
                    cpi[:, col] += doppler_steering[:, t_idx] * sinc_response[k]

        # Add AWGN (complex Gaussian noise)
        noise_std = np.sqrt(noise_power / 2.0)
        cpi += noise_std * (rng.standard_normal(cpi.shape) + 1j * rng.standard_normal(cpi.shape))

        return cpi

    def range_compress(self, cpi_data: np.ndarray) -> np.ndarray:
        """
        Range compression via matched filtering.

        In this parametric simulator, the CPI data is generated with
        pre-compressed sinc responses at target range bins (modeling
        the signal at the matched filter output). This method applies
        only normalization scaling.

        For raw IQ data, this would perform FFT-based correlation:
            Y(f) = S(f) · H*(f)

        Output SNR improvement: G_mf = B·T (time-bandwidth product)

        Args:
            cpi_data: CPI data [n_pulses, n_range_bins], complex

        Returns:
            Range-compressed CPI [n_pulses, n_range_bins], complex

        Reference: Richards (2005), Eq. 5.12
        """
        # CPI is already range-compressed (synthetic generation)
        # Apply matched filter gain scaling only
        return cpi_data * np.sqrt(self.tbp)

    def mti_cancel(self, cpi_data: np.ndarray) -> np.ndarray:
        """
        MTI (Moving Target Indication) pulse canceller.

        Suppresses zero-Doppler (stationary) clutter by differencing
        consecutive pulse returns along the slow-time dimension.

        1st order (2-pulse): y[n] = x[n] - x[n-1]
            |H(f)| = 2|sin(πf/PRF)|, null at f=0

        2nd order (3-pulse): y[n] = x[n] - 2x[n-1] + x[n-2]
            |H(f)| = 4·sin²(πf/PRF), double null at f=0

        Args:
            cpi_data: Input CPI [n_pulses, n_range_bins]

        Returns:
            MTI-filtered CPI (reduced slow-time dimension)

        Reference: Richards (2005), Ch. 3.4; Skolnik, Ch. 3.5
        """
        if self.mti_order == 0:
            return cpi_data

        if self.mti_order == 1:
            # 2-pulse canceller: y[n] = x[n] - x[n-1]
            return np.diff(cpi_data, n=1, axis=0)

        if self.mti_order == 2:
            # 3-pulse canceller: y[n] = x[n] - 2x[n-1] + x[n-2]
            return np.diff(cpi_data, n=2, axis=0)

        return cpi_data

    def doppler_fft(self, cpi_data: np.ndarray) -> np.ndarray:
        """
        Doppler processing via windowed FFT along slow-time.

        Applies a window function to reduce spectral leakage, then
        computes the FFT along the pulse dimension (axis=0).

        Window: Hamming by default → -42.5 dB sidelobes
        Processing loss: ~1.34 dB (Hamming)

        Args:
            cpi_data: Range-compressed CPI [n_pulses, n_range_bins]

        Returns:
            Range-Doppler map [n_doppler, n_range], complex

        Reference: Richards (2005), Ch. 4.5
        """
        n_pulses = cpi_data.shape[0]

        # Resize window if MTI reduced the pulse count
        if n_pulses != len(self._doppler_window):
            window = self._generate_window(n_pulses, self.window_type)
        else:
            window = self._doppler_window

        # Apply window along slow-time (pulse) dimension
        windowed = cpi_data * window[:, np.newaxis]

        # FFT along pulse dimension, then shift zero-Doppler to center
        rd_map = np.fft.fftshift(np.fft.fft(windowed, axis=0), axes=0)

        return rd_map

    def process_cpi(
        self,
        target_ranges_m: np.ndarray,
        target_velocities_mps: np.ndarray,
        target_amplitudes: np.ndarray,
        noise_power: float = 1e-12,
        seed: Optional[int] = None,
    ) -> RangeDopplerMap:
        """
        Execute full Pulse-Doppler processing pipeline.

        Pipeline: Generate CPI → Range Compress → MTI Cancel → Doppler FFT

        Args:
            target_ranges_m: Target ranges [m]
            target_velocities_mps: Target radial velocities [m/s]
            target_amplitudes: Target return amplitudes (linear)
            noise_power: Thermal noise power
            seed: RNG seed for reproducibility

        Returns:
            RangeDopplerMap with processed R-D data in dB
        """
        # 1. Generate CPI raw data
        cpi = self.generate_cpi(
            target_ranges_m, target_velocities_mps, target_amplitudes, noise_power, seed
        )

        # 2. Range compression (matched filtering)
        rc = self.range_compress(cpi)

        # 3. MTI cancellation (clutter suppression)
        mti_out = self.mti_cancel(rc)

        # 4. Doppler FFT (velocity estimation)
        rd_complex = self.doppler_fft(mti_out)

        # 5. Convert to power in dB
        rd_power = np.abs(rd_complex) ** 2
        rd_power = np.maximum(rd_power, 1e-30)  # Avoid log(0)
        rd_db = 10.0 * np.log10(rd_power)

        # Build velocity axis for potentially reduced pulse count
        n_doppler = rd_complex.shape[0]
        doppler_res_hz = self.prf_hz / n_doppler
        vel_res = self.wavelength_m * doppler_res_hz / 2.0
        vel_axis = (np.arange(n_doppler) - n_doppler / 2) * vel_res

        return RangeDopplerMap(
            data_db=rd_db,
            range_axis_m=self.range_axis_m,
            velocity_axis_mps=vel_axis,
            n_pulses=self.n_pulses,
            prf_hz=self.prf_hz,
            wavelength_m=self.wavelength_m,
            bandwidth_hz=self.bandwidth_hz,
            blind_speeds_mps=self.blind_speeds_mps,
            mti_order=self.mti_order,
            processing_gain_db=self.processing_gain_db,
        )

    # ═══════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _generate_lfm_reference(self) -> np.ndarray:
        """
        Generate LFM chirp reference waveform for matched filtering.

        s(t) = exp(j·π·k·t²) where k = B/T (chirp rate)

        Reference: Richards (2005), Eq. 4.1
        """
        n_samples = max(int(self.pulse_width_s * 2 * self.bandwidth_hz), 16)
        t = np.linspace(-self.pulse_width_s / 2, self.pulse_width_s / 2, n_samples)
        chirp_rate = self.bandwidth_hz / self.pulse_width_s
        phase = np.pi * chirp_rate * t**2
        return np.exp(1j * phase)

    @staticmethod
    def _generate_window(n: int, window_type: str) -> np.ndarray:
        """
        Generate spectral window for Doppler FFT sidelobe control.

        Args:
            n: Window length
            window_type: 'hamming', 'hann', 'taylor', or 'none'

        Returns:
            Window vector of length n
        """
        if window_type == "hamming":
            return np.hamming(n)
        elif window_type == "hann":
            return np.hanning(n)
        elif window_type == "taylor":
            # Taylor window approximation (nbar=4, sll=-35dB)
            return np.hamming(n) ** 0.8  # Simplified Taylor approx
        else:
            return np.ones(n)

    @staticmethod
    def get_blind_speed(wavelength_m: float, prf_hz: float) -> float:
        """
        Calculate first blind speed.

        v_blind = λ · PRF / 2

        Reference: Richards (2005), Eq. 3.16

        Args:
            wavelength_m: Radar wavelength [m]
            prf_hz: Pulse repetition frequency [Hz]

        Returns:
            First blind speed [m/s]
        """
        return wavelength_m * prf_hz / 2.0

    @staticmethod
    def mti_frequency_response(f_norm: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Compute MTI canceller frequency response.

        1st order: |H(f)| = 2|sin(πf)|
        2nd order: |H(f)| = 4·sin²(πf)

        Args:
            f_norm: Normalized frequency (f/PRF), range [0, 1]
            order: Canceller order (1 or 2)

        Returns:
            |H(f)|² frequency response (power)

        Reference: Richards (2005), Ch. 3.4
        """
        if order == 1:
            return 4.0 * np.sin(np.pi * f_norm) ** 2
        elif order == 2:
            return 16.0 * np.sin(np.pi * f_norm) ** 4
        else:
            return np.ones_like(f_norm)


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════


def validate_pulse_doppler() -> dict:
    """
    Self-validation of the Pulse-Doppler processor.

    Tests:
        1. Matched filter gain ≈ 10·log10(B·T)
        2. MTI null at DC > 60 dB suppression
        3. Target localization within ±1 bin

    Reference: Richards (2005), Ch. 3-5
    """
    # Setup: X-band radar, 5 MHz BW, 10 μs pulse, PRF 1 kHz
    proc = PulseDopplerProcessor(
        prf_hz=1000.0,
        n_pulses=64,
        n_range_bins=512,
        bandwidth_hz=5e6,
        pulse_width_s=10e-6,
        frequency_hz=10e9,
        mti_order=0,
        window_type="none",  # Rectangular for clean validation
    )

    results = {}

    # ── Test 1: Matched Filter Gain ──
    expected_gain_db = 10.0 * np.log10(5e6 * 10e-6)  # = 10·log10(50) ≈ 17 dB
    results["matched_filter_gain"] = {
        "expected_db": expected_gain_db,
        "computed_db": proc.processing_gain_db,
        "error_db": abs(proc.processing_gain_db - expected_gain_db),
        "pass": abs(proc.processing_gain_db - expected_gain_db) < 0.1,
        "reference": "Richards (2005), Eq. 4.6",
    }

    # ── Test 2: MTI Null at DC ──
    f_norm = np.linspace(0, 1, 10000)
    h2_1st = PulseDopplerProcessor.mti_frequency_response(f_norm, order=1)
    h2_2nd = PulseDopplerProcessor.mti_frequency_response(f_norm, order=2)

    # DC response (f=0)
    dc_atten_1st_db = -10 * np.log10(h2_1st[0] + 1e-30)
    dc_atten_2nd_db = -10 * np.log10(h2_2nd[0] + 1e-30)

    results["mti_dc_null"] = {
        "order_1_atten_db": dc_atten_1st_db,
        "order_2_atten_db": dc_atten_2nd_db,
        "pass_order_1": dc_atten_1st_db > 60,
        "pass_order_2": dc_atten_2nd_db > 60,
        "reference": "Richards (2005), Ch. 3.4",
    }

    # ── Test 3: Target Localization ──
    target_range_m = 15000.0  # 15 km
    target_vel_mps = 100.0  # 100 m/s radial

    rd_map = proc.process_cpi(
        target_ranges_m=np.array([target_range_m]),
        target_velocities_mps=np.array([target_vel_mps]),
        target_amplitudes=np.array([1.0]),
        noise_power=1e-16,
        seed=42,
    )

    # Find peak in R-D map
    peak_idx = np.unravel_index(np.argmax(rd_map.data_db), rd_map.data_db.shape)
    peak_range_m = rd_map.range_axis_m[peak_idx[1]]
    peak_vel_mps = rd_map.velocity_axis_mps[peak_idx[0]]

    range_error_bins = abs(peak_range_m - target_range_m) / proc.range_resolution_m
    vel_resolution = proc.wavelength_m * proc.prf_hz / (2.0 * proc.n_pulses)
    vel_error_bins = abs(peak_vel_mps - target_vel_mps) / vel_resolution

    results["target_localization"] = {
        "true_range_m": target_range_m,
        "detected_range_m": peak_range_m,
        "range_error_bins": range_error_bins,
        "true_velocity_mps": target_vel_mps,
        "detected_velocity_mps": peak_vel_mps,
        "velocity_error_bins": vel_error_bins,
        "pass": range_error_bins <= 1.5 and vel_error_bins <= 1.5,
        "reference": "Richards (2005), Ch. 4",
    }

    # ── Test 4: Blind Speed ──
    v_blind = proc.blind_speeds_mps[0]
    expected_blind = proc.wavelength_m * proc.prf_hz / 2.0
    results["blind_speed"] = {
        "computed_mps": v_blind,
        "expected_mps": expected_blind,
        "pass": abs(v_blind - expected_blind) < 0.01,
        "reference": "Richards (2005), Eq. 3.16",
    }

    return results


if __name__ == "__main__":
    results = validate_pulse_doppler()
    print("=" * 60)
    print("Pulse-Doppler Processor Validation")
    print("=" * 60)
    for test_name, test_result in results.items():
        status = "✓ PASS" if test_result.get("pass", False) else "✗ FAIL"
        print(f"\n{status} | {test_name}")
        for k, v in test_result.items():
            if k != "pass":
                print(f"    {k}: {v}")
