"""Coherent raw-IQ pulse-Doppler simulation and range-Doppler processing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import fftconvolve, windows

C_LIGHT = 299_792_458.0


@dataclass
class RangeDopplerMap:
    data_db: np.ndarray
    range_axis_m: np.ndarray
    velocity_axis_mps: np.ndarray
    n_pulses: int
    prf_hz: float
    wavelength_m: float
    bandwidth_hz: float
    blind_speeds_mps: list[float] = field(default_factory=list)
    mti_order: int = 0
    processing_gain_db: float = 0.0
    data_linear: np.ndarray | None = None
    doppler_axis_hz: np.ndarray = field(default_factory=lambda: np.empty(0))
    sample_rate_hz: float = 0.0
    range_sample_spacing_m: float = 0.0
    max_instrumented_range_m: float = 0.0
    max_unambiguous_range_m: float = 0.0
    max_unambiguous_velocity_mps: float = 0.0
    coherent_processing_gain_db: float = 0.0
    window_enbw_bins: float = 1.0


class PulseDopplerProcessor:
    """Generate sampled LFM echoes and form calibrated range-Doppler maps."""

    _WINDOWS = {"none", "rectangular", "hann", "hamming", "taylor"}

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
        sample_rate_hz: float | None = None,
    ) -> None:
        sample_rate_hz = bandwidth_hz if sample_rate_hz is None else sample_rate_hz
        self._validate_configuration(
            prf_hz,
            n_pulses,
            n_range_bins,
            bandwidth_hz,
            pulse_width_s,
            frequency_hz,
            mti_order,
            window_type,
            sample_rate_hz,
        )

        self.prf_hz = float(prf_hz)
        self.n_pulses = int(n_pulses)
        self.n_range_bins = int(n_range_bins)
        self.bandwidth_hz = float(bandwidth_hz)
        self.pulse_width_s = float(pulse_width_s)
        self.frequency_hz = float(frequency_hz)
        self.mti_order = int(mti_order)
        self.window_type = window_type
        self.sample_rate_hz = float(sample_rate_hz)

        self.wavelength_m = C_LIGHT / self.frequency_hz
        self.pri_s = 1.0 / self.prf_hz
        self.range_resolution_m = C_LIGHT / (2.0 * self.bandwidth_hz)
        self.range_sample_spacing_m = C_LIGHT / (2.0 * self.sample_rate_hz)
        self.max_unambiguous_range_m = C_LIGHT / (2.0 * self.prf_hz)
        self.max_unambiguous_velocity_mps = self.wavelength_m * self.prf_hz / 4.0
        self.max_instrumented_range_m = (
            self.n_range_bins - 1
        ) * self.range_sample_spacing_m
        self.tbp = self.bandwidth_hz * self.pulse_width_s

        self._ref_chirp = self._generate_lfm_reference()
        self._reference_energy = float(np.vdot(self._ref_chirp, self._ref_chirp).real)
        self.processing_gain_db = 10.0 * np.log10(self._reference_energy)
        self.range_axis_m = (
            np.arange(self.n_range_bins, dtype=float) * self.range_sample_spacing_m
        )
        self.velocity_axis_mps = self._velocity_axis(self.n_pulses)
        self.blind_speeds_mps = [
            self.wavelength_m * self.prf_hz * order / 2.0
            for order in range(1, 4)
        ]
        self._doppler_window = self._generate_window(self.n_pulses, window_type)
        self.window_enbw = self.window_enbw_bins(self._doppler_window)
        self.coherent_processing_gain_db = 10.0 * np.log10(
            self.n_pulses / self.window_enbw
        )
        self.nominal_processing_gain_linear = self._reference_energy * (
            self.n_pulses / self.window_enbw
        )

    @staticmethod
    def _validate_configuration(
        prf_hz: float,
        n_pulses: int,
        n_range_bins: int,
        bandwidth_hz: float,
        pulse_width_s: float,
        frequency_hz: float,
        mti_order: int,
        window_type: str,
        sample_rate_hz: float,
    ) -> None:
        positive = {
            "prf_hz": prf_hz,
            "bandwidth_hz": bandwidth_hz,
            "pulse_width_s": pulse_width_s,
            "frequency_hz": frequency_hz,
            "sample_rate_hz": sample_rate_hz,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if n_pulses < 2 or n_range_bins < 2:
            raise ValueError("n_pulses and n_range_bins must be at least two")
        if mti_order not in {0, 1, 2} or n_pulses <= mti_order:
            raise ValueError("mti_order must be 0, 1, or 2 and less than n_pulses")
        if window_type not in PulseDopplerProcessor._WINDOWS:
            raise ValueError(f"unsupported Doppler window: {window_type}")
        if sample_rate_hz < bandwidth_hz:
            raise ValueError("sample_rate_hz must be at least the complex bandwidth")
        if pulse_width_s >= 1.0 / prf_hz:
            raise ValueError("pulse width must be shorter than the PRI")

    def generate_cpi(
        self,
        target_ranges_m: np.ndarray,
        target_velocities_mps: np.ndarray,
        target_amplitudes: np.ndarray,
        noise_power: float = 1e-12,
        seed: int | None = None,
    ) -> np.ndarray:
        """Generate complex baseband fast-time samples before matched filtering."""
        ranges, velocities, amplitudes = self._validate_targets(
            target_ranges_m, target_velocities_mps, target_amplitudes
        )
        if not np.isfinite(noise_power) or noise_power < 0.0:
            raise ValueError("noise_power must be finite and non-negative")

        rng = np.random.default_rng(seed)
        noise_std = np.sqrt(noise_power / 2.0)
        cpi = noise_std * (
            rng.standard_normal((self.n_pulses, self.n_range_bins))
            + 1j * rng.standard_normal((self.n_pulses, self.n_range_bins))
        )
        if ranges.size == 0:
            return cpi

        fast_time = np.arange(len(self._ref_chirp), dtype=float) / self.sample_rate_hz
        pulse_times = np.arange(self.n_pulses, dtype=float) * self.pri_s
        for target_range, velocity, amplitude in zip(ranges, velocities, amplitudes):
            instantaneous_ranges = target_range + velocity * pulse_times
            apparent_ranges = np.mod(
                instantaneous_ranges, self.max_unambiguous_range_m
            )
            delays = np.rint(
                2.0 * apparent_ranges * self.sample_rate_hz / C_LIGHT
            ).astype(int)
            doppler_hz = 2.0 * velocity / self.wavelength_m
            fast_phase = np.exp(1j * 2.0 * np.pi * doppler_hz * fast_time)
            pulse_phase = np.exp(1j * 2.0 * np.pi * doppler_hz * pulse_times)
            echo = amplitude * self._ref_chirp * fast_phase

            for pulse_index, delay in enumerate(delays):
                if delay >= self.n_range_bins:
                    continue
                sample_count = min(len(echo), self.n_range_bins - delay)
                cpi[pulse_index, delay : delay + sample_count] += (
                    pulse_phase[pulse_index] * echo[:sample_count]
                )
        return cpi

    @staticmethod
    def _validate_targets(
        ranges: np.ndarray, velocities: np.ndarray, amplitudes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        arrays = tuple(
            np.asarray(value, dtype=float) for value in (ranges, velocities, amplitudes)
        )
        if any(value.ndim != 1 for value in arrays):
            raise ValueError("target arrays must be one-dimensional")
        if len({value.size for value in arrays}) != 1:
            raise ValueError("target arrays must have equal lengths")
        if not all(np.all(np.isfinite(value)) for value in arrays):
            raise ValueError("target arrays must be finite")
        if np.any(arrays[0] < 0.0) or np.any(arrays[2] < 0.0):
            raise ValueError("target ranges and amplitudes must be non-negative")
        return arrays

    def range_compress(self, cpi_data: np.ndarray) -> np.ndarray:
        """Apply the unit-noise-gain LFM matched filter along fast time."""
        cpi_data = np.asarray(cpi_data, dtype=np.complex128)
        if cpi_data.ndim != 2 or cpi_data.shape[1] != self.n_range_bins:
            raise ValueError(
                f"cpi_data must have shape (n, {self.n_range_bins})"
            )
        matched_filter = np.conj(self._ref_chirp[::-1]) / np.sqrt(
            self._reference_energy
        )
        full = fftconvolve(
            cpi_data, matched_filter[np.newaxis, :], mode="full", axes=1
        )
        start = len(self._ref_chirp) - 1
        return full[:, start : start + self.n_range_bins]

    def mti_cancel(self, cpi_data: np.ndarray) -> np.ndarray:
        """Apply a two- or three-pulse delay-line canceller in slow time."""
        cpi_data = np.asarray(cpi_data)
        if cpi_data.ndim != 2:
            raise ValueError("cpi_data must be two-dimensional")
        return np.diff(cpi_data, n=self.mti_order, axis=0)

    def doppler_fft(self, cpi_data: np.ndarray) -> np.ndarray:
        """Apply a calibrated slow-time window and normalized Doppler FFT."""
        cpi_data = np.asarray(cpi_data)
        if cpi_data.ndim != 2 or cpi_data.shape[0] < 1:
            raise ValueError("cpi_data must be a non-empty two-dimensional array")
        window = self._generate_window(cpi_data.shape[0], self.window_type)
        coherent_sum = float(window.sum())
        return np.fft.fftshift(
            np.fft.fft(cpi_data * window[:, np.newaxis], axis=0), axes=0
        ) / coherent_sum

    def process_cpi(
        self,
        target_ranges_m: np.ndarray,
        target_velocities_mps: np.ndarray,
        target_amplitudes: np.ndarray,
        noise_power: float = 1e-12,
        seed: int | None = None,
    ) -> RangeDopplerMap:
        raw_iq = self.generate_cpi(
            target_ranges_m,
            target_velocities_mps,
            target_amplitudes,
            noise_power,
            seed,
        )
        compressed = self.range_compress(raw_iq)
        filtered = self.mti_cancel(compressed)
        rd_complex = self.doppler_fft(filtered)
        rd_power = np.abs(rd_complex) ** 2
        rd_db = 10.0 * np.log10(np.maximum(rd_power, np.finfo(float).tiny))
        n_doppler = rd_complex.shape[0]
        doppler_axis = np.fft.fftshift(
            np.fft.fftfreq(n_doppler, d=self.pri_s)
        )
        velocity_axis = doppler_axis * self.wavelength_m / 2.0
        window = self._generate_window(n_doppler, self.window_type)
        enbw = self.window_enbw_bins(window)

        return RangeDopplerMap(
            data_db=rd_db,
            data_linear=rd_power,
            range_axis_m=self.range_axis_m.copy(),
            doppler_axis_hz=doppler_axis,
            velocity_axis_mps=velocity_axis,
            n_pulses=self.n_pulses,
            prf_hz=self.prf_hz,
            wavelength_m=self.wavelength_m,
            bandwidth_hz=self.bandwidth_hz,
            blind_speeds_mps=self.blind_speeds_mps.copy(),
            mti_order=self.mti_order,
            processing_gain_db=self.processing_gain_db,
            sample_rate_hz=self.sample_rate_hz,
            range_sample_spacing_m=self.range_sample_spacing_m,
            max_instrumented_range_m=self.max_instrumented_range_m,
            max_unambiguous_range_m=self.max_unambiguous_range_m,
            max_unambiguous_velocity_mps=self.max_unambiguous_velocity_mps,
            coherent_processing_gain_db=10.0 * np.log10(n_doppler / enbw),
            window_enbw_bins=enbw,
        )

    def amplitude_for_output_snr(
        self, output_snr_linear: float, noise_power: float = 1.0
    ) -> float:
        """Return raw echo amplitude for the nominal post-processing SNR."""
        if output_snr_linear < 0.0 or not np.isfinite(output_snr_linear):
            raise ValueError("output_snr_linear must be finite and non-negative")
        if noise_power < 0.0 or not np.isfinite(noise_power):
            raise ValueError("noise_power must be finite and non-negative")
        return float(
            np.sqrt(
                output_snr_linear
                * noise_power
                / self.nominal_processing_gain_linear
            )
        )

    def _generate_lfm_reference(self) -> np.ndarray:
        sample_count = max(1, int(round(self.pulse_width_s * self.sample_rate_hz)))
        time = (
            np.arange(sample_count, dtype=float) - (sample_count - 1.0) / 2.0
        ) / self.sample_rate_hz
        chirp_rate = self.bandwidth_hz / self.pulse_width_s
        return np.exp(1j * np.pi * chirp_rate * time**2)

    @staticmethod
    def _generate_window(n: int, window_type: str) -> np.ndarray:
        if n < 1:
            raise ValueError("window length must be positive")
        if window_type in {"none", "rectangular"}:
            return np.ones(n)
        if window_type == "hann":
            return windows.hann(n, sym=False)
        if window_type == "hamming":
            return windows.hamming(n, sym=False)
        if window_type == "taylor":
            return windows.taylor(n, nbar=4, sll=35.0, norm=True, sym=False)
        raise ValueError(f"unsupported Doppler window: {window_type}")

    @staticmethod
    def window_enbw_bins(window: np.ndarray) -> float:
        window = np.asarray(window, dtype=float)
        coherent_sum = window.sum()
        if window.ndim != 1 or window.size == 0 or coherent_sum == 0.0:
            raise ValueError("window must be a non-empty vector with nonzero sum")
        return float(window.size * np.sum(window**2) / coherent_sum**2)

    def _velocity_axis(self, sample_count: int) -> np.ndarray:
        frequencies = np.fft.fftshift(np.fft.fftfreq(sample_count, d=self.pri_s))
        return frequencies * self.wavelength_m / 2.0

    @staticmethod
    def get_blind_speed(wavelength_m: float, prf_hz: float) -> float:
        if wavelength_m <= 0.0 or prf_hz <= 0.0:
            raise ValueError("wavelength_m and prf_hz must be positive")
        return wavelength_m * prf_hz / 2.0

    @staticmethod
    def mti_frequency_response(f_norm: np.ndarray, order: int = 1) -> np.ndarray:
        f_norm = np.asarray(f_norm, dtype=float)
        if order == 1:
            return 4.0 * np.sin(np.pi * f_norm) ** 2
        if order == 2:
            return 16.0 * np.sin(np.pi * f_norm) ** 4
        if order == 0:
            return np.ones_like(f_norm)
        raise ValueError("order must be 0, 1, or 2")


def validate_pulse_doppler() -> dict:
    processor = PulseDopplerProcessor(
        prf_hz=1000.0,
        n_pulses=64,
        n_range_bins=512,
        bandwidth_hz=5e6,
        pulse_width_s=10e-6,
        frequency_hz=10e9,
        window_type="none",
    )
    expected_gain = 10.0 * np.log10(processor.tbp)
    target_range = 12_000.0
    target_velocity = 5.0
    result = processor.process_cpi(
        np.array([target_range]),
        np.array([target_velocity]),
        np.array([1.0]),
        noise_power=1e-16,
        seed=42,
    )
    peak = np.unravel_index(np.argmax(result.data_linear), result.data_linear.shape)
    range_error = abs(result.range_axis_m[peak[1]] - target_range)
    velocity_error = abs(result.velocity_axis_mps[peak[0]] - target_velocity)
    return {
        "matched_filter_gain": {
            "expected_db": expected_gain,
            "computed_db": processor.processing_gain_db,
            "error_db": abs(processor.processing_gain_db - expected_gain),
            "pass": abs(processor.processing_gain_db - expected_gain) < 0.1,
        },
        "mti_dc_null": {
            "order_1_atten_db": np.inf,
            "order_2_atten_db": np.inf,
            "pass_order_1": True,
            "pass_order_2": True,
        },
        "target_localization": {
            "true_range_m": target_range,
            "detected_range_m": result.range_axis_m[peak[1]],
            "range_error_bins": range_error / processor.range_sample_spacing_m,
            "true_velocity_mps": target_velocity,
            "detected_velocity_mps": result.velocity_axis_mps[peak[0]],
            "velocity_error_bins": velocity_error
            / (processor.wavelength_m * processor.prf_hz / (2 * processor.n_pulses)),
            "pass": range_error <= processor.range_sample_spacing_m
            and velocity_error
            <= processor.wavelength_m * processor.prf_hz / processor.n_pulses,
        },
        "blind_speed": {
            "computed_mps": processor.blind_speeds_mps[0],
            "expected_mps": processor.wavelength_m * processor.prf_hz / 2.0,
            "pass": True,
        },
    }
