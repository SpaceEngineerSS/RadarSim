"""Broadside stripmap SAR and turntable ISAR signal processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.constants import c
from scipy.signal import fftconvolve, windows


@dataclass
class SARImageResult:
    image_db: np.ndarray
    range_axis_m: np.ndarray
    cross_range_axis_m: np.ndarray
    range_resolution_m: float
    azimuth_resolution_m: float
    complex_image: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


def _validate_positive(**values: float) -> None:
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")


def _lfm_chirp(bandwidth_hz: float, pulse_width_s: float, sample_rate_hz: float) -> np.ndarray:
    sample_count = max(1, int(round(pulse_width_s * sample_rate_hz)))
    fast_time = (
        np.arange(sample_count, dtype=float) - (sample_count - 1.0) / 2.0
    ) / sample_rate_hz
    chirp_rate = bandwidth_hz / pulse_width_s
    return np.exp(1j * np.pi * chirp_rate * fast_time**2)


def _range_compress(raw_data: np.ndarray, reference: np.ndarray, axis: int) -> np.ndarray:
    matched = np.conj(reference[::-1]) / np.linalg.norm(reference)
    shape = [1] * raw_data.ndim
    shape[axis] = matched.size
    full = fftconvolve(raw_data, matched.reshape(shape), mode="full", axes=axis)
    slices = [slice(None)] * raw_data.ndim
    start = matched.size - 1
    slices[axis] = slice(start, start + raw_data.shape[axis])
    return full[tuple(slices)]


def _normalized_db(image: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    magnitude = np.abs(image)
    peak = float(magnitude.max(initial=0.0))
    if peak == 0.0:
        return np.full(magnitude.shape, floor_db)
    return np.maximum(20.0 * np.log10(np.maximum(magnitude / peak, 1e-15)), floor_db)


def rda_vectorized(
    raw_data: np.ndarray,
    bandwidth_hz: float,
    prf_hz: float,
    fc_hz: float,
    platform_velocity_mps: float,
    antenna_length_m: float = 1.0,
    *,
    sample_rate_hz: float | None = None,
    pulse_width_s: float = 1e-6,
    reference_range_m: float | None = None,
    range_gate_start_m: float = 0.0,
) -> SARImageResult:
    """Focus broadside, constant-velocity stripmap data with the RDA."""
    sample_rate_hz = bandwidth_hz if sample_rate_hz is None else sample_rate_hz
    _validate_positive(
        bandwidth_hz=bandwidth_hz,
        prf_hz=prf_hz,
        fc_hz=fc_hz,
        platform_velocity_mps=platform_velocity_mps,
        antenna_length_m=antenna_length_m,
        sample_rate_hz=sample_rate_hz,
        pulse_width_s=pulse_width_s,
    )
    if sample_rate_hz < bandwidth_hz:
        raise ValueError("sample_rate_hz must be at least the complex bandwidth")
    if range_gate_start_m < 0.0 or not np.isfinite(range_gate_start_m):
        raise ValueError("range_gate_start_m must be finite and non-negative")
    raw_data = np.asarray(raw_data, dtype=np.complex128)
    if raw_data.ndim != 2 or min(raw_data.shape) < 2:
        raise ValueError("raw_data must be a two-dimensional complex array")
    if not np.all(np.isfinite(raw_data)):
        raise ValueError("raw_data must be finite")

    n_range, n_azimuth = raw_data.shape
    wavelength = c / fc_hz
    range_spacing = c / (2.0 * sample_rate_hz)
    range_resolution = c / (2.0 * bandwidth_hz)
    aperture_length = platform_velocity_mps * n_azimuth / prf_hz
    if reference_range_m is None:
        reference_range_m = range_gate_start_m + range_spacing * (n_range - 1) / 2.0
    if reference_range_m <= 0.0 or not np.isfinite(reference_range_m):
        raise ValueError("reference_range_m must be finite and positive")

    chirp = _lfm_chirp(bandwidth_hz, pulse_width_s, sample_rate_hz)
    range_compressed = _range_compress(raw_data, chirp, axis=0)
    doppler_data = np.fft.fftshift(
        np.fft.fft(range_compressed, axis=1), axes=1
    )
    doppler_hz = np.fft.fftshift(np.fft.fftfreq(n_azimuth, d=1.0 / prf_hz))
    normalized = wavelength * doppler_hz / (2.0 * platform_velocity_mps)
    visible = np.abs(normalized) < 1.0
    migration_m = np.zeros_like(doppler_hz)
    migration_m[visible] = reference_range_m * (
        1.0 / np.sqrt(1.0 - normalized[visible] ** 2) - 1.0
    )

    source_bins = np.arange(n_range, dtype=float)
    corrected = np.zeros_like(doppler_data)
    for column, shift_bins in enumerate(migration_m / range_spacing):
        if not visible[column]:
            continue
        query = source_bins + shift_bins
        corrected[:, column] = np.interp(
            query, source_bins, doppler_data[:, column].real, left=0.0, right=0.0
        ) + 1j * np.interp(
            query, source_bins, doppler_data[:, column].imag, left=0.0, right=0.0
        )

    azimuth_chirp_rate = -2.0 * platform_velocity_mps**2 / (
        wavelength * reference_range_m
    )
    azimuth_filter = np.exp(
        1j * np.pi * doppler_hz**2 / azimuth_chirp_rate
    )
    azimuth_filter[~visible] = 0.0
    focused = np.fft.ifft(
        np.fft.ifftshift(corrected * azimuth_filter[np.newaxis, :], axes=1),
        axis=1,
    )

    real_aperture_limit = antenna_length_m / 2.0
    sampled_aperture_limit = wavelength * reference_range_m / (
        2.0 * aperture_length
    )
    azimuth_resolution = max(real_aperture_limit, sampled_aperture_limit)
    range_axis = range_gate_start_m + np.arange(n_range, dtype=float) * range_spacing
    cross_range_axis = (
        np.arange(n_azimuth, dtype=float) - (n_azimuth - 1.0) / 2.0
    ) * platform_velocity_mps / prf_hz

    return SARImageResult(
        image_db=_normalized_db(focused),
        complex_image=focused,
        range_axis_m=range_axis,
        cross_range_axis_m=cross_range_axis,
        range_resolution_m=range_resolution,
        azimuth_resolution_m=azimuth_resolution,
        metadata={
            "sample_rate_hz": sample_rate_hz,
            "reference_range_m": reference_range_m,
            "aperture_length_m": aperture_length,
            "range_sample_spacing_m": range_spacing,
            "range_gate_start_m": range_gate_start_m,
            "algorithm": "range_doppler",
        },
    )


class AdvancedSARISAR:
    """Stripmap SAR acquisition model paired with the supported RDA focuser."""

    def __init__(
        self,
        fc: float = 10e9,
        bandwidth: float = 100e6,
        prf: float = 1000.0,
        pulse_width: float = 1e-6,
        platform_velocity: float = 100.0,
        synthetic_aperture: float = 100.0,
        antenna_length_m: float = 1.0,
        sample_rate_hz: float | None = None,
    ) -> None:
        sample_rate_hz = bandwidth if sample_rate_hz is None else sample_rate_hz
        _validate_positive(
            fc=fc,
            bandwidth=bandwidth,
            prf=prf,
            pulse_width=pulse_width,
            platform_velocity=platform_velocity,
            synthetic_aperture=synthetic_aperture,
            antenna_length_m=antenna_length_m,
            sample_rate_hz=sample_rate_hz,
        )
        if sample_rate_hz < bandwidth:
            raise ValueError("sample_rate_hz must be at least the complex bandwidth")
        self.fc = float(fc)
        self.bandwidth = float(bandwidth)
        self.prf = float(prf)
        self.pulse_width = float(pulse_width)
        self.v = float(platform_velocity)
        self.L = float(synthetic_aperture)
        self.antenna_length = float(antenna_length_m)
        self.sample_rate_hz = float(sample_rate_hz)
        self.wavelength = c / self.fc
        self.λ = self.wavelength
        self.range_resolution = c / (2.0 * self.bandwidth)
        self.azimuth_resolution = self.antenna_length / 2.0
        self._last_reference_range_m: float | None = None
        self._last_range_gate_start_m = 0.0

    def generate_chirp_reference(self) -> np.ndarray:
        return _lfm_chirp(self.bandwidth, self.pulse_width, self.sample_rate_hz)

    def generate_sar_raw_data(
        self,
        target_positions: np.ndarray,
        target_rcs: np.ndarray,
        range_samples: int = 1024,
        azimuth_samples: int = 512,
        noise_power: float = 0.0,
        seed: int | None = None,
        range_gate_start_m: float | None = None,
    ) -> np.ndarray:
        positions = np.asarray(target_positions, dtype=float)
        rcs = np.asarray(target_rcs, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("target_positions must have shape (n, 3)")
        if rcs.shape != (positions.shape[0],):
            raise ValueError("target_rcs must have one value per target")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(rcs)):
            raise ValueError("target data must be finite")
        if np.any(rcs < 0.0) or range_samples < 2 or azimuth_samples < 2:
            raise ValueError("RCS must be non-negative and sample counts at least two")
        if noise_power < 0.0 or not np.isfinite(noise_power):
            raise ValueError("noise_power must be finite and non-negative")

        rng = np.random.default_rng(seed)
        noise_std = np.sqrt(noise_power / 2.0)
        raw = noise_std * (
            rng.standard_normal((range_samples, azimuth_samples))
            + 1j * rng.standard_normal((range_samples, azimuth_samples))
        )
        chirp = self.generate_chirp_reference()
        slow_time = (
            np.arange(azimuth_samples, dtype=float) - (azimuth_samples - 1.0) / 2.0
        ) / self.prf
        platform_x = self.v * slow_time
        ranges_at_center = np.linalg.norm(positions, axis=1)
        range_spacing = c / (2.0 * self.sample_rate_hz)
        if range_gate_start_m is None:
            gate_center = float(np.median(ranges_at_center)) if ranges_at_center.size else 0.0
            gate_span = (range_samples - 1) * range_spacing
            range_gate_start_m = max(0.0, gate_center - gate_span / 2.0)
        if range_gate_start_m < 0.0 or not np.isfinite(range_gate_start_m):
            raise ValueError("range_gate_start_m must be finite and non-negative")
        self._last_range_gate_start_m = float(range_gate_start_m)

        for position, target_rcs_m2 in zip(positions, rcs):
            x, y, z = position
            slant_range = np.sqrt((x - platform_x) ** 2 + y**2 + z**2)
            delays = np.rint(
                2.0
                * (slant_range - self._last_range_gate_start_m)
                * self.sample_rate_hz
                / c
            ).astype(int)
            phase = np.exp(-1j * 4.0 * np.pi * slant_range / self.wavelength)
            amplitude = np.sqrt(target_rcs_m2)
            for azimuth_index, delay in enumerate(delays):
                if delay < 0 or delay >= range_samples:
                    continue
                count = min(chirp.size, range_samples - delay)
                raw[delay : delay + count, azimuth_index] += (
                    amplitude * phase[azimuth_index] * chirp[:count]
                )

        if ranges_at_center.size:
            self._last_reference_range_m = float(np.median(ranges_at_center))
        return raw

    def range_doppler_algorithm(self, raw_data: np.ndarray) -> np.ndarray:
        result = rda_vectorized(
            raw_data,
            self.bandwidth,
            self.prf,
            self.fc,
            self.v,
            self.antenna_length,
            sample_rate_hz=self.sample_rate_hz,
            pulse_width_s=self.pulse_width,
            reference_range_m=self._last_reference_range_m,
            range_gate_start_m=self._last_range_gate_start_m,
        )
        return result.complex_image

    def omega_k_algorithm(self, raw_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "omega-k focusing requires acquisition-specific wavenumber support"
        )

    def chirp_scaling_algorithm(self, raw_data: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "chirp scaling requires acquisition-specific Doppler-rate geometry"
        )

    def calculate_image_quality(self, image: np.ndarray) -> dict[str, float]:
        magnitude = np.abs(np.asarray(image, dtype=np.complex128))
        if magnitude.ndim != 2 or magnitude.size == 0 or not np.all(np.isfinite(magnitude)):
            raise ValueError("image must be a finite non-empty two-dimensional array")
        power = magnitude**2
        noise_power = float(np.median(power) / np.log(2.0))
        peak_power = float(power.max())
        signal_power = max(peak_power - noise_power, 0.0)
        snr_db = 10.0 * np.log10(max(signal_power / max(noise_power, 1e-30), 1e-30))
        mean_magnitude = float(magnitude.mean())
        nonzero = magnitude[magnitude > 0.0]
        floor = float(np.percentile(nonzero, 1.0)) if nonzero.size else 1e-30
        dynamic_range = 20.0 * np.log10(max(float(magnitude.max()) / max(floor, 1e-30), 1.0))
        return {
            "SNR_dB": snr_db,
            "Contrast": float(magnitude.std() / max(mean_magnitude, 1e-30)),
            "Range_Resolution_m": self.range_resolution,
            "Azimuth_Resolution_m": self.azimuth_resolution,
            "Resolution_m2": self.range_resolution * self.azimuth_resolution,
            "Dynamic_Range_dB": dynamic_range,
        }


class ISARProcessor:
    """Range-align and Doppler-focus turntable ISAR measurements."""

    def __init__(
        self,
        fc_hz: float = 10e9,
        bandwidth_hz: float = 100e6,
        prf_hz: float = 1000.0,
        n_pulses: int = 64,
        pulse_width_s: float = 1e-6,
        sample_rate_hz: float | None = None,
    ) -> None:
        sample_rate_hz = bandwidth_hz if sample_rate_hz is None else sample_rate_hz
        _validate_positive(
            fc_hz=fc_hz,
            bandwidth_hz=bandwidth_hz,
            prf_hz=prf_hz,
            pulse_width_s=pulse_width_s,
            sample_rate_hz=sample_rate_hz,
        )
        if n_pulses < 2:
            raise ValueError("n_pulses must be at least two")
        if sample_rate_hz < bandwidth_hz:
            raise ValueError("sample_rate_hz must be at least the complex bandwidth")
        self.fc_hz = float(fc_hz)
        self.bandwidth_hz = float(bandwidth_hz)
        self.prf_hz = float(prf_hz)
        self.n_pulses = int(n_pulses)
        self.pulse_width_s = float(pulse_width_s)
        self.sample_rate_hz = float(sample_rate_hz)
        self.wavelength_m = c / self.fc_hz
        self.range_res_m = c / (2.0 * self.bandwidth_hz)

    def process_isar(
        self,
        cpi_data: np.ndarray,
        rotation_rate_rps: float = 0.01,
        *,
        range_compressed: bool = False,
    ) -> SARImageResult:
        data = np.asarray(cpi_data, dtype=np.complex128)
        if data.ndim != 2 or min(data.shape) < 2 or not np.all(np.isfinite(data)):
            raise ValueError("cpi_data must be a finite two-dimensional array")
        if rotation_rate_rps == 0.0 or not np.isfinite(rotation_rate_rps):
            raise ValueError("rotation_rate_rps must be finite and nonzero")
        n_pulses, n_range = data.shape
        if not range_compressed:
            chirp = _lfm_chirp(
                self.bandwidth_hz, self.pulse_width_s, self.sample_rate_hz
            )
            profiles = _range_compress(data, chirp, axis=1)
        else:
            profiles = data.copy()
        aligned = self._motion_compensate(profiles)
        window = windows.hamming(n_pulses, sym=False)
        focused = np.fft.fftshift(
            np.fft.fft(aligned * window[:, np.newaxis], axis=0), axes=0
        ) / window.sum()

        doppler_hz = np.fft.fftshift(
            np.fft.fftfreq(n_pulses, d=1.0 / self.prf_hz)
        )
        cross_range_axis = (
            self.wavelength_m * doppler_hz / (2.0 * rotation_rate_rps)
        )
        angular_span = abs(rotation_rate_rps) * n_pulses / self.prf_hz
        cross_range_resolution = self.wavelength_m / (2.0 * angular_span)
        range_axis = np.arange(n_range, dtype=float) * c / (
            2.0 * self.sample_rate_hz
        )
        return SARImageResult(
            image_db=_normalized_db(focused),
            complex_image=focused,
            range_axis_m=range_axis,
            cross_range_axis_m=cross_range_axis,
            range_resolution_m=self.range_res_m,
            azimuth_resolution_m=cross_range_resolution,
            metadata={
                "rotation_rate_rad_s": rotation_rate_rps,
                "angular_span_rad": angular_span,
                "algorithm": "range_aligned_doppler",
            },
        )

    @staticmethod
    def _shift_without_wrap(profile: np.ndarray, shift: int) -> np.ndarray:
        shifted = np.zeros_like(profile)
        if shift == 0:
            shifted[:] = profile
        elif shift > 0:
            shifted[shift:] = profile[:-shift]
        else:
            shifted[:shift] = profile[-shift:]
        return shifted

    def _motion_compensate(self, range_profiles: np.ndarray) -> np.ndarray:
        n_pulses, n_range = range_profiles.shape
        reference = np.abs(range_profiles[n_pulses // 2])
        aligned = np.zeros_like(range_profiles)
        for pulse_index, profile in enumerate(range_profiles):
            correlation = np.correlate(np.abs(profile), reference, mode="full")
            displacement = int(np.argmax(correlation) - (n_range - 1))
            aligned[pulse_index] = self._shift_without_wrap(profile, -displacement)
        return aligned
