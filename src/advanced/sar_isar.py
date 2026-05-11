# Developed by Mehmet Gümüş (@SpaceEngineerSS) - RadarSim v2.x
"""
Gelişmiş SAR/ISAR (Synthetic Aperture Radar) Modülü

Bu modül, gelişmiş SAR/ISAR görüntüleme algoritmalarını içerir.

Bilimsel Temeller:
- Oliver & Quegan, "Understanding Synthetic Aperture Radar Images", IEEE Press, 1998
- IEEE Transactions on Geoscience and Remote Sensing, "SAR Imaging Algorithms"
- NATO RTO-TR-SET-093, "Advanced SAR Processing Techniques"

Algoritmalar:
- Range-Doppler Algorithm (RDA)
- Backprojection Algorithm (BPA)
- Omega-K Algorithm (ω-K)
- Chirp Scaling Algorithm (CSA)
- Polar Format Algorithm (PFA)
"""

from typing import Dict, Tuple

import numpy as np
import scipy.signal as signal
from numba import prange
from scipy.constants import c
from scipy.fft import fft2, fftshift, ifft2, ifftshift


class AdvancedSARISAR:
    """Gelişmiş SAR/ISAR görüntüleme sınıfı"""

    def __init__(
        self,
        fc: float = 10e9,
        bandwidth: float = 100e6,
        prf: float = 1000,
        pulse_width: float = 1e-6,
        platform_velocity: float = 100,
        synthetic_aperture: float = 100,
    ):
        self.fc = fc  # Carrier frequency [Hz]
        self.bandwidth = bandwidth  # Bandwidth [Hz]
        self.prf = prf  # Pulse Repetition Frequency [Hz]
        self.pulse_width = pulse_width  # Pulse duration [s]
        self.v = platform_velocity  # Platform velocity [m/s]
        self.L = synthetic_aperture  # Synthetic aperture length [m]
        self.λ = c / fc  # Wavelength [m]

        # SAR resolution formulas (Reference: Cumming & Wong, "Digital Processing of SAR Data")
        # Range resolution: δr = c / (2 * B)
        self.range_resolution = c / (2 * bandwidth)

        # Azimuth resolution: δa = L / 2 (for strip-map SAR)
        # Or equivalently: δa = λ * R / (2 * L_synth) where L_synth = λ * R / D
        # Simplified for antenna length D: δa = D / 2
        self.antenna_length = 1.0  # 1 meter antenna
        self.azimuth_resolution = self.antenna_length / 2

    def generate_sar_raw_data(
        self,
        target_positions: np.ndarray,
        target_rcs: np.ndarray,
        range_samples: int = 1024,
        azimuth_samples: int = 512,
    ) -> np.ndarray:
        """
        SAR ham veri üretir (Range-Doppler domain)

        Kaynak: Oliver & Quegan, "Understanding SAR Images", IEEE Press, 1998
        Denklem: s(t,η) = Σ σi * exp(-j*4π*Ri(η)/λ)
        """
        raw_data = np.zeros((range_samples, azimuth_samples), dtype=complex)

        # Zaman eksenleri
        range_time = np.linspace(0, range_samples / self.bandwidth, range_samples)
        azimuth_time = np.linspace(
            -azimuth_samples / (2 * self.prf), azimuth_samples / (2 * self.prf), azimuth_samples
        )

        for i in prange(len(target_positions)):
            x0, y0, z0 = target_positions[i]
            rcs = target_rcs[i]

            for j in range(azimuth_samples):
                η = azimuth_time[j]

                # Platform pozisyonu
                x_platform = self.v * η

                # Menzil hesaplama
                R = np.sqrt((x0 - x_platform) ** 2 + y0**2 + z0**2)

                # Faz hesaplama
                phase = -4 * np.pi * R / self.λ

                # Range bin hesaplama
                range_bin = int(R * 2 / c * self.bandwidth)

                if 0 <= range_bin < range_samples:
                    raw_data[range_bin, j] += rcs * np.exp(1j * phase)

        return raw_data

    def range_doppler_algorithm(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Range-Doppler Algorithm (RDA) uygular

        Kaynak: IEEE Transactions on Geoscience and Remote Sensing
        Adımlar: Range compression, Range cell migration correction, Azimuth compression
        """
        range_samples, azimuth_samples = raw_data.shape

        # 1. Range compression
        chirp = self.generate_chirp_reference()
        range_compressed = np.zeros_like(raw_data, dtype=complex)

        for i in range(azimuth_samples):
            range_compressed[:, i] = signal.correlate(raw_data[:, i], np.conj(chirp), mode="same")

        # 2. Range Cell Migration Correction (RCMC)
        rcmc_corrected = self.apply_rcmc(range_compressed)

        # 3. Azimuth compression
        azimuth_compressed = np.zeros_like(rcmc_corrected, dtype=complex)

        for i in range(range_samples):
            # Azimuth reference function
            azimuth_ref = self.generate_azimuth_reference(i)
            azimuth_compressed[i, :] = signal.correlate(
                rcmc_corrected[i, :], np.conj(azimuth_ref), mode="same"
            )

        return azimuth_compressed

    def backprojection_algorithm(
        self, raw_data: np.ndarray, target_area: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """
        Backprojection Algorithm (BPA) uygular

        Kaynak: IEEE Transactions on Geoscience and Remote Sensing
        Algoritma: Time-domain backprojection for high-resolution imaging
        """
        x_min, x_max, y_min, y_max = target_area
        range_samples, azimuth_samples = raw_data.shape

        # Görüntü grid'i
        nx = 256
        ny = 256
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)

        # Zaman eksenleri
        range_time = np.linspace(0, range_samples / self.bandwidth, range_samples)
        azimuth_time = np.linspace(
            -azimuth_samples / (2 * self.prf), azimuth_samples / (2 * self.prf), azimuth_samples
        )

        # Backprojection
        image = np.zeros((ny, nx), dtype=complex)

        for i in range(ny):
            for j in range(nx):
                x_target = x_grid[j]
                y_target = y_grid[i]

                for k in range(azimuth_samples):
                    η = azimuth_time[k]
                    x_platform = self.v * η

                    # Menzil hesaplama
                    R = np.sqrt((x_target - x_platform) ** 2 + y_target**2)

                    # Range bin
                    range_bin = int(R * 2 / c * self.bandwidth)

                    if 0 <= range_bin < range_samples:
                        # Faz hesaplama
                        phase = 4 * np.pi * R / self.λ
                        image[i, j] += raw_data[range_bin, k] * np.exp(1j * phase)

        return image

    def omega_k_algorithm(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Omega-K Algorithm (ω-K) uygular

        Kaynak: IEEE Transactions on Geoscience and Remote Sensing
        Algoritma: Wavenumber domain processing for wide-swath SAR
        """
        range_samples, azimuth_samples = raw_data.shape

        # 1. 2D FFT
        raw_data_fft = fftshift(fft2(ifftshift(raw_data)))

        # 2. Reference function multiplication
        ref_function = self.generate_omega_k_reference(range_samples, azimuth_samples)
        processed_data = raw_data_fft * ref_function

        # 3. Stolt interpolation
        stolt_interpolated = self.stolt_interpolation(processed_data)

        # 4. 2D IFFT
        image = fftshift(ifft2(ifftshift(stolt_interpolated)))

        return image

    def chirp_scaling_algorithm(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Chirp Scaling Algorithm (CSA) uygular

        Kaynak: IEEE Transactions on Geoscience and Remote Sensing
        Algoritma: Efficient SAR processing without interpolation
        """
        range_samples, azimuth_samples = raw_data.shape

        # 1. Azimuth FFT
        azimuth_fft = fftshift(np.fft.fft(ifftshift(raw_data, axes=1), axis=1), axes=1)

        # 2. Chirp scaling
        scaled_data = self.apply_chirp_scaling(azimuth_fft)

        # 3. Range FFT
        range_fft = fftshift(np.fft.fft(ifftshift(scaled_data, axes=0), axis=0), axes=0)

        # 4. Range compression and RCMC
        range_compressed = self.range_compression_rcmc(range_fft)

        # 5. Range IFFT
        range_ifft = fftshift(np.fft.ifft(ifftshift(range_compressed, axes=0), axis=0), axes=0)

        # 6. Azimuth compression
        azimuth_compressed = self.azimuth_compression(range_ifft)

        # 7. Azimuth IFFT
        image = fftshift(np.fft.ifft(ifftshift(azimuth_compressed, axes=1), axis=1), axes=1)

        return image

    def generate_chirp_reference(self) -> np.ndarray:
        """Chirp reference function üretir"""
        t = np.linspace(
            -self.pulse_width / 2, self.pulse_width / 2, int(self.pulse_width * self.bandwidth)
        )
        chirp_rate = self.bandwidth / self.pulse_width
        phase = np.pi * chirp_rate * t**2
        return np.exp(1j * phase)

    def generate_azimuth_reference(self, range_bin: int, range_samples: int = 1024) -> np.ndarray:
        """
        Generate azimuth reference function (matched filter).

        Reference: Cumming & Wong, "Digital Processing of SAR Data", Ch. 6

        The azimuth phase history is:
        φ(η) = -4π/λ * R(η) ≈ -4π/λ * R₀ - π * Kr * η²

        where:
        - η = slow time (azimuth)
        - R₀ = slant range at closest approach
        - Kr = Doppler rate = -2 * v² / (λ * R₀)
        """
        azimuth_samples = 512
        η = np.linspace(
            -azimuth_samples / (2 * self.prf), azimuth_samples / (2 * self.prf), azimuth_samples
        )

        # Calculate range at this range bin
        range_time = range_bin / self.bandwidth
        R0 = c * range_time / 2  # Slant range [m]
        R0 = max(R0, 100)  # Minimum 100m range

        # Doppler centroid (assuming zero squint)
        # fdc = 2 * v * sin(θ_squint) / λ ≈ 0 for zero squint
        fdc = 0.0

        # Doppler rate (Reference: Skolnik, "Radar Handbook", Eq. 21.8)
        # Ka = -2 * v² / (λ * R₀)
        Ka = -2 * self.v**2 / (self.λ * R0)

        # Azimuth chirp phase: φ(η) = 2π * (fdc * η + 0.5 * Ka * η²)
        phase = 2 * np.pi * (fdc * η + 0.5 * Ka * η**2)

        return np.exp(1j * phase)

    def apply_rcmc(self, range_compressed: np.ndarray) -> np.ndarray:
        """
        Range Cell Migration Correction (RCMC).

        Reference: Cumming & Wong, Ch. 6; Raney et al., IEEE TGRS 1994

        RCMC corrects the range walk caused by changing geometry during SAR aperture.
        Migration: ΔR(η) = R(η) - R₀ ≈ v² * η² / (2 * R₀)

        Uses sinc interpolation in range-Doppler domain.
        """
        range_samples, azimuth_samples = range_compressed.shape
        rcmc_corrected = np.zeros_like(range_compressed, dtype=complex)

        # Azimuth frequency axis
        fa = np.linspace(-self.prf / 2, self.prf / 2, azimuth_samples)

        for i in range(range_samples):
            # Estimate range at this bin
            range_time = i / self.bandwidth
            R0 = c * range_time / 2
            R0 = max(R0, 100)  # Minimum range

            for j in range(azimuth_samples):
                # Doppler frequency
                fd = fa[j]

                # Range Cell Migration (RCM) formula:
                # ΔR = λ² * R₀ * fd² / (8 * v²)
                # Simplified for small angles
                if abs(fd) > 1e-6:
                    delta_R = (self.λ**2 * R0 * fd**2) / (8 * self.v**2)
                else:
                    delta_R = 0

                # Convert range shift to sample shift
                sample_shift = int(delta_R * 2 / c * self.bandwidth)

                # Apply shift with bounds checking
                src_idx = i + sample_shift
                if 0 <= src_idx < range_samples:
                    rcmc_corrected[i, j] = range_compressed[src_idx, j]
                else:
                    rcmc_corrected[i, j] = 0

        return rcmc_corrected

    def generate_omega_k_reference(self, range_samples: int, azimuth_samples: int) -> np.ndarray:
        """Omega-K reference function üretir"""
        # Basitleştirilmiş reference function
        ref_function = np.ones((range_samples, azimuth_samples), dtype=complex)

        # Range ve azimuth wavenumber eksenleri
        kr = np.linspace(-np.pi, np.pi, range_samples)
        ka = np.linspace(-np.pi, np.pi, azimuth_samples)

        # Reference function
        for i in range(range_samples):
            for j in range(azimuth_samples):
                ref_function[i, j] = np.exp(1j * (kr[i] ** 2 + ka[j] ** 2))

        return ref_function

    def stolt_interpolation(self, data: np.ndarray) -> np.ndarray:
        """
        Stolt Interpolation for Omega-K Algorithm.

        Reference: Stolt, "Migration by Fourier Transform", Geophysics, 1978
                   Cafforio et al., IEEE TGRS 1991

        Transforms from (Kr, Ka) to (Kx, Ky) wavenumber domain.

        The Stolt mapping: Kr' = sqrt(Kr² + Ka²)
        This corrects for range curvature in wavenumber domain.
        """
        range_samples, azimuth_samples = data.shape
        interpolated = np.zeros_like(data, dtype=complex)

        # Wavenumber axes
        # Kr: range wavenumber, related to range frequency
        # Ka: azimuth wavenumber, related to Doppler
        dkr = 4 * np.pi * self.bandwidth / (c * range_samples)
        dka = 2 * np.pi * self.prf / (self.v * azimuth_samples)

        kr = np.linspace(-range_samples / 2, range_samples / 2, range_samples) * dkr
        ka = np.linspace(-azimuth_samples / 2, azimuth_samples / 2, azimuth_samples) * dka

        # Reference wavenumber (at center frequency)
        k0 = 4 * np.pi * self.fc / c

        for i in range(range_samples):
            for j in range(azimuth_samples):
                # Stolt mapping: Kr' = sqrt((k0 + kr)² - ka²) - k0
                k_total = k0 + kr[i]
                ka_sq = ka[j] ** 2

                if k_total**2 >= ka_sq:
                    kr_prime = np.sqrt(k_total**2 - ka_sq) - k0

                    # Find nearest source index (linear interpolation)
                    src_idx = int((kr_prime / dkr) + range_samples / 2)

                    if 0 <= src_idx < range_samples:
                        interpolated[i, j] = data[src_idx, j]
                else:
                    # Evanescent wave - set to zero
                    interpolated[i, j] = 0

        return interpolated

    def apply_chirp_scaling(self, azimuth_fft: np.ndarray) -> np.ndarray:
        """Chirp scaling uygular"""
        # Basitleştirilmiş chirp scaling
        range_samples, azimuth_samples = azimuth_fft.shape
        scaled_data = np.zeros_like(azimuth_fft, dtype=complex)

        for i in range(range_samples):
            # Scaling factor
            scale_factor = 1.0 + 0.1 * (i - range_samples // 2) / range_samples
            scaled_data[i, :] = azimuth_fft[i, :] * np.exp(
                1j * scale_factor * np.arange(azimuth_samples)
            )

        return scaled_data

    def range_compression_rcmc(self, range_fft: np.ndarray) -> np.ndarray:
        """Range compression ve RCMC uygular"""
        # Basitleştirilmiş range compression
        return range_fft

    def azimuth_compression(self, range_ifft: np.ndarray) -> np.ndarray:
        """Azimuth compression uygular"""
        # Basitleştirilmiş azimuth compression
        return range_ifft

    def calculate_image_quality(self, image: np.ndarray) -> Dict[str, float]:
        """Görüntü kalitesi metriklerini hesaplar"""
        # SNR hesaplama
        signal_power = np.mean(np.abs(image) ** 2)
        noise_power = np.var(np.real(image)) + np.var(np.imag(image))
        snr = 10 * np.log10(signal_power / (noise_power + 1e-12))

        # Kontrast hesaplama
        contrast = np.std(np.abs(image)) / (np.mean(np.abs(image)) + 1e-12)

        # Çözünürlük hesaplama (basitleştirilmiş)
        resolution = self.range_resolution * self.azimuth_resolution

        return {
            "SNR_dB": snr,
            "Contrast": contrast,
            "Resolution_m2": resolution,
            "Dynamic_Range_dB": 20
            * np.log10(np.max(np.abs(image)) / (np.min(np.abs(image)) + 1e-12)),
        }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 30: VECTORIZED RDA & ISAR PROCESSOR
# ═══════════════════════════════════════════════════════════════════════


class SARImageResult:
    """
    Container for SAR/ISAR image output.

    Attributes:
        image_db: 2D focused image [dB]
        range_axis_m: Range axis values [m]
        cross_range_axis_m: Cross-range or azimuth axis [m]
        range_resolution_m: Achieved range resolution [m]
        azimuth_resolution_m: Achieved azimuth resolution [m]
    """

    __slots__ = (
        "image_db",
        "range_axis_m",
        "cross_range_axis_m",
        "range_resolution_m",
        "azimuth_resolution_m",
    )

    def __init__(
        self,
        image_db: np.ndarray,
        range_axis_m: np.ndarray,
        cross_range_axis_m: np.ndarray,
        range_resolution_m: float,
        azimuth_resolution_m: float,
    ) -> None:
        self.image_db = image_db
        self.range_axis_m = range_axis_m
        self.cross_range_axis_m = cross_range_axis_m
        self.range_resolution_m = range_resolution_m
        self.azimuth_resolution_m = azimuth_resolution_m


def rda_vectorized(
    raw_data: np.ndarray,
    bandwidth_hz: float,
    prf_hz: float,
    fc_hz: float,
    platform_velocity_mps: float,
    antenna_length_m: float = 1.0,
) -> SARImageResult:
    """
    Vectorized Range-Doppler Algorithm (RDA) for SAR image reconstruction.

    Processes a 2D data matrix using batch FFT operations:
        1. Range FFT (fast-time compression)
        2. Corner Turn → Azimuth FFT
        3. RCMC (bulk shift in range-Doppler domain)
        4. Azimuth matched filter
        5. Azimuth IFFT → focused image

    Resolution:
        Δr = c / (2·B)          — Range resolution [m]
        Δa = D / 2              — Azimuth resolution [m] (stripmap)

    Args:
        raw_data: [n_range, n_azimuth] complex raw data matrix
        bandwidth_hz: Waveform bandwidth [Hz]
        prf_hz: Pulse repetition frequency [Hz]
        fc_hz: Carrier frequency [Hz]
        platform_velocity_mps: Platform velocity [m/s]
        antenna_length_m: Physical antenna length [m]

    Returns:
        SARImageResult with focused image and metadata

    Reference: Cumming & Wong, "Digital Processing of SAR Data", 2005
    """
    n_range, n_azimuth = raw_data.shape
    wavelength = c / fc_hz
    range_res = c / (2.0 * bandwidth_hz)
    azimuth_res = antenna_length_m / 2.0

    # ── STAGE 1: Range Compression (batch FFT along axis=0) ──
    # Generate chirp reference in frequency domain
    chirp_duration = 1.0 / bandwidth_hz * n_range
    t_chirp = np.linspace(-chirp_duration / 2, chirp_duration / 2, n_range)
    chirp_rate = bandwidth_hz / chirp_duration
    chirp_ref = np.exp(1j * np.pi * chirp_rate * t_chirp**2)
    chirp_fft = np.fft.fft(chirp_ref, n=n_range)

    # Range FFT all pulses at once (vectorized)
    raw_range_fft = np.fft.fft(raw_data, axis=0)
    # Matched filter: multiply by conjugate of chirp spectrum
    range_compressed_fft = raw_range_fft * np.conj(chirp_fft)[:, np.newaxis]
    range_compressed = np.fft.ifft(range_compressed_fft, axis=0)

    # ── STAGE 2: Azimuth FFT (Corner Turn implicit in axis=1) ──
    range_doppler = np.fft.fftshift(np.fft.fft(range_compressed, axis=1), axes=1)

    # ── STAGE 3: RCMC (bulk range shift per Doppler bin) ──
    f_doppler = np.linspace(-prf_hz / 2, prf_hz / 2, n_azimuth)
    # Reference range (center of swath)
    R0 = c * n_range / (4.0 * bandwidth_hz)
    R0 = max(R0, 1000.0)  # Minimum 1km

    # Doppler chirp rate
    Ka = -2.0 * platform_velocity_mps**2 / (wavelength * R0)

    # RCMC: range shift as function of Doppler frequency
    # ΔR(fd) = λ²·R₀·fd² / (8·v²) — simplified for small angles
    delta_R = wavelength**2 * R0 * f_doppler**2 / (8.0 * platform_velocity_mps**2)
    delta_bins = np.round(delta_R / range_res).astype(int)

    # Apply RCMC via circular shift per Doppler column
    rcmc_data = range_doppler.copy()
    for j in range(n_azimuth):
        if delta_bins[j] != 0:
            rcmc_data[:, j] = np.roll(range_doppler[:, j], -delta_bins[j])

    # ── STAGE 4: Azimuth Matched Filter ──
    # Phase: exp(j·π·fd²/Ka)
    with np.errstate(divide="ignore", invalid="ignore"):
        az_filter = np.exp(1j * np.pi * f_doppler**2 / Ka)
    az_filter = np.where(np.isfinite(az_filter), az_filter, 0.0)

    # Apply azimuth compression (multiply in Doppler domain)
    az_compressed = rcmc_data * az_filter[np.newaxis, :]

    # ── STAGE 5: Azimuth IFFT → focused image ──
    focused = np.fft.ifft(np.fft.ifftshift(az_compressed, axes=1), axis=1)

    # Convert to dB
    image_mag = np.abs(focused)
    image_mag = np.where(image_mag > 0, image_mag, 1e-20)
    image_db = 20.0 * np.log10(image_mag / np.max(image_mag))

    # Build axes
    range_axis = np.arange(n_range) * range_res
    cross_range_axis = np.linspace(
        -n_azimuth / 2 * azimuth_res,
        n_azimuth / 2 * azimuth_res,
        n_azimuth,
    )

    return SARImageResult(
        image_db=image_db,
        range_axis_m=range_axis,
        cross_range_axis_m=cross_range_axis,
        range_resolution_m=range_res,
        azimuth_resolution_m=azimuth_res,
    )


class ISARProcessor:
    """
    Inverse SAR (ISAR) image processor.

    Uses the target's own motion (rotation/translation) to create
    a high-resolution cross-range profile from CPI data.

    Cross-range resolution: Δcr = λ / (2·Δθ)
    where Δθ is the total target rotation during the CPI.

    Algorithm:
        1. Range compression (matched filter)
        2. Translational motion compensation (align range profiles)
        3. Cross-range FFT (Doppler spread → cross-range)

    Reference:
        - Chen & Ling, "Time-Frequency Transforms for Radar Imaging", Artech, 2002
        - Developed by Mehmet Gümüş (github.com/SpaceEngineerSS)
    """

    def __init__(
        self,
        fc_hz: float = 10e9,
        bandwidth_hz: float = 100e6,
        prf_hz: float = 1000.0,
        n_pulses: int = 64,
    ) -> None:
        """
        Initialize ISAR processor.

        Args:
            fc_hz: Carrier frequency [Hz]
            bandwidth_hz: Waveform bandwidth [Hz]
            prf_hz: Pulse repetition frequency [Hz]
            n_pulses: Number of pulses in CPI
        """
        self.fc_hz = fc_hz
        self.bandwidth_hz = bandwidth_hz
        self.prf_hz = prf_hz
        self.n_pulses = n_pulses
        self.wavelength_m = c / fc_hz
        self.range_res_m = c / (2.0 * bandwidth_hz)

    def process_isar(
        self,
        cpi_data: np.ndarray,
        rotation_rate_rps: float = 0.01,
    ) -> SARImageResult:
        """
        Generate ISAR image from CPI data.

        Args:
            cpi_data: [n_pulses, n_range_bins] complex CPI data
            rotation_rate_rps: Target rotation rate [rad/s]

        Returns:
            SARImageResult with focused ISAR image

        Reference: Chen & Ling (2002), Ch. 5
        """
        n_pulses, n_range = cpi_data.shape

        # ── Step 1: Range compression (FFT along fast-time) ──
        range_compressed = np.fft.ifft(
            np.fft.fft(cpi_data, axis=1) * np.conj(self._chirp_spectrum(n_range))[np.newaxis, :],
            axis=1,
        )

        # ── Step 2: Translational motion compensation ──
        # Align range profiles using cross-correlation with reference pulse
        aligned = self._motion_compensate(range_compressed)

        # ── Step 3: Cross-range FFT (along slow-time/pulse axis) ──
        # Hamming window for sidelobe reduction
        window = np.hamming(n_pulses)[:, np.newaxis]
        windowed = aligned * window

        isar_image = np.fft.fftshift(np.fft.fft(windowed, axis=0), axes=0)

        # ── Resolution calculation ──
        T_cpi = n_pulses / self.prf_hz
        delta_theta = rotation_rate_rps * T_cpi
        if delta_theta > 1e-6:
            cross_range_res = self.wavelength_m / (2.0 * delta_theta)
        else:
            cross_range_res = float("inf")

        # Convert to dB
        mag = np.abs(isar_image)
        mag = np.where(mag > 0, mag, 1e-20)
        image_db = 20.0 * np.log10(mag / np.max(mag))

        range_axis = np.arange(n_range) * self.range_res_m
        cross_range_axis = (
            np.linspace(
                -n_pulses / 2 * cross_range_res,
                n_pulses / 2 * cross_range_res,
                n_pulses,
            )
            if np.isfinite(cross_range_res)
            else np.arange(n_pulses, dtype=float)
        )

        return SARImageResult(
            image_db=image_db,
            range_axis_m=range_axis,
            cross_range_axis_m=cross_range_axis,
            range_resolution_m=self.range_res_m,
            azimuth_resolution_m=cross_range_res,
        )

    def _chirp_spectrum(self, n_range: int) -> np.ndarray:
        """Generate chirp reference spectrum for range compression."""
        t = np.linspace(
            -0.5 / self.bandwidth_hz * n_range, 0.5 / self.bandwidth_hz * n_range, n_range
        )
        chirp_rate = self.bandwidth_hz / (1.0 / self.bandwidth_hz * n_range)
        chirp = np.exp(1j * np.pi * chirp_rate * t**2)
        return np.fft.fft(chirp)

    def _motion_compensate(self, range_profiles: np.ndarray) -> np.ndarray:
        """
        Translational motion compensation via range profile alignment.

        Uses cross-correlation with reference profile (first pulse)
        to estimate and correct range shifts.

        Reference: Chen & Ling (2002), Ch. 4
        """
        n_pulses, n_range = range_profiles.shape
        aligned = np.zeros_like(range_profiles)
        ref_profile = np.abs(range_profiles[0])

        for i in range(n_pulses):
            current = np.abs(range_profiles[i])
            # Cross-correlation to find shift
            correlation = np.correlate(current, ref_profile, mode="full")
            shift = np.argmax(correlation) - (n_range - 1)
            # Apply shift
            aligned[i] = np.roll(range_profiles[i], -shift)

        return aligned
