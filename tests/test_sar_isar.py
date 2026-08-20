import time

import numpy as np
import pytest
from scipy.constants import c

from src.advanced.sar_isar import (
    AdvancedSARISAR,
    ISARProcessor,
    SARImageResult,
    rda_vectorized,
)


@pytest.mark.parametrize("bandwidth_hz", [50e6, 100e6, 200e6])
def test_rda_reports_theoretical_range_resolution(bandwidth_hz: float) -> None:
    result = rda_vectorized(
        np.zeros((128, 64), dtype=complex),
        bandwidth_hz,
        1000.0,
        10e9,
        100.0,
    )
    assert result.range_resolution_m == pytest.approx(c / (2.0 * bandwidth_hz))


@pytest.mark.parametrize(
    ("antenna_length_m", "expected_resolution_m"), [(1.0, 0.5), (2.0, 1.0)]
)
def test_fully_sampled_stripmap_azimuth_limit(
    antenna_length_m: float, expected_resolution_m: float
) -> None:
    result = rda_vectorized(
        np.zeros((128, 64), dtype=complex),
        100e6,
        1000.0,
        10e9,
        100.0,
        antenna_length_m,
    )
    assert result.azimuth_resolution_m == pytest.approx(expected_resolution_m)


def test_short_sampled_aperture_limits_azimuth_resolution() -> None:
    result = rda_vectorized(
        np.zeros((128, 8), dtype=complex),
        100e6,
        1000.0,
        10e9,
        100.0,
        antenna_length_m=0.1,
        reference_range_m=1000.0,
    )
    aperture_length = 100.0 * 8 / 1000.0
    expected = (c / 10e9) * 1000.0 / (2.0 * aperture_length)
    assert result.azimuth_resolution_m == pytest.approx(expected)


def test_rda_output_structure_and_normalization() -> None:
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((256, 128)) + 1j * rng.standard_normal((256, 128))
    result = rda_vectorized(raw, 100e6, 1000.0, 10e9, 100.0)
    assert isinstance(result, SARImageResult)
    assert result.image_db.shape == raw.shape
    assert result.complex_image.shape == raw.shape
    assert result.range_axis_m.shape == (256,)
    assert result.cross_range_axis_m.shape == (128,)
    assert result.image_db.max() == pytest.approx(0.0)
    assert np.all(np.isfinite(result.image_db))


def test_physical_point_target_focuses_at_expected_coordinates() -> None:
    sar = AdvancedSARISAR(
        fc=10e9,
        bandwidth=100e6,
        prf=1000.0,
        pulse_width=1e-6,
        platform_velocity=100.0,
        synthetic_aperture=12.8,
    )
    raw = sar.generate_sar_raw_data(
        np.array([[2.0, 200.0, 0.0]]),
        np.array([1.0]),
        range_samples=256,
        azimuth_samples=128,
    )
    result = rda_vectorized(
        raw,
        100e6,
        1000.0,
        10e9,
        100.0,
        pulse_width_s=1e-6,
        reference_range_m=np.hypot(2.0, 200.0),
        range_gate_start_m=sar._last_range_gate_start_m,
    )
    peak = np.unravel_index(np.argmax(result.image_db), result.image_db.shape)
    assert result.range_axis_m[peak[0]] == pytest.approx(
        np.hypot(2.0, 200.0), abs=c / (2.0 * 100e6)
    )
    assert result.cross_range_axis_m[peak[1]] == pytest.approx(2.0, abs=0.1)


def test_isar_resolution_and_output_shape() -> None:
    processor = ISARProcessor(
        fc_hz=10e9, bandwidth_hz=100e6, prf_hz=1000.0, n_pulses=64
    )
    rotation_rate = 0.01
    result = processor.process_isar(
        np.zeros((64, 256), dtype=complex), rotation_rate_rps=rotation_rate
    )
    angular_span = rotation_rate * 64 / 1000.0
    assert result.image_db.shape == (64, 256)
    assert result.azimuth_resolution_m == pytest.approx(
        processor.wavelength_m / (2.0 * angular_span)
    )
    assert result.range_resolution_m == pytest.approx(c / (2.0 * 100e6))


def test_sar_and_isar_processing_performance() -> None:
    rng = np.random.default_rng(42)
    raw = rng.standard_normal((1024, 512)) + 1j * rng.standard_normal((1024, 512))
    start = time.perf_counter()
    sar_result = rda_vectorized(raw, 100e6, 1000.0, 10e9, 100.0)
    sar_elapsed = time.perf_counter() - start
    assert sar_elapsed < 2.0
    assert sar_result.image_db.shape == raw.shape

    cpi = raw[:64, :256]
    processor = ISARProcessor()
    start = time.perf_counter()
    isar_result = processor.process_isar(cpi)
    isar_elapsed = time.perf_counter() - start
    assert isar_elapsed < 1.0
    assert isar_result.image_db.shape == cpi.shape
