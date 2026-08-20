import numpy as np
import pytest
from scipy.constants import c

from src.advanced.sar_isar import AdvancedSARISAR, ISARProcessor, rda_vectorized


def test_stripmap_raw_data_contains_delayed_lfm_phase_history() -> None:
    sar = AdvancedSARISAR(
        fc=10e9,
        bandwidth=20e6,
        prf=1000.0,
        pulse_width=2e-6,
        platform_velocity=100.0,
        synthetic_aperture=3.2,
    )
    raw = sar.generate_sar_raw_data(
        np.array([[0.0, 1000.0, 0.0]]),
        np.array([4.0]),
        range_samples=256,
        azimuth_samples=32,
    )

    center = raw[:, 16]
    assert np.count_nonzero(np.abs(center) > 0.0) == len(sar.generate_chirp_reference())
    assert np.sum(np.abs(center) ** 2) == pytest.approx(
        4.0 * len(sar.generate_chirp_reference())
    )
    assert not np.allclose(raw[:, 0], raw[:, 16])


def test_rda_range_axis_uses_sample_rate_not_nominal_resolution() -> None:
    result = rda_vectorized(
        np.zeros((128, 64), dtype=complex),
        bandwidth_hz=20e6,
        prf_hz=1000.0,
        fc_hz=10e9,
        platform_velocity_mps=100.0,
        sample_rate_hz=40e6,
        reference_range_m=200.0,
    )
    assert result.range_axis_m[1] == pytest.approx(c / (2.0 * 40e6))
    assert result.range_resolution_m == pytest.approx(c / (2.0 * 20e6))


def test_rda_zero_input_has_finite_floor_and_no_false_peak() -> None:
    result = rda_vectorized(
        np.zeros((64, 32), dtype=complex), 20e6, 1000.0, 10e9, 100.0
    )
    assert np.all(result.image_db == -120.0)
    assert np.all(result.complex_image == 0.0)


def test_unimplemented_focusers_do_not_return_placeholder_images() -> None:
    sar = AdvancedSARISAR()
    raw = np.zeros((16, 16), dtype=complex)
    with pytest.raises(NotImplementedError, match="wavenumber"):
        sar.omega_k_algorithm(raw)
    with pytest.raises(NotImplementedError, match="Doppler-rate"):
        sar.chirp_scaling_algorithm(raw)


def test_isar_doppler_axis_maps_scatterer_to_cross_range() -> None:
    processor = ISARProcessor(
        fc_hz=10e9,
        bandwidth_hz=100e6,
        prf_hz=1000.0,
        n_pulses=128,
    )
    rotation_rate = 1.0
    cross_range = 2.0
    doppler_hz = 2.0 * rotation_rate * cross_range / processor.wavelength_m
    slow_time = np.arange(128) / processor.prf_hz
    profiles = np.zeros((128, 128), dtype=complex)
    profiles[:, 64] = np.exp(1j * 2.0 * np.pi * doppler_hz * slow_time)

    result = processor.process_isar(
        profiles, rotation_rate_rps=rotation_rate, range_compressed=True
    )
    peak = np.unravel_index(np.argmax(result.image_db), result.image_db.shape)
    assert result.cross_range_axis_m[peak[0]] == pytest.approx(
        cross_range, abs=result.azimuth_resolution_m
    )
    assert result.range_axis_m[peak[1]] == pytest.approx(
        64 * c / (2.0 * processor.sample_rate_hz)
    )


def test_isar_range_alignment_zero_fills_instead_of_wrapping() -> None:
    processor = ISARProcessor(n_pulses=5)
    profiles = np.zeros((5, 32), dtype=complex)
    for pulse, position in enumerate([12, 13, 14, 15, 16]):
        profiles[pulse, position] = 1.0

    aligned = processor._motion_compensate(profiles)
    peaks = np.argmax(np.abs(aligned), axis=1)
    assert np.all(peaks == peaks[0])
    assert np.all(aligned[:, 0] == 0.0)


def test_sar_parameters_and_inputs_are_validated() -> None:
    with pytest.raises(ValueError, match="sample_rate_hz"):
        AdvancedSARISAR(bandwidth=20e6, sample_rate_hz=10e6)
    with pytest.raises(ValueError, match="shape"):
        AdvancedSARISAR().generate_sar_raw_data(
            np.zeros((2, 2)), np.ones(2), range_samples=16, azimuth_samples=16
        )
