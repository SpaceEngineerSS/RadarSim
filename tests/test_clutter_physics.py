import numpy as np
import pytest

from src.physics.clutter import ClutterModel


@pytest.mark.parametrize(
    ("grazing_deg", "sea_state", "frequency_ghz", "polarization", "expected_db"),
    (
        (0.1, 0, 0.5, "HH", -112.42869977373226),
        (3.0, 3, 9.3, "HH", -40.888859947001855),
        (10.0, 4, 17.0, "VV", -29.211852646254712),
        (60.0, 6, 35.0, "VV", -14.770213739853158),
    ),
)
def test_nrl_five_parameter_sea_reflectivity(
    grazing_deg, sea_state, frequency_ghz, polarization, expected_db
):
    sigma0_db = ClutterModel.sea_clutter_sigma0(
        np.radians(grazing_deg), sea_state, frequency_ghz, polarization
    )
    assert sigma0_db == pytest.approx(expected_db, abs=1e-12)


def test_nrl_sea_reflectivity_increases_with_sea_state():
    values = [
        ClutterModel.sea_clutter_sigma0(np.radians(3.0), state, 10.0, "HH")
        for state in range(7)
    ]
    assert np.all(np.diff(values) > 0.0)


@pytest.mark.parametrize(
    "args",
    (
        (np.radians(0.09), 3, 10.0, "HH"),
        (np.radians(60.01), 3, 10.0, "HH"),
        (np.radians(3.0), 3, 0.49, "HH"),
        (np.radians(3.0), 7, 10.0, "HH"),
        (np.radians(3.0), 3, 10.0, "HV"),
    ),
)
def test_nrl_model_rejects_inputs_outside_validation_domain(args):
    with pytest.raises(ValueError):
        ClutterModel.sea_clutter_sigma0(*args)


def test_surface_resolution_cell_uses_pulse_and_both_beamwidths():
    area = ClutterModel.surface_resolution_cell_area(
        range_m=20_000.0,
        pulse_width_s=1e-6,
        azimuth_beamwidth_rad=np.radians(2.0),
        elevation_beamwidth_rad=np.radians(3.0),
        grazing_angle_rad=np.radians(5.0),
    )
    range_extent = 299_792_458.0e-6 / 2.0 / np.cos(np.radians(5.0))
    cross_range = 40_000.0 * np.tan(np.radians(1.0))
    expected = np.pi / 4.0 * range_extent * cross_range
    assert area == pytest.approx(expected)


def test_noise_and_clutter_are_combined_as_powers():
    sinr_db = ClutterModel.signal_to_noise_plus_clutter_db(
        snr_db=20.0, target_rcs_m2=1.0, clutter_rcs_m2=0.1
    )
    assert sinr_db == pytest.approx(-10.0 * np.log10(0.01 + 0.1))


def test_zero_clutter_preserves_thermal_snr():
    assert ClutterModel.signal_to_noise_plus_clutter_db(
        13.5, 2.0, 0.0
    ) == pytest.approx(13.5)
