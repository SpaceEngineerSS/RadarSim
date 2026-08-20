import numpy as np
import pytest

from src.physics.clutter import ClutterModel


def test_constant_gamma_ground_reflectivity_uses_sine_grazing_law():
    sigma0_db = ClutterModel.ground_clutter_sigma0(
        np.radians(30.0), gamma_db=-20.0
    )
    assert sigma0_db == pytest.approx(-23.010299956639813)


def test_constant_gamma_ground_reflectivity_increases_with_grazing_angle():
    values = [
        ClutterModel.ground_clutter_sigma0(np.radians(angle), terrain_type="rural")
        for angle in (0.1, 1.0, 10.0, 45.0)
    ]
    assert np.all(np.diff(values) > 0.0)


def test_constant_gamma_rejects_unknown_uncalibrated_terrain():
    with pytest.raises(ValueError, match="unknown terrain type"):
        ClutterModel.ground_clutter_sigma0(np.radians(5.0), "tundra")


@pytest.mark.parametrize(
    ("polarization", "expected_db"),
    (
        ("HH", -14.152949283094422),
        ("VV", -12.747674016980767),
        ("HV", -24.20084898920081),
    ),
)
def test_oh1992_bare_soil_reference_case(polarization, expected_db):
    sigma0_db = ClutterModel.bare_soil_oh1992_sigma0(
        np.radians(40.0),
        5.0,
        complex(8.0, -0.8),
        0.01,
        polarization,
    )
    assert sigma0_db == pytest.approx(expected_db, abs=1e-12)


@pytest.mark.parametrize(
    "args",
    (
        (np.radians(19.9), 5.0, complex(8.0, -0.8), 0.01, "HH"),
        (np.radians(80.1), 5.0, complex(8.0, -0.8), 0.01, "HH"),
        (np.radians(40.0), 0.9, complex(8.0, -0.8), 0.01, "HH"),
        (np.radians(40.0), 5.0, complex(0.8, -0.1), 0.01, "HH"),
        (np.radians(40.0), 5.0, complex(8.0, 0.8), 0.01, "HH"),
        (np.radians(40.0), 5.0, complex(8.0, -0.8), 0.0005, "HH"),
    ),
)
def test_oh1992_rejects_inputs_outside_measurement_domain(args):
    with pytest.raises(ValueError):
        ClutterModel.bare_soil_oh1992_sigma0(*args)


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
