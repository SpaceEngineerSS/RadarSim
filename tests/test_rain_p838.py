import pytest

from src.physics.rain import ITU_R_P838


ITU_VALIDATION_POINTS = (
    (
        14.25,
        31.076991235657,
        26.48052,
        0.0397548797329313,
        1.12418042813791,
        1.58130839366869,
    ),
    (
        14.25,
        40.2320359963616,
        33.936232,
        0.0400762407546957,
        1.11804137747879,
        2.06173213126362,
    ),
    (
        29.0,
        46.3596926118634,
        27.13586832,
        0.219780969654757,
        0.948485097658449,
        5.03135479362085,
    ),
)


@pytest.mark.parametrize(
    (
        "frequency",
        "elevation",
        "rain_rate",
        "expected_k",
        "expected_alpha",
        "expected_gamma",
    ),
    ITU_VALIDATION_POINTS,
)
def test_p838_3_matches_itu_validation_workbook(
    frequency, elevation, rain_rate, expected_k, expected_alpha, expected_gamma
):
    k, alpha = ITU_R_P838.polarization_coefficients(frequency, elevation, 0.0)
    gamma = ITU_R_P838.specific_attenuation(frequency, rain_rate, elevation, 0.0)
    assert k == pytest.approx(expected_k, rel=1e-12)
    assert alpha == pytest.approx(expected_alpha, rel=1e-12)
    assert gamma == pytest.approx(expected_gamma, rel=1e-12)


@pytest.mark.parametrize(
    ("frequency", "k_h", "alpha_h", "k_v", "alpha_v"),
    (
        (1.0, 0.0000259, 0.9691, 0.0000308, 0.8592),
        (10.0, 0.01217, 1.2571, 0.01129, 1.2156),
        (35.0, 0.3374, 0.9047, 0.3224, 0.8761),
        (100.0, 1.3671, 0.6815, 1.3680, 0.6765),
        (1000.0, 1.3795, 0.6396, 1.3822, 0.6365),
    ),
)
def test_horizontal_and_vertical_coefficients_match_p838_table_5(
    frequency, k_h, alpha_h, k_v, alpha_v
):
    computed_k_h, computed_alpha_h = ITU_R_P838.polarization_coefficients(
        frequency, 0.0, 0.0
    )
    computed_k_v, computed_alpha_v = ITU_R_P838.polarization_coefficients(
        frequency, 0.0, 90.0
    )
    assert computed_k_h == pytest.approx(k_h, rel=6e-4)
    assert computed_alpha_h == pytest.approx(alpha_h, rel=6e-4)
    assert computed_k_v == pytest.approx(k_v, rel=6e-4)
    assert computed_alpha_v == pytest.approx(alpha_v, rel=6e-4)


def test_circular_polarization_is_independent_of_tilt_sign():
    positive = ITU_R_P838.polarization_coefficients(20.0, 15.0, 45.0)
    negative = ITU_R_P838.polarization_coefficients(20.0, 15.0, -45.0)
    assert positive == pytest.approx(negative)


def test_zero_rain_and_zero_path_have_zero_loss():
    assert ITU_R_P838.specific_attenuation(10.0, 0.0) == 0.0
    assert ITU_R_P838.path_attenuation(0.0, 10.0, 50.0) == 0.0


def test_two_way_loss_is_twice_one_way_loss():
    one_way = ITU_R_P838.path_attenuation(10.0, 10.0, 25.0, two_way=False)
    two_way = ITU_R_P838.path_attenuation(10.0, 10.0, 25.0, two_way=True)
    assert two_way == pytest.approx(2.0 * one_way)


@pytest.mark.parametrize("frequency", (0.999, 1000.001))
def test_frequency_outside_recommendation_band_is_rejected(frequency):
    with pytest.raises(ValueError):
        ITU_R_P838.polarization_coefficients(frequency)


def test_nonphysical_inputs_are_rejected():
    with pytest.raises(ValueError):
        ITU_R_P838.specific_attenuation(10.0, -1.0)
    with pytest.raises(ValueError):
        ITU_R_P838.path_attenuation(-1.0, 10.0, 1.0)
    with pytest.raises(ValueError):
        ITU_R_P838.polarization_coefficients(10.0, elevation_angle_deg=91.0)
