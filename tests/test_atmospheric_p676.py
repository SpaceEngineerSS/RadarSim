import math

import pytest

from src.physics.atmospheric import ITU_R_P676


ITU_VALIDATION_POINTS = (
    (12.0, 0.00869826406877357, 0.00953538822024593),
    (20.0, 0.0118835504778076, 0.0970473048151117),
    (60.0, 14.6234747964861, 0.154841840636247),
    (90.0, 0.0388697110724235, 0.341973394422181),
    (130.0, 0.0415090835995228, 0.751844703646129),
)


@pytest.mark.parametrize(("frequency", "oxygen", "water"), ITU_VALIDATION_POINTS)
def test_p676_13_matches_itu_validation_workbook(frequency, oxygen, water):
    assert ITU_R_P676.specific_attenuation_oxygen(frequency) == pytest.approx(
        oxygen, rel=1e-12
    )
    assert ITU_R_P676.specific_attenuation_water_vapor(frequency) == pytest.approx(
        water, rel=1e-12
    )


def test_zero_water_vapor_has_no_water_attenuation():
    assert ITU_R_P676.specific_attenuation_water_vapor(
        22.23508, water_vapor_density=0.0
    ) == pytest.approx(0.0)


def test_two_way_path_loss_is_twice_one_way_loss():
    one_way = ITU_R_P676.total_attenuation(25.0, 10.0, two_way=False)
    two_way = ITU_R_P676.total_attenuation(25.0, 10.0, two_way=True)
    assert two_way == pytest.approx(2.0 * one_way)


def test_line_by_line_model_is_finite_across_valid_band():
    for frequency in (1.0, 22.23508, 60.0, 118.750334, 183.310087, 1000.0):
        oxygen = ITU_R_P676.specific_attenuation_oxygen(frequency)
        water = ITU_R_P676.specific_attenuation_water_vapor(frequency)
        assert math.isfinite(oxygen) and oxygen >= 0.0
        assert math.isfinite(water) and water >= 0.0


@pytest.mark.parametrize("frequency", (0.999, 1000.001))
def test_frequency_outside_recommendation_band_is_rejected(frequency):
    with pytest.raises(ValueError):
        ITU_R_P676.specific_attenuation_oxygen(frequency)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"temperature_c": -273.15}, "absolute zero"),
        ({"pressure_hpa": 0.0}, "pressure_hpa"),
        ({"water_vapor_density": -0.1}, "water_vapor_density"),
    ),
)
def test_invalid_atmospheric_state_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ITU_R_P676.specific_attenuation_oxygen(10.0, **kwargs)
