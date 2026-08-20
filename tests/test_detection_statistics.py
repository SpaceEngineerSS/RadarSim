import numpy as np
import pytest

from src.physics.metrics import albersheim_snr, calculate_pd_swerling


class TestAlbersheim:
    def test_canonical_single_pulse_operating_point(self):
        required_snr = albersheim_snr(pd=0.9, pfa=1e-6, n_pulses=1)

        assert required_snr == pytest.approx(13.1145, abs=0.01)

    def test_noncoherent_integration_reduces_per_pulse_snr(self):
        single_pulse = albersheim_snr(pd=0.9, pfa=1e-6, n_pulses=1)
        ten_pulse = albersheim_snr(pd=0.9, pfa=1e-6, n_pulses=10)

        assert ten_pulse < single_pulse
        assert ten_pulse == pytest.approx(4.9904, abs=0.01)

    @pytest.mark.parametrize(
        ("pd", "pfa", "n_pulses", "error"),
        [
            (0.0, 1e-6, 1, ValueError),
            (0.9, 0.0, 1, ValueError),
            (0.9, 1e-6, 0, ValueError),
            (0.9, 1e-6, 1.5, TypeError),
        ],
    )
    def test_invalid_inputs_are_rejected(self, pd, pfa, n_pulses, error):
        with pytest.raises(error):
            albersheim_snr(pd, pfa, n_pulses)


class TestSquareLawDetection:
    @pytest.mark.parametrize("swerling_case", range(5))
    def test_no_signal_limit_equals_false_alarm_probability(self, swerling_case):
        pd = calculate_pd_swerling(-np.inf, 1e-6, swerling_case, 16)

        assert pd == pytest.approx(1e-6)

    @pytest.mark.parametrize("swerling_case", range(5))
    def test_probability_is_monotonic_with_snr(self, swerling_case):
        snr_axis = np.linspace(-15.0, 30.0, 91)
        probabilities = np.array(
            [calculate_pd_swerling(snr, 1e-6, swerling_case, 8) for snr in snr_axis]
        )

        assert np.all(np.diff(probabilities) >= -1e-12)
        assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    def test_nonfluctuating_reference_point(self):
        pd = calculate_pd_swerling(13.14, pfa=1e-6, swerling_case=0, n_pulses=1)

        assert pd == pytest.approx(0.8943, abs=5e-4)

    def test_slow_and_fast_cases_match_for_one_pulse(self):
        assert calculate_pd_swerling(10.0, 1e-6, 1, 1) == pytest.approx(
            calculate_pd_swerling(10.0, 1e-6, 2, 1), rel=1e-12
        )
        assert calculate_pd_swerling(10.0, 1e-6, 3, 1) == pytest.approx(
            calculate_pd_swerling(10.0, 1e-6, 4, 1), rel=1e-12
        )

    def test_fast_fluctuation_benefits_from_pulse_diversity(self):
        slow_case = calculate_pd_swerling(5.0, 1e-6, 1, 10)
        fast_case = calculate_pd_swerling(5.0, 1e-6, 2, 10)

        assert fast_case > slow_case

    def test_false_alarm_probability_controls_threshold(self):
        strict = calculate_pd_swerling(8.0, 1e-8, 0, 1)
        permissive = calculate_pd_swerling(8.0, 1e-4, 0, 1)

        assert permissive > strict
