import numpy as np
import pytest

from radarsim.physics.metrics import albersheim_snr, calculate_pd_swerling


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


@pytest.mark.parametrize("case", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("n_pulses,snr_db", [(1, 14.0), (8, 9.0)])
def test_swerling_pd_matches_pulse_level_monte_carlo(case, n_pulses, snr_db):
    """Square-law detector on simulated complex samples with fluctuating RCS."""
    from scipy import special

    rng = np.random.default_rng(100 + case)
    looks, pfa = 200_000, 1e-4
    snr = 10 ** (snr_db / 10)
    threshold = special.gammainccinv(n_pulses, pfa)
    if case == 0:
        rcs = np.ones((looks, 1))
    elif case in (1, 3):
        shape = 1.0 if case == 1 else 2.0
        rcs = np.repeat(rng.gamma(shape, 1 / shape, (looks, 1)), n_pulses, axis=1)
    else:
        shape = 1.0 if case == 2 else 2.0
        rcs = rng.gamma(shape, 1 / shape, (looks, n_pulses))
    noise = (rng.standard_normal((looks, n_pulses)) + 1j * rng.standard_normal((looks, n_pulses))) / np.sqrt(2)
    statistic = np.sum(np.abs(np.sqrt(snr * rcs) + noise) ** 2, axis=1)
    empirical = np.mean(statistic > threshold)
    theory = calculate_pd_swerling(snr_db, pfa=pfa, swerling_case=case, n_pulses=n_pulses)
    assert theory == pytest.approx(empirical, abs=4 * np.sqrt(empirical * (1 - empirical) / looks) + 2e-3)


def test_cfar_loss_ordering_and_large_window_limit():
    from radarsim.physics.metrics import cfar_loss_db

    losses = {kind: cfar_loss_db(kind, 16, 1e-6) for kind in ("CA", "GO", "SO", "OS")}
    assert losses["SO"] > losses["OS"] > losses["GO"] > losses["CA"] > 0.0
    # CA-CFAR, single pulse Swerling 1 (Richards 2005, Fig. 16-12): about 2 dB for N = 16
    assert losses["CA"] == pytest.approx(2.0, abs=0.05)
    # The loss vanishes as the reference window grows
    assert cfar_loss_db("CA", 512, 1e-6) < 0.1


def test_cfar_threshold_matches_simulated_ca_false_alarm_rate():
    from scipy import optimize

    from radarsim.physics.metrics import _cfar_swerling1_pd

    n_ref, pfa = 16, 1e-3
    threshold = optimize.brentq(lambda t: _cfar_swerling1_pd(t, "CA", n_ref, 0) - pfa, 1e-6, 10.0)
    rng = np.random.default_rng(9)
    noise = rng.exponential(1.0, (400_000, n_ref + 1))
    measured = np.mean(noise[:, 0] > threshold * noise[:, 1:].sum(axis=1))
    assert measured == pytest.approx(pfa, rel=0.1)
