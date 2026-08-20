from math import erfc

import numpy as np
import pytest

from src.physics.ecm import (
    DRFMConfig,
    DRFMJammer,
    ECMSimulator,
    apply_receiver_hard_limiter,
)
from src.physics.rcs import SwerlingModel
from src.simulation.engine import SimulationEngine
from src.simulation.objects import Radar, Target


def calculate_jsr(range_m, **kwargs):
    simulator = ECMSimulator(radar_wavelength=0.03)
    radar = np.zeros(3)
    target = np.array([range_m, 0.0, 0.0])
    return simulator.calculate_jsr(
        radar_pos=radar,
        target_pos=target,
        jammer_pos=target,
        radar_power=100_000.0,
        radar_gain=1000.0,
        jammer_power=1000.0,
        target_rcs=1.0,
        radar_bandwidth=1e6,
        jammer_bandwidth=100e6,
        **kwargs,
    )


def test_self_screening_jsr_matches_one_way_and_two_way_radar_equations():
    jsr_db = calculate_jsr(100_000.0)
    expected_linear = (
        4.0
        * np.pi
        * 1000.0
        * 100_000.0**4
        / (100_000.0 * 1000.0 * 1.0 * 100_000.0**2)
        * 0.01
    )
    assert jsr_db == pytest.approx(10.0 * np.log10(expected_linear))


def test_self_screening_jsr_increases_six_db_when_range_doubles():
    near = calculate_jsr(50_000.0)
    far = calculate_jsr(100_000.0)
    assert far - near == pytest.approx(20.0 * np.log10(2.0))


def test_barrage_bandwidth_dilutes_in_band_jammer_power():
    simulator = ECMSimulator()
    common = dict(
        radar_pos=np.zeros(3),
        target_pos=np.array([10_000.0, 0.0, 0.0]),
        jammer_pos=np.array([10_000.0, 0.0, 0.0]),
        radar_power=100_000.0,
        radar_gain=1000.0,
        jammer_power=100.0,
        target_rcs=1.0,
        radar_bandwidth=1e6,
    )
    spot = simulator.calculate_jsr(**common, jammer_bandwidth=1e6)
    barrage = simulator.calculate_jsr(**common, jammer_bandwidth=100e6)
    assert spot - barrage == pytest.approx(20.0)


def test_one_way_propagation_advantage_increases_jsr():
    no_loss = calculate_jsr(100_000.0)
    with_loss = calculate_jsr(
        100_000.0, signal_path_loss_db=12.0, jammer_path_loss_db=6.0
    )
    assert with_loss - no_loss == pytest.approx(6.0)


def test_sjnr_combines_noise_and_jamming_as_powers():
    sjnr = ECMSimulator.calculate_sjnr_db(snr_db=20.0, jsr_db=-10.0)
    assert sjnr == pytest.approx(-10.0 * np.log10(0.01 + 0.1))


def test_invalid_jammer_parameters_are_rejected():
    simulator = ECMSimulator()
    with pytest.raises(ValueError):
        simulator.calculate_jsr(
            np.zeros(3),
            np.ones(3),
            np.ones(3),
            1.0,
            1.0,
            1.0,
            1.0,
            jammer_bandwidth=0.0,
        )


def test_complex_gaussian_hard_limiter_matches_closed_form_at_full_scale():
    result = apply_receiver_hard_limiter(0.1, 0.9, 1.0)
    alpha = 1.0 - np.exp(-1.0) + 0.5 * np.sqrt(np.pi) * erfc(1.0)
    output_power = 1.0 - np.exp(-1.0)
    distortion = output_power - alpha**2
    expected_sinr = alpha**2 * 0.1 / (alpha**2 * 0.9 + distortion)

    assert result.coherent_gain == pytest.approx(alpha)
    assert result.output_power_watts == pytest.approx(output_power)
    assert result.distortion_power_watts == pytest.approx(distortion)
    assert result.sinr_db == pytest.approx(10.0 * np.log10(expected_sinr))
    assert result.headroom_db == pytest.approx(0.0)
    assert result.overloaded is False


def test_receiver_limiter_is_transparent_with_large_headroom():
    result = apply_receiver_hard_limiter(1e-12, 1e-13, 1e-3)
    assert result.sinr_db == pytest.approx(10.0)
    assert result.clipping_loss_db == 0.0
    assert result.distortion_power_watts == 0.0
    assert result.overloaded is False


def test_drfm_delay_is_consistent_with_apparent_range_offset():
    jammer = DRFMJammer(
        DRFMConfig(
            capture_dwell_s=0.0,
            pull_rate_mps=150.0,
            max_pull_m=1000.0,
            inherent_delay_s=0.4e-6,
        )
    )
    jammer.activate()
    jammer.step(0.1)
    jammer.step(2.0)

    expected_offset = 299_792_458.0 * 0.4e-6 / 2.0 + 300.0
    assert jammer.false_range_offset_m == pytest.approx(expected_offset)
    assert jammer.retransmission_delay_s == pytest.approx(
        0.4e-6 + 2.0 * 300.0 / 299_792_458.0
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"mode": "invalid"},
        {"capture_dwell_s": -1.0},
        {"pull_rate_mps": -1.0},
        {"max_pull_m": 0.0},
        {"inherent_delay_s": -1e-6},
    ),
)
def test_drfm_configuration_rejects_nonphysical_values(kwargs):
    with pytest.raises(ValueError):
        DRFMConfig(**kwargs)


def build_noise_jamming_engine(frequency_agility=False, receiver_full_scale_dbm=-20.0):
    radar = Radar(
        "test",
        np.zeros(3),
        frequency_hz=10e9,
        power_watts=100_000.0,
        antenna_gain_db=30.0,
        receiver_bandwidth_hz=1e6,
    )
    target = Target(
        1,
        np.array([10_000.0, 0.0, 0.0]),
        rcs_m2=1.0,
        swerling_model=SwerlingModel.SWERLING_0,
        has_jammer=True,
        jammer_power_watts=100.0,
        jammer_bandwidth_hz=1e6,
    )
    engine = SimulationEngine(
        radar,
        [target],
        enable_atmospheric=False,
        receiver_full_scale_dbm=receiver_full_scale_dbm,
    )
    engine.set_ecm_mode(True, "noise_spot")
    engine.frequency_agility_enabled = frequency_agility
    return engine


def test_engine_applies_noise_jammer_to_detection_statistic():
    result = build_noise_jamming_engine().step()[0]
    assert result.jammer_jsr_db is not None
    assert result.jammer_loss_db > 0.0
    expected = ECMSimulator.calculate_sjnr_db(
        result.snr_db + result.jammer_loss_db, result.jammer_jsr_db
    )
    assert result.snr_db == pytest.approx(expected)


def test_frequency_agility_dilutes_spot_jammer_spectral_density():
    static = build_noise_jamming_engine(frequency_agility=False).step()[0]
    agile = build_noise_jamming_engine(frequency_agility=True).step()[0]
    assert agile.jammer_jsr_db < static.jammer_jsr_db
    assert agile.snr_db > static.snr_db


def test_engine_reports_receiver_overload_and_clipping_loss():
    result = build_noise_jamming_engine(receiver_full_scale_dbm=-140.0).step()[0]
    assert result.receiver_overloaded is True
    assert result.receiver_headroom_db < 0.0
    assert result.receiver_clipping_loss_db > 0.0
    assert result.to_dict()["receiver_overloaded"] is True


def test_engine_drfm_ghost_uses_radar_relative_delay_and_pull_kinematics():
    radar = Radar("offset-radar", np.array([1000.0, 1000.0, 0.0]))
    target = Target(
        7,
        np.array([11_000.0, 1000.0, 0.0]),
        velocity=np.zeros(3),
        rcs_m2=1.0,
        has_jammer=True,
        jammer_power_watts=100.0,
        ecm_type="drfm",
        drfm_capture_dwell_s=0.0,
        drfm_pull_rate_mps=100.0,
        drfm_max_pull_m=1000.0,
        drfm_inherent_delay_s=1e-6,
    )
    engine = SimulationEngine(radar, [target], dt=0.5, enable_atmospheric=False)
    engine.set_ecm_mode(True, "drfm")

    engine.step()
    engine.step()

    ghost = engine.false_targets[-1]
    expected_offset = 299_792_458.0 * 1e-6 / 2.0 + 50.0
    assert ghost.position[0] == pytest.approx(target.position[0] + expected_offset)
    assert ghost.position[1] == pytest.approx(target.position[1])
    assert ghost.velocity[0] == pytest.approx(100.0)
    assert ghost.rcs_m2 == pytest.approx(10.0)
