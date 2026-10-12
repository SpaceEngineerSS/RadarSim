"""
Engine-level physics invariants added with the v3.1 overhaul.

Covers scan-gated detection, the in-engine track-while-scan loop, clutter
entering through the elevation pattern, MTI improvement factor, rain height,
coordinated-turn kinematics, Doppler sign in the range-Doppler map, and
reproducibility (seed + reset).
"""

import numpy as np
import pytest

from radarsim.physics.clutter import ClutterModel
from radarsim.physics.rcs import SwerlingModel
from radarsim.simulation.engine import SimulationEngine
from radarsim.simulation.objects import MotionModel, Radar, Target
from radarsim.tracking.tracker import TrackStatus


def _radar(**kwargs) -> Radar:
    defaults = dict(
        radar_id="test",
        position=np.array([0.0, 0.0, 10.0]),
        frequency_hz=3e9,
        power_watts=500e3,
        antenna_gain_db=40.0,
        beamwidth_deg=1.5,
    )
    defaults.update(kwargs)
    return Radar(**defaults)


def _target(position, velocity=(0.0, 0.0, 0.0), **kwargs) -> Target:
    defaults = dict(rcs_m2=5.0, swerling_model=SwerlingModel.SWERLING_0)
    defaults.update(kwargs)
    return Target(1, np.array(position, float), np.array(velocity, float), **defaults)


# ----------------------------------------------------------------- scanning


def test_target_is_looked_at_once_per_revolution():
    radar = _radar(scan_rate_rpm=6.0)
    target = _target([20_000.0, 20_000.0, 1_000.0])  # azimuth 45 deg
    engine = SimulationEngine(radar, [target], dt=0.05, seed=1)
    log = engine.run(30.0)  # three revolutions
    looks = [r for r in log.detection_history if r.illuminated]
    assert len(looks) == 3  # beam centre crosses 45 deg at 1.25 s, 11.25 s, 21.25 s
    # every look happens while the antenna points at ~45 deg
    for look in looks:
        assert look.true_azimuth_rad == pytest.approx(np.pi / 4, abs=np.radians(5))


def test_ungated_engine_looks_every_step():
    engine = SimulationEngine(
        _radar(), [_target([0.0, 30_000.0, 1_000.0])], dt=0.1, scan_gated_detection=False
    )
    assert engine.run(1.0).total_opportunities == 10


# ----------------------------------------------------------------- tracking


def test_engine_tracker_confirms_straight_flying_target():
    radar = _radar(scan_rate_rpm=60.0)  # 1 s revisit keeps the test short
    target = _target([0.0, 30_000.0, 3_000.0], velocity=[250.0, 0.0, 0.0])
    engine = SimulationEngine(radar, [target], dt=0.02, seed=7)
    engine.run(12.0)

    confirmed = [t for t in engine.tracks if t.status == TrackStatus.CONFIRMED]
    assert len(confirmed) == 1
    track = confirmed[0]
    # Track is in the ground plane (ENU x, y); truth moves East at 250 m/s.
    assert np.hypot(
        track.position[0] - target.position[0], track.position[1] - target.position[1]
    ) < 300.0
    assert track.velocity[0] == pytest.approx(250.0, abs=40.0)
    assert track.heading_rad == pytest.approx(np.pi / 2, abs=np.radians(10))


def test_tracks_outside_the_beam_are_not_charged_misses():
    radar = _radar(scan_rate_rpm=6.0)
    target = _target([0.0, 30_000.0, 3_000.0], velocity=[200.0, 0.0, 0.0])
    engine = SimulationEngine(radar, [target], dt=0.05, seed=3)
    engine.run(35.0)  # 3.5 revolutions: 4 looks at the target
    assert any(t.hits >= 3 for t in engine.tracks)


# ------------------------------------------------------------------ clutter


def test_high_target_sees_less_surface_clutter_than_low_target():
    def clutter_loss(altitude_m: float) -> float:
        engine = SimulationEngine(
            _radar(beamwidth_el_deg=3.0),
            [_target([0.0, 20_000.0, altitude_m])],
            enable_clutter=True,
            scan_gated_detection=False,
            enable_atmospheric=False,
        )
        return engine.step()[0].surface_clutter_loss_db

    assert clutter_loss(50.0) > clutter_loss(6_000.0) + 10.0


def test_mti_improvement_factor_matches_skolnik_single_canceller():
    # sigma_c = 2*0.25/0.1 = 5 Hz; I = 2*(1000/(2*pi*5))^2 = 2026 -> 33.07 dB
    spectral_only = ClutterModel.mti_improvement_factor_db(
        1, 1000.0, 0.1, 0.25, stability_limit_db=200.0
    )
    assert spectral_only == pytest.approx(33.07, abs=0.02)
    double = ClutterModel.mti_improvement_factor_db(2, 1000.0, 0.1, 0.25, 200.0)
    assert double > spectral_only
    limited = ClutterModel.mti_improvement_factor_db(2, 1000.0, 0.1, 0.25, 40.0)
    assert limited < 40.0


def test_mti_reduces_clutter_loss():
    def loss(mti: bool) -> float:
        engine = SimulationEngine(
            _radar(),
            [_target([0.0, 15_000.0, 30.0], velocity=[0.0, -150.0, 0.0])],
            enable_clutter=True,
            scan_gated_detection=False,
        )
        engine.mti_enabled = mti
        return engine.step()[0].surface_clutter_loss_db

    assert loss(True) < loss(False)


def test_two_way_elevation_gain_is_half_power_squared_at_half_beamwidth():
    gain = ClutterModel.elevation_two_way_gain(np.radians(1.5), np.radians(3.0))
    assert gain == pytest.approx(0.25, rel=1e-6)
    floor = ClutterModel.elevation_two_way_gain(np.radians(45.0), np.radians(3.0), -30.0)
    assert floor == pytest.approx(1e-6)


# --------------------------------------------------------------------- rain


def test_rain_only_attenuates_path_below_rain_height():
    def result(altitude_m: float):
        engine = SimulationEngine(
            _radar(frequency_hz=10e9),
            [_target([0.0, 40_000.0, altitude_m])],
            rain_rate_mm_hr=10.0,
            rain_height_m=3_000.0,
            scan_gated_detection=False,
            enable_atmospheric=False,
        )
        return engine.step()[0]

    low, high = result(1_000.0), result(9_000.0)
    assert high.rain_attenuation_db < low.rain_attenuation_db
    assert high.rain_clutter_loss_db == 0.0
    assert low.rain_clutter_loss_db > 0.0


# ------------------------------------------------------------- kinematics


def test_coordinated_turn_follows_circle():
    speed, rate = 200.0, np.radians(3.0)  # standard-rate right turn
    target = Target(
        1,
        np.zeros(3),
        np.array([0.0, speed, 0.0]),
        motion_model=MotionModel.COORDINATED_TURN,
        turn_rate_rad_s=rate,
    )
    quarter_turn_s = (np.pi / 2) / rate
    steps = 300
    for _ in range(steps):
        target.update(quarter_turn_s / steps)
    radius = speed / rate
    # Right turn from North: centre at (+radius, 0); after 90 deg heading East.
    np.testing.assert_allclose(target.position[:2], [radius, radius], rtol=1e-9)
    assert target.speed == pytest.approx(speed)
    assert target.state.heading == pytest.approx(np.pi / 2)


# ------------------------------------------------------------ pulse-Doppler


def test_range_doppler_map_reports_closing_target_at_negative_velocity():
    radar = _radar(frequency_hz=10e9, prf_hz=4000.0)
    # lambda = 3 cm, PRF 4 kHz: blind speed 60 m/s, so use 20 m/s closing.
    target = _target([0.0, 6_000.0, 10.0], velocity=[0.0, -20.0, 0.0])
    engine = SimulationEngine(radar, [target], scan_gated_detection=False, seed=5)
    engine.set_pulse_doppler_mode(True, n_pulses=64, mti_order=0)
    # Point the antenna at the target and process one CPI.
    engine.radar.antenna_azimuth = 0.0
    for _ in range(5):
        engine.step(0.001)
        engine.radar.antenna_azimuth = 0.0
    rd = engine._rd_map
    assert rd is not None
    peak_row = np.unravel_index(np.argmax(rd.data_linear), rd.data_linear.shape)[0]
    bin_width = rd.velocity_axis_mps[1] - rd.velocity_axis_mps[0]
    assert rd.velocity_axis_mps[peak_row] == pytest.approx(-20.0, abs=bin_width)


# --------------------------------------------------------- reproducibility


def test_snr_is_deterministic_for_fluctuating_target():
    """Swerling fluctuation lives in Pd only; SNR uses the mean RCS."""
    engine = SimulationEngine(
        _radar(),
        [_target([0.0, 40_000.0, 1_000.0], swerling_model=SwerlingModel.SWERLING_1)],
        scan_gated_detection=False,
    )
    snrs = {round(engine.step(1e-6)[0].snr_db, 6) for _ in range(20)}
    assert len(snrs) == 1


def test_reset_reproduces_the_same_run():
    radar = _radar(scan_rate_rpm=30.0)
    target = _target(
        [0.0, 50_000.0, 2_000.0], velocity=[0.0, -300.0, 0.0],
        swerling_model=SwerlingModel.SWERLING_1,
    )
    engine = SimulationEngine(radar, [target], dt=0.05, seed=11)

    def record():
        history = engine.run(10.0).detection_history
        return [(r.time, r.is_detected) for r in history], [t.id for t in engine.tracks]

    first = record()
    engine.reset()
    assert engine.current_time == 0.0
    assert engine.tracks == []
    assert record() == first


# ---------------------------------------------------------------- monopulse


def test_monopulse_error_signal_is_linear_near_boresight_and_saturates():
    from radarsim.tracking.monopulse import MonopulseEstimator

    est = MonopulseEstimator(beamwidth_deg=2.0, monopulse_slope=1.6)
    small = np.radians(0.05)
    assert est.error_signal(small) == pytest.approx(
        1.6 * small / est.beamwidth_rad, rel=1e-3
    )
    assert est.error_signal(-small) == pytest.approx(-est.error_signal(small))
    assert abs(est.error_signal(np.radians(10.0))) < 1.0
    assert est.inverse_error_signal(est.error_signal(np.radians(0.7))) == pytest.approx(
        np.radians(0.7)
    )
    assert est.difference_pattern(0.0) == pytest.approx(0.0)
    assert est.sum_pattern(0.0) == pytest.approx(1.0)


def test_monopulse_angle_error_shrinks_with_snr():
    from radarsim.tracking.monopulse import MonopulseEstimator

    est = MonopulseEstimator(beamwidth_deg=1.5)
    rng = np.random.default_rng(0)

    def rms_error(snr_db):
        errors = [
            est.measure_angle(0.0, 0.0, 0.002, 0.0, snr_db=snr_db, rng=rng).measured_azimuth_rad
            - 0.002
            for _ in range(2000)
        ]
        return float(np.sqrt(np.mean(np.square(errors))))

    high, low = rms_error(30.0), rms_error(15.0)
    assert high < low
    assert high == pytest.approx(est.angular_accuracy(1000.0), rel=0.15)


def test_engine_uses_monopulse_when_enabled():
    def azimuth_errors(monopulse: bool):
        engine = SimulationEngine(
            _radar(),
            [_target([0.0, 20_000.0, 1_000.0])],
            scan_gated_detection=False,
            angle_noise_std_rad=0.01,
            seed=2,
        )
        engine.monopulse_enabled = monopulse
        errors = []
        for _ in range(200):
            engine.radar.antenna_azimuth = 0.0
            result = engine.step(1e-4)[0]
            if result.is_detected:
                errors.append(result.measured_azimuth_rad - result.true_azimuth_rad)
        return float(np.std(errors))

    assert azimuth_errors(True) < 0.25 * azimuth_errors(False)
