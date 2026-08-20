import numpy as np
import pytest

from src.simulation.objects import Radar
from src.tracking.tracker import Track, TrackManager, TrackStatus


def make_track(manager, track_id, position, position_variance):
    state = manager.kf.initialize(position, position_uncertainty=1.0)
    state.P[:2, :2] = np.eye(2) * position_variance
    return Track(id=track_id, state=state)


def test_association_gate_uses_innovation_covariance():
    detection = [(100.0, 0.0)]
    uncertain = TrackManager(gate_distance=500.0, measurement_noise=1.0)
    uncertain.tracks[1] = make_track(uncertain, 1, (0.0, 0.0), 10_000.0)
    associations, _, _ = uncertain._associate(detection)
    assert associations == {1: 0}

    precise = TrackManager(gate_distance=500.0, measurement_noise=1.0)
    precise.tracks[1] = make_track(precise, 1, (0.0, 0.0), 1.0)
    associations, _, _ = precise._associate(detection)
    assert associations == {}


def test_global_assignment_preserves_two_feasible_matches():
    manager = TrackManager(gate_distance=10.0, measurement_noise=10.0)
    manager.tracks[1] = make_track(manager, 1, (0.0, 0.0), 100.0)
    manager.tracks[2] = make_track(manager, 2, (6.0, 0.0), 100.0)

    associations, unassigned_detections, unassigned_tracks = manager._associate(
        [(4.0, 0.0), (-5.0, 0.0)]
    )
    assert associations == {1: 1, 2: 0}
    assert unassigned_detections == []
    assert unassigned_tracks == []


def test_confirmation_requires_consecutive_hits():
    manager = TrackManager(
        gate_distance=500.0,
        confirm_hits=3,
        max_misses=5,
        measurement_noise=20.0,
    )
    track = manager.update([(1000.0, 0.0)], dt=1.0)[0]
    manager.update([(1000.0, 0.0)], dt=1.0)
    manager.update([], dt=1.0)
    manager.update([(1000.0, 0.0)], dt=1.0)
    track = manager.update([(1000.0, 0.0)], dt=1.0)[0]
    assert track.status == TrackStatus.TENTATIVE

    track = manager.update([(1000.0, 0.0)], dt=1.0)[0]
    assert track.status == TrackStatus.CONFIRMED


def test_track_is_deleted_at_configured_miss_count():
    manager = TrackManager(gate_distance=500.0, max_misses=2)
    manager.update([(1000.0, 0.0)], dt=0.5)
    manager.update([], dt=0.5)
    assert len(manager.tracks) == 1
    manager.update([], dt=0.5)
    assert manager.tracks == {}


def test_track_age_uses_simulation_time():
    manager = TrackManager()
    track = manager.update([(0.0, 0.0)], dt=0.25)[0]
    assert track.age_seconds == pytest.approx(0.0)
    manager.update([(0.0, 0.0)], dt=0.75)
    assert track.age_seconds == pytest.approx(0.75)


def test_confirmed_track_remains_reported_while_coasting():
    manager = TrackManager(confirm_hits=2, max_misses=3)
    manager.update([(100.0, 50.0)], dt=1.0)
    track = manager.update([(100.0, 50.0)], dt=1.0)[0]
    assert track.status == TrackStatus.CONFIRMED

    manager.update([], dt=1.0)
    assert track.status == TrackStatus.COASTING
    assert manager.get_confirmed_tracks() == [track]


def test_radar_elevation_and_heading_follow_north_east_up_convention():
    radar = Radar("coordinate-test", np.zeros(3))
    geometry = radar.calculate_target_geometry(np.array([1000.0, 1000.0, 1000.0]))
    assert geometry["azimuth_deg"] == pytest.approx(45.0)
    assert geometry["elevation_deg"] > 0.0

    manager = TrackManager()
    state = manager.kf.initialize((0.0, 0.0), velocity=(100.0, 0.0))
    assert manager.kf.get_heading(state) == pytest.approx(0.0)
    state = manager.kf.initialize((0.0, 0.0), velocity=(0.0, 100.0))
    assert manager.kf.get_heading(state) == pytest.approx(np.pi / 2.0)
