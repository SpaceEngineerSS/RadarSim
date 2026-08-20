import numpy as np
import pytest

from src.advanced.sensor_fusion import AdvancedSensorFusion, SensorMeasurement


def measurement(
    sensor_id: str,
    state: np.ndarray,
    variance: float,
    *,
    timestamp: float = 0.0,
    target_id: str = "T1",
) -> SensorMeasurement:
    return SensorMeasurement(
        sensor_id=sensor_id,
        timestamp=timestamp,
        position=state[:3],
        velocity=state[3:],
        measurement_type="radar",
        uncertainty=np.eye(6) * variance,
        confidence=0.9,
        target_id=target_id,
    )


def test_information_fusion_matches_closed_form_gaussian_product() -> None:
    first = measurement("A", np.zeros(6), 4.0)
    second = measurement("B", np.full(6, 10.0), 1.0)

    result = AdvancedSensorFusion(
        fusion_method="independent_gaussian", correlation_model="independent"
    ).fuse([first, second])

    np.testing.assert_allclose(result["fused_state"], np.full(6, 8.0))
    np.testing.assert_allclose(result["fused_covariance"], np.eye(6) * 0.8)


def test_covariance_intersection_does_not_double_count_identical_information() -> None:
    first = measurement("A", np.zeros(6), 4.0)
    second = measurement("B", np.full(6, 2.0), 4.0)

    result = AdvancedSensorFusion().fuse([first, second])

    np.testing.assert_allclose(result["fused_state"], np.ones(6), atol=1e-8)
    np.testing.assert_allclose(result["fused_covariance"], np.eye(6) * 4.0)
    assert result["weights"].sum() == pytest.approx(1.0)


def test_measurements_are_propagated_to_latest_timestamp() -> None:
    old_state = np.array([100.0, 20.0, -5.0, 10.0, -2.0, 1.0])
    old = measurement("A", old_state, 1.0, timestamp=2.0)

    result = AdvancedSensorFusion(process_noise_spectral_density=0.0).fuse([old])

    np.testing.assert_allclose(result["fused_state"], old_state)
    assert result["timestamp"] == pytest.approx(2.0)

    later = measurement(
        "B",
        np.array([130.0, 14.0, -2.0, 10.0, -2.0, 1.0]),
        1.0,
        timestamp=5.0,
    )
    aligned = AdvancedSensorFusion(
        fusion_method="independent_gaussian",
        process_noise_spectral_density=0.0,
        correlation_model="independent",
    ).fuse([old, later])
    np.testing.assert_allclose(aligned["fused_state"], later.state, atol=1e-10)


def test_covariance_intersection_is_order_invariant() -> None:
    measurements = [
        measurement("A", np.arange(6, dtype=float), 2.0),
        measurement("B", np.arange(6, dtype=float) + 3.0, 5.0),
        measurement("C", np.arange(6, dtype=float) - 1.0, 3.0),
    ]
    fusion = AdvancedSensorFusion()

    forward = fusion.fuse(measurements)
    reverse = fusion.fuse(list(reversed(measurements)))

    np.testing.assert_allclose(forward["fused_state"], reverse["fused_state"])
    np.testing.assert_allclose(
        forward["fused_covariance"], reverse["fused_covariance"]
    )


def test_different_target_measurements_are_rejected() -> None:
    first = measurement("A", np.zeros(6), 1.0, target_id="T1")
    second = measurement("B", np.zeros(6), 1.0, target_id="T2")

    with pytest.raises(ValueError, match="different targets"):
        AdvancedSensorFusion().fuse([first, second])


@pytest.mark.parametrize(
    ("uncertainty", "message"),
    [
        (np.eye(5), "shape"),
        (np.diag([1.0, 1.0, 1.0, 1.0, 1.0, -1.0]), "positive definite"),
    ],
)
def test_measurement_covariance_is_validated(
    uncertainty: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SensorMeasurement(
            sensor_id="A",
            timestamp=0.0,
            position=np.zeros(3),
            velocity=np.zeros(3),
            measurement_type="radar",
            uncertainty=uncertainty,
        )


def test_unsupported_fusion_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        AdvancedSensorFusion(fusion_method="unsupported")
