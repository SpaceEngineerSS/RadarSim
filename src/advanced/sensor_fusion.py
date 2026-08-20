"""Timestamp-aligned Cartesian state fusion for heterogeneous sensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize


@dataclass
class SensorMeasurement:
    """A six-state Cartesian estimate and its error covariance."""

    sensor_id: str
    timestamp: float
    position: np.ndarray
    velocity: np.ndarray
    measurement_type: str
    uncertainty: np.ndarray
    confidence: float = 1.0
    target_id: str | None = None

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)
        self.uncertainty = np.asarray(self.uncertainty, dtype=float)

        if not self.sensor_id:
            raise ValueError("sensor_id must not be empty")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if self.position.shape != (3,) or self.velocity.shape != (3,):
            raise ValueError("position and velocity must have shape (3,)")
        if self.uncertainty.shape != (6, 6):
            raise ValueError("uncertainty must have shape (6, 6)")
        if not all(
            np.all(np.isfinite(value))
            for value in (self.position, self.velocity, self.uncertainty)
        ):
            raise ValueError("state and uncertainty must be finite")
        if not np.allclose(self.uncertainty, self.uncertainty.T, atol=1e-10):
            raise ValueError("uncertainty must be symmetric")
        try:
            np.linalg.cholesky(self.uncertainty)
        except np.linalg.LinAlgError as exc:
            raise ValueError("uncertainty must be positive definite") from exc
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

    @property
    def state(self) -> np.ndarray:
        return np.concatenate((self.position, self.velocity))


class AdvancedSensorFusion:
    """Fuse Cartesian estimates with explicit correlation assumptions."""

    _METHODS = {
        "adaptive",
        "independent_gaussian",
        "covariance_intersection",
        "ci",
    }

    def __init__(
        self,
        fusion_method: str = "adaptive",
        process_noise_spectral_density: float = 1.0,
        correlation_model: str = "unknown",
    ) -> None:
        if fusion_method not in self._METHODS:
            raise ValueError(f"unsupported fusion method: {fusion_method}")
        if process_noise_spectral_density < 0.0:
            raise ValueError("process_noise_spectral_density must be non-negative")
        if correlation_model not in {"unknown", "independent"}:
            raise ValueError("correlation_model must be 'unknown' or 'independent'")

        self.fusion_method = fusion_method
        self.process_noise_spectral_density = float(process_noise_spectral_density)
        self.correlation_model = correlation_model

    def fuse(self, measurements: list[SensorMeasurement]) -> dict[str, Any]:
        if self.fusion_method == "adaptive":
            return self.adaptive_fusion(measurements)
        if self.fusion_method in {"covariance_intersection", "ci"}:
            return self.covariance_intersection_fusion(measurements)
        return self.independent_gaussian_fusion(measurements)

    def adaptive_fusion(
        self, measurements: list[SensorMeasurement]
    ) -> dict[str, Any]:
        if self.correlation_model == "independent":
            return self.independent_gaussian_fusion(measurements)
        return self.covariance_intersection_fusion(measurements)

    def independent_gaussian_fusion(
        self, measurements: list[SensorMeasurement]
    ) -> dict[str, Any]:
        aligned, timestamp = self._align_measurements(measurements)
        if not aligned:
            return {}

        identity = np.eye(6)
        information_matrix = np.zeros((6, 6))
        information_state = np.zeros(6)
        for state, covariance, _ in aligned:
            precision = np.linalg.solve(covariance, identity)
            information_matrix += precision
            information_state += precision @ state

        covariance = np.linalg.solve(information_matrix, identity)
        state = np.linalg.solve(information_matrix, information_state)
        return self._result(
            state,
            covariance,
            "independent_gaussian",
            timestamp,
            len(aligned),
        )

    def covariance_intersection_fusion(
        self, measurements: list[SensorMeasurement]
    ) -> dict[str, Any]:
        aligned, timestamp = self._align_measurements(measurements)
        if not aligned:
            return {}
        if len(aligned) == 1:
            state, covariance, _ = aligned[0]
            result = self._result(
                state, covariance, "covariance_intersection", timestamp, 1
            )
            result["weights"] = np.ones(1)
            return result

        identity = np.eye(6)
        precisions = [np.linalg.solve(covariance, identity) for _, covariance, _ in aligned]
        information_states = [
            precision @ state
            for precision, (state, _, _) in zip(precisions, aligned)
        ]
        count = len(aligned)

        def objective(weights: np.ndarray) -> float:
            information = sum(
                weight * precision
                for weight, precision in zip(weights, precisions)
            )
            sign, log_determinant = np.linalg.slogdet(information)
            return -log_determinant if sign > 0 else np.inf

        optimization = minimize(
            objective,
            np.full(count, 1.0 / count),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * count,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if not optimization.success:
            raise RuntimeError(f"covariance intersection failed: {optimization.message}")

        weights = np.clip(optimization.x, 0.0, 1.0)
        weights /= weights.sum()
        information = sum(
            weight * precision
            for weight, precision in zip(weights, precisions)
        )
        information_state = sum(
            weight * vector
            for weight, vector in zip(weights, information_states)
        )
        covariance = np.linalg.solve(information, identity)
        state = np.linalg.solve(information, information_state)
        result = self._result(
            state,
            covariance,
            "covariance_intersection",
            timestamp,
            count,
        )
        result["weights"] = weights
        return result

    def _align_measurements(
        self, measurements: list[SensorMeasurement]
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, SensorMeasurement]], float]:
        if not measurements:
            return [], 0.0

        target_ids = {measurement.target_id for measurement in measurements}
        if len(target_ids) > 1:
            raise ValueError("measurements from different targets cannot be fused")

        timestamp = max(measurement.timestamp for measurement in measurements)
        aligned = []
        for measurement in sorted(
            measurements, key=lambda item: (item.sensor_id, item.timestamp)
        ):
            dt = timestamp - measurement.timestamp
            transition = np.eye(6)
            transition[:3, 3:] = np.eye(3) * dt
            state = transition @ measurement.state
            covariance = (
                transition @ measurement.uncertainty @ transition.T
                + self._process_noise(dt)
            )
            aligned.append((state, covariance, measurement))
        return aligned, timestamp

    def _process_noise(self, dt: float) -> np.ndarray:
        q = self.process_noise_spectral_density
        noise = np.zeros((6, 6))
        noise[:3, :3] = np.eye(3) * q * dt**3 / 3.0
        noise[:3, 3:] = np.eye(3) * q * dt**2 / 2.0
        noise[3:, :3] = noise[:3, 3:].T
        noise[3:, 3:] = np.eye(3) * q * dt
        return noise

    @staticmethod
    def _result(
        state: np.ndarray,
        covariance: np.ndarray,
        method: str,
        timestamp: float,
        sensor_count: int,
    ) -> dict[str, Any]:
        covariance = 0.5 * (covariance + covariance.T)
        return {
            "fused_state": state,
            "fused_covariance": covariance,
            "fusion_method": method,
            "sensor_count": sensor_count,
            "timestamp": timestamp,
        }
