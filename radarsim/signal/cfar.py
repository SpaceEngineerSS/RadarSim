"""Statistically calibrated CFAR detectors for square-law radar data."""

from __future__ import annotations

from enum import Enum
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import gamma


class CFARType(Enum):
    CA = "cell_averaging"
    GO = "greatest_of"
    SO = "smallest_of"
    OS = "ordered_statistic"
    CAGO = "cell_averaging_go"


@lru_cache(maxsize=128)
def _half_window_multiplier(reference_cells: int, pfa: float, greatest: bool) -> float:
    def achieved_pfa(multiplier: float) -> float:
        def integrand(value: float) -> float:
            selection_probability = (
                gamma.cdf(value, reference_cells)
                if greatest
                else gamma.sf(value, reference_cells)
            )
            return (
                2.0
                * np.exp(-multiplier * value / reference_cells)
                * gamma.pdf(value, reference_cells)
                * selection_probability
            )

        return quad(integrand, 0.0, np.inf, epsabs=1e-12, limit=250)[0]

    return float(brentq(lambda value: achieved_pfa(value) - pfa, 0.0, 1e5))


@lru_cache(maxsize=128)
def _os_multiplier(total_reference_cells: int, rank: int, pfa: float) -> float:
    def achieved_pfa(multiplier: float) -> float:
        terms = [
            (total_reference_cells - index)
            / (total_reference_cells - index + multiplier)
            for index in range(rank)
        ]
        return float(np.prod(terms))

    return float(brentq(lambda value: achieved_pfa(value) - pfa, 0.0, 1e6))


class CFARDetector:
    """One- and two-dimensional CFAR for exponential power samples."""

    def __init__(
        self,
        guard_cells: int = 2,
        reference_cells: int = 8,
        pfa: float = 1e-6,
        cfar_type: CFARType = CFARType.CA,
        os_rank: int | None = None,
    ) -> None:
        if not isinstance(guard_cells, int) or guard_cells < 0:
            raise ValueError("guard_cells must be a non-negative integer")
        if not isinstance(reference_cells, int) or reference_cells < 1:
            raise ValueError("reference_cells must be a positive integer")
        if not 0.0 < pfa < 1.0 or not np.isfinite(pfa):
            raise ValueError("pfa must be finite and strictly between zero and one")
        if not isinstance(cfar_type, CFARType):
            raise ValueError("cfar_type must be a CFARType")

        total_reference_cells = 2 * reference_cells
        if os_rank is None:
            os_rank = int(np.ceil(0.75 * total_reference_cells))
        if not 1 <= os_rank <= total_reference_cells:
            raise ValueError("os_rank must select one of the reference cells")

        self.guard_cells = guard_cells
        self.reference_cells = reference_cells
        self.pfa = float(pfa)
        self.cfar_type = cfar_type
        self.os_rank = int(os_rank)

    @staticmethod
    def ca_multiplier(total_reference_cells: int, pfa: float) -> float:
        if total_reference_cells < 1 or not 0.0 < pfa < 1.0:
            raise ValueError("invalid CFAR cell count or probability")
        return total_reference_cells * (
            pfa ** (-1.0 / total_reference_cells) - 1.0
        )

    @property
    def threshold_multiplier(self) -> float:
        total = 2 * self.reference_cells
        if self.cfar_type == CFARType.CA:
            return self.ca_multiplier(total, self.pfa)
        if self.cfar_type in {CFARType.GO, CFARType.CAGO}:
            return _half_window_multiplier(
                self.reference_cells, self.pfa, greatest=True
            )
        if self.cfar_type == CFARType.SO:
            return _half_window_multiplier(
                self.reference_cells, self.pfa, greatest=False
            )
        return _os_multiplier(total, self.os_rank, self.pfa)

    def detect(
        self, signal: np.ndarray, db_input: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        power = self._as_power(signal, db_input)
        detections = np.zeros(power.size, dtype=bool)
        thresholds = np.zeros(power.size, dtype=float)
        margin = self.guard_cells + self.reference_cells
        if power.size <= 2 * margin:
            return detections, thresholds

        cut_indices = np.arange(margin, power.size - margin)
        if self.cfar_type == CFARType.OS:
            noise_estimate = self._ordered_statistics(power)
        else:
            cumulative = np.concatenate(([0.0], np.cumsum(power)))
            left = cumulative[cut_indices - self.guard_cells] - cumulative[
                cut_indices - margin
            ]
            right = cumulative[cut_indices + margin + 1] - cumulative[
                cut_indices + self.guard_cells + 1
            ]
            if self.cfar_type == CFARType.CA:
                noise_estimate = (left + right) / (2 * self.reference_cells)
            elif self.cfar_type in {CFARType.GO, CFARType.CAGO}:
                noise_estimate = np.maximum(left, right) / self.reference_cells
            else:
                noise_estimate = np.minimum(left, right) / self.reference_cells

        valid_thresholds = self.threshold_multiplier * noise_estimate
        thresholds[cut_indices] = valid_thresholds
        detections[cut_indices] = power[cut_indices] > valid_thresholds
        return detections, thresholds

    def _ordered_statistics(self, power: np.ndarray) -> np.ndarray:
        window_length = 2 * (self.reference_cells + self.guard_cells) + 1
        windows_view = np.lib.stride_tricks.sliding_window_view(power, window_length)
        left = windows_view[:, : self.reference_cells]
        right = windows_view[:, -self.reference_cells :]
        reference = np.concatenate((left, right), axis=1)
        return np.partition(reference, self.os_rank - 1, axis=1)[:, self.os_rank - 1]

    @staticmethod
    def _as_power(signal: np.ndarray, db_input: bool) -> np.ndarray:
        signal = np.asarray(signal, dtype=float)
        if signal.ndim != 1 or not np.all(np.isfinite(signal)):
            raise ValueError("signal must be a finite one-dimensional array")
        power = 10.0 ** (signal / 10.0) if db_input else signal
        if np.any(power < 0.0):
            raise ValueError("linear power samples must be non-negative")
        return power

    def detect_2d(
        self, rd_map: np.ndarray, db_input: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply true rectangular 2-D CA-CFAR, excluding a guard rectangle."""
        if self.cfar_type != CFARType.CA:
            raise NotImplementedError("calibrated 2-D processing currently supports CA-CFAR")
        data = np.asarray(rd_map, dtype=float)
        if data.ndim != 2 or not np.all(np.isfinite(data)):
            raise ValueError("rd_map must be a finite two-dimensional array")
        power = 10.0 ** (data / 10.0) if db_input else data
        if np.any(power < 0.0):
            raise ValueError("linear power samples must be non-negative")

        detections = np.zeros(power.shape, dtype=bool)
        thresholds = np.zeros(power.shape, dtype=float)
        guard = self.guard_cells
        reference = self.reference_cells
        margin = guard + reference
        outer_width = 2 * margin + 1
        guard_width = 2 * guard + 1
        training_count = outer_width**2 - guard_width**2
        if min(power.shape) <= 2 * margin:
            return detections, thresholds

        integral = np.pad(power, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

        def rectangle_sum(r0: int, c0: int, r1: int, c1: int) -> float:
            return float(
                integral[r1, c1]
                - integral[r0, c1]
                - integral[r1, c0]
                + integral[r0, c0]
            )

        multiplier = self.ca_multiplier(training_count, self.pfa)
        for row in range(margin, power.shape[0] - margin):
            for column in range(margin, power.shape[1] - margin):
                outer = rectangle_sum(
                    row - margin,
                    column - margin,
                    row + margin + 1,
                    column + margin + 1,
                )
                inner = rectangle_sum(
                    row - guard,
                    column - guard,
                    row + guard + 1,
                    column + guard + 1,
                )
                threshold = multiplier * (outer - inner) / training_count
                thresholds[row, column] = threshold
                detections[row, column] = power[row, column] > threshold
        return detections, thresholds

    @staticmethod
    def calculate_cfar_loss(n_ref_cells: int, pfa: float) -> float:
        """Return CA-CFAR threshold loss relative to known mean noise."""
        alpha = CFARDetector.ca_multiplier(n_ref_cells, pfa)
        fixed_threshold = -np.log(pfa)
        return float(10.0 * np.log10(alpha / fixed_threshold))
