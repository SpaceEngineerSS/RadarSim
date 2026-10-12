"""
Monopulse Angle Estimation

Amplitude-comparison monopulse with two squinted Gaussian beams per axis.

With one-way voltage patterns g(θ) = exp(-2 ln2 · θ²/θ3²) squinted to ±θs:
    Σ(θ) = g(θ - θs) + g(θ + θs)
    Δ(θ) = g(θ - θs) - g(θ + θs)
    ε(θ) = Δ/Σ = tanh(4 ln2 · θ · θs / θ3²)

The error signal is linear near boresight with normalized slope
km = 4 ln2 · θs/θ3 (per beamwidth) and saturates at ±1 off axis, so large
off-boresight angles cannot be recovered. Thermal noise gives the angle
accuracy σθ = θ3 / (km · sqrt(2 · SNR)).

References:
    - Sherman & Barton, "Monopulse Principles and Techniques", 2nd Ed., Ch. 3
    - Barton, "Radar System Analysis and Modeling" (2005), Eq. 9.2.4
    - Skolnik, "Radar Handbook", 3rd Ed., Ch. 9
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.physics.geometry import wrap_to_pi

_FOUR_LN2 = 4.0 * np.log(2.0)


@dataclass
class MonopulseResult:
    """Result of monopulse angle measurement."""

    measured_azimuth_rad: float
    measured_elevation_rad: float
    error_signal_az: float  # Normalized error ε = Δ/Σ in azimuth
    error_signal_el: float  # Normalized error ε = Δ/Σ in elevation
    snr_db: float
    angular_accuracy_rad: float  # 1-sigma thermal-noise accuracy


class MonopulseEstimator:
    """Amplitude-comparison monopulse angle estimator (see module docstring)."""

    def __init__(self, beamwidth_deg: float = 1.5, monopulse_slope: float = 1.6):
        """
        Args:
            beamwidth_deg: 3-dB beamwidth θ3 [degrees]
            monopulse_slope: Normalized boresight slope km = dε/d(θ/θ3),
                typically 1.2-2.0; sets the squint θs = km·θ3/(4 ln2)
        """
        if beamwidth_deg <= 0.0 or monopulse_slope <= 0.0:
            raise ValueError("beamwidth_deg and monopulse_slope must be positive")
        self.beamwidth_deg = beamwidth_deg
        self.beamwidth_rad = np.radians(beamwidth_deg)
        self.monopulse_slope = monopulse_slope
        self.squint_angle_rad = monopulse_slope * self.beamwidth_rad / _FOUR_LN2

    def _voltage_pattern(self, theta_rad: float) -> float:
        return float(np.exp(-0.5 * _FOUR_LN2 * (theta_rad / self.beamwidth_rad) ** 2))

    def sum_pattern(self, theta_rad: float) -> float:
        """Σ(θ), normalized to 1 at boresight."""
        s = self.squint_angle_rad
        return (
            self._voltage_pattern(theta_rad - s) + self._voltage_pattern(theta_rad + s)
        ) / (2.0 * self._voltage_pattern(s))

    def difference_pattern(self, theta_rad: float) -> float:
        """Δ(θ) with the same normalization as Σ; odd, null at boresight."""
        s = self.squint_angle_rad
        return (
            self._voltage_pattern(theta_rad - s) - self._voltage_pattern(theta_rad + s)
        ) / (2.0 * self._voltage_pattern(s))

    def error_signal(self, theta_rad: float) -> float:
        """ε = Δ/Σ = tanh(4 ln2 θ θs / θ3²); bounded to (-1, 1)."""
        return float(
            np.tanh(_FOUR_LN2 * theta_rad * self.squint_angle_rad / self.beamwidth_rad**2)
        )

    def inverse_error_signal(self, epsilon: float) -> float:
        """Off-boresight angle for a measured ε (clipped inside the saturation)."""
        epsilon = float(np.clip(epsilon, -0.999999, 0.999999))
        return float(
            np.arctanh(epsilon) * self.beamwidth_rad**2 / (_FOUR_LN2 * self.squint_angle_rad)
        )

    def angular_accuracy(self, snr_linear: float) -> float:
        """Thermal-noise accuracy σθ = θ3 / (km · sqrt(2 SNR)) [rad]."""
        if snr_linear <= 0.0:
            return self.beamwidth_rad
        return min(
            self.beamwidth_rad,
            self.beamwidth_rad / (self.monopulse_slope * np.sqrt(2.0 * snr_linear)),
        )

    def measure_angle(
        self,
        beam_azimuth_rad: float,
        beam_elevation_rad: float,
        target_azimuth_rad: float,
        target_elevation_rad: float,
        snr_db: float = 20.0,
        rng: Optional[np.random.Generator] = None,
    ) -> MonopulseResult:
        """
        Measure target angles from the Σ/Δ error signals plus thermal noise.

        The noise is applied to ε (slope km/θ3 converts σθ to σε) and the
        estimate is obtained by inverting the error-signal curve.
        """
        if rng is None:
            rng = np.random.default_rng()
        sigma_angle = self.angular_accuracy(10 ** (snr_db / 10))
        sigma_epsilon = sigma_angle * self.monopulse_slope / self.beamwidth_rad

        delta_az = wrap_to_pi(target_azimuth_rad - beam_azimuth_rad)
        delta_el = target_elevation_rad - beam_elevation_rad
        eps_az = self.error_signal(delta_az) + rng.normal(0.0, sigma_epsilon)
        eps_el = self.error_signal(delta_el) + rng.normal(0.0, sigma_epsilon)

        return MonopulseResult(
            measured_azimuth_rad=wrap_to_pi(
                beam_azimuth_rad + self.inverse_error_signal(eps_az)
            ),
            measured_elevation_rad=beam_elevation_rad + self.inverse_error_signal(eps_el),
            error_signal_az=eps_az,
            error_signal_el=eps_el,
            snr_db=snr_db,
            angular_accuracy_rad=sigma_angle,
        )
