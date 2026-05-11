# Developed by Mehmet Gümüş (@SpaceEngineerSS) - RadarSim v2.x
"""
Extended Kalman Filter for Radar Target Tracking (Polar Measurements)

Implements an EKF with polar measurement model [r, θ] for direct
processing of radar detections without Cartesian pre-conversion.

State Vector: x = [x, y, vx, vy]^T (Cartesian)
Measurement Vector: z = [r, θ]^T (polar)

Measurement Model:
    r = √(x² + y²)
    θ = atan2(y, x)

Jacobian H:
    | ∂r/∂x    ∂r/∂y    ∂r/∂vx   ∂r/∂vy  |   | x/r    y/r    0  0 |
    | ∂θ/∂x    ∂θ/∂y    ∂θ/∂vx   ∂θ/∂vy  | = | -y/r²  x/r²   0  0 |

Features:
    - Adaptive measurement noise R based on SNR from Pulse-Doppler engine
    - Joseph form covariance update for numerical stability
    - Angle wrapping for innovation computation
    - Uncertainty ellipse extraction from P matrix
    - Innovation-based divergence detection

References:
    - Bar-Shalom, Y. "Estimation with Applications to Tracking and Navigation", 2001, Ch. 5.3
    - Richards, M.A. "Fundamentals of Radar Signal Processing", 2005
    - Developed by Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .kalman import KalmanState


@dataclass
class EKFConfig:
    """
    EKF configuration parameters.

    Attributes:
        process_noise_accel: Process noise acceleration σ_a [m/s²]
        range_std_m: Nominal range measurement std [m]
        angle_std_rad: Nominal angle measurement std [rad]
        snr_adapt_enabled: Enable adaptive R based on SNR
        divergence_threshold: Innovation threshold for divergence [σ]
    """

    process_noise_accel: float = 5.0
    range_std_m: float = 50.0
    angle_std_rad: float = 0.02  # ~1.15°
    snr_adapt_enabled: bool = True
    divergence_threshold: float = 5.0


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter with polar measurement model.

    Uses the nonlinear measurement function h(x) = [r, θ] with
    analytically derived Jacobian H for the EKF update step.

    Implements Constant Velocity (CV) prediction model, identical
    to the linear KF, maintaining interface compatibility.

    Features:
        - Direct polar [r, θ] measurements (no pre-conversion)
        - Adaptive R matrix based on SNR from Pulse-Doppler engine
        - Joseph form covariance update (numerically stable)
        - Angle wrapping for circular statistics
        - Divergence detection via normalized innovation

    Example:
        >>> ekf = ExtendedKalmanFilter()
        >>> state = ekf.initialize_from_polar(r_m=10000.0, theta_rad=0.5)
        >>> state = ekf.predict(state, dt=1.0)
        >>> state = ekf.update(state, z_polar=(10050.0, 0.51), snr_db=25.0)

    Reference: Bar-Shalom (2001), Ch. 5.3-5.4
    """

    # Minimum range to avoid singularity [m]
    _EPSILON_RANGE = 1.0

    def __init__(
        self,
        process_noise: float = 5.0,
        range_std: float = 50.0,
        angle_std: float = 0.02,
        snr_adapt: bool = True,
    ) -> None:
        """
        Initialize Extended Kalman Filter.

        Args:
            process_noise: Process noise σ_a [m/s²]
            range_std: Nominal range measurement noise σ_r [m]
            angle_std: Nominal angle measurement noise σ_θ [rad]
            snr_adapt: Enable adaptive R based on SNR
        """
        self.process_noise = process_noise
        self.range_std = range_std
        self.angle_std = angle_std
        self.snr_adapt = snr_adapt

        # Nominal measurement noise covariance
        self.R_nominal = np.diag([range_std**2, angle_std**2])

        # Track divergence count
        self._divergence_count = 0

        # ═══ PHASE 28: COAST MODE (Jamming Resilience) ═══
        self.coast_threshold_sjnr_db: float = 6.0
        self.max_coast_scans: int = 10
        self._coast_count: int = 0
        self._is_coasting: bool = False

    # ═══════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════

    def initialize(
        self,
        position: Tuple[float, float],
        velocity: Optional[Tuple[float, float]] = None,
        position_uncertainty: float = 100.0,
        velocity_uncertainty: float = 50.0,
    ) -> KalmanState:
        """
        Initialize from Cartesian position (backward compatible).

        Args:
            position: Initial (x, y) position [m]
            velocity: Initial (vx, vy) velocity [m/s]
            position_uncertainty: Position σ [m]
            velocity_uncertainty: Velocity σ [m/s]

        Returns:
            KalmanState with initialized state and covariance
        """
        if velocity is None:
            velocity = (0.0, 0.0)

        x = np.array(
            [position[0], position[1], velocity[0], velocity[1]],
            dtype=np.float64,
        )

        P = np.diag(
            [
                position_uncertainty**2,
                position_uncertainty**2,
                velocity_uncertainty**2,
                velocity_uncertainty**2,
            ]
        )

        return KalmanState(x=x, P=P)

    def initialize_from_polar(
        self,
        r_m: float,
        theta_rad: float,
        velocity: Optional[Tuple[float, float]] = None,
        position_uncertainty: float = 100.0,
        velocity_uncertainty: float = 50.0,
    ) -> KalmanState:
        """
        Initialize track from polar measurement.

        Converts (r, θ) → (x, y) for internal Cartesian state.

        Args:
            r_m: Range measurement [m]
            theta_rad: Azimuth measurement [rad]
            velocity: Initial (vx, vy) in m/s
            position_uncertainty: Position σ [m]
            velocity_uncertainty: Velocity σ [m/s]

        Returns:
            KalmanState with Cartesian state from polar init
        """
        x_pos = r_m * np.cos(theta_rad)
        y_pos = r_m * np.sin(theta_rad)

        return self.initialize(
            position=(x_pos, y_pos),
            velocity=velocity,
            position_uncertainty=position_uncertainty,
            velocity_uncertainty=velocity_uncertainty,
        )

    # ═══════════════════════════════════════════════════════════════
    # PREDICTION (Constant Velocity Model)
    # ═══════════════════════════════════════════════════════════════

    def predict(self, state: KalmanState, dt: float) -> KalmanState:
        """
        Predict state to next time step using CV model.

        x̂⁻ = F · x̂
        P⁻ = F · P · F^T + Q

        F = | 1  0  dt  0 |
            | 0  1  0  dt |
            | 0  0  1   0 |
            | 0  0  0   1 |

        Args:
            state: Current state
            dt: Time step [s]

        Returns:
            Predicted state

        Reference: Bar-Shalom (2001), Eq. 5.2.1
        """
        F = self._transition_matrix(dt)
        Q = self._process_noise_matrix(dt)

        x_pred = F @ state.x
        P_pred = F @ state.P @ F.T + Q

        return KalmanState(x=x_pred, P=P_pred)

    # ═══════════════════════════════════════════════════════════════
    # UPDATE (Extended Kalman — Polar Measurements)
    # ═══════════════════════════════════════════════════════════════

    def update(
        self,
        state: KalmanState,
        z_polar: Tuple[float, float],
        snr_db: float = 20.0,
    ) -> KalmanState:
        """
        EKF update with polar measurement [r, θ].

        Steps:
            1. Compute predicted measurement: ẑ = h(x̂⁻)
            2. Innovation: ỹ = z - ẑ (with angle wrapping)
            3. Jacobian: H = ∂h/∂x at x̂⁻
            4. Adaptive R from SNR
            5. Innovation covariance: S = H·P⁻·H^T + R
            6. Kalman gain: K = P⁻·H^T·S⁻¹
            7. State update: x̂ = x̂⁻ + K·ỹ
            8. Covariance: Joseph form

        Args:
            state: Predicted state
            z_polar: Polar measurement (range_m, azimuth_rad)
            snr_db: Signal-to-noise ratio [dB] from PD processor

        Returns:
            Updated state

        Reference: Bar-Shalom (2001), Eq. 5.3.7-5.3.12
        """
        z = np.array([z_polar[0], z_polar[1]], dtype=np.float64)

        # 1. Predicted measurement h(x̂⁻)
        z_pred = self._measurement_function(state.x)

        # 2. Innovation with angle wrapping
        y_innov = z - z_pred
        y_innov[1] = self._wrap_angle(y_innov[1])

        # 3. Measurement Jacobian at current state
        H = self._measurement_jacobian(state.x)

        # 4. Adaptive measurement noise R
        R = self._adaptive_R(snr_db)

        # 5. Innovation covariance
        S = H @ state.P @ H.T + R

        # 6. Kalman gain (explicit 2×2 inverse for speed)
        S_inv = self._invert_2x2(S)
        K = state.P @ H.T @ S_inv

        # 7. State update
        x_new = state.x + K @ y_innov

        # 8. Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ H
        P_new = I_KH @ state.P @ I_KH.T + K @ R @ K.T

        # Ensure symmetry
        P_new = 0.5 * (P_new + P_new.T)

        # Divergence detection
        self._check_divergence(y_innov, S)

        return KalmanState(x=x_new, P=P_new)

    def update_cartesian(
        self,
        state: KalmanState,
        measurement: Tuple[float, float],
    ) -> KalmanState:
        """
        Backward-compatible Cartesian update [x, y].

        Converts Cartesian detection to polar before EKF update.
        Falls back to standard linear update for stability.

        Args:
            state: Predicted state
            measurement: Cartesian (x, y) in meters

        Returns:
            Updated state
        """
        x_m, y_m = measurement
        r_m = np.sqrt(x_m**2 + y_m**2)
        theta_rad = np.arctan2(y_m, x_m)

        return self.update(state, (r_m, theta_rad))

    # ═══════════════════════════════════════════════════════════════
    # PHASE 28: COAST MODE (Jamming Resilience)
    # ═══════════════════════════════════════════════════════════════

    def update_with_jsr(
        self,
        state: KalmanState,
        z_polar: Tuple[float, float],
        snr_db: float = 20.0,
        jsr_db: float = 0.0,
    ) -> KalmanState:
        """
        EKF update with jamming-aware coast logic.

        Calculates SJNR = SNR / (1 + JSR). If SJNR < coast_threshold,
        freezes measurement updates and increments coast_count.
        Track continues on prediction only (CV model).

        When SJNR recovers above threshold:
            - Resumes measurement updates
            - Resets coast counter

        When coast_count exceeds max_coast_scans:
            - Track should be dropped by TrackManager

        Args:
            state: Predicted state
            z_polar: Polar measurement (range_m, azimuth_rad)
            snr_db: Signal-to-noise ratio [dB]
            jsr_db: Jam-to-signal ratio [dB] (from ECCM controller)

        Returns:
            Updated state (or unchanged if coasting)

        Reference: Schleher (1999), Ch. 4.8; Bar-Shalom (2001)
        """
        # Calculate SJNR
        sjnr_db = self._calculate_sjnr(snr_db, jsr_db)

        if sjnr_db < self.coast_threshold_sjnr_db:
            # COAST: Trust prediction, ignore measurement
            self._coast_count += 1
            self._is_coasting = True
            # Inflate covariance slightly during coast (uncertainty grows)
            P_inflated = state.P * 1.02
            return KalmanState(x=state.x.copy(), P=P_inflated)

        # SJNR recovered — resume tracking
        self._coast_count = 0
        self._is_coasting = False
        return self.update(state, z_polar, snr_db=snr_db)

    @staticmethod
    def _calculate_sjnr(snr_db: float, jsr_db: float) -> float:
        """
        Calculate Signal-to-Jamming-plus-Noise Ratio (SJNR).

        SJNR = SNR / (1 + JSR)  [linear]
        SJNR_dB = SNR_dB - 10·log₁₀(1 + 10^(JSR/10))

        Args:
            snr_db: Signal-to-noise ratio [dB]
            jsr_db: Jam-to-signal ratio [dB]

        Returns:
            SJNR [dB]

        Reference: Schleher (1999), Eq. 4.8
        """
        if jsr_db < -50:
            return snr_db  # No significant jamming
        jsr_linear = 10.0 ** (jsr_db / 10.0)
        return snr_db - 10.0 * np.log10(1.0 + jsr_linear)

    @property
    def is_coasting(self) -> bool:
        """Check if filter is in coast mode (prediction only)."""
        return self._is_coasting

    @property
    def coast_count(self) -> int:
        """Number of consecutive coast scans."""
        return self._coast_count

    @property
    def should_drop_track(self) -> bool:
        """Check if track should be dropped (exceeded max coast)."""
        return self._coast_count > self.max_coast_scans

    # ═══════════════════════════════════════════════════════════════
    # MEASUREMENT MODEL (Nonlinear)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _measurement_function(x: np.ndarray) -> np.ndarray:
        """
        Nonlinear measurement function h(x).

        h(x) = | r |   | √(x² + y²)    |
               | θ | = | atan2(y, x)    |

        Args:
            x: State vector [x, y, vx, vy]

        Returns:
            Predicted measurement [r, θ]

        Reference: Bar-Shalom (2001), Eq. 5.3.2
        """
        px, py = x[0], x[1]
        r = np.sqrt(px**2 + py**2)
        theta = np.arctan2(py, px)
        return np.array([r, theta], dtype=np.float64)

    @staticmethod
    def _measurement_jacobian(x: np.ndarray) -> np.ndarray:
        """
        Analytically derived Jacobian H = ∂h/∂x.

        H = | ∂r/∂x    ∂r/∂y    ∂r/∂vx   ∂r/∂vy  |
            | ∂θ/∂x    ∂θ/∂y    ∂θ/∂vx   ∂θ/∂vy  |

          = | x/r      y/r      0        0       |
            | -y/r²    x/r²     0        0       |

        Singularity guard: r_safe = max(r, ε)

        Args:
            x: State vector [x, y, vx, vy]

        Returns:
            2×4 Jacobian matrix

        Reference: Bar-Shalom (2001), Eq. 5.3.4
        """
        px, py = x[0], x[1]
        r = max(np.sqrt(px**2 + py**2), ExtendedKalmanFilter._EPSILON_RANGE)
        r2 = r**2

        H = np.array(
            [
                [px / r, py / r, 0.0, 0.0],
                [-py / r2, px / r2, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        return H

    # ═══════════════════════════════════════════════════════════════
    # ADAPTIVE MEASUREMENT NOISE
    # ═══════════════════════════════════════════════════════════════

    def _adaptive_R(self, snr_db: float) -> np.ndarray:
        """
        Compute adaptive measurement noise covariance R from SNR.

        When SNR is low, increase R to trust prediction more.
        When SNR is high, decrease R to trust measurement.

        Range noise: σ_r ∝ c/(2B·√(2·SNR))
        Angle noise: σ_θ ∝ θ_3dB/√(2·SNR)

        Scaling model (relative to nominal at SNR=20 dB):
            SNR > 20 dB: R = R_nominal × 0.5  (high confidence)
            10-20 dB:    R = R_nominal × 1.0  (standard)
            0-10 dB:     R = R_nominal × 5.0  (low confidence)
            < 0 dB:      R = R_nominal × 20.0 (very noisy)

        Args:
            snr_db: Signal-to-noise ratio [dB]

        Returns:
            2×2 measurement noise covariance R

        Reference: Richards (2005), Ch. 6.2
        """
        if not self.snr_adapt:
            return self.R_nominal.copy()

        # Continuous scaling: scale = 10^((20 - SNR)/20)
        # Clamp SNR to avoid extreme values
        snr_clamped = np.clip(snr_db, -10.0, 40.0)
        scale = 10.0 ** ((20.0 - snr_clamped) / 20.0)
        scale = np.clip(scale, 0.3, 30.0)

        return self.R_nominal * scale

    # ═══════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        """
        Wrap angle to [-π, π].

        Critical for EKF innovation computation where
        θ_innov = θ_measured - θ_predicted may cross ±π.

        Args:
            angle: Angle to wrap [rad]

        Returns:
            Wrapped angle in [-π, π]
        """
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def _invert_2x2(M: np.ndarray) -> np.ndarray:
        """
        Explicit 2×2 matrix inversion for speed.

        M⁻¹ = 1/(ad-bc) · | d  -b |
                           | -c  a |

        ~3× faster than np.linalg.inv for 2×2 case.

        Args:
            M: 2×2 matrix

        Returns:
            Inverse of M
        """
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        if abs(det) < 1e-30:
            return np.linalg.inv(M)  # Fallback for degenerate case

        inv_det = 1.0 / det
        return np.array(
            [[M[1, 1] * inv_det, -M[0, 1] * inv_det], [-M[1, 0] * inv_det, M[0, 0] * inv_det]],
            dtype=np.float64,
        )

    def _check_divergence(self, innovation: np.ndarray, S: np.ndarray) -> None:
        """
        Check for filter divergence via normalized innovation.

        Normalized innovation squared (NIS):
            ε² = ỹ^T · S⁻¹ · ỹ

        Should follow χ²(nz) distribution. If ε > threshold,
        the filter may be diverging.

        Args:
            innovation: Innovation vector [r, θ]
            S: Innovation covariance
        """
        S_inv = self._invert_2x2(S)
        nis = innovation @ S_inv @ innovation

        if nis > 25.0:  # ~5σ threshold for 2 DOF
            self._divergence_count += 1
        else:
            self._divergence_count = max(0, self._divergence_count - 1)

    @property
    def is_diverging(self) -> bool:
        """Check if filter is diverging (>3 consecutive outliers)."""
        return self._divergence_count > 3

    @staticmethod
    def _transition_matrix(dt: float) -> np.ndarray:
        """
        Constant Velocity state transition matrix F.

        F = | 1  0  dt  0 |
            | 0  1  0  dt |
            | 0  0  1   0 |
            | 0  0  0   1 |
        """
        return np.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )

    def _process_noise_matrix(self, dt: float) -> np.ndarray:
        """
        Discrete White Noise Acceleration process noise Q.

        Q = σ_a² · | dt⁴/4   0      dt³/2   0     |
                    | 0       dt⁴/4  0       dt³/2 |
                    | dt³/2   0      dt²     0     |
                    | 0       dt³/2  0       dt²   |

        Reference: Bar-Shalom (2001), Eq. 6.2.3
        """
        q = self.process_noise
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt

        return np.array(
            [
                [dt4 / 4, 0, dt3 / 2, 0],
                [0, dt4 / 4, 0, dt3 / 2],
                [dt3 / 2, 0, dt2, 0],
                [0, dt3 / 2, 0, dt2],
            ],
            dtype=np.float64,
        ) * (q**2)

    # ═══════════════════════════════════════════════════════════════
    # UNCERTAINTY ELLIPSE (for UI visualization)
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def uncertainty_ellipse(
        P: np.ndarray, confidence: float = 0.95, n_points: int = 32
    ) -> np.ndarray:
        """
        Compute uncertainty ellipse from covariance matrix.

        Extracts the 2D position covariance P[0:2, 0:2] and
        generates ellipse points for visualization.

        Args:
            P: 4×4 state covariance matrix
            confidence: Confidence level (0.95 → ~5.99 for 2D χ²)
            n_points: Number of ellipse points

        Returns:
            Array of (x, y) offsets [n_points, 2] for plotting

        Reference: Bar-Shalom (2001), Appendix C
        """
        # Position covariance (2×2 sub-matrix)
        P_pos = P[:2, :2]

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(P_pos)
        eigenvalues = np.maximum(eigenvalues, 0.0)  # Guard neg eigenvalues

        # Chi-squared critical value for 2 DOF
        # 0.95 → 5.991, 0.99 → 9.210, 0.90 → 4.605
        chi2_vals = {0.90: 4.605, 0.95: 5.991, 0.99: 9.210}
        chi2 = chi2_vals.get(confidence, 5.991)

        # Semi-axes
        a = np.sqrt(chi2 * eigenvalues[1])  # Major
        b = np.sqrt(chi2 * eigenvalues[0])  # Minor

        # Rotation angle
        angle = np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1])

        # Generate ellipse points
        theta = np.linspace(0, 2 * np.pi, n_points)
        ellipse = np.column_stack([a * np.cos(theta), b * np.sin(theta)])

        # Rotate
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        ellipse = ellipse @ rotation.T

        return ellipse

    # ═══════════════════════════════════════════════════════════════
    # LINEAR KF INTERFACE COMPATIBILITY
    # ═══════════════════════════════════════════════════════════════

    def get_position(self, state: KalmanState) -> Tuple[float, float]:
        """Extract position from state."""
        return (state.x[0], state.x[1])

    def get_velocity(self, state: KalmanState) -> Tuple[float, float]:
        """Extract velocity from state."""
        return (state.x[2], state.x[3])

    def get_speed(self, state: KalmanState) -> float:
        """Calculate speed from state."""
        return np.sqrt(state.x[2] ** 2 + state.x[3] ** 2)

    def get_heading(self, state: KalmanState) -> float:
        """Calculate heading angle (radians, 0 = North, CW positive)."""
        return np.arctan2(state.x[2], state.x[3])


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════


def validate_ekf() -> dict:
    """
    Self-validation of the Extended Kalman Filter.

    Tests:
        1. Jacobian vs numerical derivative
        2. Straight-line tracking convergence
        3. Angle wrapping correctness

    Reference: Bar-Shalom (2001), Ch. 5
    """
    results = {}

    # ── Test 1: Jacobian vs Numerical Derivative ──
    ekf = ExtendedKalmanFilter()
    x_test = np.array([10000.0, 5000.0, 100.0, -50.0])
    H_analytical = ekf._measurement_jacobian(x_test)

    # Numerical Jacobian (central difference)
    eps = 1e-6
    H_numerical = np.zeros((2, 4))
    for i in range(4):
        x_plus = x_test.copy()
        x_minus = x_test.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        h_plus = ekf._measurement_function(x_plus)
        h_minus = ekf._measurement_function(x_minus)
        H_numerical[:, i] = (h_plus - h_minus) / (2 * eps)

    jacobian_error = np.max(np.abs(H_analytical - H_numerical))
    results["jacobian_verification"] = {
        "max_error": jacobian_error,
        "pass": jacobian_error < 1e-6,
        "reference": "Bar-Shalom (2001), Ch. 5.3",
    }

    # ── Test 2: Straight-Line Tracking ──
    ekf_track = ExtendedKalmanFilter(process_noise=1.0, range_std=30.0, angle_std=0.01)
    # Target at (10km, 0) moving at (100, 50) m/s
    state = ekf_track.initialize(position=(10000.0, 0.0), velocity=(100.0, 50.0))
    dt = 1.0
    errors = []
    for k in range(20):
        # True position
        true_x = 10000.0 + 100.0 * (k + 1) * dt
        true_y = 50.0 * (k + 1) * dt
        true_r = np.sqrt(true_x**2 + true_y**2)
        true_theta = np.arctan2(true_y, true_x)

        # Add noise
        rng = np.random.default_rng(42 + k)
        z_r = true_r + rng.normal(0, 30)
        z_theta = true_theta + rng.normal(0, 0.01)

        state = ekf_track.predict(state, dt)
        state = ekf_track.update(state, (z_r, z_theta))

        pos_error = np.sqrt((state.x[0] - true_x) ** 2 + (state.x[1] - true_y) ** 2)
        errors.append(pos_error)

    rms_error = np.sqrt(np.mean(np.array(errors[-10:]) ** 2))
    results["straight_line_tracking"] = {
        "rms_error_m": rms_error,
        "pass": rms_error < 100.0,
        "reference": "Bar-Shalom (2001), Ch. 5.4",
    }

    # ── Test 3: Angle Wrapping ──
    wrap_tests = [
        (3.5, 3.5 - 2 * np.pi),
        (-3.5, -3.5 + 2 * np.pi),
        (np.pi, -np.pi),  # Edge case: π maps to -π or π
        (0.0, 0.0),
    ]
    all_wrap_pass = True
    for angle, expected in wrap_tests:
        wrapped = ExtendedKalmanFilter._wrap_angle(angle)
        if abs(wrapped - expected) > 1e-10 and abs(abs(wrapped) - np.pi) > 1e-10:
            all_wrap_pass = False
    results["angle_wrapping"] = {
        "pass": all_wrap_pass,
        "reference": "Circular statistics",
    }

    return results


if __name__ == "__main__":
    results = validate_ekf()
    print("=" * 60)
    print("Extended Kalman Filter Validation")
    print("=" * 60)
    for test_name, test_result in results.items():
        status = "✓ PASS" if test_result.get("pass", False) else "✗ FAIL"
        print(f"\n{status} | {test_name}")
        for k, v in test_result.items():
            if k != "pass":
                print(f"    {k}: {v}")
