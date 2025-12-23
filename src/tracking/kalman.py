"""
Linear Kalman Filter for Radar Target Tracking

Implements a Constant Velocity (CV) motion model for Track-While-Scan (TWS)
radar systems. Uses standard Kalman filter equations for prediction and update.

State Vector: [x, y, vx, vy]^T
    - x, y: Position in Cartesian coordinates (meters)
    - vx, vy: Velocity components (m/s)

Reference:
    - Bar-Shalom, Y. "Estimation with Applications to Tracking and Navigation", 2001
    - Blackman, S. "Design and Analysis of Modern Tracking Systems", 1999
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class KalmanState:
    """
    State container for Kalman Filter.
    
    Attributes:
        x: State vector [x, y, vx, vy]
        P: State covariance matrix (4x4)
    """
    x: np.ndarray  # State vector
    P: np.ndarray  # Covariance matrix


class LinearKalmanFilter:
    """
    Linear Kalman Filter for 2D target tracking.
    
    Uses Constant Velocity (CV) motion model:
        x_{k+1} = x_k + vx * dt
        y_{k+1} = y_k + vy * dt
        vx_{k+1} = vx_k (constant)
        vy_{k+1} = vy_k (constant)
    
    Measurement model:
        z = [x, y] (position only from radar)
    
    Example:
        >>> kf = LinearKalmanFilter(process_noise=1.0, measurement_noise=50.0)
        >>> initial_state = kf.initialize([1000, 2000], [100, 50])
        >>> predicted = kf.predict(initial_state, dt=1.0)
        >>> updated = kf.update(predicted, [1005, 2055])
    """
    
    def __init__(
        self,
        process_noise: float = 1.0,
        measurement_noise: float = 50.0
    ) -> None:
        """
        Initialize Kalman Filter.
        
        Args:
            process_noise: Process noise standard deviation (m/s^2)
                          Higher = more responsive to maneuvers
            measurement_noise: Measurement noise standard deviation (meters)
                              Higher = smoother tracks, slower response
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        
        # Measurement matrix H: We only observe position [x, y]
        # z = H * x  where x = [x, y, vx, vy]
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        # Measurement noise covariance R
        self.R = np.eye(2) * (measurement_noise ** 2)
    
    def initialize(
        self,
        position: Tuple[float, float],
        velocity: Optional[Tuple[float, float]] = None,
        position_uncertainty: float = 100.0,
        velocity_uncertainty: float = 50.0
    ) -> KalmanState:
        """
        Initialize a new track state.
        
        Args:
            position: Initial position (x, y) in meters
            velocity: Initial velocity (vx, vy) in m/s, defaults to (0, 0)
            position_uncertainty: Initial position uncertainty (meters)
            velocity_uncertainty: Initial velocity uncertainty (m/s)
            
        Returns:
            KalmanState with initialized state and covariance
        """
        if velocity is None:
            velocity = (0.0, 0.0)
        
        # State vector [x, y, vx, vy]
        x = np.array([
            position[0],
            position[1],
            velocity[0],
            velocity[1]
        ], dtype=np.float64)
        
        # Initial covariance (diagonal)
        P = np.diag([
            position_uncertainty ** 2,
            position_uncertainty ** 2,
            velocity_uncertainty ** 2,
            velocity_uncertainty ** 2
        ])
        
        return KalmanState(x=x, P=P)
    
    def _get_transition_matrix(self, dt: float) -> np.ndarray:
        """
        Get state transition matrix F for time step dt.
        
        Constant velocity model:
        | 1  0  dt  0 |
        | 0  1  0  dt |
        | 0  0  1   0 |
        | 0  0  0   1 |
        """
        return np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
    
    def _get_process_noise(self, dt: float) -> np.ndarray:
        """
        Get process noise covariance Q for time step dt.
        
        Uses discrete white noise acceleration model:
        Q = G * G^T * q^2
        
        where G = [dt^2/2, dt^2/2, dt, dt]^T
        and q = process noise intensity
        """
        q = self.process_noise
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        # Discrete white noise acceleration model
        Q = np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ]
        ], dtype=np.float64) * (q ** 2)
        
        return Q
    
    def predict(self, state: KalmanState, dt: float) -> KalmanState:
        """
        Predict state to next time step.
        
        Prediction equations:
            x_pred = F * x
            P_pred = F * P * F^T + Q
        
        Args:
            state: Current state
            dt: Time step (seconds)
            
        Returns:
            Predicted state
        """
        F = self._get_transition_matrix(dt)
        Q = self._get_process_noise(dt)
        
        # State prediction
        x_pred = F @ state.x
        
        # Covariance prediction
        P_pred = F @ state.P @ F.T + Q
        
        return KalmanState(x=x_pred, P=P_pred)
    
    def update(
        self,
        state: KalmanState,
        measurement: Tuple[float, float]
    ) -> KalmanState:
        """
        Update state with measurement.
        
        Update equations:
            y = z - H * x          (innovation)
            S = H * P * H^T + R    (innovation covariance)
            K = P * H^T * S^-1     (Kalman gain)
            x_new = x + K * y
            P_new = (I - K * H) * P
        
        Args:
            state: Predicted state
            measurement: Position measurement (x, y) in meters
            
        Returns:
            Updated state
        """
        z = np.array(measurement, dtype=np.float64)
        
        # Innovation (measurement residual)
        y = z - self.H @ state.x
        
        # Innovation covariance
        S = self.H @ state.P @ self.H.T + self.R
        
        # Kalman gain
        K = state.P @ self.H.T @ np.linalg.inv(S)
        
        # State update
        x_new = state.x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(4) - K @ self.H
        P_new = I_KH @ state.P @ I_KH.T + K @ self.R @ K.T
        
        return KalmanState(x=x_new, P=P_new)
    
    def get_position(self, state: KalmanState) -> Tuple[float, float]:
        """Extract position from state."""
        return (state.x[0], state.x[1])
    
    def get_velocity(self, state: KalmanState) -> Tuple[float, float]:
        """Extract velocity from state."""
        return (state.x[2], state.x[3])
    
    def get_speed(self, state: KalmanState) -> float:
        """Calculate speed from state."""
        return np.sqrt(state.x[2]**2 + state.x[3]**2)
    
    def get_heading(self, state: KalmanState) -> float:
        """Calculate heading angle (radians, 0 = North, CW positive)."""
        return np.arctan2(state.x[2], state.x[3])
