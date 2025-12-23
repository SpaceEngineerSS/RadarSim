"""
CFAR (Constant False Alarm Rate) Detectors

Implements various CFAR algorithms for radar target detection
in the presence of clutter and interference.

References:
    - Rohling, "Radar CFAR Thresholding in Clutter and Multiple Target Situations",
      IEEE Trans. AES, Vol. 19, No. 4, July 1983
    - Skolnik, "Radar Handbook", 3rd Ed., Chapter 6
    - Richards, "Fundamentals of Radar Signal Processing", 2nd Ed., Chapter 7
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum
import numba


class CFARType(Enum):
    """CFAR detector variants."""
    CA = "cell_averaging"       # Cell-Averaging CFAR
    GO = "greatest_of"          # Greatest-Of CFAR
    SO = "smallest_of"          # Smallest-Of CFAR
    OS = "ordered_statistic"    # Order-Statistic CFAR
    CAGO = "cell_averaging_go"  # CA-GO hybrid


@numba.jit(nopython=True, cache=True)
def _ca_cfar_1d_jit(
    signal: np.ndarray,
    guard_cells: int,
    reference_cells: int,
    pfa: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled 1D Cell-Averaging CFAR.
    
    Threshold = α × (1/N) × Σ(reference cell values)
    where α is determined by desired Pfa
    
    Args:
        signal: Input signal power (linear)
        guard_cells: Number of guard cells on each side
        reference_cells: Number of reference cells on each side
        pfa: Probability of false alarm
        
    Returns:
        Tuple of (detection mask, threshold values)
        
    Reference: Rohling, 1983, Eq. 2
    """
    n = len(signal)
    detections = np.zeros(n, dtype=np.bool_)
    thresholds = np.zeros(n)
    
    # Total reference window size
    total_ref = 2 * reference_cells
    
    # CFAR threshold multiplier for exponential noise
    # Pfa = (1 + α/N)^(-N) → α = N × (Pfa^(-1/N) - 1)
    alpha = total_ref * (pfa ** (-1.0 / total_ref) - 1)
    
    for i in range(guard_cells + reference_cells, n - guard_cells - reference_cells):
        # Reference window indices
        # Leading cells: i - guard - ref to i - guard - 1
        # Lagging cells: i + guard + 1 to i + guard + ref
        
        sum_ref = 0.0
        count = 0
        
        # Leading reference cells
        for j in range(i - guard_cells - reference_cells, i - guard_cells):
            sum_ref += signal[j]
            count += 1
        
        # Lagging reference cells
        for j in range(i + guard_cells + 1, i + guard_cells + reference_cells + 1):
            sum_ref += signal[j]
            count += 1
        
        # Average and threshold
        if count > 0:
            noise_estimate = sum_ref / count
            threshold = alpha * noise_estimate
            thresholds[i] = threshold
            
            if signal[i] > threshold:
                detections[i] = True
    
    return detections, thresholds


@numba.jit(nopython=True, cache=True)
def _go_cfar_1d_jit(
    signal: np.ndarray,
    guard_cells: int,
    reference_cells: int,
    pfa: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled 1D Greatest-Of CFAR.
    
    Uses maximum of leading/lagging estimates.
    Better at clutter edges but higher loss.
    
    Args:
        signal: Input signal power (linear)
        guard_cells: Number of guard cells on each side
        reference_cells: Number of reference cells on each side
        pfa: Probability of false alarm
        
    Returns:
        Tuple of (detection mask, threshold values)
        
    Reference: Rohling, 1983, Eq. 8
    """
    n = len(signal)
    detections = np.zeros(n, dtype=np.bool_)
    thresholds = np.zeros(n)
    
    # CFAR multiplier (for half-window)
    alpha = reference_cells * (pfa ** (-1.0 / reference_cells) - 1)
    
    for i in range(guard_cells + reference_cells, n - guard_cells - reference_cells):
        # Leading reference cells
        sum_leading = 0.0
        for j in range(i - guard_cells - reference_cells, i - guard_cells):
            sum_leading += signal[j]
        leading_est = sum_leading / reference_cells
        
        # Lagging reference cells
        sum_lagging = 0.0
        for j in range(i + guard_cells + 1, i + guard_cells + reference_cells + 1):
            sum_lagging += signal[j]
        lagging_est = sum_lagging / reference_cells
        
        # Greatest-Of selection
        noise_estimate = max(leading_est, lagging_est)
        threshold = alpha * noise_estimate
        thresholds[i] = threshold
        
        if signal[i] > threshold:
            detections[i] = True
    
    return detections, thresholds


@numba.jit(nopython=True, cache=True)
def _so_cfar_1d_jit(
    signal: np.ndarray,
    guard_cells: int,
    reference_cells: int,
    pfa: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    JIT-compiled 1D Smallest-Of CFAR.
    
    Uses minimum of leading/lagging estimates.
    Better for multiple closely-spaced targets.
    
    Args:
        signal: Input signal power (linear)
        guard_cells: Number of guard cells on each side
        reference_cells: Number of reference cells on each side
        pfa: Probability of false alarm
        
    Returns:
        Tuple of (detection mask, threshold values)
        
    Reference: Rohling, 1983, Eq. 10
    """
    n = len(signal)
    detections = np.zeros(n, dtype=np.bool_)
    thresholds = np.zeros(n)
    
    # CFAR multiplier
    alpha = reference_cells * (pfa ** (-1.0 / reference_cells) - 1)
    
    for i in range(guard_cells + reference_cells, n - guard_cells - reference_cells):
        # Leading reference cells
        sum_leading = 0.0
        for j in range(i - guard_cells - reference_cells, i - guard_cells):
            sum_leading += signal[j]
        leading_est = sum_leading / reference_cells
        
        # Lagging reference cells
        sum_lagging = 0.0
        for j in range(i + guard_cells + 1, i + guard_cells + reference_cells + 1):
            sum_lagging += signal[j]
        lagging_est = sum_lagging / reference_cells
        
        # Smallest-Of selection
        noise_estimate = min(leading_est, lagging_est)
        threshold = alpha * noise_estimate
        thresholds[i] = threshold
        
        if signal[i] > threshold:
            detections[i] = True
    
    return detections, thresholds


class CFARDetector:
    """
    Constant False Alarm Rate (CFAR) Detector
    
    Implements CA-CFAR, GO-CFAR, and SO-CFAR variants for
    adaptive threshold detection in radar systems.
    
    References:
        - Rohling, IEEE Trans. AES, 1983
        - Skolnik, "Radar Handbook", Ch. 6
    """
    
    def __init__(
        self,
        guard_cells: int = 2,
        reference_cells: int = 8,
        pfa: float = 1e-6,
        cfar_type: CFARType = CFARType.CA
    ):
        """
        Initialize CFAR detector.
        
        Args:
            guard_cells: Guard cells on each side of CUT
            reference_cells: Reference/training cells on each side
            pfa: Probability of false alarm
            cfar_type: CFAR algorithm variant
        """
        self.guard_cells = guard_cells
        self.reference_cells = reference_cells
        self.pfa = pfa
        self.cfar_type = cfar_type
    
    def detect(
        self,
        signal: np.ndarray,
        db_input: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply CFAR detection to signal.
        
        Args:
            signal: Input signal (power)
            db_input: True if signal is in dB (will convert to linear)
            
        Returns:
            Tuple of (detection_mask, threshold_array)
        """
        # Convert to linear if needed
        if db_input:
            signal_linear = 10 ** (signal / 10)
        else:
            signal_linear = signal
        
        # Ensure positive values
        signal_linear = np.maximum(signal_linear, 1e-30)
        
        # Select CFAR algorithm
        if self.cfar_type == CFARType.CA:
            detections, thresholds = _ca_cfar_1d_jit(
                signal_linear, self.guard_cells, self.reference_cells, self.pfa
            )
        elif self.cfar_type == CFARType.GO:
            detections, thresholds = _go_cfar_1d_jit(
                signal_linear, self.guard_cells, self.reference_cells, self.pfa
            )
        elif self.cfar_type == CFARType.SO:
            detections, thresholds = _so_cfar_1d_jit(
                signal_linear, self.guard_cells, self.reference_cells, self.pfa
            )
        else:
            # Default to CA
            detections, thresholds = _ca_cfar_1d_jit(
                signal_linear, self.guard_cells, self.reference_cells, self.pfa
            )
        
        return detections, thresholds
    
    def detect_2d(
        self,
        rd_map: np.ndarray,
        db_input: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply 2D CFAR detection to Range-Doppler map.
        
        Uses separable 1D CFAR (faster than true 2D).
        
        Args:
            rd_map: 2D Range-Doppler map [doppler x range]
            db_input: True if input is in dB
            
        Returns:
            Tuple of (2D detection mask, 2D threshold array)
        """
        n_doppler, n_range = rd_map.shape
        
        # Convert to linear
        if db_input:
            rd_linear = 10 ** (rd_map / 10)
        else:
            rd_linear = rd_map
        
        # Apply CFAR along range dimension first
        range_detections = np.zeros_like(rd_map, dtype=bool)
        range_thresholds = np.zeros_like(rd_map)
        
        for d in range(n_doppler):
            det, thresh = self.detect(rd_linear[d, :], db_input=False)
            range_detections[d, :] = det
            range_thresholds[d, :] = thresh
        
        # Apply CFAR along Doppler dimension
        doppler_detections = np.zeros_like(rd_map, dtype=bool)
        
        for r in range(n_range):
            det, _ = self.detect(rd_linear[:, r], db_input=False)
            doppler_detections[:, r] = det
        
        # Combined detection: require detection in both dimensions
        combined_detections = range_detections & doppler_detections
        
        return combined_detections, range_thresholds
    
    @staticmethod
    def calculate_cfar_loss(
        n_ref_cells: int,
        pfa: float
    ) -> float:
        """
        Calculate CFAR detection loss compared to fixed threshold.
        
        Loss = 10 * log10(α/N) where α is the CFAR multiplier
        
        Args:
            n_ref_cells: Total number of reference cells
            pfa: Probability of false alarm
            
        Returns:
            CFAR loss in dB
            
        Reference: Skolnik, Eq. 6.8
        """
        alpha = n_ref_cells * (pfa ** (-1.0 / n_ref_cells) - 1)
        
        # Loss relative to ideal Neyman-Pearson detector
        # For exponential noise: ~1 dB for N=16, Pfa=1e-6
        loss_db = 10 * np.log10(alpha / n_ref_cells)
        
        return loss_db
