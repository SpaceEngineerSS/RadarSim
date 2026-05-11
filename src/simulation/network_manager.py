"""
Multi-Radar Network Manager

Manages data exchange, track fusion, and jammer triangulation across
multiple radar systems in a distributed sensor network.

Algorithms:
    - Covariance Intersection (CI): Julier & Uhlmann (1997)
    - Strobe Triangulation: Least Squares bearing intersection
    - Track-to-Track Association: Euclidean gating
    - Latency Model: Configurable FIFO delay (Link-16 simulation)

Architecture:
    RadarNode_1 ─┐
    RadarNode_2 ─┤─→ NetworkManager ─→ FusedTrackList ─→ CTP Display
    RadarNode_N ─┘

References:
    - Julier, S. & Uhlmann, J. "A Non-divergent Estimation Algorithm in the
      Presence of Unknown Correlations", ACC, 1997
    - Poisel, R. "Electronic Warfare Target Location Methods", Artech House, 2012
    - Blackman, S. "Multiple-Target Tracking with Radar Applications", 1986
    - Developed by Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class NetworkTrack:
    """
    Track data exchanged between radar nodes.

    Attributes:
        track_id: Unique track identifier (node_id:local_id)
        node_id: Source radar node
        state: State vector [x, y, vx, vy]
        covariance: 4×4 covariance matrix P
        timestamp: Time of last update [s]
        snr_db: SNR at source radar [dB]
    """

    track_id: str
    node_id: str
    state: np.ndarray  # [x, y, vx, vy]
    covariance: np.ndarray  # 4×4 P matrix
    timestamp: float
    snr_db: float = 20.0


@dataclass
class StrobeReport:
    """
    Noise strobe (AOA-only) report from a radar detecting a jammer.

    Attributes:
        node_id: Source radar node
        radar_position: Radar (x, y) position [m]
        bearing_rad: Bearing to jammer [rad]
        timestamp: Time of measurement [s]
        jsr_db: Jam-to-signal ratio [dB]
    """

    node_id: str
    radar_position: np.ndarray  # [x, y]
    bearing_rad: float
    timestamp: float
    jsr_db: float = 20.0


@dataclass
class FusedTrack:
    """
    Track produced by Covariance Intersection fusion.

    Attributes:
        fused_id: Unique fused track identifier
        state: Fused state vector [x, y, vx, vy]
        covariance: Fused 4×4 covariance matrix
        source_nodes: List of contributing radar node IDs
        fusion_gain_db: Covariance reduction vs best single source [dB]
        timestamp: Fusion timestamp [s]
    """

    fused_id: int
    state: np.ndarray
    covariance: np.ndarray
    source_nodes: List[str]
    fusion_gain_db: float = 0.0
    timestamp: float = 0.0


@dataclass
class RadarNode:
    """
    A radar node in the sensor network.

    Attributes:
        node_id: Unique identifier (e.g., "RADAR_01")
        position_xy: Radar position [x, y] in meters
        tracks: Current local track list
        strobes: Current AOA-only jammer strobe reports
        lat_lon: Geographic coordinates (optional)
    """

    node_id: str
    position_xy: np.ndarray
    tracks: List[NetworkTrack] = field(default_factory=list)
    strobes: List[StrobeReport] = field(default_factory=list)
    lat_lon: Optional[Tuple[float, float]] = None


# ═══════════════════════════════════════════════════════════════════════
# LATENCY MODEL (Link-16 / JTIDS Simulation)
# ═══════════════════════════════════════════════════════════════════════


class LatencyModel:
    """
    Configurable delay model for tactical data link simulation.

    Simulates real-world communication latency:
        - Link-16: ~12s update cycle
        - JTIDS:   ~1-3s update cycle
        - Direct:  ~50-200ms

    Uses a FIFO queue with time-delayed release.

    Reference: MIL-STD-6016 (Link-16)
    """

    def __init__(self, delay_ms: float = 100.0) -> None:
        """
        Initialize latency model.

        Args:
            delay_ms: Communication delay [ms]
        """
        self.delay_s = delay_ms / 1000.0
        self._queue: deque = deque()

    def enqueue(self, data: object, timestamp: float) -> None:
        """
        Enqueue data with timestamp.

        Args:
            data: Track/strobe data to delay
            timestamp: Current simulation time [s]
        """
        self._queue.append((data, timestamp))

    def dequeue(self, current_time: float) -> List[object]:
        """
        Release data that has exceeded the delay.

        Args:
            current_time: Current simulation time [s]

        Returns:
            List of data items ready for consumption
        """
        ready = []
        while self._queue and (current_time - self._queue[0][1]) >= self.delay_s:
            data, _ = self._queue.popleft()
            ready.append(data)
        return ready

    @property
    def pending_count(self) -> int:
        """Number of items still in the delay queue."""
        return len(self._queue)


# ═══════════════════════════════════════════════════════════════════════
# COVARIANCE INTERSECTION (CI)
# ═══════════════════════════════════════════════════════════════════════


class CovarianceIntersection:
    """
    Covariance Intersection fusion algorithm.

    Provides CONSISTENT track fusion when cross-correlations between
    estimates are unknown. Standard Kalman fusion assumes P_cross = 0,
    which leads to overconfident (divergent) estimates.

    CI guarantees:
        tr(P_fused) ≤ min(tr(P₁), tr(P₂))

    Algorithm:
        P_fused = (ω·P₁⁻¹ + (1-ω)·P₂⁻¹)⁻¹
        x_fused = P_fused · (ω·P₁⁻¹·x₁ + (1-ω)·P₂⁻¹·x₂)

        where ω ∈ [0, 1] minimizes tr(P_fused)

    Reference: Julier & Uhlmann (1997)
    """

    @staticmethod
    def fuse_two(
        x1: np.ndarray,
        P1: np.ndarray,
        x2: np.ndarray,
        P2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fuse two estimates using Covariance Intersection.

        Args:
            x1: State estimate 1 [n]
            P1: Covariance 1 [n×n]
            x2: State estimate 2 [n]
            P2: Covariance 2 [n×n]

        Returns:
            (x_fused, P_fused, omega): Fused state, covariance, optimal weight

        Reference: Julier & Uhlmann (1997), Eq. 3-5
        """
        P1_inv = np.linalg.inv(P1)
        P2_inv = np.linalg.inv(P2)

        def _trace_objective(omega: float) -> float:
            """Minimize tr(P_fused) over omega."""
            P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
            P_fused = np.linalg.inv(P_fused_inv)
            return np.trace(P_fused)

        # Optimize omega ∈ [0, 1]
        result = minimize_scalar(
            _trace_objective,
            bounds=(0.001, 0.999),
            method="bounded",
        )
        omega = result.x

        # Compute fused estimate
        P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv
        P_fused = np.linalg.inv(P_fused_inv)
        x_fused = P_fused @ (omega * P1_inv @ x1 + (1 - omega) * P2_inv @ x2)

        # Ensure symmetry
        P_fused = 0.5 * (P_fused + P_fused.T)

        return x_fused, P_fused, omega

    @staticmethod
    def fuse_multiple(
        states: List[np.ndarray],
        covariances: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fuse N estimates using sequential pairwise CI.

        Args:
            states: List of state vectors
            covariances: List of covariance matrices

        Returns:
            (x_fused, P_fused): Fused state and covariance
        """
        if len(states) == 0:
            raise ValueError("Need at least one estimate to fuse")
        if len(states) == 1:
            return states[0].copy(), covariances[0].copy()

        x_fused = states[0].copy()
        P_fused = covariances[0].copy()

        for i in range(1, len(states)):
            x_fused, P_fused, _ = CovarianceIntersection.fuse_two(
                x_fused, P_fused, states[i], covariances[i]
            )

        return x_fused, P_fused

    @staticmethod
    def fusion_gain_db(P_fused: np.ndarray, P_best: np.ndarray) -> float:
        """
        Calculate fusion gain in dB.

        Gain = 10·log₁₀(tr(P_best) / tr(P_fused))

        Args:
            P_fused: Fused covariance
            P_best: Best single-source covariance

        Returns:
            Fusion gain [dB] (positive = improvement)
        """
        tr_fused = np.trace(P_fused)
        tr_best = np.trace(P_best)
        if tr_fused <= 0 or tr_best <= 0:
            return 0.0
        return 10.0 * np.log10(tr_best / tr_fused)


# ═══════════════════════════════════════════════════════════════════════
# STROBE TRIANGULATION (Jammer Localization)
# ═══════════════════════════════════════════════════════════════════════


class StrobeTriangulator:
    """
    Jammer localization via AOA (Angle of Arrival) triangulation.

    When a jammer is active, radars can measure bearing (AOA) but not
    range. With 2+ radars, the jammer position can be estimated by
    intersecting the bearing lines.

    Algorithm (Least Squares):
        For each radar i at (x_i, y_i) measuring bearing θ_i:

        Line equation: sin(θ_i)·x - cos(θ_i)·y = x_i·sin(θ_i) - y_i·cos(θ_i)

        System: A·p = b
        Solution: p = (A^T·A)⁻¹·A^T·b

    For 3+ bearings, the system is overdetermined → LS gives robust estimate.

    Reference: Poisel, "EW Target Location Methods", 2012, Ch. 3
    """

    @staticmethod
    def triangulate(strobes: List[StrobeReport]) -> Optional[Tuple[np.ndarray, float]]:
        """
        Triangulate jammer position from multiple bearing strobes.

        Args:
            strobes: List of StrobeReport (≥2 required)

        Returns:
            ((x, y), residual_m) or None if insufficient data

        Reference: Poisel (2012), Eq. 3.14
        """
        if len(strobes) < 2:
            return None

        n = len(strobes)
        A = np.zeros((n, 2))
        b = np.zeros(n)

        for i, strobe in enumerate(strobes):
            sin_t = np.sin(strobe.bearing_rad)
            cos_t = np.cos(strobe.bearing_rad)
            xi, yi = strobe.radar_position[0], strobe.radar_position[1]

            A[i, 0] = sin_t
            A[i, 1] = -cos_t
            b[i] = xi * sin_t - yi * cos_t

        # Least Squares: p = (A^T·A)⁻¹·A^T·b
        ATA = A.T @ A
        det = np.linalg.det(ATA)
        if abs(det) < 1e-12:
            return None  # Degenerate geometry (parallel bearings)

        p = np.linalg.solve(ATA, A.T @ b)

        # Residual (RMS distance from bearing lines)
        residuals = A @ p - b
        rms_residual = np.sqrt(np.mean(residuals**2))

        return p, rms_residual

    @staticmethod
    def gdop(
        radar_positions: List[np.ndarray],
        target_position: np.ndarray,
    ) -> float:
        """
        Calculate Geometric Dilution of Precision (GDOP).

        GDOP quantifies how radar geometry affects triangulation accuracy.
        Lower GDOP = better geometry.

        GDOP ∝ 1/sin(Δθ) for 2 radars.

        Args:
            radar_positions: List of radar [x, y] positions
            target_position: Target [x, y] position

        Returns:
            GDOP factor (1.0 = ideal, >5 = poor)

        Reference: Blackman (1986), Ch. 8
        """
        if len(radar_positions) < 2:
            return float("inf")

        # Compute bearings from each radar to target
        bearings = []
        for pos in radar_positions:
            dx = target_position[0] - pos[0]
            dy = target_position[1] - pos[1]
            bearings.append(np.arctan2(dy, dx))

        # GDOP from angular separation matrix
        n = len(bearings)
        H = np.zeros((n, 2))
        for i, theta in enumerate(bearings):
            H[i, 0] = np.cos(theta)
            H[i, 1] = np.sin(theta)

        HTH = H.T @ H
        det = np.linalg.det(HTH)
        if det < 1e-12:
            return float("inf")

        G = np.linalg.inv(HTH)
        gdop = np.sqrt(np.trace(G))
        return float(gdop)


# ═══════════════════════════════════════════════════════════════════════
# TRACK-TO-TRACK ASSOCIATION
# ═══════════════════════════════════════════════════════════════════════


class TrackAssociator:
    """
    Track-to-Track Association (T2TA) for multi-radar fusion.

    Associates tracks from different radars that correspond to the
    same physical target using Euclidean distance gating.

    Args:
        gate_distance_m: Maximum association distance [m]

    Reference: Blackman (1986), Ch. 6
    """

    def __init__(self, gate_distance_m: float = 1000.0) -> None:
        self.gate_distance_m = gate_distance_m

    def associate(
        self,
        tracks_a: List[NetworkTrack],
        tracks_b: List[NetworkTrack],
    ) -> List[Tuple[NetworkTrack, NetworkTrack]]:
        """
        Associate tracks between two radar nodes.

        Uses greedy nearest-neighbor matching with distance gating.

        Args:
            tracks_a: Tracks from radar A
            tracks_b: Tracks from radar B

        Returns:
            List of matched (track_a, track_b) pairs
        """
        if not tracks_a or not tracks_b:
            return []

        pairs = []
        used_b = set()

        for ta in tracks_a:
            best_dist = self.gate_distance_m
            best_tb = None
            best_idx = -1

            for idx, tb in enumerate(tracks_b):
                if idx in used_b:
                    continue

                dist = np.linalg.norm(ta.state[:2] - tb.state[:2])
                if dist < best_dist:
                    best_dist = dist
                    best_tb = tb
                    best_idx = idx

            if best_tb is not None:
                pairs.append((ta, best_tb))
                used_b.add(best_idx)

        return pairs


# ═══════════════════════════════════════════════════════════════════════
# NETWORK MANAGER
# ═══════════════════════════════════════════════════════════════════════


class NetworkManager:
    """
    Multi-Radar Network Manager.

    Orchestrates data exchange, track fusion, and jammer triangulation
    across multiple radar systems.

    Features:
        1. Radar node registration with geographic/Cartesian positions
        2. Track exchange with configurable Link-16 latency
        3. Track-to-Track Association (T2TA)
        4. Covariance Intersection fusion
        5. Strobe triangulation for jammer localization
        6. Common Tactical Picture (CTP) generation

    Example:
        >>> nm = NetworkManager(link_delay_ms=100)
        >>> nm.register_node("R1", position_xy=np.array([0, 0]))
        >>> nm.register_node("R2", position_xy=np.array([50000, 0]))
        >>> nm.submit_tracks("R1", [track1])
        >>> nm.submit_tracks("R2", [track2])
        >>> fused = nm.fuse(current_time=1.0)

    Reference: Blackman (1986); Julier & Uhlmann (1997)
    """

    def __init__(
        self,
        link_delay_ms: float = 100.0,
        association_gate_m: float = 1000.0,
    ) -> None:
        """
        Initialize network manager.

        Args:
            link_delay_ms: Communication delay [ms]
            association_gate_m: Track association gate [m]
        """
        self.nodes: Dict[str, RadarNode] = {}
        self.latency = LatencyModel(delay_ms=link_delay_ms)
        self.associator = TrackAssociator(gate_distance_m=association_gate_m)
        self.ci = CovarianceIntersection()
        self.triangulator = StrobeTriangulator()

        self._fused_tracks: List[FusedTrack] = []
        self._jammer_positions: List[Tuple[np.ndarray, float]] = []
        self._fused_id_counter = 0

    def register_node(
        self,
        node_id: str,
        position_xy: np.ndarray,
        lat_lon: Optional[Tuple[float, float]] = None,
    ) -> RadarNode:
        """
        Register a radar node in the network.

        Args:
            node_id: Unique node identifier
            position_xy: Position [x, y] in meters
            lat_lon: Geographic coordinates (optional)

        Returns:
            Registered RadarNode
        """
        node = RadarNode(
            node_id=node_id,
            position_xy=np.asarray(position_xy, dtype=np.float64),
            lat_lon=lat_lon,
        )
        self.nodes[node_id] = node
        return node

    def submit_tracks(
        self,
        node_id: str,
        tracks: List[NetworkTrack],
        current_time: float = 0.0,
    ) -> None:
        """
        Submit track data from a radar node (enters latency queue).

        Args:
            node_id: Source node ID
            tracks: Track list from this node
            current_time: Current simulation time [s]
        """
        if node_id in self.nodes:
            self.nodes[node_id].tracks = tracks
            self.latency.enqueue(
                {"node_id": node_id, "tracks": tracks, "type": "tracks"},
                current_time,
            )

    def submit_strobes(
        self,
        node_id: str,
        strobes: List[StrobeReport],
        current_time: float = 0.0,
    ) -> None:
        """
        Submit jammer strobe reports from a radar node.

        Args:
            node_id: Source node ID
            strobes: Strobe reports
            current_time: Current simulation time [s]
        """
        if node_id in self.nodes:
            self.nodes[node_id].strobes = strobes
            self.latency.enqueue(
                {"node_id": node_id, "strobes": strobes, "type": "strobes"},
                current_time,
            )

    def fuse(self, current_time: float) -> List[FusedTrack]:
        """
        Execute fusion cycle: process delayed data, associate, fuse.

        Args:
            current_time: Current simulation time [s]

        Returns:
            List of fused tracks (Common Tactical Picture)
        """
        # Release delayed data
        self.latency.dequeue(current_time)

        # Collect all tracks from all nodes
        all_node_tracks: Dict[str, List[NetworkTrack]] = {}
        for node_id, node in self.nodes.items():
            if node.tracks:
                all_node_tracks[node_id] = node.tracks

        # Pairwise association and fusion
        node_ids = list(all_node_tracks.keys())
        self._fused_tracks = []

        if len(node_ids) < 2:
            # Single radar — pass through as "fused"
            for nid in node_ids:
                for trk in all_node_tracks[nid]:
                    self._fused_id_counter += 1
                    self._fused_tracks.append(
                        FusedTrack(
                            fused_id=self._fused_id_counter,
                            state=trk.state.copy(),
                            covariance=trk.covariance.copy(),
                            source_nodes=[nid],
                            fusion_gain_db=0.0,
                            timestamp=current_time,
                        )
                    )
            return self._fused_tracks

        # Multi-radar CI fusion
        # Start with first node's tracks as base
        base_tracks = all_node_tracks[node_ids[0]]

        for trk in base_tracks:
            states = [trk.state.copy()]
            covs = [trk.covariance.copy()]
            sources = [node_ids[0]]

            # Find matches in other nodes
            for other_nid in node_ids[1:]:
                other_tracks = all_node_tracks[other_nid]
                matched = self.associator.associate([trk], other_tracks)
                if matched:
                    _, match_trk = matched[0]
                    states.append(match_trk.state.copy())
                    covs.append(match_trk.covariance.copy())
                    sources.append(other_nid)

            # Fuse
            x_fused, P_fused = self.ci.fuse_multiple(states, covs)

            # Compute fusion gain
            best_trace = min(np.trace(c) for c in covs)
            gain = self.ci.fusion_gain_db(P_fused, covs[0])

            self._fused_id_counter += 1
            self._fused_tracks.append(
                FusedTrack(
                    fused_id=self._fused_id_counter,
                    state=x_fused,
                    covariance=P_fused,
                    source_nodes=sources,
                    fusion_gain_db=gain,
                    timestamp=current_time,
                )
            )

        return self._fused_tracks

    def triangulate_jammers(self) -> List[Tuple[np.ndarray, float]]:
        """
        Triangulate jammer positions from all strobe reports.

        Returns:
            List of (position_xy, residual_m) for each jammer
        """
        all_strobes: List[StrobeReport] = []
        for node in self.nodes.values():
            all_strobes.extend(node.strobes)

        if len(all_strobes) < 2:
            self._jammer_positions = []
            return []

        result = self.triangulator.triangulate(all_strobes)
        if result is not None:
            self._jammer_positions = [result]
        else:
            self._jammer_positions = []

        return self._jammer_positions

    @property
    def fused_tracks(self) -> List[FusedTrack]:
        """Get current fused track list."""
        return self._fused_tracks

    @property
    def jammer_positions(self) -> List[Tuple[np.ndarray, float]]:
        """Get triangulated jammer positions."""
        return self._jammer_positions

    def get_status(self) -> dict:
        """Get network status for UI."""
        return {
            "n_nodes": len(self.nodes),
            "node_ids": list(self.nodes.keys()),
            "n_fused_tracks": len(self._fused_tracks),
            "n_jammer_locs": len(self._jammer_positions),
            "link_delay_ms": self.latency.delay_s * 1000,
            "pending_messages": self.latency.pending_count,
        }


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════


def validate_network_fusion() -> dict:
    """
    Self-validation of network fusion algorithms.

    Reference: Julier & Uhlmann (1997); Poisel (2012)
    """
    results = {}

    # Test 1: CI — fused covariance ≤ min(P1, P2) by trace
    P1 = np.diag([100.0, 200.0, 10.0, 20.0])
    P2 = np.diag([150.0, 100.0, 15.0, 10.0])
    x1 = np.array([1000.0, 2000.0, 50.0, 30.0])
    x2 = np.array([1010.0, 1990.0, 48.0, 32.0])

    x_f, P_f, omega = CovarianceIntersection.fuse_two(x1, P1, x2, P2)
    tr_fused = np.trace(P_f)
    tr_min = min(np.trace(P1), np.trace(P2))

    results["ci_trace_reduction"] = {
        "tr_P1": np.trace(P1),
        "tr_P2": np.trace(P2),
        "tr_fused": tr_fused,
        "omega": omega,
        "pass": tr_fused <= tr_min + 1e-6,
    }

    # Test 2: Triangulation
    jammer_true = np.array([25000.0, 15000.0])
    strobes = []
    radar_positions = [
        np.array([0.0, 0.0]),
        np.array([50000.0, 0.0]),
        np.array([25000.0, 40000.0]),
    ]
    for pos in radar_positions:
        bearing = np.arctan2(jammer_true[1] - pos[1], jammer_true[0] - pos[0])
        strobes.append(
            StrobeReport(
                node_id="test",
                radar_position=pos,
                bearing_rad=bearing,
                timestamp=0.0,
            )
        )

    tri_result = StrobeTriangulator.triangulate(strobes)
    if tri_result:
        pos_est, residual = tri_result
        error = np.linalg.norm(pos_est - jammer_true)
        rel_error = error / np.linalg.norm(jammer_true)
        results["triangulation"] = {
            "error_m": error,
            "relative_error": rel_error,
            "pass": rel_error < 0.01,  # <1% for perfect bearings
        }

    return results


if __name__ == "__main__":
    results = validate_network_fusion()
    print("=" * 60)
    print("Network Fusion Validation")
    print("=" * 60)
    for test_name, test_result in results.items():
        status = "✓ PASS" if test_result.get("pass", False) else "✗ FAIL"
        print(f"\n{status} | {test_name}")
        for k, v in test_result.items():
            if k != "pass":
                print(f"    {k}: {v}")
