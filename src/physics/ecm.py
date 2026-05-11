"""
Electronic Countermeasures (ECM) Simulation Module

Implements noise jamming, deception jamming, chaff, and decoy effects
for realistic radar electronic warfare simulation.

References:
    - Schleher, "Electronic Warfare in the Information Age", Artech House, 1999
    - Skolnik, "Radar Handbook", 3rd Ed., Chapter 24
    - Van Brunt, "Applied ECM", Vol. 1-3, EW Engineering, 1978
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numba
import numpy as np


class ECMType(Enum):
    """Electronic countermeasure types."""

    NOISE_BARRAGE = "noise_barrage"  # Broadband noise
    NOISE_SPOT = "noise_spot"  # Narrowband noise
    DRFM = "drfm"  # Digital RF memory repeater
    RANGE_GATE_PULL = "range_gate_pull"  # Range deception
    VELOCITY_GATE_PULL = "vgp"  # Velocity deception
    CHAFF = "chaff"  # Passive reflectors
    DECOY = "decoy"  # Active/passive decoys


@dataclass
class ECMSource:
    """
    ECM emitter configuration.

    Attributes:
        position: [x, y, z] position [m]
        power_watts: Effective radiated power [W]
        bandwidth_hz: Jamming bandwidth [Hz]
        ecm_type: Type of ECM
        active: Whether currently active
        start_time: Activation time [s]
    """

    position: np.ndarray
    power_watts: float = 100.0
    bandwidth_hz: float = 100e6
    ecm_type: ECMType = ECMType.NOISE_BARRAGE
    active: bool = False
    start_time: float = 0.0

    # DRFM parameters
    drfm_delay_s: float = 0.0
    drfm_range_pull_rate: float = 0.0  # m/s

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64)


@dataclass
class ChaffCloud:
    """
    Chaff cloud model.

    Chaff consists of thin metallic dipoles that create radar returns.
    RCS depends on number of dipoles and radar wavelength.

    Reference: Skolnik, Chapter 24.4
    """

    position: np.ndarray
    velocity: np.ndarray
    rcs_m2: float = 10.0
    deploy_time: float = 0.0
    dispersion_rate: float = 5.0  # m/s radial expansion
    lifetime_s: float = 60.0  # Typical chaff bloom time

    def update(self, dt: float, current_time: float) -> bool:
        """
        Update chaff cloud state.

        Returns:
            True if chaff is still active, False if expired
        """
        # Move with wind/inertia
        self.position = self.position + self.velocity * dt

        # Chaff falls slowly
        self.velocity[2] -= 0.5 * dt  # Simplified drag

        # RCS decreases as chaff disperses
        age = current_time - self.deploy_time
        dispersion_factor = np.exp(-age / (self.lifetime_s / 2))
        self.rcs_m2 *= dispersion_factor**dt

        return age < self.lifetime_s and self.rcs_m2 > 0.1


@numba.jit(nopython=True, cache=True)
def _calculate_jsr_jit(
    jammer_power: float,
    jammer_gain: float,
    radar_power: float,
    radar_gain: float,
    rcs: float,
    jammer_range: float,
    target_range: float,
    wavelength: float,
    bandwidth_ratio: float,
) -> float:
    """
    JIT-compiled Jam-to-Signal Ratio calculation.

    JSR = (Pj * Gj * 4π * R_t^4) / (Pt * Gt * σ * R_j^2) * (Bj/Br)

    Args:
        jammer_power: Jammer ERP [W]
        jammer_gain: Jammer antenna gain [linear]
        radar_power: Radar peak power [W]
        radar_gain: Radar antenna gain [linear]
        rcs: Target RCS [m²]
        jammer_range: Jammer to radar range [m]
        target_range: Target to radar range [m]
        wavelength: Radar wavelength [m]
        bandwidth_ratio: Bj/Br ratio

    Returns:
        JSR [linear]

    Reference: Schleher, Eq. 4.1
    """
    if target_range < 1 or jammer_range < 1:
        return 0.0

    # Signal power (from radar equation)
    signal_power = (radar_power * radar_gain**2 * wavelength**2 * rcs) / (
        (4 * np.pi) ** 3 * target_range**4
    )

    # Jam power at radar receiver
    jam_power = (jammer_power * jammer_gain) / (4 * np.pi * jammer_range**2)

    # Apply bandwidth ratio (Bj/Br)
    jam_power *= min(1.0, bandwidth_ratio)

    if signal_power > 0:
        return jam_power / signal_power
    return 1e10  # Very high JSR if no signal


class ECMSimulator:
    """
    Electronic Countermeasures Simulator

    Simulates various ECM effects on radar systems including:
    - Noise jamming (barrage and spot)
    - DRFM-based deception
    - Chaff and decoys

    Reference: Schleher, "Electronic Warfare in the Information Age"
    """

    def __init__(self, radar_wavelength: float = 0.03):
        """
        Initialize ECM simulator.

        Args:
            radar_wavelength: Radar operating wavelength [m]
        """
        self.wavelength = radar_wavelength
        self.ecm_sources: List[ECMSource] = []
        self.chaff_clouds: List[ChaffCloud] = []
        self.jamming_active = False
        self.jamming_power = 0.0

    def add_jammer(
        self,
        position: np.ndarray,
        power_watts: float = 100.0,
        ecm_type: ECMType = ECMType.NOISE_BARRAGE,
        bandwidth_hz: float = 100e6,
    ) -> ECMSource:
        """Add an ECM source to the simulation."""
        source = ECMSource(
            position=position,
            power_watts=power_watts,
            bandwidth_hz=bandwidth_hz,
            ecm_type=ecm_type,
            active=False,
        )
        self.ecm_sources.append(source)
        return source

    def activate_jamming(self, current_time: float = 0.0) -> None:
        """Activate all ECM sources."""
        self.jamming_active = True
        for source in self.ecm_sources:
            source.active = True
            source.start_time = current_time

    def deactivate_jamming(self) -> None:
        """Deactivate all ECM sources."""
        self.jamming_active = False
        for source in self.ecm_sources:
            source.active = False

    def deploy_chaff(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        rcs_m2: float = 10.0,
        current_time: float = 0.0,
    ) -> ChaffCloud:
        """Deploy a chaff cloud."""
        cloud = ChaffCloud(
            position=np.asarray(position, dtype=np.float64),
            velocity=np.asarray(velocity, dtype=np.float64),
            rcs_m2=rcs_m2,
            deploy_time=current_time,
        )
        self.chaff_clouds.append(cloud)
        return cloud

    def calculate_jsr(
        self,
        radar_pos: np.ndarray,
        target_pos: np.ndarray,
        jammer_pos: np.ndarray,
        radar_power: float,
        radar_gain: float,
        jammer_power: float,
        target_rcs: float,
        radar_bandwidth: float = 1e6,
    ) -> float:
        """
        Calculate Jam-to-Signal Ratio.

        Args:
            radar_pos: Radar position [m]
            target_pos: Target position [m]
            jammer_pos: Jammer position [m]
            radar_power: Radar peak power [W]
            radar_gain: Radar antenna gain [linear]
            jammer_power: Jammer ERP [W]
            target_rcs: Target RCS [m²]
            radar_bandwidth: Radar receiver bandwidth [Hz]

        Returns:
            JSR in dB
        """
        jammer_range = np.linalg.norm(jammer_pos - radar_pos)
        target_range = np.linalg.norm(target_pos - radar_pos)

        # Assume jammer has omnidirectional antenna (gain = 1)
        jammer_gain = 1.0

        # Bandwidth ratio (assuming barrage jammer covers radar band)
        bw_ratio = radar_bandwidth / 100e6  # 100 MHz barrage

        jsr_linear = _calculate_jsr_jit(
            jammer_power,
            jammer_gain,
            radar_power,
            radar_gain,
            target_rcs,
            jammer_range,
            target_range,
            self.wavelength,
            bw_ratio,
        )

        return 10 * np.log10(max(jsr_linear, 1e-10))

    def generate_noise_strobes(
        self, n_range_bins: int, n_azimuth_bins: int, jammer_azimuth_deg: float, jsr_db: float
    ) -> np.ndarray:
        """
        Generate noise jamming strobes for display.

        Creates radial strobe pattern from jammer direction.
        OPTIMIZED: Limits array size and uses efficient generation.

        Args:
            n_range_bins: Number of range bins (capped at 100)
            n_azimuth_bins: Number of azimuth bins (capped at 72)
            jammer_azimuth_deg: Jammer bearing from radar
            jsr_db: Jam-to-Signal ratio [dB]

        Returns:
            2D array of jamming returns [range x azimuth]
        """
        # ═══ PERFORMANCE: Cap array size ═══
        n_range_bins = min(n_range_bins, 100)
        n_azimuth_bins = min(n_azimuth_bins, 72)

        strobes = np.zeros((n_range_bins, n_azimuth_bins), dtype=np.float32)

        if jsr_db < 0:
            return strobes  # No visible jamming

        # Calculate strobe width based on jammer power (cap at 30 deg)
        strobe_width = min(30, int(5 + jsr_db / 3))  # degrees

        # Azimuth bin of jammer
        jammer_bin = int((jammer_azimuth_deg % 360) / 360 * n_azimuth_bins)

        # Create strobe pattern - OPTIMIZED: vectorized
        intensity = min(1.0, jsr_db / 30)
        half_width = strobe_width * n_azimuth_bins // 720  # Convert degrees to bins

        # Pre-generate noise for affected bins only (not full array)
        for i in range(-half_width, half_width + 1):
            az_bin = (jammer_bin + i) % n_azimuth_bins
            # Use pre-generated uniform noise instead of exponential (faster)
            strobes[:, az_bin] = intensity * (0.25 + 0.5 * np.random.random(n_range_bins))

        return strobes

    def apply_jamming_to_signal(
        self, signal_db: np.ndarray, jsr_db: float, noise_floor_db: float = -60.0
    ) -> np.ndarray:
        """
        Apply jamming effects to received signal.

        Args:
            signal_db: Original signal [dB]
            jsr_db: Jam-to-Signal ratio [dB]
            noise_floor_db: Receiver noise floor [dB]

        Returns:
            Jammed signal [dB]
        """
        if jsr_db < 0:
            return signal_db

        # Elevated noise floor due to jamming
        jammed_floor = noise_floor_db + max(0, jsr_db - 10)

        # Add noise proportional to JSR
        noise_std = min(10, jsr_db / 3)
        jamming_noise = np.random.normal(0, noise_std, len(signal_db))

        # Mask signal with elevated noise
        jammed = np.maximum(signal_db, jammed_floor + jamming_noise)

        return jammed

    def update(self, dt: float, current_time: float) -> None:
        """Update all ECM elements."""
        # Update chaff clouds
        active_chaff = []
        for cloud in self.chaff_clouds:
            if cloud.update(dt, current_time):
                active_chaff.append(cloud)
        self.chaff_clouds = active_chaff

    def get_total_jamming_power(self) -> float:
        """Get total active jamming power."""
        total = 0.0
        for source in self.ecm_sources:
            if source.active:
                total += source.power_watts
        return total

    def get_chaff_returns(self) -> List[Tuple[np.ndarray, float]]:
        """Get all chaff cloud positions and RCS values."""
        return [(cloud.position.copy(), cloud.rcs_m2) for cloud in self.chaff_clouds]

    def calculate_burn_through_range(
        self,
        radar_power_watts: float,
        radar_gain_linear: float,
        jammer_power_watts: float,
        jammer_gain_linear: float = 1.0,
        target_rcs_m2: float = 1.0,
        required_snr_linear: float = 20.0,
        radar_bandwidth_hz: float = 1e6,
        jammer_bandwidth_hz: float = 100e6,
    ) -> float:
        """
        Calculate burn-through range.

        Burn-through range is where radar can detect target despite jamming.

        Reference: Schleher (1999), Eq. 4.5

        R_bt = sqrt((Pt * Gt * σ) / (Pj * Gj * 4π * SNR_req * Bj/Br))

        Args:
            radar_power_watts: Radar peak power [W]
            radar_gain_linear: Radar antenna gain [linear]
            jammer_power_watts: Jammer ERP [W]
            jammer_gain_linear: Jammer antenna gain [linear]
            target_rcs_m2: Target RCS [m²]
            required_snr_linear: Required SNR for detection [linear]
            radar_bandwidth_hz: Radar receiver bandwidth [Hz]
            jammer_bandwidth_hz: Jammer bandwidth [Hz]

        Returns:
            Burn-through range [m]
        """
        # Bandwidth ratio (barrage jammer spreads power)
        bw_ratio = min(1.0, radar_bandwidth_hz / jammer_bandwidth_hz)

        numerator = radar_power_watts * radar_gain_linear * target_rcs_m2
        denominator = (
            jammer_power_watts * jammer_gain_linear * 4 * np.pi * required_snr_linear * bw_ratio
        )

        if denominator > 1e-30:
            r_bt_squared = numerator / denominator
            return np.sqrt(r_bt_squared)
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# PHASE 28: DRFM JAMMER (Digital Radio Frequency Memory)
# ═══════════════════════════════════════════════════════════════════════

C_LIGHT = 299_792_458.0


class DRFMState(Enum):
    """DRFM jammer operational states."""

    IDLE = "idle"  # Not active
    CAPTURE = "capture"  # Capturing radar's range gate
    PULL = "pull"  # Pulling range/velocity gate away
    RELEASE = "release"  # Releasing — radar loses track


@dataclass
class DRFMConfig:
    """
    DRFM Jammer configuration.

    Attributes:
        power_watts: Jammer effective radiated power [W]
        gain_over_skin_db: J/S advantage over skin return [dB]
        pull_rate_mps: Range gate pull rate [m/s]
        max_pull_m: Maximum pull distance [m]
        capture_dwell_s: Time to establish gate capture [s]
        mode: 'rgpo' for range pull, 'vgpo' for velocity pull
        vgpo_accel_hz_per_s: Velocity gate pull acceleration [Hz/s]

    Reference: Schleher (1999), Ch. 7
    """

    power_watts: float = 500.0
    gain_over_skin_db: float = 10.0
    pull_rate_mps: float = 100.0
    max_pull_m: float = 5000.0
    capture_dwell_s: float = 2.0
    mode: str = "rgpo"  # "rgpo" or "vgpo"
    vgpo_accel_hz_per_s: float = 50.0


class DRFMJammer:
    """
    Digital Radio Frequency Memory (DRFM) Jammer.

    Implements coherent deception jamming via Range Gate Pull-Off (RGPO)
    and Velocity Gate Pull-Off (VGPO) techniques.

    State Machine:
        IDLE → CAPTURE → PULL → RELEASE

    RGPO Physics:
        1. Jammer receives radar pulse, stores in DRFM memory
        2. Retransmits with controlled delay Δτ(t)
        3. Apparent range: R_app(t) = R_true + pull_rate · t
        4. Delay per update: Δτ_step = 2 · pull_rate · dt / c

    VGPO Physics:
        1. Jammer modulates retransmitted pulse with Doppler offset
        2. f_d_jam(t) = f_d_true + accel · t
        3. Pulls velocity gate away from true target Doppler

    CPI Integration:
        inject_into_cpi() adds phase-coherent false returns into the
        Pulse-Doppler CPI data matrix at the offset range/Doppler bin.

    Reference:
        - Schleher, "Electronic Warfare in the Information Age", 1999, Ch. 7
        - Skolnik, "Radar Handbook", 3rd Ed., Ch. 24.3
        - Developed by Mehmet Gümüş (github.com/SpaceEngineerSS)
    """

    def __init__(self, config: Optional[DRFMConfig] = None) -> None:
        """
        Initialize DRFM jammer.

        Args:
            config: Jammer configuration (defaults to standard RGPO)
        """
        self.config = config or DRFMConfig()
        self.state = DRFMState.IDLE

        # Internal state
        self._capture_timer: float = 0.0
        self._pull_offset_m: float = 0.0
        self._pull_offset_hz: float = 0.0
        self._total_time: float = 0.0
        self._active: bool = False

    def activate(self) -> None:
        """Activate jammer — begins capture sequence."""
        if self.state == DRFMState.IDLE:
            self.state = DRFMState.CAPTURE
            self._capture_timer = 0.0
            self._pull_offset_m = 0.0
            self._pull_offset_hz = 0.0
            self._active = True

    def deactivate(self) -> None:
        """Deactivate jammer — returns to IDLE."""
        self.state = DRFMState.IDLE
        self._active = False
        self._pull_offset_m = 0.0
        self._pull_offset_hz = 0.0

    def step(self, dt: float) -> None:
        """
        Advance jammer state machine by one time step.

        State transitions:
            IDLE:    No action
            CAPTURE: Wait for capture_dwell_s → transition to PULL
            PULL:    Increase offset → transition to RELEASE at max_pull
            RELEASE: Abrupt stop (radar loses track)

        Args:
            dt: Time step [s]
        """
        self._total_time += dt

        if self.state == DRFMState.IDLE:
            return

        if self.state == DRFMState.CAPTURE:
            self._capture_timer += dt
            if self._capture_timer >= self.config.capture_dwell_s:
                self.state = DRFMState.PULL

        elif self.state == DRFMState.PULL:
            if self.config.mode == "rgpo":
                # RGPO: Increase range offset
                # Δτ_step = 2 · pull_rate · dt / c
                self._pull_offset_m += self.config.pull_rate_mps * dt
                if self._pull_offset_m >= self.config.max_pull_m:
                    self.state = DRFMState.RELEASE

            elif self.config.mode == "vgpo":
                # VGPO: Increase Doppler offset
                self._pull_offset_hz += self.config.vgpo_accel_hz_per_s * dt
                # Max pull at 500 Hz offset (reasonable for X-band)
                if abs(self._pull_offset_hz) > 500.0:
                    self.state = DRFMState.RELEASE

        elif self.state == DRFMState.RELEASE:
            # Abrupt release — radar loses track
            self._active = False
            self.state = DRFMState.IDLE
            self._pull_offset_m = 0.0
            self._pull_offset_hz = 0.0

    def inject_into_cpi(
        self,
        cpi_data: np.ndarray,
        true_range_m: float,
        true_velocity_mps: float,
        amplitude: float,
        range_resolution_m: float,
        wavelength_m: float,
        pri_s: float,
        n_range_bins: int,
    ) -> np.ndarray:
        """
        Inject DRFM false return into CPI data matrix.

        Adds a phase-coherent pulse replica at the offset range/Doppler
        position determined by the current pull state.

        The injected signal maintains coherence with the radar's
        LFM chirp (matched filter compatible) to survive range
        compression and Doppler processing.

        Args:
            cpi_data: Raw CPI data [n_pulses, n_range_bins], complex
            true_range_m: True target range [m]
            true_velocity_mps: True target radial velocity [m/s]
            amplitude: Signal amplitude (linear, from J/S)
            range_resolution_m: Range resolution [m]
            wavelength_m: Radar wavelength [m]
            pri_s: Pulse repetition interval [s]
            n_range_bins: Number of range bins

        Returns:
            Modified CPI data with injected false return

        Reference: Schleher (1999), Eq. 7.4
        """
        if self.state not in (DRFMState.CAPTURE, DRFMState.PULL):
            return cpi_data

        n_pulses = cpi_data.shape[0]

        # Calculate false target position
        if self.config.mode == "rgpo":
            false_range_m = true_range_m + self._pull_offset_m
            false_velocity_mps = true_velocity_mps
        else:  # VGPO
            false_range_m = true_range_m
            false_velocity_mps = true_velocity_mps + self._pull_offset_hz * wavelength_m / 2.0

        # J/S gain: jammer is stronger than skin return
        jammer_amplitude = amplitude * 10.0 ** (self.config.gain_over_skin_db / 20.0)

        # Range bin for false target
        false_range_bin = int(round(false_range_m / range_resolution_m))
        if false_range_bin < 0 or false_range_bin >= n_range_bins:
            return cpi_data

        # Doppler phase for false target (phase-coherent with radar LFM)
        n_idx = np.arange(n_pulses, dtype=np.float64)
        doppler_phase = 4.0 * np.pi * false_velocity_mps * n_idx * pri_s / wavelength_m
        steering = jammer_amplitude * np.exp(1j * doppler_phase)

        # Inject false return at offset range bin
        # Use a narrow sinc response (same as legitimate target)
        sinc_halfwidth = 2
        sinc_idx = np.arange(-sinc_halfwidth, sinc_halfwidth + 1)
        sinc_response = np.sinc(sinc_idx / 1.2).astype(np.complex128)
        sinc_response /= np.linalg.norm(sinc_response)

        for k, offset in enumerate(sinc_idx):
            col = false_range_bin + offset
            if 0 <= col < n_range_bins:
                cpi_data[:, col] += steering * sinc_response[k]

        return cpi_data

    @property
    def is_active(self) -> bool:
        """Check if jammer is producing false returns."""
        return self.state in (DRFMState.CAPTURE, DRFMState.PULL)

    @property
    def pull_offset_m(self) -> float:
        """Current range pull offset [m]."""
        return self._pull_offset_m

    @property
    def pull_offset_hz(self) -> float:
        """Current Doppler pull offset [Hz]."""
        return self._pull_offset_hz

    @property
    def false_range_offset_m(self) -> float:
        """Range offset of false target from true position [m]."""
        if self.config.mode == "rgpo":
            return self._pull_offset_m
        return 0.0

    def get_status(self) -> dict:
        """Get jammer status for UI display."""
        return {
            "state": self.state.value,
            "mode": self.config.mode.upper(),
            "active": self.is_active,
            "pull_offset_m": self._pull_offset_m,
            "pull_offset_hz": self._pull_offset_hz,
            "power_watts": self.config.power_watts,
        }


def calculate_jamming_effectiveness(jsr_db: float, detection_threshold_db: float = 13.0) -> float:
    """
    Calculate jamming effectiveness as probability of mask.

    Reference: Schleher (1999), Chapter 5

    Args:
        jsr_db: Jam-to-Signal ratio [dB]
        detection_threshold_db: Required SNR for detection [dB]

    Returns:
        Probability that jamming masks the target (0-1)
    """
    margin = jsr_db - detection_threshold_db
    # Sigmoid around the threshold
    return 1.0 / (1.0 + np.exp(-margin / 3.0))


def validate_burn_through_range() -> dict:
    """
    Validate burn-through range against textbook example.

    Reference: Schleher (1999), Example 4.1

    Setup:
        - Radar: Pt = 100 kW, G = 35 dB
        - Jammer: Pj = 1 kW, Gj = 6 dBi, Bj = 100 MHz
        - Target: σ = 1 m², Required SNR = 13 dB
        - Radar BW = 1 MHz

    Returns:
        Validation result dictionary
    """
    # Parameters
    radar_power = 100e3  # 100 kW
    radar_gain_db = 35.0
    radar_gain = 10 ** (radar_gain_db / 10)
    radar_bw = 1e6

    jammer_power = 1000.0  # 1 kW
    jammer_gain_db = 6.0
    jammer_gain = 10 ** (jammer_gain_db / 10)
    jammer_bw = 100e6

    target_rcs = 1.0
    snr_db = 13.0
    snr_linear = 10 ** (snr_db / 10)

    # Create simulator and calculate
    sim = ECMSimulator(radar_wavelength=0.03)
    r_bt = sim.calculate_burn_through_range(
        radar_power_watts=radar_power,
        radar_gain_linear=radar_gain,
        jammer_power_watts=jammer_power,
        jammer_gain_linear=jammer_gain,
        target_rcs_m2=target_rcs,
        required_snr_linear=snr_linear,
        radar_bandwidth_hz=radar_bw,
        jammer_bandwidth_hz=jammer_bw,
    )

    r_bt_km = r_bt / 1000.0

    # Expected: 15-30 km for these parameters
    expected_min_km = 10.0
    expected_max_km = 35.0
    is_valid = expected_min_km <= r_bt_km <= expected_max_km

    return {
        "test_name": "Burn-Through Range Validation",
        "reference": "Schleher (1999), Chapter 4",
        "inputs": {
            "radar_power_kw": radar_power / 1000,
            "radar_gain_db": radar_gain_db,
            "jammer_power_kw": jammer_power / 1000,
            "jammer_gain_dbi": jammer_gain_db,
            "target_rcs_m2": target_rcs,
        },
        "results": {
            "burn_through_range_m": r_bt,
            "burn_through_range_km": r_bt_km,
        },
        "validation": {
            "expected_range_km": f"{expected_min_km}-{expected_max_km}",
            "is_valid": is_valid,
        },
    }


# Module validation
if __name__ == "__main__":
    result = validate_burn_through_range()
    print("\n=== ECM Validation ===")
    print(f"Test: {result['test_name']}")
    print(f"Burn-through range: {result['results']['burn_through_range_km']:.2f} km")
    print(f"Expected: {result['validation']['expected_range_km']} km")
    print(f"Valid: {'✓ PASS' if result['validation']['is_valid'] else '✗ FAIL'}")
