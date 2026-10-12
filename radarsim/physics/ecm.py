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
from math import erfc, exp, log10, pi, sqrt
from typing import Optional

import numba
import numpy as np


@dataclass(frozen=True)
class ReceiverLimiterResult:
    """Input-referred result of a memoryless complex-envelope hard limiter."""

    sinr_db: float
    clipping_loss_db: float
    input_power_dbm: float
    headroom_db: float
    coherent_gain: float
    distortion_power_watts: float
    output_power_watts: float
    overloaded: bool


def apply_receiver_hard_limiter(
    signal_power_watts: float,
    interference_power_watts: float,
    full_scale_power_watts: float,
) -> ReceiverLimiterResult:
    """Apply a radial hard limiter to a circular complex-Gaussian receiver input."""
    if signal_power_watts <= 0.0:
        raise ValueError("signal_power_watts must be positive")
    if interference_power_watts <= 0.0:
        raise ValueError("interference_power_watts must be positive")
    if full_scale_power_watts <= 0.0:
        raise ValueError("full_scale_power_watts must be positive")

    total_input = signal_power_watts + interference_power_watts
    clipping_ratio = full_scale_power_watts / total_input
    input_sinr = signal_power_watts / interference_power_watts

    if clipping_ratio >= 50.0:
        coherent_gain = 1.0
        output_power = total_input
        distortion_power = 0.0
    else:
        root_ratio = sqrt(clipping_ratio)
        tail = exp(-clipping_ratio)
        coherent_gain = 1.0 - tail + 0.5 * sqrt(pi * clipping_ratio) * erfc(
            root_ratio
        )
        output_power = total_input * (1.0 - tail)
        distortion_power = max(
            output_power - coherent_gain**2 * total_input, 0.0
        )

    output_signal = coherent_gain**2 * signal_power_watts
    output_interference = (
        coherent_gain**2 * interference_power_watts + distortion_power
    )
    output_sinr = output_signal / output_interference
    sinr_db = 10.0 * log10(output_sinr)
    input_sinr_db = 10.0 * log10(input_sinr)
    headroom_db = 10.0 * log10(clipping_ratio)
    return ReceiverLimiterResult(
        sinr_db=sinr_db,
        clipping_loss_db=max(input_sinr_db - sinr_db, 0.0),
        input_power_dbm=10.0 * log10(total_input / 1e-3),
        headroom_db=headroom_db,
        coherent_gain=coherent_gain,
        distortion_power_watts=distortion_power,
        output_power_watts=output_power,
        overloaded=headroom_db < 0.0,
    )


@numba.jit(nopython=True, cache=True)
def _calculate_jsr_jit(
    jammer_power: float,
    jammer_gain: float,
    radar_power: float,
    radar_gain: float,
    rcs: float,
    jammer_range: float,
    target_range: float,
    bandwidth_ratio: float,
) -> float:
    """
    JIT-compiled Jam-to-Signal Ratio calculation.

    JSR = (Pj * Gj * 4π * R_t^4) / (Pt * Gt * σ * R_j^2) * min(1, Br/Bj)

    Args:
        jammer_power: Jammer ERP [W]
        jammer_gain: Jammer antenna gain [linear]
        radar_power: Radar peak power [W]
        radar_gain: Radar antenna gain [linear]
        rcs: Target RCS [m²]
        jammer_range: Jammer to radar range [m]
        target_range: Target to radar range [m]
        bandwidth_ratio: Receiver-to-jammer bandwidth ratio Br/Bj

    Returns:
        JSR [linear]

    Reference: Schleher, Eq. 4.1
    """
    if target_range < 1 or jammer_range < 1:
        return 0.0

    numerator = 4.0 * np.pi * jammer_power * jammer_gain * target_range**4
    denominator = radar_power * radar_gain * rcs * jammer_range**2
    if denominator > 0.0:
        return numerator / denominator * min(1.0, bandwidth_ratio)
    return np.inf


class ECMSimulator:
    """
    Electronic Countermeasures Simulator

    Jam-to-signal ratio, signal-to-jamming-plus-noise ratio and burn-through
    range for noise jamming (barrage and spot).

    Reference: Schleher, "Electronic Warfare in the Information Age"
    """

    def __init__(self, radar_wavelength: float = 0.03):
        """
        Initialize ECM simulator.

        Args:
            radar_wavelength: Radar operating wavelength [m]
        """
        self.wavelength = radar_wavelength

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
        jammer_bandwidth: float = 100e6,
        jammer_gain: float = 1.0,
        signal_path_loss_db: float = 0.0,
        jammer_path_loss_db: float = 0.0,
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

        if min(radar_power, radar_gain, jammer_power, target_rcs) <= 0.0:
            raise ValueError("powers, radar gain, and target RCS must be positive")
        if radar_bandwidth <= 0.0 or jammer_bandwidth <= 0.0 or jammer_gain <= 0.0:
            raise ValueError("bandwidths and jammer gain must be positive")
        bw_ratio = radar_bandwidth / jammer_bandwidth

        jsr_linear = _calculate_jsr_jit(
            jammer_power,
            jammer_gain,
            radar_power,
            radar_gain,
            target_rcs,
            jammer_range,
            target_range,
            bw_ratio,
        )

        return (
            10 * np.log10(max(jsr_linear, 1e-10))
            + signal_path_loss_db
            - jammer_path_loss_db
        )

    @staticmethod
    def calculate_sjnr_db(snr_db: float, jsr_db: float) -> float:
        """Combine thermal noise and jamming powers relative to the signal."""
        inverse_sjnr = 10.0 ** (-snr_db / 10.0) + 10.0 ** (jsr_db / 10.0)
        return float(-10.0 * np.log10(inverse_sjnr))

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

        Self-screening jammer: S/J = Pt·Gt·σ·Bj / (4π·R²·Pj·Gj·Br) = SNR_req, so
        R_bt = sqrt((Pt · Gt · σ) / (Pj · Gj · 4π · SNR_req · Br/Bj)),
        with Br/Bj capped at 1 when the jammer is narrower than the receiver.

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
            jammer_power_watts
            * jammer_gain_linear
            * 4
            * np.pi
            * required_snr_linear
            * bw_ratio
        )

        if denominator > 1e-30:
            r_bt_squared = numerator / denominator
            return np.sqrt(r_bt_squared)
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
# Digital radio-frequency memory deception
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
    max_doppler_pull_hz: float = 500.0
    inherent_delay_s: float = 0.0

    def __post_init__(self) -> None:
        self.mode = self.mode.lower()
        if self.mode not in {"rgpo", "vgpo"}:
            raise ValueError("mode must be 'rgpo' or 'vgpo'")
        if self.power_watts <= 0.0:
            raise ValueError("power_watts must be positive")
        if self.pull_rate_mps < 0.0 or self.vgpo_accel_hz_per_s < 0.0:
            raise ValueError("pull rates cannot be negative")
        if self.max_pull_m <= 0.0 or self.max_doppler_pull_hz <= 0.0:
            raise ValueError("maximum pull offsets must be positive")
        if self.capture_dwell_s < 0.0 or self.inherent_delay_s < 0.0:
            raise ValueError("capture dwell and inherent delay cannot be negative")


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
                    self._pull_offset_m = self.config.max_pull_m
                    self.state = DRFMState.RELEASE

            elif self.config.mode == "vgpo":
                # VGPO: Increase Doppler offset
                self._pull_offset_hz += self.config.vgpo_accel_hz_per_s * dt
                if abs(self._pull_offset_hz) >= self.config.max_doppler_pull_hz:
                    self._pull_offset_hz = self.config.max_doppler_pull_hz
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
            false_range_m = true_range_m + self.false_range_offset_m
            false_velocity_mps = true_velocity_mps
        else:  # VGPO
            false_range_m = true_range_m
            # A Doppler offset dfd moves the apparent radial velocity by -dfd*lambda/2.
            false_velocity_mps = (
                true_velocity_mps - self._pull_offset_hz * wavelength_m / 2.0
            )

        # J/S gain: jammer is stronger than skin return
        jammer_amplitude = amplitude * 10.0 ** (self.config.gain_over_skin_db / 20.0)

        # Range bin for false target
        false_range_bin = int(round(false_range_m / range_resolution_m))
        if false_range_bin < 0 or false_range_bin >= n_range_bins:
            return cpi_data

        # Doppler phase for false target (phase-coherent with radar LFM)
        n_idx = np.arange(n_pulses, dtype=np.float64)
        # Same convention as PulseDopplerProcessor: fd = -2 v_r / lambda.
        doppler_phase = -4.0 * np.pi * false_velocity_mps * n_idx * pri_s / wavelength_m
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
            return C_LIGHT * self.config.inherent_delay_s / 2.0 + self._pull_offset_m
        return 0.0

    @property
    def retransmission_delay_s(self) -> float:
        """Controlled two-way-equivalent delay applied to the captured pulse [s]."""
        return self.config.inherent_delay_s + 2.0 * self._pull_offset_m / C_LIGHT

    def get_status(self) -> dict:
        """Get jammer status for UI display."""
        return {
            "state": self.state.value,
            "mode": self.config.mode.upper(),
            "active": self.is_active,
            "pull_offset_m": self._pull_offset_m,
            "pull_offset_hz": self._pull_offset_hz,
            "retransmission_delay_s": self.retransmission_delay_s,
            "power_watts": self.config.power_watts,
        }


def validate_burn_through_range() -> dict:
    """
    Self-consistency check of the self-screening burn-through range.

    At R_bt the signal-to-jamming ratio from ``calculate_jsr`` must equal the
    required SNR, i.e. JSR(R_bt) = -SNR_req [dB]; inside R_bt the radar burns
    through (JSR lower), outside it is jammed.

    Setup: Pt = 100 kW, Gt = 35 dB, Pj = 1 kW ERP, Gj = 6 dBi, Bj = 100 MHz,
    Br = 1 MHz, sigma = 1 m^2, SNR_req = 13 dB.

    Reference: Schleher (1999), Sec. 4.2 (self-screening jammer equation)
    """
    radar_power = 100e3
    radar_gain_db = 35.0
    radar_gain = 10 ** (radar_gain_db / 10)
    radar_bw = 1e6

    jammer_power = 1000.0
    jammer_gain_db = 6.0
    jammer_gain = 10 ** (jammer_gain_db / 10)
    jammer_bw = 100e6

    target_rcs = 1.0
    snr_db = 13.0
    snr_linear = 10 ** (snr_db / 10)

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

    def jsr_at(range_m: float) -> float:
        position = np.array([range_m, 0.0, 0.0])
        return sim.calculate_jsr(
            radar_pos=np.zeros(3),
            target_pos=position,
            jammer_pos=position,
            radar_power=radar_power,
            radar_gain=radar_gain,
            jammer_power=jammer_power,
            target_rcs=target_rcs,
            radar_bandwidth=radar_bw,
            jammer_bandwidth=jammer_bw,
            jammer_gain=jammer_gain,
        )

    jsr_at_burn_through_db = jsr_at(r_bt)
    is_valid = (
        abs(jsr_at_burn_through_db + snr_db) < 0.01
        and jsr_at(0.5 * r_bt) < -snr_db
        and jsr_at(2.0 * r_bt) > -snr_db
    )

    return {
        "test_name": "Burn-Through Range Validation",
        "reference": "Schleher (1999), Sec. 4.2",
        "inputs": {
            "radar_power_kw": radar_power / 1000,
            "radar_gain_db": radar_gain_db,
            "jammer_power_kw": jammer_power / 1000,
            "jammer_gain_dbi": jammer_gain_db,
            "target_rcs_m2": target_rcs,
            "required_snr_db": snr_db,
        },
        "results": {
            "burn_through_range_m": r_bt,
            "burn_through_range_km": r_bt / 1000.0,
            "jsr_at_burn_through_db": jsr_at_burn_through_db,
        },
        "validation": {
            "expected_jsr_db": -snr_db,
            "is_valid": is_valid,
        },
    }


# Module validation
if __name__ == "__main__":
    result = validate_burn_through_range()
    print("\n=== ECM Validation ===")
    print(f"Test: {result['test_name']}")
    print(f"Burn-through range: {result['results']['burn_through_range_km']:.2f} km")
    print(f"J/S at burn-through: {result['results']['jsr_at_burn_through_db']:.2f} dB")
    print(f"Valid: {'✓ PASS' if result['validation']['is_valid'] else '✗ FAIL'}")
