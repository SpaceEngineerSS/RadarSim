# Developed by Mehmet Gümüş (@SpaceEngineerSS) - RadarSim v2.x
"""
Electronic Counter-Countermeasures (ECCM) Module

Implements radar techniques to mitigate electronic attack (EA):
    - Frequency Agility: Hop carrier frequency to reduce narrowband J/S
    - PRF Stagger: Vary PRI to defeat RGPO and resolve ambiguities
    - ECCM Controller: Orchestrates all techniques

Physics:
    - Freq Agility: J/S_agile = J/S_static - 10·log₁₀(N_hops)
    - PRF Stagger:  PRI_n = T_PRI · (1 + δ·uniform(-1, 1))
    - RGPO discrimination: Cross-PRI correlation < 0.3 → jamming

References:
    - Schleher, "Electronic Warfare in the Information Age", 1999, Ch. 8-9
    - Skolnik, "Radar Handbook", 3rd Ed., Ch. 24.5
    - Developed by Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

from typing import List, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# FREQUENCY AGILITY
# ═══════════════════════════════════════════════════════════════════════


class FrequencyAgility:
    """
    Frequency agility ECCM technique.

    Randomizes radar carrier frequency per CPI to reduce effectiveness
    of narrowband noise jamming (spot jammer).

    Physics:
        A spot jammer concentrates power in bandwidth B_j ≈ B_r.
        With N frequency hops spread across B_hop:
            - Jammer must spread power across B_hop to cover all hops
            - Effective J/S reduced by: 10·log₁₀(N_hops) [dB]
            - Or jammer covers only 1/N of the hops (probability)

    Example:
        >>> fa = FrequencyAgility(center_freq_hz=10e9, n_hops=10)
        >>> f = fa.get_next_frequency()  # Random hop
        >>> js_reduction = fa.js_reduction_db  # 10 dB

    Reference: Schleher (1999), Ch. 8.2
    """

    def __init__(
        self,
        center_freq_hz: float = 10e9,
        hop_bandwidth_hz: float = 500e6,
        n_hops: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize frequency agility.

        Args:
            center_freq_hz: Center frequency [Hz]
            hop_bandwidth_hz: Total hop bandwidth [Hz]
            n_hops: Number of discrete frequencies in hop set
            seed: RNG seed for reproducibility
        """
        self.center_freq_hz = center_freq_hz
        self.hop_bandwidth_hz = hop_bandwidth_hz
        self.n_hops = max(2, n_hops)
        self._rng = np.random.default_rng(seed)

        # Generate hop set (equally spaced across bandwidth)
        self.hop_set = np.linspace(
            center_freq_hz - hop_bandwidth_hz / 2,
            center_freq_hz + hop_bandwidth_hz / 2,
            self.n_hops,
        )

        self._current_hop_idx = 0
        self._enabled = False

    def enable(self) -> None:
        """Enable frequency agility."""
        self._enabled = True

    def disable(self) -> None:
        """Disable frequency agility."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if frequency agility is active."""
        return self._enabled

    def get_next_frequency(self) -> float:
        """
        Get next frequency from hop set.

        Returns random frequency from the hop set each call.

        Returns:
            Carrier frequency [Hz]
        """
        if not self._enabled:
            return self.center_freq_hz

        self._current_hop_idx = self._rng.integers(0, self.n_hops)
        return self.hop_set[self._current_hop_idx]

    @property
    def js_reduction_db(self) -> float:
        """
        J/S reduction from frequency agility [dB].

        J/S_agile = J/S_static - 10·log₁₀(N_hops)

        Returns:
            J/S reduction in dB (positive = benefit)

        Reference: Schleher (1999), Eq. 8.3
        """
        if not self._enabled:
            return 0.0
        return 10.0 * np.log10(self.n_hops)

    @property
    def current_frequency_hz(self) -> float:
        """Current operating frequency [Hz]."""
        if not self._enabled:
            return self.center_freq_hz
        return self.hop_set[self._current_hop_idx]

    def get_status(self) -> dict:
        """Get status for UI display."""
        return {
            "enabled": self._enabled,
            "n_hops": self.n_hops,
            "js_reduction_db": self.js_reduction_db,
            "current_freq_ghz": self.current_frequency_hz / 1e9,
            "hop_bw_mhz": self.hop_bandwidth_hz / 1e6,
        }


# ═══════════════════════════════════════════════════════════════════════
# PRF STAGGER
# ═══════════════════════════════════════════════════════════════════════


class PRFStagger:
    """
    PRF Stagger/Jitter ECCM technique.

    Varies Pulse Repetition Interval (PRI) to:
    1. Defeat RGPO jamming (delayed pulse arrives at wrong time)
    2. Resolve range ambiguities (staggered unambiguous ranges)
    3. Improve MTI blind speed coverage

    Physics:
        PRI_n = T_PRI · (1 + δ · u_n)  where u_n ∈ [-1, 1]

        RGPO discrimination:
        - Real target: correlation across staggered PRIs ≈ 1.0
        - RGPO jammer: correlation ≈ 0 (delay is fixed, PRI varies)

    Example:
        >>> stagger = PRFStagger(nominal_pri_s=1e-3, jitter_percent=5.0)
        >>> pri = stagger.get_next_pri()
        >>> is_rgpo = stagger.discriminate_rgpo(range_measurements)

    Reference: Schleher (1999), Ch. 8.4; Skolnik, Ch. 24.5
    """

    def __init__(
        self,
        nominal_pri_s: float = 1e-3,
        jitter_percent: float = 5.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Initialize PRF stagger.

        Args:
            nominal_pri_s: Nominal PRI [s]
            jitter_percent: Jitter magnitude [% of PRI]
            seed: RNG seed for reproducibility
        """
        self.nominal_pri_s = nominal_pri_s
        self.jitter_percent = jitter_percent
        self._rng = np.random.default_rng(seed)
        self._enabled = False

        # History for RGPO discrimination
        self._pri_history: List[float] = []
        self._range_history: List[float] = []
        self._max_history = 16

    def enable(self) -> None:
        """Enable PRF stagger."""
        self._enabled = True

    def disable(self) -> None:
        """Disable PRF stagger."""
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        """Check if PRF stagger is active."""
        return self._enabled

    def get_next_pri(self) -> float:
        """
        Get next staggered PRI.

        PRI_n = T_PRI · (1 + δ · u_n)

        Returns:
            Staggered PRI [s]
        """
        if not self._enabled:
            return self.nominal_pri_s

        delta = self.jitter_percent / 100.0
        jitter = delta * self._rng.uniform(-1.0, 1.0)
        pri = self.nominal_pri_s * (1.0 + jitter)

        self._pri_history.append(pri)
        if len(self._pri_history) > self._max_history:
            self._pri_history.pop(0)

        return pri

    def record_range_measurement(self, range_m: float) -> None:
        """
        Record range measurement for RGPO discrimination.

        Args:
            range_m: Measured range [m]
        """
        self._range_history.append(range_m)
        if len(self._range_history) > self._max_history:
            self._range_history.pop(0)

    def discriminate_rgpo(self) -> float:
        """
        RGPO discrimination via cross-PRI correlation.

        Real target: range is consistent across PRI variations → corr ≈ 1
        RGPO jammer: fixed delay + variable PRI → range varies → corr ≈ 0

        Returns:
            Correlation coefficient [0, 1]:
                > 0.7: likely real target
                < 0.3: likely RGPO jammer

        Reference: Schleher (1999), Ch. 8.4
        """
        if len(self._range_history) < 4:
            return 1.0  # Insufficient data, assume real

        ranges = np.array(self._range_history[-8:])
        if np.std(ranges) < 1e-6:
            return 1.0  # Zero variance = perfectly consistent

        # Normalized range consistency (1 = consistent, 0 = varying)
        # Use coefficient of variation (CV = std/mean)
        cv = np.std(ranges) / max(np.mean(np.abs(ranges)), 1.0)
        # Map CV to correlation: low CV → high correlation
        correlation = np.exp(-cv * 10.0)
        return float(np.clip(correlation, 0.0, 1.0))

    def get_status(self) -> dict:
        """Get status for UI display."""
        return {
            "enabled": self._enabled,
            "jitter_percent": self.jitter_percent,
            "nominal_prf_hz": 1.0 / self.nominal_pri_s if self.nominal_pri_s > 0 else 0,
        }


# ═══════════════════════════════════════════════════════════════════════
# ECCM CONTROLLER
# ═══════════════════════════════════════════════════════════════════════


class ECCMController:
    """
    Electronic Counter-Countermeasures Controller.

    Orchestrates all ECCM techniques and provides a unified interface
    for the simulation engine.

    Techniques:
        1. Frequency Agility: Reduces narrowband J/S by N_hops factor
        2. PRF Stagger: Defeats RGPO, resolves ambiguities
        3. Burn-through: Radar power overcomes jamming at close range

    Example:
        >>> eccm = ECCMController(center_freq_hz=10e9)
        >>> eccm.enable_frequency_agility()
        >>> eccm.enable_prf_stagger()
        >>> freq, pri = eccm.get_cpi_parameters()

    Reference: Schleher (1999), Ch. 8-9
    """

    def __init__(
        self,
        center_freq_hz: float = 10e9,
        nominal_prf_hz: float = 1000.0,
        n_freq_hops: int = 10,
        prf_jitter_percent: float = 5.0,
    ) -> None:
        """
        Initialize ECCM controller.

        Args:
            center_freq_hz: Radar center frequency [Hz]
            nominal_prf_hz: Nominal PRF [Hz]
            n_freq_hops: Number of frequency agility hops
            prf_jitter_percent: PRF jitter magnitude [%]
        """
        self.freq_agility = FrequencyAgility(
            center_freq_hz=center_freq_hz,
            n_hops=n_freq_hops,
        )
        self.prf_stagger = PRFStagger(
            nominal_pri_s=1.0 / nominal_prf_hz,
            jitter_percent=prf_jitter_percent,
        )

        # Jamming environment state
        self.jamming_active = False
        self.jsr_db = 0.0  # Jam-to-Signal ratio

    def enable_frequency_agility(self) -> None:
        """Enable frequency agility ECCM."""
        self.freq_agility.enable()

    def disable_frequency_agility(self) -> None:
        """Disable frequency agility ECCM."""
        self.freq_agility.disable()

    def enable_prf_stagger(self) -> None:
        """Enable PRF stagger ECCM."""
        self.prf_stagger.enable()

    def disable_prf_stagger(self) -> None:
        """Disable PRF stagger ECCM."""
        self.prf_stagger.disable()

    def set_jamming_environment(self, active: bool, jsr_db: float = 20.0) -> None:
        """
        Set jamming environment state.

        Args:
            active: Whether jamming is present
            jsr_db: Static Jam-to-Signal ratio [dB]
        """
        self.jamming_active = active
        self.jsr_db = jsr_db if active else 0.0

    def get_effective_jsr_db(self) -> float:
        """
        Get effective J/S after ECCM reduction.

        Applies frequency agility reduction to static JSR.

        Returns:
            Effective J/S [dB]
        """
        if not self.jamming_active:
            return -100.0  # No jamming

        effective = self.jsr_db - self.freq_agility.js_reduction_db
        return effective

    def get_cpi_parameters(self) -> Tuple[float, float]:
        """
        Get frequency and PRI for next CPI.

        Returns:
            (carrier_frequency_hz, pri_s) tuple
        """
        freq = self.freq_agility.get_next_frequency()
        pri = self.prf_stagger.get_next_pri()
        return freq, pri

    def calculate_sjnr_db(self, snr_db: float) -> float:
        """
        Calculate Signal-to-Jamming-plus-Noise Ratio.

        SJNR = SNR / (1 + J/S)  [linear]
        SJNR_dB = SNR_dB - 10·log₁₀(1 + 10^(JSR_eff/10))

        Args:
            snr_db: Signal-to-noise ratio [dB]

        Returns:
            SJNR [dB]

        Reference: Schleher (1999), Eq. 4.8
        """
        jsr_eff = self.get_effective_jsr_db()
        if jsr_eff < -50:
            return snr_db  # No significant jamming

        jsr_linear = 10.0 ** (jsr_eff / 10.0)
        sjnr_db = snr_db - 10.0 * np.log10(1.0 + jsr_linear)
        return sjnr_db

    def get_status(self) -> dict:
        """Get combined ECCM status for UI."""
        return {
            "jamming_active": self.jamming_active,
            "jsr_db": self.jsr_db,
            "effective_jsr_db": self.get_effective_jsr_db(),
            "freq_agility": self.freq_agility.get_status(),
            "prf_stagger": self.prf_stagger.get_status(),
        }


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════


def validate_eccm() -> dict:
    """
    Self-validation of ECCM techniques.

    Tests:
        1. Frequency agility reduces J/S by 10·log₁₀(N)
        2. PRF stagger produces varying PRIs
        3. SJNR calculation with jamming

    Reference: Schleher (1999), Ch. 8-9
    """
    results = {}

    # ── Test 1: Frequency Agility J/S Reduction ──
    fa = FrequencyAgility(center_freq_hz=10e9, n_hops=10)
    fa.enable()
    expected_reduction = 10.0 * np.log10(10)  # = 10 dB
    actual_reduction = fa.js_reduction_db
    results["freq_agility_js_reduction"] = {
        "expected_db": expected_reduction,
        "actual_db": actual_reduction,
        "error_db": abs(actual_reduction - expected_reduction),
        "pass": abs(actual_reduction - expected_reduction) < 0.01,
    }

    # ── Test 2: PRF Stagger Variation ──
    stagger = PRFStagger(nominal_pri_s=1e-3, jitter_percent=5.0, seed=42)
    stagger.enable()
    pris = [stagger.get_next_pri() for _ in range(100)]
    pri_std = np.std(pris)
    results["prf_stagger_variation"] = {
        "mean_pri_us": np.mean(pris) * 1e6,
        "std_pri_us": pri_std * 1e6,
        "pass": pri_std > 0,  # Must have variation
    }

    # ── Test 3: SJNR with Jamming ──
    ctrl = ECCMController(center_freq_hz=10e9, n_freq_hops=10)
    ctrl.set_jamming_environment(True, jsr_db=20.0)
    sjnr_no_eccm = 20.0 - 10.0 * np.log10(1.0 + 10.0 ** (20.0 / 10.0))
    sjnr_with_eccm = ctrl.calculate_sjnr_db(snr_db=20.0)

    # Without ECCM, SJNR should be much lower
    ctrl.enable_frequency_agility()
    sjnr_agile = ctrl.calculate_sjnr_db(snr_db=20.0)

    results["sjnr_improvement"] = {
        "sjnr_no_eccm_db": sjnr_no_eccm,
        "sjnr_with_agility_db": sjnr_agile,
        "improvement_db": sjnr_agile - sjnr_no_eccm,
        "pass": sjnr_agile > sjnr_no_eccm,
    }

    return results


if __name__ == "__main__":
    results = validate_eccm()
    print("=" * 60)
    print("ECCM Validation")
    print("=" * 60)
    for test_name, test_result in results.items():
        status = "✓ PASS" if test_result.get("pass", False) else "✗ FAIL"
        print(f"\n{status} | {test_name}")
        for k, v in test_result.items():
            if k != "pass":
                print(f"    {k}: {v}")
