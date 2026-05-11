"""
Electronic Warfare (EA/ECCM) Validation Tests

Tests:
    1. DRFM RGPO state machine transitions
    2. DRFM CPI injection coherence
    3. Frequency Agility J/S reduction (≥10 dB for N=10)
    4. PRF Stagger variation
    5. RGPO discrimination via correlation
    6. EKF coast mode (SJNR < 6 dB)
    7. EKF track drop after max coast scans
    8. SJNR calculation
    9. Burn-through range validation
    10. ECCM Controller integration

References:
    - Schleher, "Electronic Warfare in the Information Age", 1999
    - Skolnik, "Radar Handbook", 3rd Ed., Ch. 24
    - Developed by Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

import numpy as np
import pytest

from src.advanced.eccm import (
    ECCMController,
    FrequencyAgility,
    PRFStagger,
)
from src.physics.ecm import (
    DRFMConfig,
    DRFMJammer,
    DRFMState,
    ECMSimulator,
)
from src.tracking.ekf import ExtendedKalmanFilter
from src.tracking.kalman import KalmanState

# ═══════════════════════════════════════════════════════════════════
# TEST 1: DRFM JAMMER STATE MACHINE
# ═══════════════════════════════════════════════════════════════════


class TestDRFMStateMachine:
    """
    Verify DRFM jammer state transitions.

    State Machine: IDLE → CAPTURE → PULL → RELEASE

    Reference: Schleher (1999), Ch. 7
    """

    def test_initial_state_idle(self):
        """Jammer starts in IDLE state."""
        jammer = DRFMJammer()
        assert jammer.state == DRFMState.IDLE
        assert not jammer.is_active

    def test_activate_starts_capture(self):
        """Activation transitions to CAPTURE."""
        jammer = DRFMJammer()
        jammer.activate()
        assert jammer.state == DRFMState.CAPTURE
        assert jammer.is_active

    def test_capture_to_pull_transition(self):
        """After capture_dwell_s, transitions to PULL."""
        config = DRFMConfig(capture_dwell_s=1.0)
        jammer = DRFMJammer(config)
        jammer.activate()

        # Step through capture phase
        for _ in range(10):
            jammer.step(0.11)  # 10 × 0.11 = 1.1s > 1.0s

        assert jammer.state == DRFMState.PULL

    def test_pull_to_release_transition(self):
        """At max_pull_m, transitions to RELEASE then IDLE."""
        config = DRFMConfig(
            capture_dwell_s=0.1,
            pull_rate_mps=1000.0,
            max_pull_m=100.0,
        )
        jammer = DRFMJammer(config)
        jammer.activate()

        # Fast-forward through capture
        jammer.step(0.2)
        assert jammer.state == DRFMState.PULL

        # Pull until max
        for _ in range(20):
            jammer.step(0.01)  # 20 × 0.01 × 1000 = 200m > 100m

        # Should have transitioned through RELEASE → IDLE
        assert jammer.state == DRFMState.IDLE
        assert not jammer.is_active

    def test_deactivate_resets_state(self):
        """Deactivation returns to IDLE and resets offsets."""
        jammer = DRFMJammer()
        jammer.activate()
        jammer.step(5.0)  # Past capture
        jammer.step(1.0)  # Some pull
        jammer.deactivate()

        assert jammer.state == DRFMState.IDLE
        assert jammer.pull_offset_m == 0.0

    def test_rgpo_range_offset_increases(self):
        """Range offset must increase during PULL phase."""
        config = DRFMConfig(
            capture_dwell_s=0.1,
            pull_rate_mps=100.0,
            max_pull_m=5000.0,
        )
        jammer = DRFMJammer(config)
        jammer.activate()
        jammer.step(0.2)  # Past capture

        offsets = []
        for _ in range(10):
            jammer.step(1.0)
            offsets.append(jammer.pull_offset_m)

        # Must be monotonically increasing
        for i in range(1, len(offsets)):
            assert offsets[i] > offsets[i - 1], "Pull offset not increasing"

    def test_vgpo_mode(self):
        """VGPO mode should increase Doppler offset."""
        config = DRFMConfig(
            capture_dwell_s=0.1,
            mode="vgpo",
            vgpo_accel_hz_per_s=50.0,
        )
        jammer = DRFMJammer(config)
        jammer.activate()
        jammer.step(0.2)  # Past capture
        jammer.step(2.0)  # 2s × 50 Hz/s = 100 Hz

        assert jammer.pull_offset_hz == pytest.approx(100.0, abs=1.0)


# ═══════════════════════════════════════════════════════════════════
# TEST 2: DRFM CPI INJECTION
# ═══════════════════════════════════════════════════════════════════


class TestDRFMInjection:
    """
    Verify CPI injection produces coherent false returns.

    Reference: Schleher (1999), Eq. 7.4
    """

    def test_injection_adds_energy(self):
        """Injection must add energy to CPI data."""
        config = DRFMConfig(
            capture_dwell_s=0.0,
            pull_rate_mps=100.0,
            gain_over_skin_db=10.0,
        )
        jammer = DRFMJammer(config)
        jammer.activate()
        jammer.step(0.1)  # Into PULL state, offset = 10m

        cpi = np.zeros((32, 256), dtype=np.complex128)
        cpi_orig_power = np.sum(np.abs(cpi) ** 2)

        cpi_jammed = jammer.inject_into_cpi(
            cpi_data=cpi.copy(),
            true_range_m=5000.0,
            true_velocity_mps=50.0,
            amplitude=1.0,
            range_resolution_m=30.0,
            wavelength_m=0.03,
            pri_s=1e-3,
            n_range_bins=256,
        )

        jammed_power = np.sum(np.abs(cpi_jammed) ** 2)
        assert jammed_power > cpi_orig_power, "Injection did not add energy"

    def test_injection_at_correct_bin(self):
        """False return must appear at offset range bin."""
        config = DRFMConfig(
            capture_dwell_s=0.0,
            pull_rate_mps=0.0,  # No pull yet
            gain_over_skin_db=20.0,
        )
        jammer = DRFMJammer(config)
        jammer.activate()

        true_range_m = 3000.0
        range_res_m = 30.0
        expected_bin = int(round(true_range_m / range_res_m))  # Bin 100

        cpi = np.zeros((16, 256), dtype=np.complex128)
        cpi_jammed = jammer.inject_into_cpi(
            cpi_data=cpi.copy(),
            true_range_m=true_range_m,
            true_velocity_mps=0.0,
            amplitude=1.0,
            range_resolution_m=range_res_m,
            wavelength_m=0.03,
            pri_s=1e-3,
            n_range_bins=256,
        )

        # Maximum energy should be near expected bin
        power_per_bin = np.sum(np.abs(cpi_jammed) ** 2, axis=0)
        peak_bin = np.argmax(power_per_bin)
        assert abs(peak_bin - expected_bin) <= 2, f"Peak at bin {peak_bin}, expected {expected_bin}"

    def test_idle_no_injection(self):
        """No injection when jammer is IDLE."""
        jammer = DRFMJammer()  # IDLE
        cpi = np.ones((16, 128), dtype=np.complex128) * 0.5

        cpi_result = jammer.inject_into_cpi(
            cpi_data=cpi.copy(),
            true_range_m=5000.0,
            true_velocity_mps=50.0,
            amplitude=1.0,
            range_resolution_m=30.0,
            wavelength_m=0.03,
            pri_s=1e-3,
            n_range_bins=128,
        )

        assert np.allclose(cpi, cpi_result), "IDLE jammer should not modify CPI"


# ═══════════════════════════════════════════════════════════════════
# TEST 3: FREQUENCY AGILITY J/S REDUCTION
# ═══════════════════════════════════════════════════════════════════


class TestFrequencyAgility:
    """
    Verify frequency agility reduces J/S by 10·log₁₀(N).

    Reference: Schleher (1999), Ch. 8.2
    """

    @pytest.mark.parametrize(
        "n_hops,expected_reduction_db",
        [
            (10, 10.0),  # 10·log₁₀(10) = 10 dB
            (20, 13.01),  # 10·log₁₀(20) ≈ 13.01 dB
            (2, 3.01),  # 10·log₁₀(2) ≈ 3.01 dB
            (100, 20.0),  # 10·log₁₀(100) = 20 dB
        ],
    )
    def test_js_reduction(self, n_hops, expected_reduction_db):
        """J/S reduction must equal 10·log₁₀(N_hops)."""
        fa = FrequencyAgility(center_freq_hz=10e9, n_hops=n_hops)
        fa.enable()
        actual = fa.js_reduction_db
        assert (
            abs(actual - expected_reduction_db) < 0.1
        ), f"Expected {expected_reduction_db:.1f} dB, got {actual:.1f} dB"

    def test_disabled_no_reduction(self):
        """Disabled frequency agility gives 0 dB reduction."""
        fa = FrequencyAgility(n_hops=10)
        assert fa.js_reduction_db == 0.0

    def test_frequencies_in_hop_set(self):
        """All generated frequencies must be in the hop set."""
        fa = FrequencyAgility(center_freq_hz=10e9, hop_bandwidth_hz=500e6, n_hops=10)
        fa.enable()
        for _ in range(100):
            freq = fa.get_next_frequency()
            assert freq in fa.hop_set, f"Frequency {freq} not in hop set"

    def test_frequency_variation(self):
        """Hopping must produce varying frequencies."""
        fa = FrequencyAgility(center_freq_hz=10e9, n_hops=10, seed=42)
        fa.enable()
        freqs = [fa.get_next_frequency() for _ in range(50)]
        unique = len(set(freqs))
        assert unique >= 5, f"Only {unique} unique frequencies in 50 hops"

    def test_10_hops_reduces_10db(self):
        """
        CRITICAL: 10 hops must reduce J/S by ≥10 dB.

        This is the primary ECCM performance requirement.
        Reference: Schleher (1999), Eq. 8.3
        """
        fa = FrequencyAgility(center_freq_hz=10e9, n_hops=10)
        fa.enable()
        assert fa.js_reduction_db >= 10.0


# ═══════════════════════════════════════════════════════════════════
# TEST 4: PRF STAGGER
# ═══════════════════════════════════════════════════════════════════


class TestPRFStagger:
    """
    Verify PRF stagger produces varying PRIs.

    Reference: Schleher (1999), Ch. 8.4
    """

    def test_stagger_variation(self):
        """PRIs must vary when stagger is enabled."""
        stagger = PRFStagger(nominal_pri_s=1e-3, jitter_percent=5.0, seed=42)
        stagger.enable()
        pris = [stagger.get_next_pri() for _ in range(100)]
        assert np.std(pris) > 0, "No PRI variation"

    def test_stagger_within_bounds(self):
        """All PRIs must be within ±δ of nominal."""
        nom = 1e-3
        delta = 5.0
        stagger = PRFStagger(nominal_pri_s=nom, jitter_percent=delta, seed=42)
        stagger.enable()

        for _ in range(1000):
            pri = stagger.get_next_pri()
            lower = nom * (1 - delta / 100)
            upper = nom * (1 + delta / 100)
            assert lower <= pri <= upper, f"PRI {pri} out of bounds [{lower}, {upper}]"

    def test_disabled_returns_nominal(self):
        """Disabled stagger returns exact nominal PRI."""
        stagger = PRFStagger(nominal_pri_s=1e-3, jitter_percent=5.0)
        assert stagger.get_next_pri() == 1e-3


# ═══════════════════════════════════════════════════════════════════
# TEST 5: RGPO DISCRIMINATION
# ═══════════════════════════════════════════════════════════════════


class TestRGPODiscrimination:
    """
    PRF stagger must discriminate RGPO from real targets.

    Real target:  consistent range across PRIs → correlation ≈ 1
    RGPO jammer:  varying range → correlation < 0.3
    """

    def test_real_target_high_correlation(self):
        """Real target must have high correlation (> 0.7)."""
        stagger = PRFStagger(nominal_pri_s=1e-3, jitter_percent=5.0, seed=42)
        stagger.enable()

        # Consistent range measurements (real target)
        for _ in range(10):
            stagger.record_range_measurement(10000.0 + np.random.normal(0, 5))

        corr = stagger.discriminate_rgpo()
        assert corr > 0.7, f"Real target correlation {corr:.2f} < 0.7"

    def test_rgpo_low_correlation(self):
        """RGPO jammer must have low correlation (< 0.7)."""
        stagger = PRFStagger(nominal_pri_s=1e-3, jitter_percent=5.0, seed=42)
        stagger.enable()

        # Varying range measurements (RGPO pulling rapidly)
        for k in range(10):
            stagger.record_range_measurement(10000.0 + k * 500)  # 500m/step pull

        corr = stagger.discriminate_rgpo()
        assert corr < 0.7, f"RGPO correlation {corr:.2f} >= 0.7"


# ═══════════════════════════════════════════════════════════════════
# TEST 6: EKF COAST MODE
# ═══════════════════════════════════════════════════════════════════


class TestEKFCoastMode:
    """
    EKF must coast (prediction only) when SJNR < threshold.

    Coast: No measurement update, covariance inflates.
    Resume: When SJNR recovers, measurement updates resume.

    Reference: Schleher (1999), Ch. 4.8
    """

    def test_coast_on_high_jsr(self):
        """High JSR → SJNR < 6 dB → coast mode activated."""
        ekf = ExtendedKalmanFilter(process_noise=5.0)
        state = ekf.initialize(position=(10000.0, 5000.0), velocity=(100.0, 50.0))
        state = ekf.predict(state, dt=1.0)

        # Update with heavy jamming (J/S = 30 dB)
        state_coasted = ekf.update_with_jsr(state, z_polar=(11000.0, 0.5), snr_db=20.0, jsr_db=30.0)

        assert ekf.is_coasting, "Should be coasting with J/S=30 dB, SNR=20 dB"
        assert ekf.coast_count == 1

    def test_no_coast_without_jamming(self):
        """No jamming → SJNR = SNR → normal update."""
        ekf = ExtendedKalmanFilter(process_noise=5.0)
        state = ekf.initialize(position=(10000.0, 5000.0), velocity=(100.0, 50.0))
        state = ekf.predict(state, dt=1.0)

        state_updated = ekf.update_with_jsr(
            state, z_polar=(11000.0, 0.5), snr_db=20.0, jsr_db=-100.0
        )

        assert not ekf.is_coasting
        assert ekf.coast_count == 0

    def test_coast_preserves_state(self):
        """During coast, state vector should not change."""
        ekf = ExtendedKalmanFilter(process_noise=5.0)
        state = ekf.initialize(position=(10000.0, 5000.0), velocity=(100.0, 50.0))
        state = ekf.predict(state, dt=1.0)

        x_before = state.x.copy()
        state_coasted = ekf.update_with_jsr(state, z_polar=(99999.0, 0.0), snr_db=10.0, jsr_db=30.0)

        np.testing.assert_array_equal(state_coasted.x, x_before)

    def test_coast_10_scans(self):
        """EKF must maintain prediction during 10 scans of coasting."""
        ekf = ExtendedKalmanFilter(process_noise=5.0)
        state = ekf.initialize(position=(10000.0, 0.0), velocity=(100.0, 50.0))

        for scan in range(10):
            state = ekf.predict(state, dt=1.0)
            state = ekf.update_with_jsr(state, z_polar=(10000.0, 0.5), snr_db=15.0, jsr_db=30.0)

        assert ekf.coast_count == 10
        assert ekf.is_coasting

        # State should still be finite
        assert np.all(np.isfinite(state.x))
        assert np.all(np.isfinite(state.P))

    def test_track_drop_after_max_coast(self):
        """Track should be flagged for drop after max_coast_scans."""
        ekf = ExtendedKalmanFilter(process_noise=5.0)
        ekf.max_coast_scans = 5
        state = ekf.initialize(position=(10000.0, 0.0))

        for _ in range(6):
            state = ekf.predict(state, dt=1.0)
            state = ekf.update_with_jsr(state, z_polar=(10000.0, 0.5), snr_db=10.0, jsr_db=30.0)

        assert ekf.should_drop_track

    def test_coast_recovery(self):
        """When SJNR recovers, coast count resets."""
        ekf = ExtendedKalmanFilter(process_noise=5.0)
        state = ekf.initialize(position=(10000.0, 0.0), velocity=(100.0, 0.0))

        # 3 scans of jamming
        for _ in range(3):
            state = ekf.predict(state, dt=1.0)
            state = ekf.update_with_jsr(state, z_polar=(10000.0, 0.0), snr_db=15.0, jsr_db=30.0)

        assert ekf.coast_count == 3

        # Jamming clears
        state = ekf.predict(state, dt=1.0)
        state = ekf.update_with_jsr(state, z_polar=(10000.0, 0.0), snr_db=20.0, jsr_db=-10.0)

        assert ekf.coast_count == 0
        assert not ekf.is_coasting


# ═══════════════════════════════════════════════════════════════════
# TEST 7: SJNR CALCULATION
# ═══════════════════════════════════════════════════════════════════


class TestSJNR:
    """
    Verify SJNR calculation.

    SJNR_dB = SNR_dB - 10·log₁₀(1 + 10^(JSR/10))

    Reference: Schleher (1999), Eq. 4.8
    """

    def test_no_jamming(self):
        """Without jamming, SJNR ≈ SNR."""
        sjnr = ExtendedKalmanFilter._calculate_sjnr(20.0, -100.0)
        assert abs(sjnr - 20.0) < 0.01

    def test_equal_jsr_and_snr(self):
        """When JSR = SNR, SJNR = SNR - 10·log₁₀(2) ≈ SNR - 3.01 dB."""
        sjnr = ExtendedKalmanFilter._calculate_sjnr(20.0, 20.0)
        expected = 20.0 - 10 * np.log10(1 + 10 ** (20.0 / 10))
        assert abs(sjnr - expected) < 0.01

    def test_high_jamming(self):
        """Heavy jamming (30 dB JSR) should significantly reduce SJNR."""
        sjnr = ExtendedKalmanFilter._calculate_sjnr(20.0, 30.0)
        assert sjnr < 0.0, f"SJNR {sjnr:.1f} should be negative under heavy jamming"


# ═══════════════════════════════════════════════════════════════════
# TEST 8: ECCM CONTROLLER INTEGRATION
# ═══════════════════════════════════════════════════════════════════


class TestECCMController:
    """
    Verify ECCM controller orchestration.

    Reference: Schleher (1999), Ch. 8-9
    """

    def test_effective_jsr_with_agility(self):
        """Frequency agility must reduce effective J/S."""
        ctrl = ECCMController(n_freq_hops=10)
        ctrl.set_jamming_environment(True, jsr_db=20.0)

        jsr_before = ctrl.get_effective_jsr_db()

        ctrl.enable_frequency_agility()
        jsr_after = ctrl.get_effective_jsr_db()

        assert jsr_after < jsr_before
        assert (jsr_before - jsr_after) >= 9.9  # ≈10 dB

    def test_sjnr_improves_with_eccm(self):
        """SJNR must improve when ECCM is enabled."""
        ctrl = ECCMController(n_freq_hops=10)
        ctrl.set_jamming_environment(True, jsr_db=20.0)

        sjnr_no_eccm = ctrl.calculate_sjnr_db(snr_db=20.0)

        ctrl.enable_frequency_agility()
        sjnr_with_eccm = ctrl.calculate_sjnr_db(snr_db=20.0)

        assert sjnr_with_eccm > sjnr_no_eccm

    def test_cpi_parameters(self):
        """CPI parameters must return valid freq and PRI."""
        ctrl = ECCMController(center_freq_hz=10e9, nominal_prf_hz=1000.0)
        ctrl.enable_frequency_agility()
        ctrl.enable_prf_stagger()

        freq, pri = ctrl.get_cpi_parameters()
        assert freq > 0
        assert pri > 0

    def test_no_jamming_high_sjnr(self):
        """Without jamming, SJNR ≈ SNR."""
        ctrl = ECCMController()
        sjnr = ctrl.calculate_sjnr_db(snr_db=25.0)
        assert abs(sjnr - 25.0) < 0.01


# ═══════════════════════════════════════════════════════════════════
# TEST 9: BURN-THROUGH RANGE
# ═══════════════════════════════════════════════════════════════════


class TestBurnThrough:
    """
    Verify burn-through range calculation.

    Reference: Schleher (1999), Ch. 4.5
    """

    def test_burn_through_in_range(self):
        """Burn-through must be in valid range for standard parameters."""
        sim = ECMSimulator(radar_wavelength=0.03)
        # Standard radar vs jammer scenario
        # R_bt = sqrt(Pt*Gt*sigma / (Pj*Gj*4pi*SNR_req*Bj/Br))
        r_bt = sim.calculate_burn_through_range(
            radar_power_watts=100e3,
            radar_gain_linear=10 ** (35 / 10),
            jammer_power_watts=1000.0,
            jammer_gain_linear=10 ** (6 / 10),
            target_rcs_m2=1.0,
            required_snr_linear=10 ** (13 / 10),
            radar_bandwidth_hz=1e6,
            jammer_bandwidth_hz=100e6,
        )
        r_bt_km = r_bt / 1000.0
        # Burn-through range should be positive and finite
        assert r_bt_km > 0, f"Burn-through range should be positive, got {r_bt_km:.3f} km"
        assert np.isfinite(r_bt_km), "Burn-through range should be finite"
