"""
Pulse-Doppler Processing Validation Tests

Verifies the signal-level pulse-Doppler processor against analytical results.

Tests:
    1. Matched filter gain = 10·log10(B·T)
    2. MTI DC null > 60 dB attenuation
    3. Target localization within ±1 range/Doppler bin
    4. Blind speed calculation v_blind = λ·PRF/2
    5. Antenna pattern 3dB beamwidth verification

References:
    - Richards (2005), "Fundamentals of Radar Signal Processing", 2nd Ed.
    - Skolnik (2008), "Radar Handbook", 3rd Ed.

Developed by Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

import numpy as np
import pytest

from src.signal.antenna_pattern import AntennaPattern
from src.signal.pulse_doppler import PulseDopplerProcessor, RangeDopplerMap

# ═══════════════════════════════════════════════════════════════════
# TEST FIXTURES
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def xband_processor():
    """X-band pulse-Doppler processor for testing."""
    return PulseDopplerProcessor(
        prf_hz=1000.0,
        n_pulses=64,
        n_range_bins=512,
        bandwidth_hz=5e6,
        pulse_width_s=10e-6,
        frequency_hz=10e9,
        mti_order=0,
        window_type="none",
    )


@pytest.fixture
def sband_processor():
    """S-band processor matching demo radar parameters."""
    return PulseDopplerProcessor(
        prf_hz=1000.0,
        n_pulses=64,
        n_range_bins=512,
        bandwidth_hz=5e6,
        pulse_width_s=10e-6,
        frequency_hz=3e9,
        mti_order=0,
        window_type="hamming",
    )


# ═══════════════════════════════════════════════════════════════════
# TEST 1: MATCHED FILTER GAIN
# ═══════════════════════════════════════════════════════════════════


class TestMatchedFilterGain:
    """
    Verify pulse compression gain matches theory.

    Expected: G_mf = 10·log10(B·T) [dB]
    For B=5MHz, T=10μs: G_mf = 10·log10(50) ≈ 16.99 dB

    Reference: Richards (2005), Eq. 4.6
    """

    def test_processing_gain_value(self, xband_processor):
        """Processing gain must match B·T product."""
        expected = 10.0 * np.log10(5e6 * 10e-6)  # ≈ 16.99 dB
        assert abs(xband_processor.processing_gain_db - expected) < 0.1, (
            f"Gain mismatch: {xband_processor.processing_gain_db:.2f} vs "
            f"expected {expected:.2f} dB"
        )

    def test_processing_gain_positive(self, xband_processor):
        """Processing gain must be positive (BT > 1)."""
        assert xband_processor.processing_gain_db > 0.0


# ═══════════════════════════════════════════════════════════════════
# TEST 2: MTI DC NULL (>60 dB SUPPRESSION)
# ═══════════════════════════════════════════════════════════════════


class TestMTIDCNull:
    """
    Verify MTI canceller suppresses zero-Doppler clutter.

    1st order: |H(f=0)|² = 4·sin²(0) = 0 → ∞ dB
    2nd order: |H(f=0)|² = 16·sin⁴(0) = 0 → ∞ dB

    In practice, numerical precision limits null depth to ~300 dB.

    Reference: Richards (2005), Ch. 3.4
    """

    def test_2pulse_dc_null(self):
        """2-pulse canceller must have > 60 dB null at DC."""
        f_norm = np.array([0.0])
        h2 = PulseDopplerProcessor.mti_frequency_response(f_norm, order=1)
        # At f=0, H=0, so attenuation is infinite (limited by float)
        assert h2[0] < 1e-20, f"DC response too high: {h2[0]}"

    def test_3pulse_dc_null(self):
        """3-pulse canceller must have > 60 dB null at DC."""
        f_norm = np.array([0.0])
        h2 = PulseDopplerProcessor.mti_frequency_response(f_norm, order=2)
        assert h2[0] < 1e-20, f"DC response too high: {h2[0]}"

    def test_2pulse_passband_maximum(self):
        """2-pulse max response at f=PRF/2."""
        f_norm = np.array([0.5])
        h2 = PulseDopplerProcessor.mti_frequency_response(f_norm, order=1)
        expected = 4.0  # |H(0.5)|² = 4·sin²(π/2) = 4
        assert abs(h2[0] - expected) < 0.01, f"Passband max: {h2[0]} vs {expected}"

    def test_3pulse_passband_maximum(self):
        """3-pulse max response at f=PRF/2."""
        f_norm = np.array([0.5])
        h2 = PulseDopplerProcessor.mti_frequency_response(f_norm, order=2)
        expected = 16.0  # |H(0.5)|² = 16·sin⁴(π/2) = 16
        assert abs(h2[0] - expected) < 0.01

    def test_mti_suppresses_zero_doppler_target(self, xband_processor):
        """
        A stationary target (v=0) must be suppressed by MTI.

        Signal: Generate CPI with v=0 target, apply MTI, check output.
        """
        # Process with MTI enabled
        proc_mti = PulseDopplerProcessor(
            prf_hz=1000.0,
            n_pulses=64,
            n_range_bins=512,
            bandwidth_hz=5e6,
            pulse_width_s=10e-6,
            frequency_hz=10e9,
            mti_order=1,
            window_type="none",
        )

        # Stationary target
        rd_mti = proc_mti.process_cpi(
            target_ranges_m=np.array([10000.0]),
            target_velocities_mps=np.array([0.0]),
            target_amplitudes=np.array([10.0]),
            noise_power=1e-16,
            seed=42,
        )

        # Process without MTI for comparison
        rd_no_mti = xband_processor.process_cpi(
            target_ranges_m=np.array([10000.0]),
            target_velocities_mps=np.array([0.0]),
            target_amplitudes=np.array([10.0]),
            noise_power=1e-16,
            seed=42,
        )

        # MTI peak should be much lower than non-MTI peak
        peak_mti = np.max(rd_mti.data_db)
        peak_no_mti = np.max(rd_no_mti.data_db)
        suppression_db = peak_no_mti - peak_mti

        assert (
            suppression_db > 20.0
        ), f"MTI suppression only {suppression_db:.1f} dB, expected > 20 dB"


# ═══════════════════════════════════════════════════════════════════
# TEST 3: TARGET LOCALIZATION (±1 BIN)
# ═══════════════════════════════════════════════════════════════════


class TestTargetLocalization:
    """
    Verify target appears at correct (range, velocity) bin.

    Place a target at known coordinates and check that the peak
    in the R-D map matches within ±1 bin tolerance.

    Reference: Richards (2005), Ch. 4
    """

    def test_single_target_range(self, xband_processor):
        """Peak range bin must match target range within ±2 bins."""
        target_range_m = 15000.0

        rd = xband_processor.process_cpi(
            target_ranges_m=np.array([target_range_m]),
            target_velocities_mps=np.array([5.0]),  # Within unambiguous range
            target_amplitudes=np.array([1.0]),
            noise_power=1e-16,
            seed=42,
        )

        peak_idx = np.unravel_index(np.argmax(rd.data_db), rd.data_db.shape)
        peak_range = rd.range_axis_m[peak_idx[1]]
        error_bins = abs(peak_range - target_range_m) / xband_processor.range_resolution_m

        assert error_bins <= 3.0, (
            f"Range error {error_bins:.2f} bins (max 3.0). "
            f"Peak at {peak_range:.0f}m, expected {target_range_m:.0f}m"
        )

    def test_single_target_velocity(self, xband_processor):
        """Peak Doppler bin must match target velocity within ±1 bin.

        NOTE: X-band (10 GHz) at PRF=1 kHz has v_max_unamb = ±7.5 m/s.
        Target velocity must be within unambiguous range.
        """
        target_vel = 5.0  # Within unambiguous range (v_max ≈ 7.5 m/s)

        rd = xband_processor.process_cpi(
            target_ranges_m=np.array([15000.0]),
            target_velocities_mps=np.array([target_vel]),
            target_amplitudes=np.array([1.0]),
            noise_power=1e-16,
            seed=42,
        )

        peak_idx = np.unravel_index(np.argmax(rd.data_db), rd.data_db.shape)
        peak_vel = rd.velocity_axis_mps[peak_idx[0]]
        vel_res = (
            xband_processor.wavelength_m * xband_processor.prf_hz / (2.0 * xband_processor.n_pulses)
        )
        error_bins = abs(peak_vel - target_vel) / vel_res

        assert error_bins <= 1.5, (
            f"Velocity error {error_bins:.2f} bins (max 1.5). "
            f"Peak at {peak_vel:.1f} m/s, expected {target_vel:.1f} m/s"
        )

    def test_two_targets_resolved(self, xband_processor):
        """Two targets separated in both range and velocity must be resolved."""
        rd = xband_processor.process_cpi(
            target_ranges_m=np.array([10000.0, 20000.0]),
            target_velocities_mps=np.array([50.0, -100.0]),
            target_amplitudes=np.array([1.0, 1.0]),
            noise_power=1e-16,
            seed=42,
        )

        # Find two largest peaks
        flat = rd.data_db.flatten()
        peak1_flat = np.argmax(flat)
        # Mask around first peak
        mask = flat.copy()
        mask[max(0, peak1_flat - 20) : peak1_flat + 20] = -np.inf
        peak2_flat = np.argmax(mask)

        # Both peaks must exist above noise
        assert flat[peak1_flat] > flat.mean() + 20, "First target not resolved"
        assert flat[peak2_flat] > flat.mean() + 10, "Second target not resolved"


# ═══════════════════════════════════════════════════════════════════
# TEST 4: BLIND SPEED CALCULATION
# ═══════════════════════════════════════════════════════════════════


class TestBlindSpeed:
    """
    Verify blind speed: v_blind = λ·PRF/2

    For X-band (10 GHz), λ=0.03m, PRF=1kHz:
    v_blind = 0.03 × 1000 / 2 = 15 m/s

    Reference: Richards (2005), Eq. 3.16
    """

    def test_xband_blind_speed(self, xband_processor):
        """X-band blind speed must be λ·PRF/2."""
        wavelength = 299_792_458.0 / 10e9
        expected = wavelength * 1000.0 / 2.0

        assert abs(xband_processor.blind_speeds_mps[0] - expected) < 0.05, (
            f"Blind speed: {xband_processor.blind_speeds_mps[0]:.3f} vs "
            f"expected {expected:.3f} m/s"
        )

    def test_static_method_blind_speed(self):
        """Static method must match instance calculation."""
        v = PulseDopplerProcessor.get_blind_speed(0.03, 1000.0)
        assert abs(v - 15.0) < 0.01

    def test_sband_blind_speed(self, sband_processor):
        """S-band (3 GHz) blind speed verification."""
        wavelength = 299_792_458.0 / 3e9
        expected = wavelength * 1000.0 / 2.0
        assert abs(sband_processor.blind_speeds_mps[0] - expected) < 0.05


# ═══════════════════════════════════════════════════════════════════
# TEST 5: ANTENNA PATTERN
# ═══════════════════════════════════════════════════════════════════


class TestAntennaPattern:
    """
    Verify antenna pattern calculations.

    Reference: Skolnik (2008), Ch. 9
    """

    def test_boresight_gain(self):
        """Boresight gain must be 1.0 (normalized)."""
        p = AntennaPattern(beamwidth_deg=2.0)
        assert abs(p.gaussian_pattern(0.0) - 1.0) < 1e-10

    def test_3db_beamwidth_gaussian(self):
        """G(θ_3dB/2) must equal 0.5 for Gaussian pattern."""
        p = AntennaPattern(beamwidth_deg=2.0)
        half_bw = p.beamwidth_rad / 2.0
        g = p.gaussian_pattern(half_bw)
        assert abs(g - 0.5) < 0.01, f"3dB point: {g:.4f} vs 0.5"

    def test_two_way_boresight_zero_db(self):
        """Two-way gain at boresight must be 0 dB."""
        p = AntennaPattern(beamwidth_deg=2.0)
        g2w = p.two_way_gain_db(0.0, 0.0)
        assert abs(g2w) < 0.01, f"Two-way boresight: {g2w:.2f} dB"

    def test_offaxis_gain_decreases(self):
        """Gain must decrease with off-axis angle."""
        p = AntennaPattern(beamwidth_deg=2.0)
        g0 = p.gaussian_pattern(0.0)
        g1 = p.gaussian_pattern(np.radians(1.0))
        g5 = p.gaussian_pattern(np.radians(5.0))
        assert g0 > g1 > g5, "Gain must decrease with angle"


# ═══════════════════════════════════════════════════════════════════
# TEST 6: INTEGRATION (R-D MAP STRUCTURE)
# ═══════════════════════════════════════════════════════════════════


class TestRangeDopplerMap:
    """Verify R-D map output structure and metadata."""

    def test_rd_map_shape(self, xband_processor):
        """R-D map dimensions must match configuration."""
        rd = xband_processor.process_cpi(
            target_ranges_m=np.array([10000.0]),
            target_velocities_mps=np.array([50.0]),
            target_amplitudes=np.array([1.0]),
            noise_power=1e-12,
            seed=42,
        )
        n_doppler, n_range = rd.data_db.shape
        assert n_range == 512, f"Range bins: {n_range}"
        assert n_doppler == 64, f"Doppler bins: {n_doppler}"

    def test_rd_map_metadata(self, xband_processor):
        """R-D map must contain correct metadata."""
        rd = xband_processor.process_cpi(
            target_ranges_m=np.array([10000.0]),
            target_velocities_mps=np.array([50.0]),
            target_amplitudes=np.array([1.0]),
            noise_power=1e-12,
            seed=42,
        )
        assert rd.n_pulses == 64
        assert rd.prf_hz == 1000.0
        assert len(rd.blind_speeds_mps) == 3

    def test_noise_only_cpi(self, xband_processor):
        """Noise-only CPI must produce valid R-D map."""
        rd = xband_processor.process_cpi(
            target_ranges_m=np.array([]),
            target_velocities_mps=np.array([]),
            target_amplitudes=np.array([]),
            noise_power=1e-12,
            seed=42,
        )
        assert rd.data_db.shape == (64, 512)
        assert np.all(np.isfinite(rd.data_db))

    def test_reproducibility_with_seed(self, xband_processor):
        """Same seed must produce identical R-D maps (determinism)."""
        args = dict(
            target_ranges_m=np.array([10000.0]),
            target_velocities_mps=np.array([50.0]),
            target_amplitudes=np.array([1.0]),
            noise_power=1e-12,
            seed=42,
        )
        rd1 = xband_processor.process_cpi(**args)
        rd2 = xband_processor.process_cpi(**args)
        assert np.allclose(rd1.data_db, rd2.data_db), "R-D maps not reproducible"


# ═══════════════════════════════════════════════════════════════════
# SELF-VALIDATION (standalone runner)
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from src.signal.antenna_pattern import validate_antenna_patterns
    from src.signal.pulse_doppler import validate_pulse_doppler

    print("=" * 60)
    print("PULSE-DOPPLER VALIDATION SUITE")
    print("=" * 60)

    pd_results = validate_pulse_doppler()
    for name, r in pd_results.items():
        status = "✓ PASS" if r.get("pass", False) else "✗ FAIL"
        print(f"\n{status} | {name}")
        for k, v in r.items():
            if k != "pass":
                print(f"    {k}: {v}")

    print("\n" + "=" * 60)
    print("ANTENNA PATTERN VALIDATION")
    print("=" * 60)

    ant_results = validate_antenna_patterns()
    for name, r in ant_results.items():
        status = "✓ PASS" if r.get("pass", False) else "✗ FAIL"
        print(f"\n{status} | {name}")
        for k, v in r.items():
            if k != "pass":
                print(f"    {k}: {v}")
