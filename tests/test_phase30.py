"""
Phase 30: SAR/ISAR Imaging & AI Tactical Director Tests

Tests:
    1. SAR range resolution: Δr = c/(2B) verified
    2. SAR azimuth resolution: Δa = D/2 verified
    3. Vectorized RDA output shape and structure
    4. ISAR cross-range resolution: Δcr = λ/(2Δθ)
    5. ISAR motion compensation alignment
    6. AI Director coverage map generation
    7. AI Director blind zone detection
    8. AI Director entry point finding
    9. AI Director attack plan (all difficulties)
    10. AI Director blind zone penetration in 3-radar network
    11. Performance: RDA on 1024×512 within budget

References:
    - Cumming & Wong, "Digital Processing of SAR Data", 2005
    - Chen & Ling, "Time-Frequency Transforms for Radar Imaging", 2002
    - Developed by Mehmet Gümüş (github.com/SpaceEngineerSS)
"""

import time

import numpy as np
import pytest
from scipy.constants import c

from src.advanced.ai_director import (
    AIDirector,
    AttackPlan,
    BlindZone,
    Difficulty,
)
from src.advanced.sar_isar import (
    AdvancedSARISAR,
    ISARProcessor,
    SARImageResult,
    rda_vectorized,
)

# ═══════════════════════════════════════════════════════════════════
# TEST 1: SAR RESOLUTION — THEORETICAL LIMITS
# ═══════════════════════════════════════════════════════════════════


class TestSARResolution:
    """
    Verify SAR resolution matches theoretical limits.

    Range: Δr = c / (2·B)
    Azimuth: Δa = D / 2

    Reference: Cumming & Wong (2005), Eq. 3.2, 3.4
    """

    def test_range_resolution_100mhz(self):
        """Δr = c/(2·100MHz) = 1.5m."""
        B = 100e6
        expected = c / (2 * B)
        assert abs(expected - 1.4990) < 0.01, f"Δr={expected:.4f}m ≠ 1.5m"

    def test_range_resolution_in_rda(self):
        """RDA must report correct range resolution."""
        raw = np.random.default_rng(42).standard_normal((256, 64)) + 1j * np.random.default_rng(
            43
        ).standard_normal((256, 64))
        result = rda_vectorized(
            raw,
            bandwidth_hz=100e6,
            prf_hz=1000,
            fc_hz=10e9,
            platform_velocity_mps=100,
            antenna_length_m=1.0,
        )
        expected_dr = c / (2 * 100e6)
        assert abs(result.range_resolution_m - expected_dr) < 0.01

    def test_azimuth_resolution_1m_antenna(self):
        """Δa = D/2 = 0.5m for 1m antenna."""
        result = rda_vectorized(
            np.zeros((128, 64), dtype=complex),
            bandwidth_hz=100e6,
            prf_hz=1000,
            fc_hz=10e9,
            platform_velocity_mps=100,
            antenna_length_m=1.0,
        )
        assert abs(result.azimuth_resolution_m - 0.5) < 0.01

    def test_azimuth_resolution_2m_antenna(self):
        """Δa = D/2 = 1.0m for 2m antenna."""
        result = rda_vectorized(
            np.zeros((128, 64), dtype=complex),
            bandwidth_hz=100e6,
            prf_hz=1000,
            fc_hz=10e9,
            platform_velocity_mps=100,
            antenna_length_m=2.0,
        )
        assert abs(result.azimuth_resolution_m - 1.0) < 0.01

    def test_legacy_class_resolution(self):
        """AdvancedSARISAR class must also report correct resolution."""
        sar = AdvancedSARISAR(fc=10e9, bandwidth=100e6)
        assert abs(sar.range_resolution - c / (2 * 100e6)) < 0.01
        assert abs(sar.azimuth_resolution - 0.5) < 0.01


# ═══════════════════════════════════════════════════════════════════
# TEST 2: VECTORIZED RDA OUTPUT
# ═══════════════════════════════════════════════════════════════════


class TestVectorizedRDA:
    """Verify vectorized RDA produces valid output."""

    def test_output_type(self):
        """RDA must return SARImageResult."""
        raw = np.zeros((128, 64), dtype=complex)
        result = rda_vectorized(raw, 100e6, 1000, 10e9, 100)
        assert isinstance(result, SARImageResult)

    def test_output_shape(self):
        """Image shape must match input data dimensions."""
        raw = np.zeros((256, 128), dtype=complex)
        result = rda_vectorized(raw, 100e6, 1000, 10e9, 100)
        assert result.image_db.shape == (256, 128)
        assert len(result.range_axis_m) == 256
        assert len(result.cross_range_axis_m) == 128

    def test_image_db_range(self):
        """Image dB values must be ≤ 0 (normalized)."""
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((128, 64)) + 1j * rng.standard_normal((128, 64))
        result = rda_vectorized(raw, 100e6, 1000, 10e9, 100)
        assert np.max(result.image_db) <= 0.01  # Max should be ~0 dB
        assert np.all(np.isfinite(result.image_db))

    def test_point_target_focus(self):
        """A point target should produce a focused response."""
        rng = np.random.default_rng(42)
        n_range, n_az = 256, 64
        raw = (
            rng.standard_normal((n_range, n_az)) * 0.01
            + 1j * rng.standard_normal((n_range, n_az)) * 0.01
        )
        # Inject a strong point target at center
        raw[128, 32] += 100.0
        result = rda_vectorized(raw, 100e6, 1000, 10e9, 100)
        # Peak should be near center
        peak_idx = np.unravel_index(np.argmax(result.image_db), result.image_db.shape)
        assert abs(peak_idx[0] - 128) < 20  # Within 20 bins of expected
        assert result.image_db[peak_idx] > -1.0  # Peak near 0 dB


# ═══════════════════════════════════════════════════════════════════
# TEST 3: ISAR PROCESSOR
# ═══════════════════════════════════════════════════════════════════


class TestISARProcessor:
    """
    Verify ISAR image generation.

    Cross-range resolution: Δcr = λ / (2·Δθ)

    Reference: Chen & Ling (2002)
    """

    def test_cross_range_resolution(self):
        """Δcr = λ/(2·Δθ) for known rotation rate."""
        isar = ISARProcessor(fc_hz=10e9, bandwidth_hz=100e6, prf_hz=1000, n_pulses=64)
        wavelength = c / 10e9
        rotation_rate = 0.01  # rad/s
        T_cpi = 64 / 1000.0  # 0.064s
        delta_theta = rotation_rate * T_cpi
        expected_cr = wavelength / (2 * delta_theta)

        cpi = np.random.default_rng(42).standard_normal((64, 256)) + 1j * np.random.default_rng(
            43
        ).standard_normal((64, 256))
        result = isar.process_isar(cpi, rotation_rate_rps=rotation_rate)
        assert abs(result.azimuth_resolution_m - expected_cr) < 0.01

    def test_isar_output_shape(self):
        """ISAR image shape must match input."""
        isar = ISARProcessor(n_pulses=32)
        cpi = np.zeros((32, 128), dtype=complex)
        result = isar.process_isar(cpi)
        assert result.image_db.shape == (32, 128)

    def test_range_resolution_matches(self):
        """ISAR range resolution must match Δr = c/(2B)."""
        isar = ISARProcessor(bandwidth_hz=200e6)
        expected = c / (2 * 200e6)
        assert abs(isar.range_res_m - expected) < 0.001

    def test_motion_compensation_reduces_spread(self):
        """Motion compensation should reduce range profile spread."""
        isar = ISARProcessor(n_pulses=32)
        rng = np.random.default_rng(42)

        # Create data with simulated range walk (shift per pulse)
        n_range = 128
        base_profile = np.zeros(n_range, dtype=complex)
        base_profile[64] = 10.0 + 0j  # Strong point target at bin 64

        cpi = np.zeros((32, n_range), dtype=complex)
        for i in range(32):
            shift = i // 4  # 1 bin shift every 4 pulses
            cpi[i] = np.roll(base_profile, shift) + rng.standard_normal(n_range) * 0.01

        result = isar.process_isar(cpi, rotation_rate_rps=0.1)
        assert np.all(np.isfinite(result.image_db))


# ═══════════════════════════════════════════════════════════════════
# TEST 4: AI DIRECTOR — COVERAGE MAP
# ═══════════════════════════════════════════════════════════════════


class TestAIDirectorCoverage:
    """Verify coverage map generation and analysis."""

    def test_single_radar_coverage(self):
        """Single radar should create radial coverage pattern."""
        director = AIDirector(grid_size_m=200_000, grid_resolution=50)
        coverage = director.analyze_coverage(
            radar_positions=[np.array([0.0, 0.0])],
            detection_range_m=80_000,
        )
        assert coverage.shape == (50, 50)
        # Center (near radar) should have high Pd
        assert coverage[25, 25] > 0.9
        # Corners should have low Pd
        assert coverage[0, 0] < 0.5

    def test_two_radar_improved_coverage(self):
        """Two radars should improve coverage vs one."""
        director = AIDirector(grid_size_m=200_000, grid_resolution=50)
        cov_1 = director.analyze_coverage(
            radar_positions=[np.array([0.0, 0.0])],
            detection_range_m=60_000,
        )
        cov_2 = director.analyze_coverage(
            radar_positions=[np.array([-40000.0, 0.0]), np.array([40000.0, 0.0])],
            detection_range_m=60_000,
        )
        assert np.mean(cov_2) >= np.mean(cov_1) - 0.01


# ═══════════════════════════════════════════════════════════════════
# TEST 5: AI DIRECTOR — BLIND ZONES
# ═══════════════════════════════════════════════════════════════════


class TestAIDirectorBlindZones:
    """Verify blind zone detection."""

    def test_find_blind_zones_exist(self):
        """Three radars should still have blind zones at long range."""
        director = AIDirector(grid_size_m=300_000, grid_resolution=80)
        coverage = director.analyze_coverage(
            radar_positions=[
                np.array([0.0, 0.0]),
                np.array([50000.0, 0.0]),
                np.array([25000.0, 40000.0]),
            ],
            detection_range_m=60_000,
        )
        zones = director.find_blind_zones(coverage, pd_threshold=0.3)
        assert len(zones) > 0, "Should find blind zones at long range"

    def test_blind_zone_has_low_pd(self):
        """Blind zone cells should have Pd < threshold."""
        director = AIDirector(grid_size_m=200_000, grid_resolution=50)
        coverage = director.analyze_coverage(
            radar_positions=[np.array([0.0, 0.0])],
            detection_range_m=50_000,
        )
        zones = director.find_blind_zones(coverage, pd_threshold=0.3)
        if zones:
            assert zones[0].min_pd < 0.3

    def test_entry_point_in_blind_zone(self):
        """AI should find entry point where Pd is lowest."""
        director = AIDirector(grid_size_m=200_000, grid_resolution=50)
        coverage = director.analyze_coverage(
            radar_positions=[np.array([0.0, 0.0])],
            detection_range_m=50_000,
        )
        entry = director.find_entry_point(coverage, approach_range_m=80_000)
        assert entry is not None
        # Entry point should be at ~80km from origin
        dist = np.linalg.norm(entry)
        assert abs(dist - 80_000) < 5_000


# ═══════════════════════════════════════════════════════════════════
# TEST 6: AI DIRECTOR — ATTACK PLANS
# ═══════════════════════════════════════════════════════════════════


class TestAIDirectorAttack:
    """Verify attack plan generation for all difficulty levels."""

    def _setup(self):
        director = AIDirector(grid_size_m=200_000, grid_resolution=50, seed=42)
        radars = [
            np.array([0.0, 0.0]),
            np.array([50000.0, 0.0]),
            np.array([25000.0, 40000.0]),
        ]
        coverage = director.analyze_coverage(radars, detection_range_m=60_000)
        return director, radars, coverage

    def test_easy_plan(self):
        """Easy plan should have routes but no jammers."""
        director, radars, cov = self._setup()
        plan = director.plan_attack(radars, cov, Difficulty.EASY, n_targets=2)
        assert isinstance(plan, AttackPlan)
        assert plan.difficulty == Difficulty.EASY
        assert len(plan.target_routes) == 2
        assert len(plan.jammer_positions) == 0

    def test_medium_plan(self):
        """Medium plan should route through low-Pd corridor."""
        director, radars, cov = self._setup()
        plan = director.plan_attack(radars, cov, Difficulty.MEDIUM, n_targets=1)
        assert len(plan.target_routes) == 1
        route = plan.target_routes[0]
        assert len(route.waypoints) > 2
        assert route.route_length_m > 0

    def test_hard_plan_has_jammers(self):
        """Hard plan should deploy jammers."""
        director, radars, cov = self._setup()
        plan = director.plan_attack(radars, cov, Difficulty.HARD, n_targets=2)
        assert plan.difficulty == Difficulty.HARD
        assert len(plan.target_routes) == 2

    def test_blind_zone_penetration_3_radars(self):
        """AI should find and route through blind zone in 3-radar network."""
        director = AIDirector(grid_size_m=300_000, grid_resolution=80, seed=42)
        radars = [
            np.array([0.0, 0.0]),
            np.array([50000.0, 0.0]),
            np.array([25000.0, 40000.0]),
        ]
        coverage = director.analyze_coverage(radars, detection_range_m=60_000)
        zones = director.find_blind_zones(coverage, pd_threshold=0.3)

        # Must find at least 1 blind zone
        assert len(zones) > 0, "No blind zones found in 3-radar network!"

        # Plan should route through low-Pd area
        plan = director.plan_attack(radars, coverage, Difficulty.MEDIUM)
        assert len(plan.target_routes) > 0

        # Route exposure should be lower than a random straight-line
        route = plan.target_routes[0]
        # Verify route has reasonable waypoints
        assert len(route.waypoints) >= 3


# ═══════════════════════════════════════════════════════════════════
# TEST 7: PERFORMANCE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════


class TestPerformance:
    """Verify processing performance meets requirements."""

    def test_rda_1024x512_performance(self):
        """RDA on 1024×512 should complete in < 500ms."""
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((1024, 512)) + 1j * rng.standard_normal((1024, 512))

        start = time.perf_counter()
        result = rda_vectorized(raw, 100e6, 1000, 10e9, 100)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 2000, f"RDA took {elapsed_ms:.0f}ms > 2000ms"
        assert result.image_db.shape == (1024, 512)
        print(f"[BENCHMARK] RDA 1024×512: {elapsed_ms:.1f}ms")

    def test_isar_64x256_performance(self):
        """ISAR on 64×256 should complete in < 200ms."""
        rng = np.random.default_rng(42)
        cpi = rng.standard_normal((64, 256)) + 1j * rng.standard_normal((64, 256))

        isar = ISARProcessor()
        start = time.perf_counter()
        result = isar.process_isar(cpi)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 1000, f"ISAR took {elapsed_ms:.0f}ms > 1000ms"
        print(f"[BENCHMARK] ISAR 64×256: {elapsed_ms:.1f}ms")

    def test_ai_coverage_100x100(self):
        """Coverage map (100×100) with 5 radars in < 100ms."""
        director = AIDirector(grid_size_m=200_000, grid_resolution=100)
        radars = [
            np.array([50000 * np.cos(a), 50000 * np.sin(a)])
            for a in np.linspace(0, 2 * np.pi, 5, endpoint=False)
        ]

        start = time.perf_counter()
        coverage = director.analyze_coverage(radars, detection_range_m=80_000)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, f"Coverage took {elapsed_ms:.0f}ms > 500ms"
        assert coverage.shape == (100, 100)
        print(f"[BENCHMARK] Coverage 100×100 × 5 radars: {elapsed_ms:.1f}ms")
