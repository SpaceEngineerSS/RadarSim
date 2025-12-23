"""
RadarSim Simulation Validation Test Suite

Tests for simulation engine, kinematics, and detection logic.

Test ID | Description                    | Reference           | Tolerance
--------|--------------------------------|---------------------|------------
1       | Linear motion CV model         | x = x0 + v*t        | ±0.1 m
2       | Constant acceleration CA       | x = x0 + v*t + 0.5at²| ±0.1 m
3       | Detection at close range       | High SNR → detect   | True
4       | No detection at far range      | Low SNR → no detect | True

References:
    - Bar-Shalom (2001). "Estimation with Applications to Tracking"
    - Skolnik (2008). "Radar Handbook", 3rd Ed.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.objects import (
    Target, Radar, SimulationState, MotionModel, KinematicState
)
from src.simulation.engine import (
    SimulationEngine, validate_linear_motion, validate_detection_logic
)
from src.physics.rcs import SwerlingModel


# =============================================================================
# TEST 1: Linear Motion (Constant Velocity)
# =============================================================================

class TestLinearMotion:
    """
    Validate constant velocity motion model.
    
    Reference: Kinematic equation x = x0 + v*t
    """
    
    def test_linear_motion_100mps(self):
        """Target at v=100 m/s should be at x=1000 m after 10 s"""
        result = validate_linear_motion(
            velocity_mps=100.0,
            duration_s=10.0,
            dt=0.01
        )
        
        assert result['validation']['is_valid'], (
            f"Linear motion failed: expected x={result['computed_values']['expected_x']:.2f}, "
            f"got x={result['computed_values']['actual_x']:.2f}"
        )
    
    def test_linear_motion_300mps(self):
        """Target at v=300 m/s (aircraft speed) should be at x=3000 m after 10 s"""
        result = validate_linear_motion(
            velocity_mps=300.0,
            duration_s=10.0,
            dt=0.01
        )
        
        assert result['validation']['is_valid']
    
    def test_target_update_cv(self):
        """Direct test of Target.update() for CV model"""
        target = Target(
            target_id=1,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([100.0, 0.0, 0.0]),
            motion_model=MotionModel.CONSTANT_VELOCITY
        )
        
        # Update for 1 second
        for _ in range(100):
            target.update(0.01)
        
        assert target.position[0] == pytest.approx(100.0, abs=0.1)


# =============================================================================
# TEST 2: Constant Acceleration Motion
# =============================================================================

class TestConstantAcceleration:
    """
    Validate constant acceleration motion model.
    
    Reference: Kinematic equation x = x0 + v*t + 0.5*a*t²
    """
    
    def test_ca_motion(self):
        """Test CA model: x = 0.5*a*t² for zero initial velocity"""
        target = Target(
            target_id=1,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            acceleration=np.array([10.0, 0.0, 0.0]),  # 10 m/s² (about 1g)
            motion_model=MotionModel.CONSTANT_ACCELERATION
        )
        
        # Update for 10 seconds
        dt = 0.01
        for _ in range(1000):
            target.update(dt)
        
        # Expected: x = 0.5 * 10 * 10² = 500 m
        expected_x = 0.5 * 10.0 * 10.0**2
        actual_x = target.position[0]
        
        assert actual_x == pytest.approx(expected_x, abs=1.0), (
            f"CA motion failed: expected {expected_x}, got {actual_x}"
        )
    
    def test_ca_velocity_update(self):
        """Velocity should increase with acceleration"""
        target = Target(
            target_id=1,
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([0.0, 0.0, 0.0]),
            acceleration=np.array([10.0, 0.0, 0.0]),
            motion_model=MotionModel.CONSTANT_ACCELERATION
        )
        
        # Update for 5 seconds
        for _ in range(500):
            target.update(0.01)
        
        # Expected velocity: v = a*t = 10 * 5 = 50 m/s
        expected_velocity = 50.0
        actual_velocity = target.velocity[0]
        
        assert actual_velocity == pytest.approx(expected_velocity, abs=0.5)


# =============================================================================
# TEST 3: Detection at Close Range
# =============================================================================

class TestDetectionLogic:
    """
    Validate detection logic based on SNR.
    
    Reference: Radar equation - SNR decreases with R⁴
    """
    
    def test_detection_close_range(self):
        """Target at 10 km with 10 m² RCS should be detected"""
        result = validate_detection_logic(
            close_range_km=10.0,
            far_range_km=500.0
        )
        
        # Close target should have high SNR
        assert result['computed_values']['close_above_threshold'], (
            f"Close target should be detected. SNR={result['computed_values']['close_snr_db']:.2f} dB"
        )
    
    def test_no_detection_far_range(self):
        """Target at 500 km with 1 m² RCS should not be detected"""
        result = validate_detection_logic(
            close_range_km=10.0,
            far_range_km=500.0
        )
        
        # Far target should have low SNR
        assert result['computed_values']['far_below_threshold'], (
            f"Far target should NOT be detected. SNR={result['computed_values']['far_snr_db']:.2f} dB"
        )
    
    def test_validation_passes(self):
        """Both detection conditions should be satisfied"""
        result = validate_detection_logic()
        assert result['validation']['is_valid']


# =============================================================================
# TEST 4: Radar Geometry Calculation
# =============================================================================

class TestRadarGeometry:
    """Test radar-target geometry calculations."""
    
    @pytest.fixture
    def radar(self):
        """Create test radar at origin."""
        return Radar(
            radar_id="test",
            position=np.array([0.0, 0.0, 0.0]),
            frequency_hz=10e9
        )
    
    def test_range_calculation(self, radar):
        """Range should equal Euclidean distance"""
        target_pos = np.array([3000.0, 4000.0, 0.0])  # 5 km (3-4-5 triangle)
        
        geom = radar.calculate_target_geometry(target_pos)
        
        assert geom['range_m'] == pytest.approx(5000.0, abs=0.1)
    
    def test_azimuth_east(self, radar):
        """Target to the east should have azimuth ~90°"""
        target_pos = np.array([0.0, 1000.0, 0.0])  # Due east
        
        geom = radar.calculate_target_geometry(target_pos)
        
        assert geom['azimuth_deg'] == pytest.approx(90.0, abs=1.0)
    
    def test_azimuth_north(self, radar):
        """Target to the north should have azimuth ~0°"""
        target_pos = np.array([1000.0, 0.0, 0.0])  # Due north
        
        geom = radar.calculate_target_geometry(target_pos)
        
        assert geom['azimuth_deg'] == pytest.approx(0.0, abs=1.0)
    
    def test_radial_velocity(self, radar):
        """Target approaching should have negative radial velocity"""
        target_pos = np.array([1000.0, 0.0, 0.0])
        target_vel = np.array([-100.0, 0.0, 0.0])  # Approaching
        
        geom = radar.calculate_target_geometry(target_pos, target_vel)
        
        # Approaching = negative radial velocity
        assert geom['radial_velocity_mps'] == pytest.approx(-100.0, abs=1.0)


# =============================================================================
# TEST 5: Simulation Engine Integration
# =============================================================================

class TestSimulationEngine:
    """Test full simulation engine."""
    
    def test_engine_runs(self):
        """Simulation engine should run without errors"""
        radar = Radar(
            radar_id="test",
            position=np.array([0.0, 0.0, 0.0])
        )
        
        target = Target(
            target_id=1,
            position=np.array([50000.0, 0.0, 0.0]),
            velocity=np.array([-100.0, 0.0, 0.0]),  # Approaching
            rcs_m2=5.0
        )
        
        engine = SimulationEngine(
            radar=radar,
            targets=[target],
            dt=0.1
        )
        
        # Run for 1 second
        log = engine.run(duration_s=1.0)
        
        assert log.total_opportunities > 0
    
    def test_target_approaches(self):
        """Target range should decrease as it approaches"""
        radar = Radar(
            radar_id="test",
            position=np.array([0.0, 0.0, 0.0])
        )
        
        target = Target(
            target_id=1,
            position=np.array([50000.0, 0.0, 0.0]),
            velocity=np.array([-100.0, 0.0, 0.0]),  # Approaching at 100 m/s
            rcs_m2=5.0
        )
        
        engine = SimulationEngine(
            radar=radar,
            targets=[target],
            dt=0.1
        )
        
        initial_range = target.range_to(radar.position)
        
        # Run for 10 seconds
        engine.run(duration_s=10.0)
        
        final_range = target.range_to(radar.position)
        
        # Should be 1 km closer
        expected_change = 100.0 * 10.0  # 1000 m
        actual_change = initial_range - final_range
        
        assert actual_change == pytest.approx(expected_change, abs=10.0)


# =============================================================================
# TEST 6: Kinematic State
# =============================================================================

class TestKinematicState:
    """Test KinematicState dataclass."""
    
    def test_state_vector_conversion(self):
        """State should convert to/from vector correctly"""
        state = KinematicState(
            position=np.array([100.0, 200.0, 300.0]),
            velocity=np.array([10.0, 20.0, 30.0]),
            acceleration=np.array([1.0, 2.0, 3.0])
        )
        
        vector = state.to_state_vector()
        assert len(vector) == 9
        assert vector[0] == 100.0
        assert vector[3] == 10.0
        assert vector[6] == 1.0
    
    def test_speed_calculation(self):
        """Speed should be magnitude of velocity"""
        state = KinematicState(
            position=np.zeros(3),
            velocity=np.array([3.0, 4.0, 0.0])  # 3-4-5 triangle
        )
        
        assert state.speed == pytest.approx(5.0, abs=0.01)
    
    def test_2d_to_3d_padding(self):
        """2D vectors should be padded to 3D"""
        state = KinematicState(
            position=np.array([100.0, 200.0]),
            velocity=np.array([10.0, 20.0])
        )
        
        assert len(state.position) == 3
        assert state.position[2] == 0.0


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v", 
        "--tb=short"
    ])
