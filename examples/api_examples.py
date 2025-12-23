"""
RadarSim API Examples

Usage examples demonstrating the radar simulation API.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_basic_radar():
    """
    Example 1: Basic Radar Setup
    
    Create radar parameters and calculate detection of a target.
    """
    from radar_physics import RadarParameters, TargetParameters, RadarPhysics
    
    # Create X-band radar (10 GHz)
    radar_params = RadarParameters(
        frequency=10e9,           # 10 GHz
        power_transmitted=1000,   # 1 kW
        antenna_gain_tx=30,       # 30 dB
        pulse_width=1e-6,         # 1 μs
        prf=1000                  # 1 kHz
    )
    
    physics = RadarPhysics(radar_params)
    
    # Create target at 20 km range
    target = TargetParameters(
        position=np.array([20000.0, 0.0, 5000.0]),  # 20 km, 5 km altitude
        velocity=np.array([-200.0, 0.0, 0.0]),      # 200 m/s approaching
        rcs=10.0                                     # 10 m² RCS
    )
    
    radar_pos = np.array([0.0, 0.0, 0.0])
    radar_vel = np.array([0.0, 0.0, 0.0])
    
    # Calculate metrics
    range_m = physics.calculate_range(target.position, radar_pos)
    snr_db = physics.calculate_snr(target, radar_pos)
    doppler_hz = physics.doppler_shift(target, radar_pos, radar_vel)
    
    print("=== Basic Radar Example ===")
    print(f"Target Range: {range_m/1000:.1f} km")
    print(f"SNR: {snr_db:.1f} dB")
    print(f"Doppler Shift: {doppler_hz:.0f} Hz")
    print(f"Radial Velocity: {doppler_hz * radar_params.wavelength / 2:.1f} m/s")


def example_atmospheric_attenuation():
    """
    Example 2: ITU-R P.676 Atmospheric Attenuation
    
    Calculate atmospheric loss for different frequencies.
    """
    from radar_physics import ITU_R_P676
    
    print("\n=== Atmospheric Attenuation (ITU-R P.676) ===")
    
    frequencies_ghz = [1, 5, 10, 22, 35, 60, 94]
    range_km = 10.0
    
    print(f"Range: {range_km} km (two-way)")
    print("-" * 40)
    
    for freq in frequencies_ghz:
        attenuation = ITU_R_P676.total_attenuation(range_km, freq, two_way=True)
        print(f"  {freq:3d} GHz: {attenuation:6.2f} dB")


def example_swerling_rcs():
    """
    Example 3: Swerling RCS Fluctuation Models
    
    Generate fluctuating RCS values for different Swerling models.
    """
    from radar_physics import SwerlingRCS, SwerlingModel
    
    print("\n=== Swerling RCS Models ===")
    
    mean_rcs = 10.0  # 10 m²
    n_samples = 1000
    
    models = [
        SwerlingModel.SWERLING_0,
        SwerlingModel.SWERLING_1,
        SwerlingModel.SWERLING_2,
        SwerlingModel.SWERLING_3,
        SwerlingModel.SWERLING_4
    ]
    
    print(f"Mean RCS: {mean_rcs} m²")
    print("-" * 50)
    
    for model in models:
        samples = [SwerlingRCS.generate_rcs(mean_rcs, model) for _ in range(n_samples)]
        mean_measured = np.mean(samples)
        std_measured = np.std(samples)
        
        print(f"  {model.name}: Mean={mean_measured:.2f}, Std={std_measured:.2f}")


def example_ekf_tracking():
    """
    Example 4: Extended Kalman Filter Tracking
    
    Track a target using the EKF with polar measurements.
    """
    from radar_physics import RadarParameters, TargetParameters, RadarPhysics
    from target_tracking import ExtendedKalmanFilter, TrackState, MotionModel
    
    print("\n=== EKF Tracking Example ===")
    
    # Create EKF
    ekf = ExtendedKalmanFilter(dt=0.1, motion_model=MotionModel.CONSTANT_VELOCITY)
    
    # Initial track state
    track = TrackState(
        position=np.array([10000.0, 0.0, 5000.0]),
        velocity=np.array([-100.0, 0.0, 0.0]),
        acceleration=np.zeros(3),
        covariance=np.eye(6) * 100,
        track_id=1,
        target_type="aircraft",
        rcs=10.0,
        last_update=0.0,
        track_quality=0.5,
        detection_count=1
    )
    
    print(f"Initial Position: {track.position}")
    
    # Simulate 5 steps
    for step in range(5):
        # Predict
        track = ekf.predict(track)
        print(f"Step {step+1} - Predicted X: {track.position[0]:.1f} m")


def example_ecm_simulation():
    """
    Example 5: ECM Simulation
    
    Demonstrate ECM effects on radar detection.
    """
    from radar_physics import RadarParameters, TargetParameters, RadarPhysics
    from ecm_simulation import ECMSimulator
    
    print("\n=== ECM Simulation Example ===")
    
    # Setup
    radar_params = RadarParameters(
        frequency=10e9,
        power_transmitted=1000,
        antenna_gain_tx=30
    )
    physics = RadarPhysics(radar_params)
    ecm = ECMSimulator(physics)
    
    target = TargetParameters(
        position=np.array([15000.0, 0.0, 5000.0]),
        velocity=np.array([-150.0, 0.0, 0.0]),
        rcs=10.0
    )
    
    radar_pos = np.array([0.0, 0.0, 0.0])
    
    # Without jamming
    snr_clear = physics.calculate_snr(target, radar_pos)
    print(f"SNR (no ECM): {snr_clear:.1f} dB")
    
    # Activate noise jamming
    ecm.activate_noise_jamming(
        jammer_position=np.array([14000.0, 0.0, 5000.0]),
        jammer_power=500.0,
        frequency_offset=0.0
    )
    
    # Calculate jammed signal
    received_power = physics.radar_equation(target, radar_pos)
    jammer_distance = physics.calculate_range(ecm.jammer_position, radar_pos)
    jammed_power = ecm.apply_noise_jamming(received_power, jammer_distance)
    
    # Jamming effectiveness
    jamming_loss_db = 10 * np.log10(received_power / jammed_power) if jammed_power > 0 else float('inf')
    print(f"Jamming Loss: {jamming_loss_db:.1f} dB")
    
    # RGPO demonstration
    ecm.activate_rgpo(pull_rate=100.0, max_offset=5000.0)
    
    for t in range(5):
        offset = ecm.update_rgpo(1.0)
        print(f"RGPO t={t+1}s: Range Offset = {offset:.0f} m")


def example_cfar_detection():
    """
    Example 6: CA-CFAR Detection
    
    Demonstrate constant false alarm rate detection.
    """
    from ecm_simulation import ECCMSystem
    
    print("\n=== CA-CFAR Detection Example ===")
    
    eccm = ECCMSystem()
    
    # Create synthetic range profile with noise and targets
    np.random.seed(42)
    num_cells = 100
    noise = np.random.exponential(1.0, num_cells)
    
    # Add two targets
    target_bins = [25, 67]
    for bin_idx in target_bins:
        noise[bin_idx] = 30.0  # Strong target return
    
    # Run CFAR
    detections, thresholds = eccm.cfar_detection(noise, pfa=1e-4)
    
    print(f"Range cells: {num_cells}")
    print(f"Target bins: {target_bins}")
    print(f"Detected bins: {detections}")
    print(f"Detection rate: {len(set(target_bins) & set(detections)) / len(target_bins) * 100:.0f}%")


def example_proportional_navigation():
    """
    Example 7: Proportional Navigation Guidance
    
    Calculate missile guidance commands.
    """
    from target_tracking import GuidanceSystem, TrackState
    
    print("\n=== Proportional Navigation Example ===")
    
    guidance = GuidanceSystem()
    
    # Missile state
    missile = TrackState(
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([500.0, 0.0, 0.0]),  # Mach 1.5
        acceleration=np.zeros(3),
        covariance=np.eye(6),
        track_id=0,
        target_type="missile",
        rcs=0.1,
        last_update=0.0,
        track_quality=1.0,
        detection_count=1
    )
    
    # Target state (aircraft, crossing)
    target = TrackState(
        position=np.array([8000.0, 2000.0, 1000.0]),
        velocity=np.array([-200.0, 100.0, 0.0]),
        acceleration=np.zeros(3),
        covariance=np.eye(6),
        track_id=1,
        target_type="aircraft",
        rcs=10.0,
        last_update=0.0,
        track_quality=1.0,
        detection_count=1
    )
    
    # Calculate guidance
    accel_cmd = guidance.calculate_guidance_command(missile, target)
    intercept_point, tgo = guidance.calculate_intercept_point(missile, target)
    
    print(f"Guidance Acceleration: [{accel_cmd[0]:.1f}, {accel_cmd[1]:.1f}, {accel_cmd[2]:.1f}] m/s²")
    print(f"Acceleration Magnitude: {np.linalg.norm(accel_cmd):.1f} m/s²")
    print(f"Predicted Intercept: [{intercept_point[0]:.0f}, {intercept_point[1]:.0f}, {intercept_point[2]:.0f}] m")
    print(f"Time to Go: {tgo:.1f} s")


if __name__ == "__main__":
    print("RadarSim API Examples")
    print("=" * 60)
    
    example_basic_radar()
    example_atmospheric_attenuation()
    example_swerling_rcs()
    example_ekf_tracking()
    example_ecm_simulation()
    example_cfar_detection()
    example_proportional_navigation()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
