"""Executable examples for RadarSim's public scientific APIs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io.scenario_loader import ScenarioLoader
from src.physics.atmospheric import ITU_R_P676
from src.physics.metrics import albersheim_snr, calculate_pd_swerling
from src.physics.radar_equation import (
    RadarParameters,
    calculate_doppler_shift,
    calculate_received_power,
    calculate_snr,
)
from src.physics.rcs import SwerlingModel, SwerlingRCS
from src.signal.cfar import CFARDetector, CFARType
from src.signal.pulse_doppler import PulseDopplerProcessor
from src.tracking.tracker import TrackManager


def radar_equation_example() -> None:
    radar = RadarParameters(
        frequency=10.0e9,
        power_transmitted=10.0e3,
        antenna_gain_tx=35.0,
        antenna_gain_rx=35.0,
        noise_figure=4.0,
        noise_bandwidth=2.0e6,
        pulse_width=2.0e-6,
        prf=2.0e3,
    )
    range_m = 25.0e3
    rcs_m2 = 3.0
    received_w = calculate_received_power(radar, rcs_m2, range_m)
    snr_db = calculate_snr(radar, rcs_m2, range_m)
    doppler_hz = calculate_doppler_shift(
        radar,
        target_pos=np.array([range_m, 0.0, 2.0e3]),
        target_vel=np.array([-180.0, 0.0, 0.0]),
    )
    print(f"Radar equation: Pr={received_w:.3e} W, SNR={snr_db:.2f} dB")
    print(f"Doppler: {doppler_hz:.2f} Hz (negative means approaching)")


def propagation_and_detection_example() -> None:
    for frequency_ghz in (10.0, 22.235, 60.0, 94.0):
        loss_db = ITU_R_P676.total_attenuation(
            range_km=10.0,
            frequency_ghz=frequency_ghz,
            two_way=True,
        )
        print(f"Gas loss at {frequency_ghz:6.3f} GHz: {loss_db:8.3f} dB")

    required_snr = albersheim_snr(pd=0.9, pfa=1e-6, n_pulses=8)
    actual_pd = calculate_pd_swerling(
        snr_db=required_snr,
        pfa=1e-6,
        swerling_case=1,
        n_pulses=8,
    )
    print(f"Albersheim SNR={required_snr:.2f} dB; Swerling-I Pd={actual_pd:.4f}")


def swerling_example() -> None:
    state = np.random.get_state()
    np.random.seed(20260820)
    try:
        for model in SwerlingModel:
            samples = np.array(
                [SwerlingRCS.generate_rcs(10.0, model) for _ in range(20_000)]
            )
            print(
                f"{model.name}: mean={samples.mean():.3f} m^2, "
                f"std={samples.std():.3f} m^2"
            )
    finally:
        np.random.set_state(state)


def pulse_doppler_and_cfar_example() -> None:
    processor = PulseDopplerProcessor(
        prf_hz=4.0e3,
        n_pulses=64,
        n_range_bins=2048,
        bandwidth_hz=5.0e6,
        sample_rate_hz=10.0e6,
        pulse_width_s=8.0e-6,
        frequency_hz=10.0e9,
        window_type="hamming",
    )
    amplitude = processor.amplitude_for_output_snr(100.0, noise_power=1.0)
    rd_map = processor.process_cpi(
        target_ranges_m=np.array([12_000.0]),
        target_velocities_mps=np.array([-45.0]),
        target_amplitudes=np.array([amplitude]),
        noise_power=1.0,
        seed=42,
    )
    peak = np.unravel_index(np.argmax(rd_map.data_linear), rd_map.data_linear.shape)
    print(
        f"Range-Doppler peak: R={rd_map.range_axis_m[peak[1]]:.1f} m, "
        f"v={rd_map.velocity_axis_mps[peak[0]]:.2f} m/s"
    )

    profile = rd_map.data_linear[:, peak[1]]
    detector = CFARDetector(
        guard_cells=2,
        reference_cells=8,
        pfa=1e-4,
        cfar_type=CFARType.CA,
    )
    detections, _ = detector.detect(profile)
    print(f"CFAR detections in target range cell: {np.flatnonzero(detections).tolist()}")


def tracking_example() -> None:
    manager = TrackManager(
        gate_distance=500.0,
        confirm_hits=3,
        max_misses=3,
        measurement_noise=25.0,
    )
    for index in range(5):
        tracks = manager.update([(1000.0 + 20.0 * index, 2000.0)], dt=1.0)
    track = tracks[0]
    print(
        f"Track {track.id}: status={track.status.value}, "
        f"position=({track.position[0]:.1f}, {track.position[1]:.1f}) m"
    )


def scenario_example() -> None:
    loader = ScenarioLoader("scenarios/basic_tracking.json")
    config = loader.get_config()
    engine = loader.create_simulation_engine()
    detections = engine.step()
    print(
        f"Scenario '{config.name}': {len(config.targets)} targets, "
        f"first-step detections={len(detections)}"
    )


def main() -> None:
    radar_equation_example()
    propagation_and_detection_example()
    swerling_example()
    pulse_doppler_and_cfar_example()
    tracking_example()
    scenario_example()


if __name__ == "__main__":
    main()
