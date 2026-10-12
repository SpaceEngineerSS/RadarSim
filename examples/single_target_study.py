#!/usr/bin/env python3
"""
Headless single-target detection study.

Runs one radar against one radially flying target through the full
SimulationEngine physics and reports empirical and theoretical Pd.

Usage:
    python examples/single_target_study.py --range 30 --rcs 1.0 --duration 10
    python examples/single_target_study.py --frequency 35 --range 20 --swerling 3 --seed 7

To run a full scenario file without the GUI use:
    radarsim --headless f16_vs_sa6
"""

import argparse
import sys

from radarsim.simulation.headless_runner import HeadlessRunner, SimulationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless single-target detection study")
    parser.add_argument("--range", type=float, default=50.0, help="Target range [km]")
    parser.add_argument("--rcs", type=float, default=1.0, help="Mean target RCS [m^2]")
    parser.add_argument(
        "--velocity", type=float, default=0.0, help="Radial velocity [m/s], + = receding"
    )
    parser.add_argument("--swerling", type=int, default=1, choices=range(5))
    parser.add_argument("--frequency", type=float, default=10.0, help="Frequency [GHz]")
    parser.add_argument("--power", type=float, default=100.0, help="Peak power [kW]")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration [s]")
    parser.add_argument("--pfa", type=float, default=1e-6, help="Probability of false alarm")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Print only Pd")
    args = parser.parse_args()

    config = SimulationConfig(
        target_range_m=args.range * 1000,
        target_rcs_m2=args.rcs,
        target_velocity_mps=args.velocity,
        swerling_case=args.swerling,
        frequency_hz=args.frequency * 1e9,
        power_watts=args.power * 1e3,
        duration_s=args.duration,
        probability_false_alarm=args.pfa,
        seed=args.seed,
    )
    result = HeadlessRunner(config).run()

    if args.quiet:
        print(f"{result.detection_ratio:.4f}")
        return 0

    print("=" * 60)
    print("RadarSim headless detection study")
    print("=" * 60)
    print(f"Target range: {config.target_range_m / 1000:.1f} km, RCS {config.target_rcs_m2:.2f} m^2")
    print(f"Swerling case: {config.swerling_case}, Pfa: {config.probability_false_alarm:g}")
    print(f"Frequency: {config.frequency_hz / 1e9:.1f} GHz, power {config.power_watts / 1e3:.0f} kW")
    print("-" * 60)
    print(f"Looks: {result.n_looks:,}   detections: {result.n_detections:,}")
    print(f"Empirical Pd: {result.detection_ratio:.3f}   theoretical mean Pd: {result.mean_pd:.3f}")
    print(f"SNR mean/min/max: {result.mean_snr_db:.1f} / {result.min_snr_db:.1f} / {result.max_snr_db:.1f} dB")
    print(f"Runtime: {result.runtime_s * 1000:.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
