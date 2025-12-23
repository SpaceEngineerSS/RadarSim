#!/usr/bin/env python3
"""
Headless Simulation CLI

Run single radar simulation without GUI.

Usage:
    python headless.py                          # Default config
    python headless.py --range 50 --rcs 2       # Custom target
    python headless.py --config scenario.yaml   # From file

Examples:
    # Quick test
    python headless.py --range 30 --rcs 1.0 --duration 10

    # High frequency test
    python headless.py --frequency 35 --range 20
"""

import argparse
import sys
import os
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.simulation.headless_runner import HeadlessRunner, SimulationConfig


def load_config_from_yaml(filepath: str) -> SimulationConfig:
    """Load configuration from YAML file."""
    with open(filepath, 'r') as f:
        data = yaml.safe_load(f)
    
    radar = data.get('radar', {})
    target = data.get('target', data.get('targets', [{}])[0] if isinstance(data.get('targets'), list) else {})
    sim = data.get('simulation', {})
    
    return SimulationConfig(
        frequency_hz=float(radar.get('frequency_hz', 10e9)),
        power_watts=float(radar.get('power_watts', 100e3)),
        antenna_gain_db=float(radar.get('antenna', {}).get('gain_db', 30)),
        target_range_m=float(target.get('range_m', target.get('position', [50000])[0] if isinstance(target.get('position'), list) else 50000)),
        target_rcs_m2=float(target.get('rcs_m2', 1.0)),
        duration_s=float(sim.get('duration_s', 10)),
        detection_threshold_db=float(sim.get('threshold_db', 13)),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run headless radar simulation"
    )
    
    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML configuration file"
    )
    
    # Target parameters
    parser.add_argument(
        "--range", type=float, default=50.0,
        help="Target range in km (default: 50)"
    )
    parser.add_argument(
        "--rcs", type=float, default=1.0,
        help="Target RCS in m² (default: 1.0)"
    )
    parser.add_argument(
        "--velocity", type=float, default=0.0,
        help="Target radial velocity in m/s (default: 0)"
    )
    
    # Radar parameters
    parser.add_argument(
        "--frequency", type=float, default=10.0,
        help="Radar frequency in GHz (default: 10)"
    )
    parser.add_argument(
        "--power", type=float, default=100.0,
        help="Transmit power in kW (default: 100)"
    )
    
    # Simulation parameters
    parser.add_argument(
        "--duration", type=float, default=10.0,
        help="Simulation duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--threshold", type=float, default=13.0,
        help="Detection threshold in dB (default: 13)"
    )
    
    # Options
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress output"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        if not os.path.exists(args.config):
            print(f"Error: Config file not found: {args.config}")
            return 1
        config = load_config_from_yaml(args.config)
    else:
        config = SimulationConfig(
            target_range_m=args.range * 1000,
            target_rcs_m2=args.rcs,
            target_velocity_mps=args.velocity,
            frequency_hz=args.frequency * 1e9,
            power_watts=args.power * 1e3,
            duration_s=args.duration,
            detection_threshold_db=args.threshold,
        )
    
    if not args.quiet:
        print("=" * 60)
        print("RadarSim Headless Mode")
        print("=" * 60)
        print(f"Target Range: {config.target_range_m / 1000:.1f} km")
        print(f"Target RCS: {config.target_rcs_m2:.2f} m²")
        print(f"Frequency: {config.frequency_hz / 1e9:.1f} GHz")
        print(f"Power: {config.power_watts / 1e3:.0f} kW")
        print(f"Duration: {config.duration_s:.1f} s")
        print(f"Threshold: {config.detection_threshold_db:.1f} dB")
        print("=" * 60)
    
    # Run simulation
    runner = HeadlessRunner(config)
    result = runner.run()
    
    if not args.quiet:
        print("\n--- RESULTS ---")
        print(f"Pulses transmitted: {result.n_pulses:,}")
        print(f"Detections: {result.n_detections:,}")
        print(f"Detection Probability (Pd): {result.detection_ratio:.3f}")
        print(f"Mean SNR: {result.mean_snr_db:.1f} dB")
        print(f"Min/Max SNR: {result.min_snr_db:.1f} / {result.max_snr_db:.1f} dB")
        print(f"Runtime: {result.runtime_s * 1000:.1f} ms")
        print("=" * 60)
    else:
        # Machine-readable output
        print(f"{result.detection_ratio:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
