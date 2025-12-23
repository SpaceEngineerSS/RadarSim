#!/usr/bin/env python3
"""
Batch Run Script - Monte Carlo Simulation Executor

Runs multiple headless simulations in parallel using multiprocessing.
Outputs results to CSV for analysis.

Usage:
    python batch_run.py                    # Run default sweep
    python batch_run.py --configs 100      # Run 100 configurations
    python batch_run.py --output results.csv

Requirements:
    pip install tqdm
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import List, Dict

# Try tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bar: pip install tqdm")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.simulation.headless_runner import SimulationConfig, SimulationResult, run_single_simulation
from src.simulation.scenario_generator import ScenarioGenerator, ParameterSpace


def run_batch(
    configs: List[SimulationConfig],
    n_workers: int = None,
    output_file: str = "output/batch_results.csv"
) -> List[SimulationResult]:
    """
    Run batch of simulations in parallel.
    
    Args:
        configs: List of simulation configurations
        n_workers: Number of parallel workers (default: CPU count)
        output_file: Output CSV file path
        
    Returns:
        List of simulation results
    """
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)
    
    print(f"=" * 60)
    print(f"RadarSim Batch Processor")
    print(f"=" * 60)
    print(f"Configurations: {len(configs)}")
    print(f"Workers: {n_workers}")
    print(f"Output: {output_file}")
    print(f"=" * 60)
    
    start_time = time.perf_counter()
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Run in parallel
    results = []
    
    with Pool(n_workers) as pool:
        if TQDM_AVAILABLE:
            # With progress bar
            iterator = pool.imap_unordered(run_single_simulation, configs)
            for result in tqdm(iterator, total=len(configs), desc="Simulating"):
                results.append(result)
        else:
            # Without progress bar
            print("Running simulations...")
            for i, result in enumerate(pool.imap_unordered(run_single_simulation, configs)):
                results.append(result)
                if (i + 1) % 10 == 0:
                    print(f"  Completed: {i + 1}/{len(configs)}")
    
    total_time = time.perf_counter() - start_time
    
    # Save results to CSV
    _save_results_csv(results, output_file)
    
    # Print summary
    _print_summary(results, total_time)
    
    return results


def _save_results_csv(results: List[SimulationResult], filepath: str) -> None:
    """Save results to CSV file."""
    if not results:
        return
    
    # Get field names from first result
    sample_dict = results[0].to_dict()
    fieldnames = list(sample_dict.keys())
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            writer.writerow(result.to_dict())
    
    print(f"\n✓ Results saved to: {filepath}")


def _print_summary(results: List[SimulationResult], total_time: float) -> None:
    """Print batch run summary."""
    if not results:
        print("No results to summarize")
        return
    
    # Calculate statistics
    total_pulses = sum(r.n_pulses for r in results)
    total_detections = sum(r.n_detections for r in results)
    avg_pd = sum(r.detection_ratio for r in results) / len(results)
    avg_snr = sum(r.mean_snr_db for r in results) / len(results)
    
    print(f"\n" + "=" * 60)
    print(f"BATCH COMPLETE")
    print(f"=" * 60)
    print(f"Total simulations: {len(results)}")
    print(f"Total pulses: {total_pulses:,}")
    print(f"Total detections: {total_detections:,}")
    print(f"Average Pd: {avg_pd:.3f}")
    print(f"Average SNR: {avg_snr:.1f} dB")
    print(f"Total time: {total_time:.2f}s")
    print(f"Sims/second: {len(results) / total_time:.1f}")
    print(f"=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo radar simulations"
    )
    parser.add_argument(
        "--configs", type=int, default=None,
        help="Number of configurations (default: auto)"
    )
    parser.add_argument(
        "--runs", type=int, default=5,
        help="Monte Carlo runs per configuration (default: 5)"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: CPU count - 1)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV file (default: output/batch_YYYYMMDD_HHMMSS.csv)"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run quick sweep (10 ranges, 5 runs each)"
    )
    
    args = parser.parse_args()
    
    # Generate output filename
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"output/batch_{timestamp}.csv"
    
    # Generate configurations
    if args.quick:
        print("Running quick Pd vs Range sweep...")
        configs = ScenarioGenerator.quick_sweep(
            range_min_km=10,
            range_max_km=100,
            n_ranges=10,
            rcs_m2=1.0,
            n_runs=5
        )
    else:
        # Full parameter space
        space = ParameterSpace(
            ranges_km=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            rcs_values_m2=[0.5, 1.0, 2.0, 5.0],
            n_runs_per_config=args.runs,
        )
        configs = ScenarioGenerator.generate(space)
    
    # Limit configs if specified
    if args.configs and len(configs) > args.configs:
        configs = configs[:args.configs]
    
    # Run batch
    results = run_batch(
        configs=configs,
        n_workers=args.workers,
        output_file=args.output
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
