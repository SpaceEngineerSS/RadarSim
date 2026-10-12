"""
Monte Carlo sweep: empirical Pd against range for several target RCS values.

Every configuration runs the full SimulationEngine physics with its own seed
(spawned from one master seed), in parallel worker processes. Results go to a
CSV file, one row per run.

    python examples/monte_carlo_sweep.py --runs 10 --output pd_sweep.csv
"""

import argparse
import csv
import time
from multiprocessing import Pool, cpu_count

from radarsim.simulation.headless_runner import run_single_simulation
from radarsim.simulation.scenario_generator import ParameterSpace, ScenarioGenerator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--runs", type=int, default=5, help="Monte Carlo runs per configuration")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--seed", type=int, default=2026, help="master seed")
    parser.add_argument("--output", default="pd_sweep.csv")
    args = parser.parse_args()

    space = ParameterSpace(
        ranges_km=[10, 15, 20, 25, 30, 40, 50],
        rcs_values_m2=[0.5, 1.0, 5.0],
        n_runs_per_config=args.runs,
        duration_s=20.0,
        master_seed=args.seed,
    )
    configs = ScenarioGenerator.generate(space)
    print(f"{len(configs)} runs on {args.workers} workers")

    started = time.perf_counter()
    with Pool(args.workers) as pool:
        results = pool.map(run_single_simulation, configs)
    elapsed = time.perf_counter() - started

    rows = [result.to_dict() for result in results]
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"done in {elapsed:.1f} s, wrote {args.output}")
    print(f"{'range km':>9}{'RCS m2':>8}{'Pd mean':>9}")
    for rcs in space.rcs_values_m2:
        for range_km in space.ranges_km:
            pd = [r["detection_ratio"] for r in rows if r["rcs_m2"] == rcs and r["range_km"] == range_km]
            print(f"{range_km:>9}{rcs:>8}{sum(pd) / len(pd):>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
