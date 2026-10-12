"""
Command-line scenario runner (no GUI).

    python -m src --headless scenarios/f16_vs_sa6.yaml --duration 120 --seed 7
    python -m src --headless scenarios/naval_battlegroup.yaml --out looks.csv

Prints one line per object (looks, detections, empirical Pd, first detection,
median SNR) and the final track picture. ``--out`` writes every detection
opportunity as CSV (or JSON when the file name ends in .json).
"""

import csv
import json
import sys
import time
from typing import List, Optional, TextIO

import numpy as np

from src.io.scenario_loader import ScenarioLoader


def run_headless(
    scenario_path: str,
    duration_s: Optional[float] = None,
    seed: Optional[int] = None,
    out_path: Optional[str] = None,
    stream: Optional[TextIO] = None,
) -> int:
    """Run a scenario without the GUI; returns a process exit code."""
    if stream is None:
        stream = sys.stdout
    loader = ScenarioLoader(scenario_path)
    for warning in loader.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    config = loader.get_config()
    engine = loader.create_simulation_engine()
    if seed is not None:
        engine.seed = seed
        engine.rng = np.random.default_rng(seed)
    duration = config.duration_s if duration_s is None else duration_s

    started = time.perf_counter()
    log = engine.run(duration)
    elapsed = time.perf_counter() - started

    looks_by_target = {}
    for result in log.detection_history:
        looks_by_target.setdefault(result.target_id, []).append(result)

    print(f"{config.name}  ({duration:g} s simulated in {elapsed:.2f} s)", file=stream)
    print(
        f"{'object':<26}{'looks':>7}{'det':>6}{'Pd':>7}{'first det [s]':>15}{'median SNR [dB]':>17}",
        file=stream,
    )
    for target in engine.targets:
        looks = looks_by_target.get(target.target_id, [])
        detections = [r for r in looks if r.is_detected]
        pd = len(detections) / len(looks) if looks else 0.0
        first = f"{detections[0].time:.1f}" if detections else "-"
        snr = f"{np.median([r.snr_db for r in looks]):.1f}" if looks else "-"
        print(
            f"{target.name[:25]:<26}{len(looks):>7}{len(detections):>6}{pd:>7.2f}{first:>15}{snr:>17}",
            file=stream,
        )

    confirmed = [t for t in engine.tracks if t.status.value == "confirmed"]
    print(f"tracks: {len(engine.tracks)} ({len(confirmed)} confirmed)", file=stream)

    if out_path:
        rows = [r.to_dict() for r in log.detection_history]
        if out_path.lower().endswith(".json"):
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=1, default=float)
        elif rows:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"wrote {len(rows)} looks to {out_path}", file=stream)
    return 0


def add_headless_arguments(parser) -> None:
    parser.add_argument(
        "--headless", action="store_true", help="run the scenario without the GUI"
    )
    parser.add_argument("--duration", type=float, default=None, help="simulated time [s]")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--out", default=None, help="write looks to a .csv or .json file")


__all__: List[str] = ["run_headless", "add_headless_arguments"]
