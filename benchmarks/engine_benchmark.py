"""
Engine throughput for every built-in scenario.

    python benchmarks/engine_benchmark.py

Reports the mean wall-clock cost of one engine step and of building one
display frame, and how many times faster than real time the scenario runs.
"""

import time

from radarsim.io.scenario_loader import ScenarioLoader, builtin_scenarios
from radarsim.simulation.frame import build_frame

SIMULATED_SECONDS = 30.0


def main() -> int:
    print(f"{'scenario':<30}{'objects':>8}{'step ms':>9}{'frame ms':>10}{'x real time':>13}")
    for path in builtin_scenarios():
        engine = ScenarioLoader(str(path)).create_simulation_engine()
        steps = int(round(SIMULATED_SECONDS / engine.dt))
        started = time.perf_counter()
        for _ in range(steps):
            engine.step()
        step_s = (time.perf_counter() - started) / steps
        started = time.perf_counter()
        for _ in range(50):
            build_frame(engine, engine.dt)
        frame_s = (time.perf_counter() - started) / 50
        speedup = engine.dt / step_s
        print(f"{path.stem:<30}{len(engine.targets):>8}{step_s * 1e3:>9.3f}{frame_s * 1e3:>10.3f}{speedup:>13.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
