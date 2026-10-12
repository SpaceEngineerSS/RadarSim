# Command line and Python

## Command line

```text
radarsim [scenario] [--headless] [--duration S] [--seed N] [--out FILE] [--list]
```

- `radarsim` opens the desktop application; `radarsim f16_vs_sa6` opens it with that
  scenario (a built-in name or a file path).
- `--list` prints the built-in scenarios.
- `--headless` runs the scenario without the interface and prints one line per
  object: looks, detections, empirical Pd, time of first detection and median SNR,
  followed by the number of tracks.
- `--out looks.csv` (or `.json`) writes every detection opportunity with all loss
  terms: SNR, Pd, atmospheric and rain loss, clutter loss, jamming loss,
  receiver headroom and so on.

```text
$ radarsim --headless f16_vs_sa6 --duration 120 --seed 3
F-16 vs SA-6 Air Defense  (120 s simulated in 0.70 s)
object                      looks   det     Pd  first det [s]  median SNR [dB]
F-16 Alpha                     12     0   0.00              -             -9.7
F-16 Bravo                     12    12   1.00            7.4             39.6
Chaff 1                         6     6   1.00           67.7             51.9
ADM-160 MALD                    9     9   1.00           37.7             42.2
tracks: 3 (3 confirmed)
```

The same seed always gives the same run.

## Python

Load a scenario and run it:

```python
from radarsim.io.scenario_loader import ScenarioLoader, resolve_scenario

engine = ScenarioLoader(resolve_scenario("naval_battlegroup")).create_simulation_engine()
engine.seed = 7
log = engine.run(60.0)

for result in log.detection_history[:5]:
    print(f"{result.time:6.1f} s  target {result.target_id}  "
          f"SNR {result.snr_db:5.1f} dB  clutter loss {result.surface_clutter_loss_db:4.1f} dB")

for event in engine.event_log:
    print(event.time, event.kind.value, event.message)
```

Build a scene in code:

```python
import numpy as np
from radarsim.simulation.engine import SimulationEngine
from radarsim.simulation.objects import Radar, Target, MotionModel
from radarsim.physics.rcs import SwerlingModel

radar = Radar("S-band", position=np.array([0, 0, 20.0]), frequency_hz=3e9,
              power_watts=200e3, antenna_gain_db=35, beamwidth_deg=1.5, scan_rate_rpm=12)
target = Target(1, position=np.array([0, 60e3, 4000.0]), velocity=np.array([0, -250.0, 0]),
                rcs_m2=2.0, swerling_model=SwerlingModel.SWERLING_1,
                motion_model=MotionModel.CONSTANT_VELOCITY, affiliation="hostile")
engine = SimulationEngine(radar, [target], dt=0.05, seed=1, enable_clutter=True)
engine.run(30.0)
print([(t.id, t.status.value) for t in engine.tracks])
```

Coverage and Monte Carlo, using the same per-look physics as a run:

```python
from radarsim.simulation.analysis import probe_engine, coverage_curve, monte_carlo_pd

probe = probe_engine(engine)            # radar and environment, no targets
curve = coverage_curve(probe, rcs_m2=1.0, swerling=1, altitude_m=3000, max_range_m=150e3)
print(curve.range_pd50_m, curve.range_pd90_m)
```

Batch studies for a single target against a parametric radar are in
`radarsim.simulation.headless_runner` and `scenario_generator`; see
`examples/single_target_study.py` and `examples/monte_carlo_sweep.py`, which runs a
seeded range × RCS sweep on all cores.

More self-contained snippets are in `examples/api_tour.py`.
