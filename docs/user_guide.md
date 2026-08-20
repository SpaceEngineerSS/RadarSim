# User guide

## Install and start

Create a Python 3.9–3.12 virtual environment and install `.[gui]`. Start the installed command with `radarsim`, or run `python run_gui.py` from a checkout. A display server with OpenGL support is recommended for the 3-D view; software rendering may be slower.

The main window groups scenario controls, radar parameters, environment controls, target management, scopes, and analysis views. Load a scenario before starting if you need repeatable inputs. Changing live parameters creates a new experiment and should be recorded with the result.

## Scenario format

YAML and JSON share the same schema. The loader recognizes `scenario`, `radar`, `targets`, `environment`, and `simulation`. Unrecognized auxiliary sections remain in the source file but do not alter the engine.

```yaml
scenario:
  name: "Reference tracking run"
  description: "One constant-velocity target"
  duration_seconds: 60
  update_rate_hz: 20

radar:
  name: "X-band test radar"
  frequency_hz: 10.0e9
  power_watts: 100000
  prf_hz: 2000
  pulse_width_s: 2.0e-6
  system_losses_db: 4.0
  antenna:
    gain_db: 35
    beamwidth_az_deg: 2.0
    beamwidth_el_deg: 3.0
    polarization_tilt_deg: 0
  receiver:
    noise_figure_db: 4.0
    bandwidth_hz: 1.0e6
    system_temperature_k: 290
    full_scale_dbm: -20
  position: {x_m: 0, y_m: 0, z_m: 20}

targets:
  - name: "Target 1"
    type: "aircraft"
    rcs_m2: 5.0
    swerling_model: 1
    initial_position: {x_m: 30000, y_m: 5000, z_m: 3000}
    velocity: {vx_mps: -150, vy_mps: 0, vz_mps: 0}
    has_ecm: false

environment:
  temperature_c: 15
  pressure_hpa: 1013.25
  water_vapor_gpm3: 7.5
  rain_rate_mm_hr: 0
  terrain_type: "rural"
  sea_state: 0
  ground_surface:
    model: "gamma"

simulation:
  enable_atmospheric_loss: true
  enable_clutter: false
  pfa: 1.0e-6
  pulses_integrated: 8
```

Valid `swerling_model` values are 0–4. `ground_surface.model` is `gamma` or `oh1992`. Sea state is 0–6. Probability of false alarm must be strictly between zero and one.

For a DRFM target, set `has_ecm: true`, `ecm_type: drfm`, and add:

```yaml
drfm:
  mode: rgpo
  gain_over_skin_db: 12
  capture_dwell_s: 1.5
  pull_rate_mps: 75
  max_pull_m: 2500
  vgpo_rate_hz_per_s: 80
  max_doppler_pull_hz: 600
  inherent_delay_s: 2.0e-7
```

Set `mode` to `rgpo` for range pull-off or `vgpo` for Doppler pull-off. A single jammer instance applies one mode at a time.

## Reading displays

The PPI shows horizontal range/azimuth; the RHI shows range/elevation; the A-scope shows amplitude or power versus range and can overlay a CFAR threshold. The range-Doppler view is ambiguous unless PRF and wavelength are considered. The target inspector reports radial velocity using positive-away convention.

Track symbols are estimates, not truth. A confirmed track has met the configured hit requirement. A coasted track is prediction-only and its covariance should grow. Deletion means the lifecycle threshold was exceeded; it does not prove the physical target disappeared.

Receiver saturation indicates aggregate input exceeded full scale. In that state, displayed post-limiter SNR includes a Gaussian-equivalent distortion diagnostic; exact deterministic spectral distortion is outside the aggregate engine path.

## SAR and ISAR

The SAR viewer consumes point-scatterer scenes and platform/radar parameters. Choose range-Doppler processing for the implemented focused result. Image magnitude is normalized to its own finite peak and displayed in dB with a controlled floor and brightness offset. Quality measures such as PSLR or entropy depend on the selected scene and crop; they are not universal sensor specifications.

ISAR needs range profiles over a coherent observation and a nonzero angular-rate estimate for metric cross range. Profile alignment removes bulk translation at integer-bin precision.

## Recording and export

HDF5 recording preserves simulation metadata and time-indexed results. Replay validates required datasets and reports malformed JSON metadata rather than accepting arbitrary code. CSV/JSON exports are useful for tables; GeoJSON/KML require a meaningful local-to-geographic conversion configured by the caller.

## Headless studies

```python
from src.io.scenario_loader import ScenarioLoader

loader = ScenarioLoader("scenarios/basic_tracking.json")
config = loader.get_config()
engine = loader.create_simulation_engine()

history = []
steps = round(config.duration_s * config.update_rate_hz)
for _ in range(steps):
    history.append(engine.step())
```

Use a fixed NumPy seed before constructing stochastic scenarios when repeatability is required. Save the exact scenario, version, seed, and dependency versions with exported results.

## Troubleshooting

If the command reports missing GUI dependencies, install `.[gui]` in the active interpreter. If OpenGL rendering fails, first verify that the 2-D scopes work and update the graphics driver. If a target never appears in a pulse-Doppler map, check the fast-time instrumented range and ambiguity intervals. If CFAR false alarms differ from `pfa`, confirm that the reference power is independent exponential noise and that enough valid edge-free cells were counted.
