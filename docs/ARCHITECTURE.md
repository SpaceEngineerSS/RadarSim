# Architecture

RadarSim separates physical models, time evolution, estimation, file formats, and presentation so that headless studies do not depend on the desktop UI.

```text
scenario YAML/JSON
        |
        v
src.io.scenario_loader
        |
        v
src.simulation.engine ----> detections ----> src.tracking
   |       |       |                            |
   |       |       +---- src.physics.ecm        v
   |       +------------ src.physics.*     network/fusion
   +-------------------- radar/targets
        |
        +---- recording/export/replay
        +---- PySide6 UI views

raw complex samples ----> src.signal ----> range-Doppler/CFAR
point-scatterer scenes --> src.advanced.sar_isar --> imagery
```

## Package responsibilities

`src.physics` contains constants, radar equation and detection functions, ITU-R propagation, RCS fluctuation, clutter, terrain, antenna-related physics, and ECM link/nonlinearity models. These modules should not import UI code.

`src.simulation` owns radar and target state, discrete-time stepping, detections, ECM integration, receiver diagnostics, track/network orchestration, and deterministic event timing.

`src.signal` owns sampled waveform generation, pulse-Doppler processing, CFAR, antenna patterns, and ambiguity-related calculations. Arrays are NumPy arrays and complex baseband uses `complex128` unless a caller deliberately converts it.

`src.tracking` owns linear and extended Kalman filters, global measurement-to-track association, track lifecycle, monopulse estimation, and guidance utilities. State and covariance timestamps must refer to the same epoch.

`src.advanced` contains SAR/ISAR imaging, multisensor fusion, recording analysis, bistatic geometry, and compatibility facades. “Advanced” is a package boundary, not a claim that every literature algorithm is implemented.

`src.io` parses scenarios and handles recording, replay, and export. Loaders validate scalar domains before constructing simulation objects.

`src.ui` contains PySide6 windows, panels, scopes, and worker threads. UI calculations call the same scientific APIs as the headless engine; they should not duplicate approximate formulas.

## Simulation step

At each engine step, target kinematics and active ECM states advance using the configured `dt`. For each target, the engine determines geometry, line of sight, fluctuated RCS, propagation loss, echo power, noise/clutter/interference, receiver limiting, SNR/SJNR, detection probability, and a stochastic detection. Measurements update the tracker at simulation time. UI and recording consumers receive snapshots rather than owning model state.

## Tracking and fusion timing

Local tracks predict to the measurement timestamp before gating and update. The Mahalanobis cost matrix is gated by a chi-square/NIS threshold and solved globally with the Hungarian algorithm. Confirmation requires consecutive hits; coasting and deletion use explicit missed-update counts.

Network messages are deep-copied into a latency queue. Ready tracks propagate with a constant-velocity state transition to the fusion time. Estimates with unknown cross-correlation use covariance intersection; explicitly independent Gaussian measurements use information-form fusion. These paths are separate to prevent accidental overconfidence.

## Extension rules

New physical models belong in the lowest package that can express them without UI dependencies. Public inputs need unit-bearing names and validation. State-changing objects must take simulation time or `dt` explicitly. Randomized APIs should accept a generator or preserve/restore global state in validation helpers. File-format changes require round-trip tests and a compatibility note in the changelog.
