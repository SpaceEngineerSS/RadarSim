# RadarSim 3.0.1

RadarSim 3.0.1 is a maintenance and stability release resolving scenario execution edge cases, dynamic parameter synchronization, and command-line scenario loading.

## Highlights in 3.0.1

- **Scenario-Bundled Distribution:** Bundled the `scenarios/` directory in release ZIP/TAR.GZ archives so standalone executables have immediate access to all 10 scenario definitions.
- **Built-in Scenarios Menu:** Added `File > ⚡ Built-in Scenarios` menu for instant one-click scenario launching without manual file browsing.
- **Rendering Performance Optimization:** Pre-cached discrete brush palettes for phosphor persistence trails and affiliation markers, eliminating on-the-fly QPainter rasterization bottlenecks and dropping phosphor update latency from 117 ms to 0.5 ms (192x speedup).
- **Sub-1GHz Attenuation Safety:** Guarded ITU-R P.676 atmospheric absorption and ITU-R P.838 rain attenuation for frequencies below 1.0 GHz (such as UHF/VHF early warning radars), applying negligible 0.0 dB attenuation rather than raising an out-of-domain exception. Resolves crashes in `stealth_deep_penetration.yaml`.
- **Terrain Type Normalization:** Added `"mountain"` alias and normalization in `ClutterModel.ground_clutter_sigma0` to match scenario specifications. Resolves crash in `mountain_ambush.yaml`.
- **Dynamic Radar Parameter Synchronization:** Converted `SimulationEngine._radar_params` to a real-time property synchronized with `self.radar`, ensuring GUI frequency/power slider changes immediately affect SNR and detection calculations.
- **Command-Line Scenario Loading:** `python run_gui.py [scenario_path]` and `radarsim [scenario_path]` now accept a scenario YAML/JSON file path directly on startup.
- **GUI File Filter:** Added `*.json` to `QFileDialog` scenario open filter alongside `*.yaml`/`*.yml`.
- **Public Scenario API:** Added `MainWindow.load_scenario(filepath)` for modular and programmatic scenario loading.
- **Windows Console Encoding:** Replaced Unicode arrow and lambda characters in console logging to prevent `UnicodeEncodeError` on CP1252/CP1254 environments.
- **Automated Verification:** Added regression tests executing multi-step simulation on all distributed scenarios. Full test suite: 343 passing tests.
