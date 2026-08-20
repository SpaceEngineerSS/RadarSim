# Changelog

RadarSim follows [Semantic Versioning](https://semver.org/). Dates use ISO 8601.

## [3.0.0] - 2026-08-20

This release replaces several approximate or placeholder paths with physically traceable implementations. It is a major release because signal-array shapes and calibration, tracking/fusion semantics, imaging behaviour, package metadata, and removed modules can affect existing callers.

### Physics and detection

- Corrected monostatic link-budget, thermal-noise, atmospheric/rain, polarization, Albersheim, Swerling, and seeded validation behaviour.
- Added validated Oh 1992 bare-soil and NRL 2012 sea-reflectivity models with documented domains.
- Added resolution-cell surface area, signal-to-noise-plus-clutter, Weibull/K clutter, and rain-volume calculations.
- Added aggregate receiver input-power accounting and a circular-complex Gaussian hard-limiter/Bussgang distortion diagnostic.

### Signal processing

- Rebuilt pulse-Doppler acquisition around delayed complex LFM samples, linear matched filtering, fast/slow-time phase, physical axes, ambiguity metadata, window coherent gain, and equivalent noise bandwidth.
- Calibrated CA, GO, SO, and OS one-dimensional CFAR thresholds and added true rectangular two-dimensional CA-CFAR.
- Reworked stripmap SAR raw-data generation and range-Doppler focusing with non-circular fractional RCMC and calibrated coordinates.
- Reworked ISAR compression, zero-filled profile alignment, and angular-rate cross-range mapping.
- Omega-k and chirp-scaling entry points now explicitly report that they are unavailable instead of returning another algorithm’s result.

### Tracking and fusion

- Added NIS/chi-square gating, globally optimal Hungarian assignment, consecutive confirmation, exact coast/delete timing, and simulation timestamps.
- Replaced explicit covariance inversions with stable solves and Joseph-form updates where applicable.
- Added latency-aware constant-velocity timestamp alignment and joint multi-estimate covariance intersection.
- Separated independent-Gaussian information fusion from unknown-correlation covariance intersection.

### ECM and simulation

- Added validated DRFM configuration, capture/pull/hold timing, RGPO/VGPO bounds, non-wrapping CPI injection, and radar-relative false-target placement.
- Corrected exact SJNR combination and receiver saturation propagation into detection diagnostics.
- Extended scenario round trips for receiver full scale, pulse integration, ground-surface parameters, and DRFM controls.

### Desktop, packaging, and documentation

- Unified the installed `radarsim` and source `run_gui.py` entry points on PySide6.
- Wired target-inspector probability of detection to the simulation’s Swerling calculation and corrected SAR display normalization and metrics.
- Removed obsolete optional modules and binary assets; RadarSim has no remote-service dependency.
- Added UI smoke tests, strict lint/test/security/package CI, and tag-driven multi-platform Nuitka release builds.
- Replaced legacy phase notes with model equations, limitations, methodology, architecture, user guide, primary references, and release notes.

### Verification

- 340 automated tests across supported Python versions.
- Ruff source/test lint, Bandit medium/high-severity scan, package build, Sphinx warnings-as-errors build, and offscreen PySide6 smoke coverage.

## [2.4.0] - 2026-05-11

- Added initial SAR/ISAR, multi-radar fusion, DRFM/ECCM, EKF, and pulse-Doppler modules.
- Added PySide6 scientific displays and scenario examples.
- Several algorithms from this line were revised or removed in 3.0.0; use the 2.4.0 tag for historical behaviour.

## [1.0.0] - 2025-12-23

- Initial open-source radar equation, target simulation, tracking, ECM, visualization, and export release.

[3.0.0]: https://github.com/SpaceEngineerSS/RadarSim/releases/tag/v3.0.0
[2.4.0]: https://github.com/SpaceEngineerSS/RadarSim/releases/tag/v2.4.0
[1.0.0]: https://github.com/SpaceEngineerSS/RadarSim/releases/tag/v1.0.0
