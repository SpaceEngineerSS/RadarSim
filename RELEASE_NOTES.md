# RadarSim 3.0.0

RadarSim 3.0 is a scientific-model and signal-chain revision. The release focuses on traceable equations, calibrated statistical processing, explicit timing/covariance semantics, and honest model boundaries.

The pulse-Doppler path now starts with delayed complex LFM echoes and performs linear matched filtering, optional MTI, windowed Doppler processing, physical range/velocity axes, and ambiguity reporting. CFAR thresholds are calibrated separately for CA, GO, SO, and OS detectors; two-dimensional CA-CFAR uses a true rectangular reference ring.

Tracking now uses chi-square innovation gating and Hungarian global assignment with exact confirmation/coasting rules. Network estimates are timestamp-aligned before fusion. Independent measurements and unknown-correlated tracks use separate information-fusion and covariance-intersection paths.

The physics engine includes corrected link/noise accounting, ITU-R gas and rain loss, Oh bare-soil and NRL sea-clutter reflectivity, statistical clutter, DRFM pull-off, and receiver hard-limiting diagnostics. SAR raw data and range-Doppler focusing now use actual phase history, range gating, and non-circular RCMC; ISAR uses compressed, aligned profiles and a rotation-rate-derived cross-range axis.

Obsolete optional modules and binary assets have been removed. RadarSim has no telemetry or remote-service dependency.

This release intentionally reports omega-k and chirp-scaling SAR processing as unimplemented. It also does not claim hardware-certified performance, real-platform signatures, refractive ray tracing, multipath, CAD electromagnetic scattering, or classified equipment behaviour. See `docs/MODEL_FIDELITY.md` before interpreting simulation results.

Verification for the source release: 340 automated tests, strict Ruff checks, offscreen PySide6 smoke tests, Bandit medium/high-severity scan, Sphinx warnings-as-errors build, and wheel/source-distribution build.

Breaking changes include removal of obsolete modules/assets, corrected signal calibration and axes, stricter input validation, and revised fusion/tracking lifecycle semantics. Existing scripts should pin 2.4.0 until their assumptions are checked against the 3.0 documentation.
