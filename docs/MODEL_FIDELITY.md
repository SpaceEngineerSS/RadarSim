# Model fidelity and limitations

This document is the boundary between implemented behaviour and effects that RadarSim does not claim to reproduce.

## Fidelity levels

| Area | Implemented level | Principal limitation |
|---|---|---|
| Kinematics | Discrete 3-D point objects with static, constant-velocity, and configured manoeuvre updates | No six-degree-of-freedom aerodynamics or control system |
| Radar link | Monostatic point-target power and thermal noise budget | No bistatic geometry, mutual coupling, hardware calibration, or scan-loss scheduler |
| Detection | Albersheim thresholding and Swerling statistics | No compound target/clutter likelihood-ratio detector |
| Atmosphere | Homogeneous-path ITU-R gas and rain specific attenuation | No refractivity ray tracing, ducting, multipath, diffraction, cloud, or fog |
| RCS | User mean, simple aspect factor, Swerling fluctuation | No CAD-based electromagnetic scattering or micro-Doppler signature |
| Land clutter | Terrain categories and Oh 1992 bare-soil backscatter | No vegetation canopy, buildings, shadowing, or spatial correlation map |
| Sea clutter | NRL 2012 mean reflectivity plus K-distributed samples | No evolving electromagnetic sea surface or coherent sea spikes |
| Pulse-Doppler | Complex-IQ LFM, delay, matched filter, MTI, FFT, ambiguity | No phase noise, timing jitter, array channels, or quantized converter model |
| CFAR | Calibrated CA/GO/SO/OS 1-D and CA 2-D | Calibration assumes independent exponential reference power |
| ECM | Generic noise, chaff, DRFM pull-off, false targets | No named-device data or waveform-recognition logic |
| Receiver | Aggregate input power and Gaussian radial limiter diagnostic | No analogue filter chain, AGC loop dynamics, intermodulation cascade, or ADC bits |
| Tracking | CV KF/EKF, NIS gate, global assignment, lifecycle | No IMM, MHT, JPDA, extended-target, or bias estimator |
| Fusion | Timestamp propagation, independent information fusion, covariance intersection | No cross-covariance transport or decentralized consensus filter |
| SAR | Broadside stripmap point-scatterer raw data and RDA focusing | No squint, topography, autofocus, polarimetry, omega-k, or chirp scaling |
| ISAR | Range compression, translational alignment, range-Doppler image | Requires usable, approximately constant rotation rate |

## Coordinate and sign conventions

Scenario vectors are Cartesian metres and metres per second. Most simulation views interpret `x` and `y` as a local horizontal plane and `z` as altitude. Some legacy docstrings use the term NED; algorithms consuming these vectors should rely on their explicit axis definition rather than assume geodetic NED. Positive radial velocity is away from the radar. Doppler follows the same sign.

Angles are radians in computational APIs and degrees in explicitly named configuration fields. Power ratios use \(10\log_{10}\); complex-voltage ratios use \(20\log_{10}\) only where an amplitude is being converted.

## Validity checks

A numerically finite output is not evidence that a model is valid. For every study:

1. verify that input frequency, angle, polarization, roughness, weather, and geometry fall inside the cited empirical domain;
2. separate instrumented range/velocity from physical unambiguous range/velocity;
3. identify whether an input is peak, average, pulse, coherent, or noncoherent power;
4. report random seed and number of Monte Carlo trials;
5. compare at least one limiting case with an analytic result; and
6. avoid assigning public example parameters to real equipment performance.

## Claims RadarSim does not support

RadarSim results alone cannot establish detection range of a real sensor, radar signature of a real vehicle, jammer effectiveness against a real system, operational tactics, safety certification, or hardware compliance. Such claims require traceable measured inputs, validated propagation and hardware models, uncertainty budgets, and independent review.
