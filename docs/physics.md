# Physics models

RadarSim uses SI units internally. Angles passed to physics functions are radians unless a name ends in `_deg`; powers are watts unless a name ends in `_dbm`; gains and losses bearing `_db` are power ratios in decibels.

## Monostatic radar equation

For a point target at slant range \(R\), the received power is

\[
P_r = \frac{P_t G_t G_r \lambda^2 \sigma}{(4\pi)^3 R^4 L_t L_r L_a}.
\]

`P_t` is peak transmit power, `G_t` and `G_r` are linear antenna gains, `lambda` is wavelength, `sigma` is RCS, `L_t` and `L_r` are transmit and receive loss factors, and `L_a` is the accumulated two-way atmospheric loss factor. The implementation floors range at 1 m to avoid a singular point-target equation at zero separation.

Receiver noise power is \(P_n=kT_sB_nF\), where \(k\) is the exact SI Boltzmann constant, \(T_s\) is system noise temperature, \(B_n\) is equivalent noise bandwidth, and \(F\) is receiver noise factor. If no explicit noise bandwidth is supplied, the scalar radar API uses \(1/\tau\). This is a modelling default, not a claim that every matched filter has exactly that bandwidth.

The displayed single-look SNR is \(10\log_{10}(P_r/P_n)\). Noncoherent pulse integration is handled by the detection model, so pulse count is not silently multiplied into received power.

## Detection threshold and probability

`calculate_required_snr_albersheim` implements the Albersheim closed-form approximation for noncoherent integration using requested \(P_d\), \(P_{fa}\), and pulse count. Inputs are constrained to open probabilities and a positive integer pulse count. It is an approximation to square-law, noncoherent detection and should not be used as an exact coherent-detector threshold.

Probability of detection supports nonfluctuating and Swerling I–IV targets. Swerling I/II use exponential RCS power; III/IV use a gamma distribution with shape two. I and III remain fixed over a coherent processing interval; II and IV decorrelate pulse to pulse. Monte Carlo tests check means and variances against the analytic distributions.

## Range and Doppler geometry

Slant range is the Euclidean norm of the radar-to-target position vector. Radial velocity is \(v_r=(\mathbf v_t-\mathbf v_r)\cdot\hat{\mathbf R}\), with positive velocity defined as receding. Monostatic Doppler is \(f_d=2v_r/\lambda\). Consequently, approaching targets have negative Doppler in the physics API.

## Atmospheric gases and rain

Gaseous loss follows the line-by-line dry-air and water-vapour specific-attenuation structure in ITU-R P.676. Frequency, pressure, temperature, and water-vapour density are explicit. The homogeneous-path result is multiplied by path length and by two for a monostatic round trip.

Rain specific attenuation follows ITU-R P.838: \(\gamma_R=kR^\alpha\) dB/km. The coefficients are interpolated by frequency and combined for the configured polarization tilt. Path integration assumes uniform rain. It does not include a melting layer, spatially varying rain cells, radome wetting, or cloud/fog attenuation.

## RCS and aspect

User-provided `rcs_m2` is a mean point-target RCS. The optional aspect model applies a simple nose/beam/tail interpolation before Swerling fluctuation. It is useful for sensitivity studies but is not a physical-optics or method-of-moments solver. Target-type defaults are generic illustrative medians and must not be interpreted as authoritative signatures for named platforms.

## Land and sea clutter

Normalized surface backscatter \(\sigma^0\) is converted to clutter RCS by multiplying by the illuminated surface resolution-cell area. The area model uses slant-range resolution, azimuth beamwidth, and grazing-angle projection; values near zero grazing are bounded to avoid a singular footprint.

Land options are:

- `gamma`: an empirical terrain-category backscatter level with grazing-angle dependence.
- `oh1992`: the Oh, Sarabandi, and Ulaby co-polarized bare-soil formulation using wavelength, incidence angle, complex relative permittivity, and RMS height. Its published domain is bare soil measured at L, C, and X bands and incidence angles of roughly 10–70 degrees; extrapolation is not validated.

The sea option follows the NRL 2012 empirical reflectivity model, including frequency, grazing angle, polarization, wind/sea-state terms, and propagation-factor selection. Inputs outside the report’s empirical region are rejected or bounded as documented in the function.

Weibull and compound K-distributed generators provide stochastic amplitudes for clutter experiments. They do not make the empirical mean-backscatter model itself stochastic. Rain reflectivity uses the Marshall–Palmer drop-size relation for a volume-clutter estimate.

## Terrain and line of sight

The terrain module provides deterministic synthetic height fields, bilinear elevation lookup, and sampled line-of-sight masking. It is a geometric obstruction model. Earth curvature, refraction, diffraction, digital-elevation-dataset accuracy, and land-cover scattering are outside its present scope.

## Numerical safeguards

Physical inputs are validated before logarithms, divisions, filter design, or distribution sampling. Covariances are symmetrized after updates; linear systems are solved rather than explicitly inverted where possible. Floors used solely to prevent floating-point singularities are documented at the corresponding API.
