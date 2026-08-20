# ECM, ECCM, and receiver effects

RadarSim represents generic public-domain mechanisms. It does not model a named jammer, operational waveform library, threat database, or classified electronic-order-of-battle data.

## Noise jamming

Jammer-to-signal ratio at the radar is calculated from separate one-way jammer and two-way target paths. The target echo follows \(R_t^{-4}\); direct jammer illumination follows \(R_j^{-2}\). Transmit powers, antenna gains, bandwidth sharing, target RCS, wavelength, and system losses remain explicit.

For signal-to-noise ratio \(S/N\) and jammer-to-noise ratio \(J/N\), RadarSim uses

\[
\frac{S}{J+N}=\frac{S/N}{1+J/N}.
\]

This exact linear relationship is converted back to dB. Subtracting JSR from SNR is only an interference-dominant approximation and is not used as the general formula.

Spot and barrage modes differ through effective jammer bandwidth overlap. Strobes are angular observations with configurable bearing error; they are not target-range measurements.

## Chaff

A chaff cloud has position, velocity, mean RCS, dispersion, deployment time, and lifetime. The simulation returns a point/volume-equivalent false echo while the cloud is active. It does not resolve individual dipoles, polarization resonance, wind shear, or cloud microphysics.

## DRFM deception

`DRFMJammer` is a state machine with capture, track, pull-off, and hold states. Configuration includes:

- gain over skin return;
- capture dwell;
- RGPO range pull rate and maximum offset;
- VGPO Doppler pull rate and maximum offset;
- inherent retransmission delay; and
- RGPO or VGPO operation per jammer instance.

Range delay and apparent offset obey \(\Delta R=c\Delta t/2\); velocity pull is mapped through \(f_d=2v/\lambda\). Pull offsets are integrated with simulation time and bounded by the configured maxima. False-target positions are generated along the instantaneous radar-to-target line, so moving or displaced radars do not create world-origin artefacts.

`inject_into_cpi` applies delayed replicas without circular wrap. A replica outside the recorded CPI is truncated or absent. This prevents a delayed false target from reappearing at an impossible early range.

The model omits DRFM quantization, ADC/DAC spurs, oscillator phase noise, memory depth, threat-word recognition, antenna coupling, and waveform-specific coherent cancellation.

## Burn-through

Burn-through range is obtained by solving the generic radar/jammer link relationship at a requested SJNR. It is meaningful only when the assumed antenna gains, bandwidth overlap, propagation, and jammer geometry are supplied consistently. It is not a universal property of a radar or jammer.

## Receiver hard limiting

The simulation sums desired signal, noise, and interference power at the receiver input and compares it with `receiver_full_scale_dbm`. Above full scale, a memoryless radial limiter constrains complex-envelope magnitude while retaining phase.

For a zero-mean circular complex Gaussian input, the limiter diagnostic uses the corresponding Bussgang decomposition: output equals a scaled input plus uncorrelated distortion. RadarSim reports input power, limited output power, linear gain, distortion power, and saturation state. This closed-form result is exact for the assumed Gaussian input statistics; deterministic multitone or pulsed inputs require sample-level nonlinear simulation for exact spectral products.

## ECCM interpretation

Frequency agility, sidelobe control, CFAR, MTI, tracking gates, and receiver headroom can be explored as counter-countermeasure mechanisms, but RadarSim does not automatically claim that a detection is “ECCM successful.” Evaluation should compare controlled runs using fixed seeds and state the metric: achieved detection probability, false-track rate, track covariance, coast duration, or image quality.
