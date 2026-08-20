# Signal processing

## LFM pulse model

`PulseDopplerProcessor` generates a unit-envelope baseband linear-FM reference pulse sampled at `sample_rate_hz`. A target echo is inserted at two-way delay \(2R/c\), scaled as a complex voltage, and phase-advanced pulse to pulse by its Doppler frequency. Fractional delay is represented by the carrier phase associated with the residual delay after integer sample placement. Independent circular complex Gaussian noise is added in I and Q.

The fast-time recording window is finite. A target whose delayed pulse does not intersect that window is rejected rather than wrapped to the beginning of the array. Target range must therefore be inside the instrumented range.

## Matched filtering and range

Range compression is a linear convolution with the conjugate time-reversed reference pulse. The valid delay origin is corrected for the reference length, so range bin \(n\) maps to \(R_n=cn/(2f_s)\). Nominal LFM range resolution is \(c/(2B)\). Sampling controls bin spacing; it does not improve physical resolution beyond the bandwidth limit.

## Slow-time processing

The Doppler transform operates across pulses after optional one- or two-delay MTI cancellation. Windows are rectangular, Hann, Hamming, and Taylor. Coherent gain is included in amplitude calibration. Equivalent noise bandwidth is \(N\sum w_n^2/(\sum w_n)^2\) bins.

Shifted FFT frequencies are calculated with `numpy.fft.fftfreq`. Velocity is \(v=\lambda f_d/2\), and the uniform-PRF unambiguous interval is \([-\lambda\mathrm{PRF}/4,\lambda\mathrm{PRF}/4)\). Unambiguous range is \(c/(2\mathrm{PRF})\). Velocities outside the interval alias; the map reports the ambiguity rather than relabelling it as true velocity.

## CFAR detectors

All CFAR input is converted to power. For exponentially distributed, independent reference-cell power, CA-CFAR uses \(\alpha=N(P_{fa}^{-1/N}-1)\), where \(N\) is total reference cells. GO- and SO-CFAR solve their split-window false-alarm equations. OS-CFAR solves the order-statistic equation for the selected rank.

One-dimensional detection excludes guard cells and the cell under test. Edge cells without a complete window are invalid. Two-dimensional CA uses a rectangular reference ring around a rectangular guard region; it is not repeated 1-D processing.

Requested false-alarm probability is achieved only under calibration assumptions. Correlated samples, spectral leakage, clutter edges, interfering targets, and non-Gaussian clutter change the achieved rate. GO is conservative at many clutter edges; SO can protect a weaker side in some transitions; OS resists a limited number of interfering targets depending on rank.

## Detection amplitude calibration

`amplitude_for_output_snr` maps a requested post-processing SNR to an input complex amplitude using matched-filter energy, Doppler coherent gain, and noise power. It is for controlled verification and does not replace the radar equation in physical scenarios.

## SAR processing

The stripmap generator synthesizes raw LFM echoes from point scatterers along a straight broadside aperture. Every pulse uses its platform-to-scatterer slant range and phase \(-4\pi R/\lambda\). Fast time is range-gated around the scene.

The range-Doppler algorithm performs range matched filtering, azimuth FFT, fractional non-circular range-cell migration correction, range-dependent azimuth filtering, inverse azimuth FFT, and coordinate calibration. Nominal slant-range resolution is \(c/(2B)\); nominal broadside stripmap azimuth resolution is approximately half the physical antenna length under these assumptions. Omega-k and chirp-scaling entry points raise `NotImplementedError`.

## ISAR processing

ISAR range profiles are compressed and aligned with zero-filled integer shifts, avoiding circular wraparound. Slow-time windowing and Doppler transformation form cross range. For constant target angular rate \(\omega\), \(x=\lambda f_d/(2\omega)\). When \(\omega\) is zero or poorly known, metric cross range is not physically identifiable. Severe nonuniform rotation requires a more advanced autofocus or time-frequency method.
