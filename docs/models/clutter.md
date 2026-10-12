# Clutter and MTI

Code: `radarsim/physics/clutter.py`, `radarsim/simulation/engine.py`.

## Surface reflectivity

| Surface | Model | Domain |
|---|---|---|
| land (`rural`, `urban`, `forest`, …) | constant-γ: \(\sigma^0 = \gamma \sin\psi\), γ from a short table or a measured value | any grazing angle |
| bare soil (`ground_surface.model: oh1992`) | Oh, Sarabandi & Ulaby (1992) empirical model | 1–10 GHz, 10–70° incidence, \(0.1 \le ks \le 6\) |
| sea | NRL five-parameter model (Gregers-Hansen & Mital 2012), HH and VV | 0.5–35 GHz, sea state 0–6, 0.1–60° grazing |

Outside the Oh-1992 incidence range (a ground radar looking at a distant target sees
grazing angles of a fraction of a degree) the engine falls back to constant-γ
instead of extrapolating. Polarisation tilt mixes the HH and VV values. Clutter
sectors in a scenario change the surface class and add a calibration offset for
regions of azimuth and range.

## Clutter in the target's cell

The surface patch sharing the target's range cell has area

\[
A_c \approx \frac{\pi}{4}\, \min\!\left(\frac{c\tau_c}{2\cos\psi},\,
\frac{2R\tan(\theta_{el}/2)}{\sin\psi}\right) \cdot 2R\tan\!\left(\frac{\theta_{az}}{2}\right)
\]

with the compressed pulse length \(\tau_c\). The beam is pointed at the target, so
the patch, at depression angle \(-\psi\), is seen through the elevation pattern
\(g(\Delta) = \max\{e^{-4\ln 2\,\Delta^2/\theta_{el}^2},\,10^{SLL/10}\}\) at
\(\Delta = \epsilon_t + \psi\), twice (transmit and receive):

\[
\sigma_c = \sigma^0 A_c\, g(\Delta)^2 .
\]

A low flyer competes with the full main-beam clutter; an aircraft several
beamwidths up sees it only through the sidelobes. The test suite checks that a
target at 6 km altitude loses at least 10 dB less to clutter than one at 50 m at the
same range.

## MTI

A two-pulse canceller removes returns near zero Doppler. Its improvement factor
against clutter with a Gaussian spectrum of width \(\sigma_c = 2\sigma_v/\lambda\)
(Skolnik 2008, eqs. 2.41–2.42) is

\[
I_1 = 2\left(\frac{f_r}{2\pi\sigma_c}\right)^2, \qquad
I = \left(\frac{1}{I_1} + \frac{1}{I_{s}}\right)^{-1},
\]

limited by the system stability \(I_s\) = 50 dB. The clutter velocity spread
defaults to 0.25 m/s over land and 1 m/s over sea (Nathanson 1999) and can be set
per scenario. Targets whose radial speed is below the MTI threshold are cancelled
with the clutter.

## Validity

- Clutter is treated as noise-like in the SINR (no Weibull/K-distribution tails in
  the detection decision); the distributions are available as samplers in
  `ClutterModel` for signal-level work.
- One clutter patch per target cell; range-ambiguous clutter from earlier pulses is
  not added.
