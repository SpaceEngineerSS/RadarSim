# Tracking

Code: `radarsim/tracking/tracker.py`, `kalman.py`, `ekf.py`, `monopulse.py`.

The engine feeds every detection, converted from measured range, azimuth and
elevation to a position on the ground plane, to a track-while-scan tracker.

## Filter

Each track is a constant-velocity Kalman filter on \([x, y, \dot x, \dot y]\) with
continuous white-noise acceleration (Bar-Shalom, Li & Kirubarajan 2001, §6.2.2):

\[
Q(\Delta t) = q\begin{bmatrix} \Delta t^3/3 & \Delta t^2/2 \\ \Delta t^2/2 & \Delta t \end{bmatrix}
\otimes I_2, \qquad q = \sigma_a^2 \cdot 1\,\mathrm{s}.
\]

Unlike the discrete-step variant, the covariance added over a revisit interval does
not depend on how many simulation steps it is split into. The default
\(\sigma_a\) = 10 m/s² suits transport aircraft; `close_air_combat` raises it to
40 m/s² to hold a 4.7 g turn. An extended Kalman filter with polar measurements and
SNR-dependent noise is available through `TrackManager.set_ekf_mode`.

## Association

Detections are assigned to tracks by global nearest neighbour (Hungarian algorithm)
with a χ² gate at 99.73 % on the normalised innovation squared. A track with only one
detection has no velocity yet, so it uses a velocity gate instead,
\(|z_2 - z_1| \le v_{max}\,\Delta t + 3\sqrt{2}\,\sigma_z\), and on the second hit is
initialised by two-point differencing (velocity from the two positions, covariance
from the measurement noise).

## Life cycle

- **tentative** after the first detection; dropped after two misses;
- **confirmed** after three consecutive hits (configurable);
- **coasting** after a miss, back to confirmed on the next hit;
- **lost** after three misses, or when it has not been updated for 3.5 revisit
  periods (a target that left a sector scan for good).

A miss is only charged when the beam actually looked at the track's predicted
position in that step and the track has not already been updated during the current
scan, so tracks between beam passes simply coast.

## Monopulse

With **Monopulse** on, angles come from an amplitude-comparison estimator instead of
fixed angle noise. Two Gaussian beams squinted by \(\pm\theta_s\) give

\[
\varepsilon(\theta) = \frac{\Delta}{\Sigma} = \tanh\!\left(\frac{4\ln 2\,\theta\,\theta_s}{\theta_3^2}\right),
\qquad k_m = \frac{4\ln 2\,\theta_s}{\theta_3},
\]

linear near boresight and saturating off axis. Thermal noise gives the accuracy
\(\sigma_\theta = \theta_3 / (k_m\sqrt{2\,\mathrm{SNR}})\) (Barton 2005), applied to
\(\varepsilon\) before inversion.

## Validity

- Single-model filters only (no IMM); hard manoeuvres need a larger \(\sigma_a\).
- Tracks are two-dimensional; altitude is not estimated.
