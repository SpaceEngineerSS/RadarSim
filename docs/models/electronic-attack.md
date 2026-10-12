# Electronic attack

Code: `radarsim/physics/ecm.py`, `radarsim/simulation/engine.py`.

Every jammer-equipped target uses its own technique from the scenario
(`ecm_type`). The **Jammers on** switch turns all of them on or off, and
**Technique** can force one technique on every jammer to compare them.

## Noise jamming

For a self-screening jammer of ERP \(P_j G_j\) and bandwidth \(B_j\) against a
radar of bandwidth \(B_r\) (Schleher 1999, §4.2):

\[
\frac{J}{S} = \frac{P_j G_j\, 4\pi R^2\, B_r}{P_t G_t \sigma\, B_j}\cdot\frac{L_s}{L_j},
\qquad B_r/B_j \le 1 .
\]

The jammer's one-way path suffers half the propagation and system loss of the
radar's two-way path. Jamming enters the SINR after the matched filter divided by the
compression gain, like thermal noise ([Detection](detection.md)). Spot and barrage
jammers differ only in bandwidth. With frequency agility on, the jammer has to spread
over 10 % of the carrier frequency.

The burn-through range, where \(S/J\) reaches the required SNR, is

\[
R_{bt} = \sqrt{\frac{P_t G_t \sigma\, B_j}{4\pi P_j G_j\, \mathrm{SNR}_{req}\, B_r}} .
\]

The self-check `validate_burn_through_range` confirms that `calculate_jsr` gives
exactly \(-\mathrm{SNR}_{req}\) at this range. The event log reports when a jammed
target is first detected despite jamming (*burn-through*).

## DRFM deception

A digital RF memory captures the radar pulse and plays it back, delayed (range gate
pull-off, RGPO) or Doppler-shifted (velocity gate pull-off, VGPO), with a gain over
the skin return. The jammer state machine (idle → capture → pull → reset) uses the
dwell, pull rate and maximum pull from the scenario. The false return appears as a
ghost target moving away from the real one along the line of sight; in the
pulse-Doppler processor it is injected phase-coherently into the CPI.

## Chaff and decoys

Chaff clouds and decoys are radar objects in their own right, released at a given
time, optionally from a named aircraft's position. Chaff has a large, Swerling 2
cross-section (many independently moving dipoles) that blooms and then decays,

\[
\sigma(t) = \sigma_{max}\,(1 - e^{-t/\tau_b})\,e^{-t/\tau_d},
\]

and falls slowly. A decoy flies its own course with its own RCS. Both are detected
and tracked exactly like aircraft, which is the point: the radar cannot tell them
apart by detection alone.

## Validity

- No jammer antenna pattern or sidelobe jamming geometry beyond the self-screening
  case; stand-off jammers are not modelled.
- Noise jamming is treated as white Gaussian noise in the receiver band.
