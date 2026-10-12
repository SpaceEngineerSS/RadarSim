# Propagation and terrain

Code: `radarsim/physics/atmospheric.py`, `rain.py`, `terrain.py`.

## Gaseous absorption

Specific attenuation of oxygen and water vapour follows the line-by-line method of
ITU-R P.676-13 Annex 1 (44 oxygen and 35 water-vapour lines plus the dry continuum),
valid from 1 to 1000 GHz. The test suite reproduces the ITU validation workbook to
12 significant digits. The Recommendation is written in terms of the dry-air
pressure \(p\); scenarios give the barometric pressure, and the engine passes
\(p = P - e\) with the water-vapour pressure \(e = \rho T / 216.7\).

The path loss is \(2\gamma R\) for the two-way path. Below 1 GHz gaseous absorption
is negligible and is set to zero.

## Rain

Specific attenuation \(\gamma_R = k R^{\alpha}\) uses the ITU-R P.838-3 coefficients
for the radar's frequency and polarisation tilt. Rain only fills the layer below the
rain height \(h_R = h_0 + 0.36\) km (ITU-R P.839-4, \(h_0 = 3\) km by default), so
the attenuated length is the part of the slant path below \(h_R\). A target above the
rain layer sees less attenuation and no rain clutter in its own cell.

Rain also returns power: the volume reflectivity from the Marshall–Palmer
Z–R relation, \(Z = 200 R^{1.6}\), \(\eta = \pi^5 |K|^2 Z / \lambda^4\), times a
resolution cell of \((\pi/4) R^2 \theta_{az}\theta_{el}\,c\tau_c/2\).

## Terrain and line of sight

Terrain height is procedural fractal noise (octaves, persistence and lacunarity are
configurable) plus optional Gaussian peaks of 10 km standard deviation. The same
function feeds the RHI profile, the 3D view and the line-of-sight test.

Line of sight marches along the straight ray from radar to target. Heights in a
scenario are altitudes above the curved surface, so at ground distance \(d_1\) from
the radar and \(d_2\) from the target the ray must clear

\[
h_{terrain}(x, y) + \frac{d_1 d_2}{2 k R_E}, \qquad k = 4/3,
\]

the Earth bulge of the 4/3-radius model (Skolnik 2008, §2.12). A masked target gets
no SNR and cannot be detected; the event log records when it becomes masked and when
it reappears.

## Validity

- Propagation is straight-line with the standard 4/3 Earth; ducting and multipath
  lobing are not modelled.
- P.676 is used along a horizontal path at the given pressure and temperature;
  slant paths through the atmosphere are not integrated layer by layer.
