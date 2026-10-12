# Conventions

These hold everywhere in the code, the scenario files and the interface.

| Quantity | Convention |
|---|---|
| Frame | local East-North-Up: x East, y North, z Up, metres |
| Altitude | z above the local ground plane; above mean sea level when terrain is loaded |
| Azimuth | from North, positive clockwise (towards East); returned in (−180°, 180°] |
| Elevation | from the horizontal plane, positive up |
| Heading | direction of horizontal motion, same convention as azimuth |
| Radial velocity | \(v_r = \dot R\), positive when the target recedes |
| Doppler | \(f_d = -2 v_r / \lambda\): closing targets have positive Doppler |
| Angles in code | radians; degrees only in fields ending `_deg` |
| Power | watts in linear expressions, dB / dBm in log form; \(10\log_{10}\) for power ratios |
| Time | seconds of simulated time; the engine advances in fixed steps `dt` |

The helpers in `radarsim.physics.geometry` (`polar_from_enu`, `enu_from_polar`,
`azimuth_from_enu`, `wrap_to_pi`) are the only place these conversions are written
out; everything else calls them.

The extended Kalman filter keeps its own internal polar angle (`atan2(y, x)`) and
is fed Cartesian positions by the tracker, so its convention never leaks out.
