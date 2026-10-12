"""
Coordinate-frame conventions and angle helpers shared by every RadarSim module.

Frame: local East-North-Up (ENU), metres.
    x = East, y = North, z = Up (altitude above the local ground plane).

Angles:
    azimuth   measured from North, positive clockwise (towards East), radians.
              Returned in (-pi, pi] so sector limits such as (-60, 60) deg work directly.
    elevation measured from the horizontal plane, positive up, radians.
    heading   direction of horizontal motion, same convention as azimuth.

Radial velocity is positive for a receding target (range rate dR/dt > 0).

Reference: IEEE Std 686-2017 (radar terminology); ENU frame as in
Bar-Shalom, Li & Kirubarajan (2001), Sec. 1.4.
"""

from typing import Tuple

import numpy as np


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    wrapped = (angle_rad + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.pi if wrapped == -np.pi else wrapped)


def azimuth_from_enu(east_m: float, north_m: float) -> float:
    """Azimuth of a horizontal ENU vector, clockwise from North, in (-pi, pi]."""
    return float(np.arctan2(east_m, north_m))


def heading_from_velocity(velocity: np.ndarray) -> float:
    """Heading of a velocity vector (same convention as azimuth)."""
    return azimuth_from_enu(velocity[0], velocity[1])


def enu_from_polar(range_m: float, azimuth_rad: float, elevation_rad: float = 0.0) -> np.ndarray:
    """Convert slant range, azimuth and elevation to an ENU offset vector."""
    ground_range = range_m * np.cos(elevation_rad)
    return np.array(
        [
            ground_range * np.sin(azimuth_rad),
            ground_range * np.cos(azimuth_rad),
            range_m * np.sin(elevation_rad),
        ]
    )


def polar_from_enu(delta: np.ndarray) -> Tuple[float, float, float]:
    """Return (slant range, azimuth, elevation) of an ENU offset vector."""
    ground_range = float(np.hypot(delta[0], delta[1]))
    return (
        float(np.linalg.norm(delta)),
        azimuth_from_enu(delta[0], delta[1]),
        float(np.arctan2(delta[2], ground_range)),
    )
