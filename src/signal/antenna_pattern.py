# Developed by Mehmet Gümüş (@SpaceEngineerSS) - RadarSim v2.x
"""
Antenna Radiation Pattern Models

Provides analytical antenna gain patterns for radar beam simulation.

Models:
    - Sinc² (uniform aperture)
    - Taylor-weighted (controlled sidelobes)
    - Gaussian (pencil beam approximation)
    - Two-way (Tx × Rx) gain for monostatic radar

Physics:
    G(θ) = sinc²(π·D·sin(θ)/λ) for uniform rectangular aperture
    3dB beamwidth: θ₃ᵈᵇ ≈ 0.886·λ/D [rad]

References:
    - Skolnik, M.I. "Radar Handbook", 3rd Ed., Ch. 9
    - Balanis, C. "Antenna Theory", 4th Ed., Ch. 6
    - Developed by Mehmet Gümüş (github.com/SpaceEngineerSS)
"""


import numpy as np


class AntennaPattern:
    """
    Antenna radiation pattern calculator.

    Supports sinc, Taylor-weighted, and Gaussian beam models
    with two-way gain computation for monostatic radar.

    Reference: Skolnik, "Radar Handbook", 3rd Ed., Ch. 9
    """

    def __init__(self, beamwidth_deg: float = 2.0, sll_db: float = -30.0):
        """
        Initialize antenna pattern.

        Args:
            beamwidth_deg: 3-dB beamwidth [degrees]
            sll_db: Design sidelobe level [dB] (negative, e.g., -30)
        """
        self.beamwidth_deg = beamwidth_deg
        self.beamwidth_rad = np.radians(beamwidth_deg)
        self.sll_db = sll_db

        # Gaussian shape constant: G(θ₃ᵈᵇ/2) = 0.5 → k = 2.776/θ₃ᵈᵇ²
        self.k_gaussian = 2.776 / (self.beamwidth_rad**2)

    def sinc_pattern(self, theta_rad: float) -> float:
        """
        Uniform aperture (sinc²) gain pattern.

        G(θ) = sinc²(π·D·sin(θ)/λ) ≈ sinc²(1.39·θ/θ₃ᵈᵇ)

        The constant 1.39 comes from: for sinc² pattern,
        the 3dB point is at u = 0.4429, and D/λ ≈ 0.886/θ₃ᵈᵇ.

        Args:
            theta_rad: Off-boresight angle [rad]

        Returns:
            Normalized gain (0 to 1), linear scale
        """
        if abs(theta_rad) < 1e-10:
            return 1.0
        u = 1.39 * theta_rad / self.beamwidth_rad
        return float(np.sinc(u) ** 2)

    def gaussian_pattern(self, theta_rad: float) -> float:
        """
        Gaussian beam pattern (pencil beam approximation).

        G(θ) = exp(-k·θ²) where k = 2.776/θ₃ᵈᵇ²

        Exact at 3dB: G(θ₃ᵈᵇ/2) = 0.5 by construction.

        Args:
            theta_rad: Off-boresight angle [rad]

        Returns:
            Normalized gain (0 to 1)
        """
        return float(np.exp(-self.k_gaussian * theta_rad**2))

    def taylor_pattern(self, theta_rad: float, nbar: int = 4) -> float:
        """
        Taylor-weighted aperture pattern (approximate).

        Provides near-constant sidelobes at the design SLL level
        over the first n̄-1 sidelobes.

        Approximated by: G(θ) = sinc²(u) · taper(u)
        where taper suppresses sidelobes to SLL.

        Args:
            theta_rad: Off-boresight angle [rad]
            nbar: Number of nearly-constant-level sidelobes

        Returns:
            Normalized gain (0 to 1)

        Reference: Skolnik, Ch. 9.3
        """
        sinc_gain = self.sinc_pattern(theta_rad)

        # Sidelobe suppression taper
        sll_linear = 10 ** (self.sll_db / 20.0)
        u = 1.39 * theta_rad / self.beamwidth_rad

        # Beyond the main lobe, apply sidelobe cap
        if abs(u) > 1.0:
            max_sl = sll_linear**2
            return min(sinc_gain, max_sl)

        return sinc_gain

    def two_way_gain_db(self, theta_az_rad: float, theta_el_rad: float) -> float:
        """
        Two-way antenna gain for monostatic radar.

        G_2way(θ) = G_tx(θ) · G_rx(θ) = G²(θ) for monostatic

        In dB: G_2way_dB = 2 · 10·log10(G(θ))

        Args:
            theta_az_rad: Off-boresight azimuth angle [rad]
            theta_el_rad: Off-boresight elevation angle [rad]

        Returns:
            Two-way gain relative to boresight [dB] (always ≤ 0)
        """
        # Combined off-axis angle
        theta_total = np.sqrt(theta_az_rad**2 + theta_el_rad**2)
        g_one_way = self.gaussian_pattern(theta_total)
        g_one_way = max(g_one_way, 1e-15)
        return float(20.0 * np.log10(g_one_way))


def validate_antenna_patterns() -> dict:
    """
    Validate antenna pattern calculations.

    Tests:
        1. 3dB beamwidth: G(θ₃ᵈᵇ/2) = 0.5 (by definition)
        2. Boresight gain: G(0) = 1.0
        3. Two-way gain at boresight = 0 dB

    Reference: Skolnik, Ch. 9
    """
    pattern = AntennaPattern(beamwidth_deg=2.0)
    half_bw = pattern.beamwidth_rad / 2.0

    results = {}

    # Test 1: Boresight gain
    g0 = pattern.gaussian_pattern(0.0)
    results["boresight_gain"] = {
        "computed": g0,
        "expected": 1.0,
        "pass": abs(g0 - 1.0) < 1e-10,
    }

    # Test 2: 3dB point (Gaussian)
    g_3db = pattern.gaussian_pattern(half_bw)
    results["3db_point_gaussian"] = {
        "computed": g_3db,
        "expected": 0.5,
        "error": abs(g_3db - 0.5),
        "pass": abs(g_3db - 0.5) < 0.01,
    }

    # Test 3: Two-way gain at boresight
    g2w = pattern.two_way_gain_db(0.0, 0.0)
    results["two_way_boresight_db"] = {
        "computed_db": g2w,
        "expected_db": 0.0,
        "pass": abs(g2w) < 0.01,
    }

    return results


if __name__ == "__main__":
    results = validate_antenna_patterns()
    for name, r in results.items():
        status = "✓" if r["pass"] else "✗"
        print(f"{status} {name}: {r}")
