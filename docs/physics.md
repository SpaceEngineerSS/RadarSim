# Physics Engine Documentation

This document describes the physics models implemented in RadarSim's core engine.

## Table of Contents
- [Radar Equation](#radar-equation)
- [Atmospheric Attenuation](#atmospheric-attenuation)
- [RCS Fluctuation Models](#rcs-fluctuation-models)
- [Terrain & Line-of-Sight](#terrain--line-of-sight)
- [References](#references)

---

## Radar Equation

### Standard Monostatic Form

The fundamental radar range equation calculates received power from a target:

$$P_r = \frac{P_t G_t G_r \lambda^2 \sigma}{(4\pi)^3 R^4 L}$$

Where:
| Symbol | Description | Units |
|--------|-------------|-------|
| $P_r$ | Received power | W |
| $P_t$ | Transmitted power | W |
| $G_t$ | Transmit antenna gain | linear |
| $G_r$ | Receive antenna gain | linear |
| $\lambda$ | Wavelength ($c/f$) | m |
| $\sigma$ | Radar Cross Section (RCS) | m² |
| $R$ | Target range | m |
| $L$ | System losses | linear |

**Implementation:** `src/physics/radar_equation.py::_calculate_signal_power_jit()`

### Signal-to-Noise Ratio (SNR)

$$SNR = \frac{P_r}{P_n} = \frac{P_r}{k T_0 B F_n}$$

Where:
- $k = 1.380649 \times 10^{-23}$ J/K (Boltzmann constant)
- $T_0 = 290$ K (Standard reference temperature)
- $B$ = Noise bandwidth (Hz)
- $F_n$ = Receiver noise figure (linear)

**Implementation:** `src/physics/radar_equation.py::_calculate_snr_jit()`

### Antenna Gain Approximation

For parabolic antennas, gain is approximated from beamwidth:

$$G \approx \frac{30,000}{\theta_{az} \times \theta_{el}}$$

Where $\theta$ is the 3-dB beamwidth in degrees.

**Reference:** Skolnik, "Radar Handbook", 3rd Ed., Eq. 6.9

---

## Atmospheric Attenuation

### ITU-R P.676-13 Model

RadarSim implements the Annex 1 line-by-line oxygen and water-vapour equations from
Recommendation ITU-R P.676-13 over 1-1000 GHz. The spectroscopic line tables, pressure
broadening, oxygen interference terms, dry-air continuum, and water-vapour continuum are
evaluated for the scenario temperature, dry-air pressure, and water-vapour density.

### Total Attenuation

$$A_{total} = (\gamma_o + \gamma_w) \times d \times 2$$

Where:
- $\gamma_o$ = Oxygen specific attenuation (dB/km)
- $\gamma_w$ = Water vapor specific attenuation (dB/km)
- $d$ = One-way path length (km)
- Factor of 2 accounts for two-way radar path

### Oxygen Absorption

Key absorption features:
- **60 GHz complex**: Strong O₂ resonance (~15 dB/km at peak)
- **118.75 GHz line**: Secondary O₂ absorption

The implementation is verified against the official ITU-R Study Group 3 validation workbook at
12, 20, 60, 90, and 130 GHz, separately for oxygen and water vapour.

### Water Vapor Absorption

Key absorption lines:
- **22.235 GHz**: Primary H₂O line
- **183.31 GHz**: Strong H₂O line (~30 dB/km in humid conditions)

**Implementation:** `src/physics/atmospheric.py::ITU_R_P676`

### Rain Attenuation (ITU-R P.838)

RadarSim now implements specific attenuation due to rain:

$$ \gamma_R = k \cdot R^\alpha \quad (\text{dB/km}) $$

Where:
- $R$: Rain rate (mm/hr)
- $k, \alpha$: Frequency-dependent coefficients (Horizontal polarization)

| Freq (GHz) | $k_H$ | $\alpha_H$ |
|------------|-------|------------|
| 10 (X-Band)| 0.01217| 1.2571 |
| 20 (K-Band)| 0.09164| 1.0568 |
| 35 (Ka-Band)| 0.3374| 0.9047 |

The implementation evaluates the P.838-3 curve-fit equations continuously from 1 to 1000 GHz
and combines horizontal and vertical coefficients for path elevation and polarization tilt.

**Implementation:** `src/physics/rain.py::ITU_R_P838`

---

## RCS Fluctuation Models

### Swerling Models

RadarSim implements all four Swerling models for realistic RCS fluctuation:

| Model | Distribution | Decorrelation | Physical Basis |
|-------|--------------|---------------|----------------|
| Swerling I | Exponential | Scan-to-scan | Many equal scatterers |
| Swerling II | Exponential | Pulse-to-pulse | Many equal, fast variation |
| Swerling III | Chi-squared (4 DOF) | Scan-to-scan | One dominant + many small |
| Swerling IV | Chi-squared (4 DOF) | Pulse-to-pulse | Dominant + small, fast |

### Implementation

```python
def apply_swerling_fluctuation(rcs_mean: float, model: int) -> float:
    if model == 1 or model == 2:
        # Exponential distribution (Rayleigh amplitude)
        return np.random.exponential(rcs_mean)
    elif model == 3 or model == 4:
        # Chi-squared with 4 DOF
        return np.random.gamma(2, rcs_mean / 2)
    else:
        return rcs_mean  # Swerling 0 (non-fluctuating)
```

**Implementation:** `src/physics/rcs.py::SwerlingModel`

---

## Terrain & Line-of-Sight

### 4/3 Earth Model

To account for atmospheric refraction, RadarSim uses the standard 4/3 effective Earth radius model:

$$R_e' = \frac{4}{3} R_e = 8495 \text{ km}$$

Where $R_e = 6371$ km is the mean Earth radius.

### Line-of-Sight Algorithm

The LOS check uses raymarching with 100+ steps:

```python
for i in range(num_steps):
    # Interpolate position along ray
    t = i / num_steps
    ray_x = radar_x + t * (target_x - radar_x)
    ray_y = radar_y + t * (target_y - radar_y)
    
    # Calculate ray height with Earth curvature
    ray_alt = radar_alt + t * (target_alt - radar_alt)
    ray_alt -= earth_curvature_drop(distance * t)
    
    # Check against terrain
    terrain_height = terrain_map.get_elevation(ray_x, ray_y)
    if ray_alt < terrain_height:
        return False  # Blocked by terrain
```

### Radar Horizon

Maximum detection range limited by horizon:

$$R_{horizon} = \sqrt{2 R_e' h}$$

For radar height $h = 100$ m: $R_{horizon} \approx 41$ km (one-way)

**Implementation:** `src/physics/terrain.py::TerrainMap`

---

## Monopulse Tracking

### Sum/Difference Patterns

RadarSim implements amplitude-comparison monopulse for sub-beamwidth angular accuracy.

$$ \Sigma(\theta) = 2 \cos\left(\frac{\pi d}{\lambda}\sin\theta\right) \cdot \text{sinc}\left(\frac{\pi D}{\lambda}\sin\theta\right) $$

$$ \Delta(\theta) = 2j \sin\left(\frac{\pi d}{\lambda}\sin\theta\right) \cdot \text{sinc}\left(\frac{\pi D}{\lambda}\sin\theta\right) $$

### Error Signal & Accuracy

The error signal $\epsilon = \Delta / \Sigma$ provides a linear response near boresight, allowing for precise angle estimation.

**Theoretical Accuracy:**
$$ \sigma_\theta = \frac{\theta_{3dB}}{k_m \sqrt{2 \cdot SNR}} $$
Where $k_m \approx 1.6$ is the monopulse slope.

**Implementation:** `src/tracking/monopulse.py`

---

## Ambiguity Analysis

### Maximum Unambiguous Values

- **Range:** $R_{max} = c / (2 \cdot PRF)$
- **Velocity:** $V_{max} = (\lambda \cdot PRF) / 4$

### Blind Speeds

Targets traveling at multiples of the blind speed will appear stationary (zero Doppler) due to aliasing:
$$ V_{blind} = n \cdot \frac{\lambda \cdot PRF}{2} $$

**Implementation:** `src/ui/analysis/ambiguity_plot.py`

---

## Environmental Clutter

Surface clutter is formed from normalized backscatter and the common range-antenna resolution
cell:

$$\sigma_c = \sigma^0 A_c$$

Thermal noise and clutter are combined as powers rather than subtracting CNR in decibels:

$$\frac{1}{\mathrm{SINR}} = \frac{1}{\mathrm{SNR}} + \frac{\sigma_c}{\sigma_t}$$

The result record exposes the selected model, $\sigma^0$, cell area, clutter RCS, and resulting
SINR loss.

### Land clutter

The default constant-gamma screening model is

$$\sigma^0 = \gamma\sin\psi$$

where $\psi$ is grazing angle. Terrain-name gamma values are nominal priors and do not encode
site moisture, season, cultivation, frequency, or spatial resolution. A measured `gamma_db`
should be supplied for quantitative studies.

The optional Oh-Sarabandi-Ulaby 1992 model computes HH, VV, and HV normalized backscatter from
complex relative permittivity and RMS surface height. Its enforced domain is bare soil, L/C/X
band, 10-70 degree incidence, and $0.1 \le ks \le 6$. It is not used below that incidence-domain
boundary and is not presented as a low-angle land-clutter model.

### Sea and rain clutter

Sea reflectivity uses the NRL five-parameter model for HH/VV polarization, 0.5-35 GHz,
0.1-60 degree grazing, Douglas sea states 0-6. Rain volume reflectivity uses the
Marshall-Palmer $Z=200R^{1.6}$ relationship. These amplitude models are separate from
ITU-R P.838 path attenuation.

**Implementation:** `src/physics/clutter.py`

---


## References

1. **Skolnik, M.I.** (2008). *Radar Handbook*, 3rd Edition. McGraw-Hill.
   - Chapter 2: Radar equation fundamentals
   - Chapter 6: Antenna theory
   - Chapter 7: RCS models

2. **ITU-R P.676-12** (2019). *Attenuation by atmospheric gases*. International Telecommunication Union.

3. **Barton, D.K.** (2005). *Radar System Analysis and Modeling*. Artech House.
   - Swerling models derivation

4. **Blake, L.V.** (1980). *Radar Range-Performance Analysis*. Artech House.
   - 4/3 Earth model origins

5. **Oh, Y., Sarabandi, K., & Ulaby, F.T.** (1992). "An Empirical Model and an
   Inversion Technique for Radar Scattering from Bare Soil Surfaces." IEEE TGRS.
   DOI: 10.1109/36.134086.

6. **Gregers-Hansen, V., & Mital, R.** (2012). "An Improved Empirical Model for Radar
   Sea Clutter Reflectivity." IEEE TAES. DOI: 10.1109/TAES.2012.6324732.

---

*Document generated for RadarSim v1.0.0*
