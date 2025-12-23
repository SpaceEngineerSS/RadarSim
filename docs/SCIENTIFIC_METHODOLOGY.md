# Scientific Methodology

This document describes the scientific basis, mathematical formulas, and validation methodology used in RadarSim.

## 1. Radar Equation

### 1.1 Standard Form

The simulation implements the monostatic radar range equation as defined in IEEE Std 686-2008:

$$P_r = \frac{P_t \cdot G_t \cdot G_r \cdot \lambda^2 \cdot \sigma}{(4\pi)^3 \cdot R^4 \cdot L_{sys}}$$

**Implementation:** `radar_physics.py` → `RadarPhysics.radar_equation()`

### 1.2 Extended Form with Atmospheric Attenuation

$$P_r = \frac{P_t \cdot G_t \cdot G_r \cdot \lambda^2 \cdot \sigma}{(4\pi)^3 \cdot R^4 \cdot L_t \cdot L_r \cdot L_a}$$

Where:
- $L_a$ = Atmospheric attenuation factor (ITU-R P.676)

**Reference:** Skolnik, M. (2008). *Radar Handbook*, 3rd Ed., Chapter 2.

---

## 2. Atmospheric Attenuation (ITU-R P.676-12)

### 2.1 Model Description

Atmospheric attenuation due to oxygen and water vapor absorption is modeled according to ITU-R Recommendation P.676-12.

$$\gamma_{total} = \gamma_o + \gamma_w \text{ (dB/km)}$$

$$A_{total} = 2 \cdot \gamma_{total} \cdot R \text{ (dB, two-way)}$$

### 2.2 Oxygen Absorption

Key resonance at 60 GHz. Simplified model for 1-100 GHz range.

### 2.3 Water Vapor Absorption

Key resonances at 22.235 GHz and 183.31 GHz.

**Implementation:** `radar_physics.py` → `ITU_R_P676` class

**Reference:** ITU-R P.676-12 (12/2017). *Attenuation by atmospheric gases*.

**Reference:** ITU-R P.676-12 (12/2017). *Attenuation by atmospheric gases*.

### 2.4 Rain Attenuation (ITU-R P.838-3)

Specific attenuation due to rain:

$$ \gamma_R = k \cdot R^\alpha \quad (\text{dB/km}) $$

Where $R$ is rain rate (mm/h) and $k, \alpha$ are frequency-dependent coefficients.

**Implementation:** `engine.py` using coefficients for X/K/Ka band.

---

## 3. Radar Cross Section (RCS)

### 3.1 Swerling Fluctuation Models

The simulation implements all four Swerling models plus the non-fluctuating (Marcum) case:

| Model | PDF | Decorrelation |
|-------|-----|---------------|
| Swerling 0 | Constant (δ) | - |
| Swerling I | Exponential | Scan-to-scan |
| Swerling II | Exponential | Pulse-to-pulse |
| Swerling III | Chi-squared (4 DoF) | Scan-to-scan |
| Swerling IV | Chi-squared (4 DoF) | Pulse-to-pulse |

### 3.2 Swerling I/II (Rayleigh)

$$p(\sigma) = \frac{1}{\sigma_{avg}} \exp\left(-\frac{\sigma}{\sigma_{avg}}\right)$$

Many independent scatterers of similar magnitude.

### 3.3 Swerling III/IV (Chi-squared)

$$p(\sigma) = \frac{4\sigma}{\sigma_{avg}^2} \exp\left(-\frac{2\sigma}{\sigma_{avg}}\right)$$

One dominant scatterer plus many small scatterers.

**Implementation:** `radar_physics.py` → `SwerlingRCS` class

**Reference:** Swerling, P. (1960). *Probability of Detection for Fluctuating Targets*. IRE Transactions.

**Reference:** Swerling, P. (1960). *Probability of Detection for Fluctuating Targets*. IRE Transactions.

---

## 4. Clutter Models

### 4.1 Ground Clutter (Weibull)
$$ P(\sigma) = \frac{k}{\lambda}\left(\frac{\sigma}{\lambda}\right)^{k-1} \exp\left[-\left(\frac{\sigma}{\lambda}\right)^k\right] $$

### 4.2 Sea Clutter (GIT Model)
Based on Douglas Sea State (SS) and grazing angle ($\psi$):
$$ \sigma^0_{sea} = f(SS, \psi, \lambda, \text{Polarization}) $$

### 4.3 Rain Volume Clutter
Based on Marshall-Palmer Z-R relationship:
$$ Z = 200 \cdot R^{1.6} $$
$$ \eta = \frac{\pi^5 |K|^2}{\lambda^4} Z \cdot 10^{-18} $$

---

## 4. Doppler Processing

### 4.1 Doppler Shift

$$f_d = \frac{2 v_r}{\lambda}$$

Where:
- $v_r$ = Radial velocity (m/s)
- $\lambda$ = Wavelength (m)

### 4.2 Radial Velocity Calculation

$$v_r = \frac{(\mathbf{p}_t - \mathbf{p}_r) \cdot (\mathbf{v}_t - \mathbf{v}_r)}{|\mathbf{p}_t - \mathbf{p}_r|}$$

**Implementation:** `radar_physics.py` → `RadarPhysics.doppler_shift()`

**Reference:** Skolnik, M. (2008). *Radar Handbook*, Chapter 3.

---

## 5. Monopulse Angle Tracking

### 5.1 Sum-Difference Processing

Angle error is extracted from the ratio of difference to sum patterns:

$$\theta_{error} = k_m \cdot \frac{\Delta}{\Sigma}$$

Where $k_m$ is the monopulse slope constant (typically 1.4-1.8).

### 5.2 Angular Accuracy

Thermal noise contribution to angle error:

$$\sigma_\theta = \frac{\theta_{3dB}}{k_m \sqrt{2 \cdot SNR}}$$

**Implementation:** `radar_physics.py` → `RadarPhysics.monopulse_angle_error()`

**Reference:** Skolnik, M. (2008). *Radar Handbook*, Chapter 9.

---

## 6. Extended Kalman Filter

### 6.1 State Vector

9-state constant acceleration model:

$$\mathbf{x} = [x, y, z, v_x, v_y, v_z, a_x, a_y, a_z]^T$$

### 6.2 Measurement Vector (Polar)

$$\mathbf{z} = [r, \theta_{az}, \theta_{el}, \dot{r}]^T$$

### 6.3 State Transition (Constant Acceleration)

$$\mathbf{F} = \begin{bmatrix} 1 & \Delta t & \frac{\Delta t^2}{2} \\ 0 & 1 & \Delta t \\ 0 & 0 & 1 \end{bmatrix} \otimes \mathbf{I}_3$$

### 6.4 Measurement Function

$$h(\mathbf{x}) = \begin{bmatrix} \sqrt{x^2 + y^2 + z^2} \\ \arctan(y/x) \\ \arctan(z/\sqrt{x^2+y^2}) \\ (xv_x + yv_y + zv_z)/r \end{bmatrix}$$

### 6.5 Jacobian

Computed analytically in `measurement_jacobian()`.

**Implementation:** `target_tracking.py` → `ExtendedKalmanFilter` class

**Reference:** Bar-Shalom, Li & Kirubarajan (2001). *Estimation with Applications to Tracking*.

---

## 7. Proportional Navigation Guidance

### 7.1 Pure Proportional Navigation (PPN)

$$\mathbf{a}_{cmd} = N' V_c \dot{\boldsymbol{\lambda}}$$

Where:
- $N'$ = Navigation ratio (typically 3-5)
- $V_c$ = Closing velocity
- $\dot{\boldsymbol{\lambda}}$ = Line-of-sight rate vector

### 7.2 LOS Rate

$$\boldsymbol{\omega} = \frac{\mathbf{R} \times \mathbf{V}}{|\mathbf{R}|^2}$$

**Implementation:** `target_tracking.py` → `GuidanceSystem` class

**Reference:** Zarchan, P. (2012). *Tactical and Strategic Missile Guidance*. AIAA.

---

## 8. Assumptions and Limitations

### 8.1 Simplifications

1. **Point targets** - No extended target modeling
2. **Flat Earth** - No Earth curvature consideration
3. **Simplified atmosphere** - No rain, clouds, or multipath
4. **Single sensor** - No multi-static configurations
5. **No terrain** - No ground clutter or shadowing

### 8.2 Numerical Precision

- All calculations use 64-bit floating point
- Covariance matrices maintain positive definiteness via Joseph form

### 8.3 Validity Range

| Parameter | Valid Range |
|-----------|-------------|
| Frequency | 1-100 GHz |
| Range | 100 m - 200 km |
| Velocity | 0 - 2000 m/s |
| Altitude | 0 - 50 km |

---

## 9. Validation

### 9.1 Unit Tests

Located in `tests/` directory:
- `test_radar_physics.py` - Radar equation validation
- `test_tracking.py` - EKF convergence tests
- `test_ecm.py` - ECM effectiveness tests

### 9.2 Reference Comparisons

Validated against:
- Published radar range equation examples (Skolnik)
- ITU-R P.676 attenuation tables
- Swerling Pd curves

---

## References

1. Skolnik, M.I. (2008). *Radar Handbook*, 3rd Edition. McGraw-Hill.
2. Bar-Shalom, Y., Li, X.R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation*. Wiley.
3. Blackman, S.S. & Popoli, R. (1999). *Design and Analysis of Modern Tracking Systems*. Artech House.
4. ITU-R P.676-12 (12/2017). *Attenuation by atmospheric gases*.
5. Swerling, P. (1960). Probability of Detection for Fluctuating Targets. *IRE Transactions on Information Theory*, IT-6(2), 269-308.
6. Zarchan, P. (2012). *Tactical and Strategic Missile Guidance*, 6th Edition. AIAA.
7. IEEE Std 686-2008. *IEEE Standard Radar Definitions*.
