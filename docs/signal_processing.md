# Signal Processing Documentation

## Overview

RadarSim implements standard radar signal processing algorithms including detection, filtering, and imaging.

---

## Table of Contents
- [1. CFAR Detection](#1-cfar-detection)
- [2. Pulse Compression](#2-pulse-compression)
- [3. Doppler Processing](#3-doppler-processing)
- [4. Range-Doppler Maps](#4-range-doppler-maps)
- [5. MTI Filtering](#5-mti-filtering)
- [6. Clutter Models](#6-clutter-models)
- [7. SAR Image Formation](#7-sar-image-formation)
- [8. Probability of Detection](#8-probability-of-detection)
- [9. B-Scope ECM Strobe](#9-b-scope-ecm-strobe-visualization)
- [10. ROC Curve Analysis](#10-roc-curve-analysis)
- [11. SNR Histogram](#11-snr-histogram-analysis)
- [References](#references)

---

## 1. CFAR Detection

**File:** `src/signal/cfar.py`

### Cell-Averaging CFAR (CA-CFAR)

```
┌─────────────────────────────────────────────────────────┐
│ Ref Cells │ Guard │ CUT │ Guard │ Ref Cells │
│   (N/2)   │  (G)  │     │  (G)  │   (N/2)   │
└─────────────────────────────────────────────────────────┘
```

**Threshold:**
$$T = \alpha \cdot \frac{1}{N} \sum_{i=1}^{N} x_i$$

Where $\alpha$ is set by desired $P_{fa}$:
$$\alpha = N \cdot (P_{fa}^{-1/N} - 1)$$

### A-Scope Visualization

The A-Scope (Phase 23) shows CFAR cells on mouse hover:
- **Red line:** Cell Under Test (CUT)
- **Grey region:** Guard cells
- **Green region:** Reference cells

---

---

## 2. Pulse Compression

### Linear Frequency Modulated (LFM) Chirp

RadarSim generates LFM waveforms for improved range resolution:

$$ s(t) = A \cdot \exp\left(j2\pi \left(f_0 t + \frac{\beta t^2}{2}\right)\right) $$

Where $\beta = B/T$ is the chirp rate.

### Matched Filter

The matched filter output maximizes SNR: $y(t) = x(t) * s^*(-t)$.

**Pulse Compression Ratio:** $PCR = B \cdot T$ (e.g., $100 = 20$ dB gain).

**Implementation:** `src/signal/waveforms.py::LFMWaveform`

---

## 3. Doppler Processing

### Doppler Shift

For monostatic radar with radial velocity $v_r$:

$$ f_d = \frac{2 v_r}{\lambda} = \frac{2 v_r f_0}{c} $$

### Velocity Resolution & Aliasing

- **Resolution:** $\Delta v = \lambda / (2 \cdot N \cdot T_{PRI})$
- **Max Unambiguous Velocity:** $v_{max} = \pm (PRF \cdot \lambda) / 4$

**Implementation:** `src/physics/radar_equation.py`

---

## 4. Range-Doppler Maps

### Generation Process

1.  **Collect CPI:** Gather N pulses of raw I/Q data.
2.  **Range FFT:** FFT along fast-time (range bins).
3.  **Doppler FFT:** FFT along slow-time (pulses).
4.  **CFAR:** Apply 2D CFAR detection.

**Implementation:** `src/ui/range_doppler.py`

---

## 5. MTI Filtering

**File:** `src/simulation/engine.py`

### Velocity Threshold

$$V_{threshold} = 2 \text{ m/s}$$ (default)

Targets with radial velocity below threshold are rejected as clutter.

**Logic:**
```python
if abs(target_velocity_radial) < mti_threshold:
    target.is_detected = False  # Clutter
```

---

## 6. Clutter Models

**File:** `src/physics/clutter.py`

### Ground Clutter (Weibull)

$$P(\sigma) = \frac{k}{\lambda}\left(\frac{\sigma}{\lambda}\right)^{k-1} \exp\left(-\left(\frac{\sigma}{\lambda}\right)^k\right)$$

### Sea Clutter (GIT Model)

Uses Georgia Tech Research Institute (GIT) empirical model based on Douglas Sea State:

$$ \sigma^0 = f(SS, \theta_{grazing}, \lambda, pol) $$

Accounts for:
- Sea State (0-6)
- Grazing angle
- Wind direction (implied)

### Rain Clutter (Marshall-Palmer)

$$Z = 200 \cdot R^{1.6}$$ (Z-R relationship)

$$\eta = \frac{\pi^5 |K|^2}{\lambda^4} Z$$

---

## 7. SAR Image Formation

**File:** `src/advanced/sar_isar.py`

### Range Resolution

$$\delta_r = \frac{c}{2B}$$

Where $B$ = bandwidth [Hz].

### Azimuth Resolution

$$\delta_a = \frac{D}{2}$$

Where $D$ = antenna length [m].

### Algorithms Implemented

| Algorithm | Description |
|-----------|-------------|
| Range-Doppler (RDA) | Standard SAR processing |
| Backprojection (BPA) | Time-domain focusing |
| Omega-K (ωK) | Wavenumber domain |
| Chirp Scaling (CSA) | Efficient, no interpolation |

### Key Equations

**Doppler Rate:**
$$K_a = -\frac{2v^2}{\lambda R_0}$$

**Range Cell Migration:**
$$\Delta R = \frac{\lambda^2 R_0 f_d^2}{8v^2}$$

**Stolt Interpolation:**
$$K_r' = \sqrt{(k_0 + K_r)^2 - K_a^2} - k_0$$

---

## 8. Probability of Detection

**File:** `src/physics/metrics.py`

### Albersheim's Equation

Required SNR for given $P_d$ and $P_{fa}$:

$$SNR = A + 0.12AB + 1.7B$$

Where:
- $A = \ln(0.62 / P_{fa})$
- $B = \ln(P_d / (1 - P_d))$

### Swerling Detection Probability

For Swerling I targets:
$$P_d = \exp\left(-\frac{T}{1 + SNR}\right)$$

---

## 9. B-Scope ECM Strobe Visualization

**File:** `src/ui/b_scope.py`

When jamming is detected, vertical "noise bars" appear at the jammer azimuth:

```
    Jammer at 30°
         │
         ▼
┌────────║─────────────────────────────────┐
│████████║████████                         │  Range
│████████║████████                         │   ↑
│████████║████████  ← Yellow strobe        │
│████████║████████                         │
└────────║─────────────────────────────────┘
         Azimuth →
```

**Implementation:**
- `LinearRegionItem` pool for up to 5 simultaneous jammers
- Yellow semi-transparent brush (255, 255, 0, 60)
- Strobe width: 5° centered on jammer azimuth

---

## 10. ROC Curve Analysis

**File:** `src/ui/analysis/roc_curve.py`

### Receiver Operating Characteristic

Plots $P_d$ vs $P_{fa}$ for different SNR values:

$$P_d = f(P_{fa}, SNR, \text{Swerling Model})$$

### Swerling Model Selection

| Model | Fluctuation Type |
|-------|------------------|
| 0 | Non-fluctuating |
| 1 | Slow, Rayleigh |
| 3 | Slow, Chi-square |

### Operating Point Display

Shows current radar operating point on ROC curve.

---

## 11. SNR Histogram Analysis

**File:** `src/ui/analysis/snr_histogram.py`

Real-time distribution of detection SNR values:

| Zone | SNR Range | Color |
|------|-----------|-------|
| Weak | < 10 dB | Red |
| Moderate | 10-20 dB | Yellow |
| Strong | > 20 dB | Green |

### Statistics Displayed

- Mean SNR
- Median SNR  
- Standard Deviation
- Total Detections

---

## References

1. Richards, M.A. - *Fundamentals of Radar Signal Processing*, 2014
2. Cumming & Wong - *Digital Processing of SAR Data*, 2005
3. Rohling, H. - "Radar CFAR Thresholding", IEEE TAES, 1983
4. Swerling, P. - "Probability of Detection", IRE Trans, 1960

