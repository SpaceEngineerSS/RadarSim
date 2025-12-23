# Electronic Countermeasures (ECM) Documentation

This document describes the electronic warfare models implemented in RadarSim.

## Table of Contents
- [ECM Overview](#ecm-overview)
- [Noise Jamming](#noise-jamming)
- [Deception Jamming](#deception-jamming)
- [Chaff & Decoys](#chaff--decoys)
- [Burn-Through Range](#burn-through-range)
- [References](#references)

---

## ECM Overview

RadarSim simulates both **noise** and **deception** jamming techniques, as well as passive countermeasures.

| ECM Type | Effect | Implementation |
|----------|--------|----------------|
| Noise Barrage | Raises noise floor across band | Additive power |
| Noise Spot | Focuses on specific frequency | Additive power |
| DRFM Repeater | Creates false targets | Ghost track injection |
| RGPO | Range deception (delay) | Range offset |
| VGPO | Velocity deception (Doppler) | Doppler offset |
| Chaff | RCS masking | False scatter cloud |

**Implementation:** `src/physics/ecm.py`

---

## Noise Jamming

### Barrage Jamming

Wideband noise that degrades SNR across the radar's entire bandwidth:

$$J/S = \frac{P_j G_j \lambda^2 R_t^2}{4\pi P_t G_t \sigma R_j^2}$$

Where:
- $P_j$ = Jammer power (W)
- $G_j$ = Jammer antenna gain
- $R_t$ = Range to target
- $R_j$ = Range to jammer
- $\sigma$ = Target RCS

### Self-Screening Jamming

When the jammer is on the target (self-protection):

$$J/S = \frac{P_j G_j \cdot 4\pi R^2}{P_t G_t \sigma}$$

The jammer advantage increases with range $R$ (vs. $R^4$ radar disadvantage).

### Spot Jamming

Concentrated power in narrow bandwidth:

$$P_{j,eff} = P_j \cdot \frac{B_r}{B_j}$$

Where:
- $B_r$ = Radar bandwidth
- $B_j$ = Jammer bandwidth

More power-efficient but requires knowledge of radar frequency.

**Implementation:** `src/physics/ecm.py::NoiseJammingModel`

---

## Deception Jamming

### DRFM (Digital RF Memory)

Captures radar pulse, modifies it, and retransmits:

```
Radar Tx → [Capture] → [Delay/Modify] → [Retransmit]
                ↓
         DRFM Memory
```

Effects:
- **Range deception**: Add time delay
- **Velocity deception**: Add Doppler shift
- **Multiple false targets**: Retransmit multiple copies

### RGPO (Range Gate Pull-Off)

Gradually increases retransmission delay to pull tracking gate off target:

$$\Delta R(t) = R_0 + v_p \cdot t$$

Where $v_p$ is the pull-off velocity (typically 10-100 m/s).

### VGPO (Velocity Gate Pull-Off)

Similar technique for Doppler tracking systems:

$$\Delta f_d(t) = f_{d,0} + a_p \cdot t$$

Where $a_p$ is the Doppler acceleration rate.

**Implementation:** `src/physics/ecm.py::DRFMRepeater`

---

## Chaff & Decoys

### Chaff

Dipole reflectors cut to radar wavelength:

$$\sigma_{chaff} = N \cdot 0.18 \lambda^2$$

Where $N$ is the number of dipoles.

**Bloom dynamics:**
- Initial cloud: ~50m radius
- Expansion: ~5 m/s radial
- Settling: Falls ~3 m/s

### Towed Decoys

Active or passive decoys towed behind aircraft:
- Larger RCS than platform
- Offset position
- May have DRFM repeater

**Implementation:** `src/physics/ecm.py::ChaffCloud`

---

## Burn-Through Range

The range at which radar SNR overcomes jammer power:

$$R_{BT} = \sqrt[4]{\frac{P_t G_t \sigma B_j}{P_j G_j (4\pi)^2 (S/J)_{min}}}$$

Where $(S/J)_{min}$ is the minimum required signal-to-jammer ratio (typically 0 dB for detection).

### Factors Affecting Burn-Through

| Factor | Effect on $R_{BT}$ |
|--------|-------------------|
| ↑ $P_t$ (Radar power) | ↑ Increases |
| ↑ $G_t$ (Radar gain) | ↑ Increases |
| ↑ $\sigma$ (Target RCS) | ↑ Increases |
| ↑ $P_j$ (Jammer power) | ↓ Decreases |
| ↑ $G_j$ (Jammer gain) | ↓ Decreases |

### Implementation

```python
def calculate_burn_through_range(
    radar_power_w: float,
    radar_gain_db: float,
    target_rcs_m2: float,
    jammer_power_w: float,
    jammer_gain_db: float,
    sj_min_db: float = 0.0
) -> float:
    """Calculate burn-through range in meters."""
    
    G_t = 10 ** (radar_gain_db / 10)
    G_j = 10 ** (jammer_gain_db / 10)
    sj_min = 10 ** (sj_min_db / 10)
    
    numerator = radar_power_w * G_t * target_rcs_m2
    denominator = jammer_power_w * G_j * (4 * np.pi) ** 2 * sj_min
    
    return (numerator / denominator) ** 0.25
```

---

## ECCM (Counter-Countermeasures)

RadarSim models these ECCM techniques:

| ECCM | Description |
|------|-------------|
| Frequency Agility | Hop frequencies to avoid spot jamming |
| PRF Jitter | Randomize PRF to defeat DRFM synchronization |
| Sidelobe Blanking | Ignore returns from sidelobe angles |
| Burn-Through | Increase power to overcome jamming |

---

## References

1. **Adamy, D.L.** (2001). *EW 101: A First Course in Electronic Warfare*. Artech House.
   - Complete ECM/ECCM taxonomy

2. **Schleher, D.C.** (1999). *Electronic Warfare in the Information Age*. Artech House.
   - Burn-through range derivation

3. **Poisel, R.A.** (2011). *Modern Communications Jamming*. Artech House.
   - Digital RF countermeasures

4. **Neri, F.** (2006). *Introduction to Electronic Defense Systems*, 2nd Ed. SciTech.
   - RGPO/VGPO techniques

---

*Document generated for RadarSim v1.0.0*
