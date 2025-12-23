# RadarSim User Guide

## Quick Start

```bash
python run_gui.py
```

---

## Main Window Layout

```
┌─────────────────────────────────────────────────────────┐
│ Menu Bar: File | Simulation | View | Advanced | Help   │
├───────────────────┬─────────────────────────────────────┤
│                   │                                     │
│  Control Panel    │         Radar Scopes                │
│  - Radar Params   │   [PPI] [RHI] [3D TACTICAL]        │
│  - Start/Stop     │                                     │
│  - Speed          │                                     │
│                   │                                     │
├───────────────────┼─────────────────────────────────────┤
│  Status Panel     │     Target Inspector / A-Scope      │
│  - Track List     │                                     │
└───────────────────┴─────────────────────────────────────┘
```

---

## Main Scopes

### PPI (Plan Position Indicator)
- Polar display centered on radar
- Sweep line with phosphor decay
- Target blips with MIL-STD-2525D colors

### RHI (Range-Height Indicator)
- Side view: Range vs Altitude
- Shows terrain masking

### 3D Tactical Map
- OpenGL terrain rendering
- Target spheres with color coding
- Rotate: Left-click drag
- Zoom: Scroll wheel

---

## Advanced Menu

### Enable Clutter
Adds environmental noise:
- Ground/Sea/Rain returns
- **Rain Attenuation:** Range reduction based on rain rate (check Console)
- SNR degradation based on terrain type and sea state

### Enable MTI Filter
Removes slow-moving targets:
- Velocity threshold: 2 m/s default
- Clutter rejection

### Enable ECCM Frequency Agility
Counters ECM jamming:
- Random frequency hopping
- Reduces jammer effectiveness

### Enable Monopulse Tracking
Precision angle estimation:
- Sub-beamwidth accuracy
- Sum/Difference pattern processing

### Generate SAR Image
Opens SAR Viewer:
- **Real Physics:** Uses Range-Doppler Algorithm on active targets
- Displays authentic heatmap (not pre-rendered)
- Contrast/brightness sliders
- Contrast/brightness sliders

---

## Analysis Window

**Open:** View → Show Analysis Window

### Tab 1: Recording Analysis
- Playback controls for HDF5 recordings
- Time slider

### Tab 2: Ambiguity (PRF)
- PRF vs Range/Velocity trade-off
- Interactive sliders
- Blind speed visualization

### Tab 3: ROC Curves
- Pd vs Pfa for different SNRs
- Swerling model selector
- Operating point marker

### Tab 4: SNR Stats
- Real-time SNR histogram
- Detection strength distribution
- Weak/Moderate/Strong zones

---

## Scenarios

Load pre-built scenarios from **File → Load Scenario**:

| Scenario | Description |
|----------|-------------|
| **Air Logic / Combat** | |
| `close_air_combat.yaml` | Dogfight scenario with fast maneuvering targets |
| `f16_vs_sa6.yaml` | SEAD mission simulation (F-16 vs Surface-to-Air) |
| `drone_swarm_saturation.yaml` | High-density small RCS targets to test saturation |
| `hypersonic_interception.yaml` | Very high speed (>Mach 5) targets |
| **Maritime / Naval** | |
| `naval_battlegroup.yaml` | Complex fleet defense environment |
| **Stealth / Advanced** | |
| `stealth_deep_penetration.yaml` | Low-RCS stealth aircraft testing detection limits |
| `ecm_environment.json` | Heavy electronic countermeasures environment |
| **Terrain / Physics** | |
| `mountain_ambush.yaml` | Terrain masking and pop-up target testing |
| `ground_clutter_filtering.yaml` | Low-altitude targets vs ground clutter (MTI test) |
| `basic_tracking.json` | Simple single target for calibration |

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play/Pause |
| R | Reset |
| 1/2/3/4 | Switch tabs |
| F11 | Fullscreen |
| Ctrl+O | Load scenario |
| Ctrl+Shift+S | Save scenario |

---

## B-Scope Display

When viewing AESA radar scenarios:
- Range vs Azimuth Cartesian display
- **ECM Strobes:** Yellow vertical bars show jammer directions
- Targets color-coded by MIL-STD-2525D

---

## Tips

1. **Enable Clutter** for realistic SNR degradation
2. **Use MTI** to filter ground clutter
3. **Check ROC Curves** to understand detection trade-offs
4. **Hover over A-Scope** to see CFAR cell visualization
5. **Watch B-Scope** for ECM strobe indicators

---

*Document generated for RadarSim v1.0.0*
