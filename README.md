<p align="center">
  <img src="docs/images/radarsim_logo.svg" alt="RadarSim Logo" width="200"/>
</p>

<h1 align="center">RadarSim v2.4.0</h1>

<p align="center">
  <b>Professional Pulse-Doppler Radar Simulation Platform</b><br>
  <i>Physics-Based • AI-Enhanced • Imaging Radar • Open Source</i>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#keyboard-shortcuts">Shortcuts</a> •
  <a href="#documentation">Docs</a> •
  <a href="#citation">Citation</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/PyQt6-6.0%2B-orange?style=flat-square" alt="PyQt6"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/>
  <img src="https://img.shields.io/badge/version-2.4.0-gold?style=flat-square" alt="v2.4.0"/>
  <img src="https://img.shields.io/badge/tests-217%20passed-brightgreen?style=flat-square" alt="217 tests"/>
  <img src="https://img.shields.io/badge/Project_Evolution-Phase_30-ff69b4?style=flat-square" alt="Phase 30"/>
</p>

---

## 📖 Overview

**RadarSim** is a scientifically-validated radar simulation engine for education, research, and professional training. Built with NumPy/Numba for performance and PyQt6 for a modern UI.

### 🔬 Scientific Core
- **Pulse-Doppler Engine:** Signal-level processing (CPI, MTI, FFT) referenced from Richards (2005).
- **Advanced Tracking:** Extended Kalman Filter (EKF) with polar coordinates.
- **Electronic Warfare:** DRFM Jamming (RGPO/VGPO) and Frequency Agility.
- **Sensor Fusion:** Networked Radar & Strobe Triangulation.
- **Imaging:** High-resolution SAR/ISAR algorithms.
- **AI/Cognitive Control:** Dynamic logic referenced from Haykin (2006).

**Key Validation:** Radar Equation calculations achieve ±0.005 dB accuracy vs. Skolnik reference values.

![PPI Scope](docs/images/ppi_scope.png)
*Figure 1: Main Plan Position Indicator (PPI) display showing detected targets.*

---

## ✨ Features

### 🎯 Physics Engine
| Feature | Implementation | Reference |
|---------|----------------|-----------|
| **Radar Equation** | Monostatic/Bistatic with Numba JIT | Skolnik, Ch. 2 |
| **Atmospheric Attenuation** | ITU-R P.676-12 (O₂ + H₂O) | IEEE Std |
| **Swerling RCS Models** | Types 0-4 fluctuation | Swerling (1960) |
| **Monopulse Tracking** | Sum/Difference patterns, sub-beamwidth accuracy | Phase 20 |
| **3D Terrain Masking** | 4/3 Earth refraction, LOS shadowing | ITU-R P.526 |

### 🌪️ Environmental Effects
| Feature | Details |
|---------|---------|
| **Ground Clutter** | Weibull distribution, σ⁰ coefficients |
| **Sea Clutter** | **GIT Model**, Douglas sea states (1-6) |
| **Rain Physics** | ITU-R P.838 Attenuation + Marshall-Palmer Clutter |
| **MTI Filtering** | Velocity threshold, slow-mover rejection |

### ⚔️ Electronic Warfare
| Technique | Type | Status |
|-----------|------|--------|
| Noise Jamming | ECM | ✅ |
| DRFM Repeater | ECM | ✅ |
| RGPO/VGPO Deception | ECM | ✅ |
| **Frequency Agility** | ECCM | ✅ |
| **Burn-Through Display** | ECM Strobe | ✅ |

### 🛰️ Imaging Radar (Phase 30)
| Feature | Implementation | Reference |
|---------|----------------|-----------|
| **SAR (RDA)** | Vectorized 5-stage Range-Doppler Algorithm | Cumming & Wong (2005) |
| **ISAR** | Cross-range imaging via target rotation | Chen & Ling (2002) |
| **Resolution** | Δr = c/(2B) = 1.5m, Δa = D/2 = 0.5m | Verified |

### 🤖 AI Tactical Director (Phase 30)
| Feature | Description |
|---------|-------------|
| **Coverage Analysis** | 2D Pd map from multi-radar network |
| **Blind Zone Detection** | Flood-fill connected component analysis |
| **Attack Planning** | 3 difficulty levels (Easy/Medium/Hard) |
| **Low-Pd Routing** | Greedy corridor navigation |
| **Jammer Deployment** | Optimal DRFM positioning |

### 📊 Visualization Scopes
| Scope | Description |
|-------|-------------|
| **PPI** | Plan Position Indicator with phosphor decay |
| **B-Scope** | Range vs Azimuth (AESA style) with ECM strobes |
| **A-Scope** | Amplitude vs Range with CFAR hover visualization |
| **RHI** | Range-Height Indicator (elevation) |
| **3D Tactical** | OpenGL terrain with target spheres |
| **SAR Viewer** | Real physics-based Synthetic Aperture Radar imaging |

### 📸 Visualization Gallery

| SAR Imaging | 3D Tactical Map |
|:-----------:|:---------------:|
| <img src="docs/images/sar_viewer.png" width="400"/> | <img src="docs/images/3d_tactical.png" width="400"/> |
| *Real-time SAR formation (Range-Doppler)* | *3D situation awareness* |

| RHI Scope | A-Scope Analysis |
|:---------:|:----------------:|
| <img src="docs/images/rhi_scope.png" width="400"/> | <img src="docs/images/a_scope_cfar.png" width="400"/> |
| *Elevation scanning (Range-Height)* | *CFAR threshold visualization* |

### 📈 Analysis Tools
| Tool | Function |
|------|----------|
| **Ambiguity Diagram** | PRF vs Range/Velocity trade-off |
| **ROC Curves** | Pd vs Pfa for Swerling models |
| **SNR Histogram** | Detection strength distribution |

### 📊 Analysis Tools

<p align="center">
  <img src="docs/images/ambiguity_plot.png" width="30%" />
  <img src="docs/images/roc_curves.png" width="30%" />
  <img src="docs/images/snr_histogram.png" width="30%" />
</p>
<p align="center">
  <i>Ambiguity Diagram • ROC Curves • Real-time SNR Statistics</i>
</p>

### 🤖 AI/ML Pipeline
- **RandomForest Classifier** trained on synthetic radar data
- **Classes:** Drone 🛸, Fighter Jet ✈️, Missile 🚀
- **Real-time inference** with confidence scoring

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/SpaceEngineerSS/RadarSim.git
cd RadarSim

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run application
python run_gui.py
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play/Pause simulation |
| `R` | Reset (stop) simulation |
| `1` | Switch to PPI Scope |
| `2` | Switch to RHI Scope |
| `3` | Switch to 3D Tactical |
| `4` | Switch to 4th tab |
| `F11` | Toggle fullscreen |
| `Ctrl+O` | Load scenario |
| `Ctrl+Shift+S` | Save scenario |
| `Ctrl+R` | Start recording |

---

## 📁 Project Structure

```
RadarSim/
├── run_gui.py              # Main entry point (PyQt6 GUI)
├── headless.py             # Headless simulation runner
├── batch_run.py            # Batch scenario executor
├── requirements.txt        # Dependencies
├── scenarios/              # YAML scenario files (10 scenarios)
│   ├── f16_vs_sa6.yaml
│   ├── drone_swarm_saturation.yaml
│   ├── naval_battlegroup.yaml
│   └── ...
├── src/
│   ├── physics/            # Core physics (radar_equation, clutter, ecm)
│   ├── signal/             # Signal processing (cfar, pulse_doppler, antenna)
│   ├── tracking/           # Target tracking (EKF, monopulse, track manager)
│   ├── simulation/         # Simulation engine & network manager
│   ├── ui/                 # PyQt6 GUI (PPI, B-Scope, RHI, A-Scope, 3D)
│   ├── advanced/           # SAR/ISAR, AI Director, Sensor Fusion, ECCM, LPI
│   └── ml/                 # AI classification pipeline
├── models/                 # Trained ML models
├── docs/                   # Documentation
└── tests/                  # 217 unit tests
```

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [Physics Engine](docs/physics.md) | Radar equation, Monopulse, Ambiguity |
| [Signal Processing](docs/signal_processing.md) | CFAR, MTI, SAR algorithms |
| [User Guide](docs/user_guide.md) | GUI walkthrough and Advanced features |

---

## 🔬 Scientific References

1. **Skolnik, M.I.** — *Radar Handbook*, 3rd Ed., McGraw-Hill, 2008
2. **Richards, M.A.** — *Fundamentals of Radar Signal Processing*, 2nd Ed., McGraw-Hill, 2014
3. **IEEE Std 686-2017** — Radar Definitions
4. **ITU-R P.676-12** — Attenuation by Atmospheric Gases
5. **Cumming & Wong** — *Digital Processing of SAR Data*, Artech House, 2005
6. **Chen & Ling** — *Time-Frequency Transforms for Radar Imaging*, Artech House, 2002
7. **Bar-Shalom, Y.** — *Estimation with Applications to Tracking*, Wiley, 2001
8. **Julier & Uhlmann** — *Non-divergent Estimation Algorithm*, ACC, 1997
9. **Schleher, D.C.** — *Electronic Warfare in the Information Age*, Artech House, 1999
10. **Poisel, R.** — *Electronic Warfare Target Location Methods*, Artech House, 2012

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 📖 Citation

If you use RadarSim in academic work, please cite:

```bibtex
@software{radarsim2025,
  title = {RadarSim: Physics-Based Pulse-Doppler Radar Simulation},
  author = {RadarSim Contributors},
  year = {2025},
  url = {https://github.com/SpaceEngineerSS/RadarSim}
}
```

---

## 👨‍💻 Developer

**Mehmet Gümüş**
* 🌐 Website: [spacegumus.com.tr](https://spacegumus.com.tr)
* 🐙 GitHub: [@SpaceEngineerSS](https://github.com/SpaceEngineerSS)

---

<p align="center">
  <b>Built with ❤️ for the Radar Community</b>
</p>