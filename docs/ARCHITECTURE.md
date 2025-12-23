# System Architecture

## Overview

RadarSim is designed as a modular, high-performance radar simulation platform. It separates core physics calculations (using Numba/SciPy) from the real-time simulation loop and the PyQt6-based user interface.

## Directory Structure

The `src/` directory is organized by domain:

```
src/
├── physics/          # Core Physics Engine
│   ├── radar_equation.py   # SNR, Range, Power calculations (Numba)
│   ├── atmospheric.py      # ITU-R P.676 Attenuation
│   ├── clutter.py          # Grund/Sea (GIT)/Rain models
│   ├── rcs.py              # Swerling fluctuation models
│   └── terrain.py          # 4/3 Earth & LOS Raymarching
│
├── signal/           # Signal Processing Chain
│   ├── cfar.py             # CA/GO/SO-CFAR Detectors
│   ├── waveforms.py        # LFM Chirp generation
│   └── doppler.py          # Doppler shift & Aliasing
│
├── tracking/         # Target Tracking
│   ├── kalman.py           # Linear/Extended Kalman Filters
│   └── monopulse.py        # Sum/Difference pattern logic
│
├── simulation/       # Simulation Runtime
│   ├── engine.py           # Main simplified simulation loop
│   ├── thread_manager.py   # QThread management
│   └── scenario.py         # Scenario state management
│
├── ui/               # User Interface (PyQt6)
│   ├── main_window.py      # Application entry
│   ├── scopes/             # Radar displays (PPI, RHI, A-Scope)
│   ├── panels/             # Control panels & Dashboards
│   └── analysis/           # Analysis tools (ROC, Ambiguity)
│
├── ml/               # AI Classification
│   ├── inference_engine.py # Real-time Random Forest inference
│   └── dataset_generator.py# Synthetic training data gen
│
├── advanced/         # Advanced Modules
│   ├── sar_isar.py         # Synthetic Aperture Radar imaging
│   └── fusion.py           # Sensor fusion stub
│
└── io/               # Input/Output
    └── exporter.py         # Data export utilities
```

## Key Components

### 1. Physics Engine (`src/physics`)
The physics engine is stateless and purely functional where possible. Critical functions (like the radar equation loop) are JIT-compiled using Numba for near-C++ performance.

### 2. Simulation Loop (`src/simulation`)
The 'Engine' runs on a separate QThread. It acts as the central clock, updating target positions, calculating detections, and emitting signals to the UI.

### 3. User Interface (`src/ui`)
The UI is event-driven.
- **Visualizations**: Use `pyqtgraph` for high-performance plotting (PPI, B-Scope).
- **3D Map**: Uses a dedicated OpenGL widget for terrain visualization.

### 4. Machine Learning (`src/ml`)
A lightweight inference pipeline that runs alongside the tracking loop. It classifies tracks based on feature vectors extracted from the physics engine (RCS, Velocity, SNR).

## Technologies

- **Language**: Python 3.10+
- **GUI Framework**: PyQt6
- **Numeric Computing**: NumPy, SciPy
- **Acceleration**: Numba (JIT)
- **Plotting**: PyQtGraph, Matplotlib
- **Machine Learning**: Scikit-Learn

## Data Flow

1.  **Input**: User loads a YAML scenario file.
2.  **Update**: Engine advances target positions (kinematics).
3.  **Physics**: Engine calculates SNR for each target (Radar Eq + Losses).
4.  **Detection**: CFAR thresholding determines detections.
5.  **Tracking**: Detections are fed into Kalman Filters.
6.  **Classification**: Tracker output is fed into ML Inference.
7.  **Display**: UI updates scopes at 30-60 Hz.