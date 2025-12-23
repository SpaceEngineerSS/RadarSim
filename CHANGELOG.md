# Changelog

All notable changes to RadarSim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-23 (Gold Release)

### Added - Phase 19: Clutter, MTI & ECCM
- **Environmental Clutter** (Ground/Sea/Rain) with SNR degradation
- **MTI Filter** with configurable velocity threshold
- **ECCM Frequency Agility** to counter jamming

### Added - Phase 20: Precision Tracking
- **Monopulse Angle Estimation** (Sum/Difference patterns)
- **Ambiguity Analysis Widget** (PRF vs Range/Velocity trade-off)
- Sub-beamwidth angular accuracy calculation

### Added - Phase 21: Statistical Analysis
- **ROC Curves** (Pd vs Pfa for Swerling models)
- **SNR Histogram** (Detection strength distribution)
- **Metrics Module** (Albersheim equation, Swerling Pd)

### Added - Phase 22: System Capabilities
- **Scenario Export** (Save simulation state to YAML)
- **Keyboard Shortcuts** (Space, R, 1-4, F11)
- **Performance Monitor** overlay widget

### Added - Phase 23: Signal Visualization
- **B-Scope ECM Strobes** (Jammer direction visualization)
- **A-Scope CFAR Hover** (CUT/Guard/Reference cell display)

### Added - Phase 24: Documentation
- Complete README.md rewrite with v1.0 feature list
- Technical docs: physics_engine.md, signal_processing.md, user_guide.md
- Automated screenshot capture (10 images)

### Previous Features (Phase 15-18)
- **Terrain Masking** with 4/3 Earth refraction
- **RHI Scope** (Range-Height Indicator)
- **3D Tactical Map** with OpenGL
- **SAR Viewer** with realistic imagery
- **Advanced Menu** (Clutter, MTI, ECCM, Monopulse toggles)
- **MIL-STD-2525D Symbology**

### Changed
- Main window now uses QTabWidget for display areas
- PPI/B-Scope use affiliation-based coloring



## [2.0.0] - 2025-12-22

### Added
- **3D Coordinate System**: Full 3D position, velocity, and acceleration support
- **ITU-R P.676 Atmospheric Model**: Oxygen and water vapor absorption (1-100 GHz)
- **Swerling RCS Models**: Implementation of Swerling I, II, III, IV fluctuation models
- **Extended Kalman Filter**: Nonlinear tracking with polar measurements (range, azimuth, elevation, range_rate)
- **Track Lifecycle Management**: TENTATIVE → CONFIRMED → COASTING → DROPPED status
- **SimulationConfig Class**: Centralized configuration management
- **SimulationStatistics**: Comprehensive statistics tracking
- **Probability of Detection**: Albersheim approximation with Swerling correction
- **Scientific Documentation**: Complete formula documentation with IEEE references
- **Open Source Files**: LICENSE (MIT), CONTRIBUTING.md, CHANGELOG.md

### Changed
- **radar_physics.py**: Complete rewrite with 3D support and scientific enhancements
- **target_tracking.py**: EKF implementation replacing simple Kalman filter
- **main.py**: Modern simulation loop with configuration and logging
- **README.md**: Professional documentation with formulas and references

### Fixed
- Atmospheric attenuation now uses scientifically accurate ITU-R model
- Monopulse angle error includes SNR-dependent thermal noise
- Track association uses proper Mahalanobis distance gating

### References
- Skolnik, "Radar Handbook", 3rd Ed., 2008
- Bar-Shalom, "Estimation with Applications to Tracking", 2001
- ITU-R P.676-12, 2017

## [1.0.0] - 2025-12-23

### Added
- Initial release
- Basic radar equation implementation
- 2D target tracking with Kalman filter
- ECM simulation (chaff, decoy, jamming)
- Pygame visualization
- Proportional Navigation guidance

---

## Version Numbering

- **MAJOR**: Incompatible API changes or fundamental algorithm changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible
