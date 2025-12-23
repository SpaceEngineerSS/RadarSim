# RadarSim Phase 25 Update - Release Notes

**Version:** 1.0.0-Phase25
**Date:** 2025-12-23

## 🌟 New Features

### 1. Scientific Rain Physics
- **Attenuation:** Implemented **ITU-R P.838** specific attenuation model. Rain now causes realistic signal loss based on frequency (X/K/Ka band) and rain rate.
- **Volume Clutter:** Implemented **Marshall-Palmer** reflectivity model ($Z=200R^{1.6}$). Raindrops now reflect radar energy, creating volume clutter that degrades SNR.
- **Visual Feedback:** Check the console/log for "Rain Loss" values when `rain_rate_mm_hr > 0`.

### 2. Intelligent Auto-Classification
- **Auto-Inference Loop:** The engine now automatically classifies all detected targets every ~300ms using the random forest model.
- **Live Results:** `DetectionResult` objects now contain `predicted_class` (Drone/Fighter/Missile) and `confidence` score.
- **Integration:** Inspecting a target shows the same classification result that the engine sees.

### 3. Real SAR Imaging
- **Physics-Based Rendering:** The "Generate SAR Image" button no longer uses demo noise. It now feeds live target positions into the **AdvancedSARISAR** engine.
- **Range-Doppler Algorithm:** Uses real phase history generation and RDA processing to create authentic SAR heatmaps.
- **Sea Clutter:** Added **GIT (Georgia Tech)** sea clutter model. Radar over water now experiences realistic sea-state dependent clutter.

## 🐛 Bug Fixes
- Fixed "fake" SAR demo generator being used in production.
- Fixed `rain_rate_mm_hr` being ignored in SNR calculations.
- Fixed generic ground clutter model being used for sea scenarios.

## 📖 Documentation
- Updated `docs/physics.md` with new ITU-R P.838 and GIT models.
- Updated `docs/ml.md` with auto-inference loop architecture.
- Updated `docs/user_guide.md` with new features.
