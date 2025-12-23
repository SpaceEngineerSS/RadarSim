# Machine Learning Pipeline Documentation

This document describes the AI/ML target classification system in RadarSim.

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Feature Engineering](#feature-engineering)
- [Dataset Generation](#dataset-generation)
- [Model Training](#model-training)
- [Real-Time Inference](#real-time-inference)
- [References](#references)

---

## Architecture Overview

RadarSim uses a **Random Forest** classifier for real-time target classification.

```
                    ┌─────────────────┐
   Radar Track ──▶  │ Feature Extract │ ──▶ [range, doppler, snr, rcs]
                    └─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │  Random Forest  │
                    │  (100 Trees)    │
                    └─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │   Prediction    │ ──▶ {class, confidence}
                    └─────────────────┘
```
                    ┌─────────────────┐
   Radar Track ──▶  │ Feature Extract │ ──▶ [range, doppler, snr, rcs]
                    └─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │  Random Forest  │
                    │  (100 Trees)    │
                    └─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │   Prediction    │ ──▶ {class, confidence}
                    └─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │  Result Update  │ ──▶ DetectionResult.predicted_class
                    └─────────────────┘
```

**Implementation:** `src/ml/`

---

## Feature Engineering

### Input Features

| Feature | Description | Units | Range |
|---------|-------------|-------|-------|
| `range` | Distance to target | km | 0 - 300 |
| `doppler` | Radial velocity | m/s | -500 to +500 |
| `snr` | Signal-to-noise ratio | dB | 0 - 40 |
| `rcs_est` | Estimated RCS | m² | 0.01 - 100 |

### Feature Scaling

All features are standardized using `StandardScaler`:

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

Where $\mu$ and $\sigma$ are learned from training data.

**Implementation:** `src/ml/inference_engine.py::_preprocess()`

---

## Dataset Generation

### Target Class Profiles

Three target classes with physics-based parameter distributions:

#### 🛸 Drone (Class 0)
```python
{
    'range': uniform(5, 50),       # Short range
    'doppler': normal(15, 10),     # Low speed
    'snr': normal(15, 5),          # Moderate SNR
    'rcs_est': lognormal(0.01, 3)  # Very small RCS
}
```

#### ✈️ Fighter Jet (Class 1)
```python
{
    'range': uniform(20, 200),     # Medium-long range
    'doppler': normal(300, 100),   # High speed
    'snr': normal(25, 5),          # Good SNR
    'rcs_est': lognormal(2, 5)     # Medium RCS
}
```

#### 🚀 Missile (Class 2)
```python
{
    'range': uniform(10, 100),     # Variable range
    'doppler': normal(800, 200),   # Very high speed
    'snr': normal(20, 8),          # Variable SNR
    'rcs_est': lognormal(0.1, 2)   # Small RCS
}
```

### Dataset Size

Default training: 1000 samples per class (3000 total)

**Implementation:** `src/ml/dataset_generator.py`

---

## Model Training

### Random Forest Configuration

```python
RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,          # Limit tree depth
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples per leaf
    random_state=42,       # Reproducibility
    n_jobs=-1              # Parallel training
)
```

### Training Pipeline

1. **Load data**: Read from `output/training_data.csv`
2. **Split**: 80% train, 20% validation
3. **Scale**: Fit `StandardScaler` on training set
4. **Train**: Fit Random Forest
5. **Evaluate**: Accuracy, precision, recall, F1
6. **Save**: Model to `models/radar_classifier.pkl`

### Expected Performance

| Metric | Target |
|--------|--------|
| Accuracy | > 90% |
| Precision | > 85% per class |
| Recall | > 85% per class |

**Implementation:** `src/ml/trainer.py`

---

## Real-Time Inference

### InferenceEngine Class

```python
class InferenceEngine:
    """Real-time target classification engine."""
    
    def __init__(self, model_path: str = "models/radar_classifier.pkl"):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load("models/feature_scaler.pkl")
        self.classes = ['Drone', 'Fighter Jet', 'Missile']
    
    def classify(self, track_data: dict) -> dict:
        """
        Classify a radar track.
        
        Args:
            track_data: {range_km, doppler_mps, snr_db, rcs_m2}
            
        Returns:
            {class_name, class_id, confidence, probabilities}
        """
        features = self._preprocess(track_data)
        probs = self.model.predict_proba(features)[0]
        class_id = np.argmax(probs)
        
        return {
            'class_name': self.classes[class_id],
            'class_id': int(class_id),
            'confidence': float(probs[class_id]),
            'probabilities': probs.tolist()
        }
```

### GUI Integration

The Target Inspector panel displays:
- **Class icon**: 🛸 / ✈️ / 🚀
- **Confidence bar**: Color-coded (green > 80%, yellow 50-80%, red < 50%)
- **Probabilities**: All class scores

### Auto-Classification Loop

In Phase 25, an automatic inference loop was added to the main simulation engine:
- **Frequency**: Every 10 frames (~3Hz)
- **Scope**: All detected targets
- **Storage**: Results stored in `DetectionResult` objects
- **Throttling**: To minimize CPU usage during large scenarios

**Implementation:** `src/simulation/engine.py::step()` and `src/ui/panels/target_inspector.py`

---

## Model Files

| File | Description |
|------|-------------|
| `models/radar_classifier.pkl` | Trained Random Forest model |
| `models/feature_scaler.pkl` | StandardScaler fitted on training data |
| `output/training_data.csv` | Generated dataset |

### Retraining

```bash
# Generate new dataset
python -m src.ml.dataset_generator --samples 2000

# Train model
python -m src.ml.trainer --output models/

# Validate
python -m pytest tests/test_ml.py -v
```

---

## References

1. **Breiman, L.** (2001). "Random Forests." *Machine Learning*, 45, 5-32.

2. **Pedregosa, F. et al.** (2011). "Scikit-learn: Machine Learning in Python." *JMLR*, 12, 2825-2830.

3. **Mitchell, R.A.** (2004). "Radar Target Classification Using Machine Learning." *IEEE Radar Conference*.

4. **Chen, V.C.** (2019). *Radar Micro-Doppler Signature Processing and Applications*. IET.
   - Feature extraction from radar signatures

---

*Document generated for RadarSim v1.0.0*
