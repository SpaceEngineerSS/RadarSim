"""
Real-time Inference Engine for Radar Target Classification

Provides fast predictions for live simulation using pre-trained models.

Features:
    - Lazy model loading (loaded on first use)
    - Thread-safe singleton pattern
    - Graceful fallback if model missing
    - Performance optimized (<10ms inference)

Usage:
    engine = InferenceEngine()
    class_name, confidence = engine.predict({
        'range_km': 50.0,
        'doppler_hz': 1200.0,
        'snr_db': 15.0,
        'rcs_est_m2': 5.0,
    })
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import threading
import time
import warnings

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from .dataset_generator import CLASS_DEFINITIONS, TargetClass


class InferenceEngine:
    """
    Real-time inference engine for target classification.
    
    Thread-safe singleton that loads models on first use.
    Falls back gracefully if models are not available.
    
    Attributes:
        model: Loaded classifier (or None)
        scaler: Loaded feature scaler (or None)
        is_ready: Whether models are loaded and ready
    """
    
    _instance: Optional['InferenceEngine'] = None
    _lock = threading.Lock()
    
    # Feature order must match training
    FEATURE_ORDER = ['range_km', 'doppler_hz', 'snr_db', 'rcs_est_m2']
    
    # Class name mapping
    CLASS_NAMES = {
        0: "Drone",
        1: "Fighter Jet", 
        2: "Missile",
    }
    
    # Emoji mapping for UI
    CLASS_ICONS = {
        0: "🛸",  # Drone
        1: "✈️",  # Fighter
        2: "🚀",  # Missile
    }
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern - only one instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        auto_load: bool = True,
    ):
        """
        Initialize inference engine.
        
        Args:
            model_dir: Directory containing model files
            auto_load: Whether to load models immediately
        """
        # Only initialize once (singleton)
        if self._initialized:
            return
        
        self._initialized = True
        
        if model_dir is None:
            self.model_dir = Path(__file__).parent.parent.parent / 'models'
        else:
            self.model_dir = Path(model_dir)
        
        self.model = None
        self.scaler = None
        self.is_ready = False
        self._load_error: Optional[str] = None
        
        if auto_load:
            self._load_models()
    
    def _load_models(self) -> bool:
        """
        Load model and scaler from disk.
        
        Returns:
            True if successful, False otherwise
        """
        if not JOBLIB_AVAILABLE:
            self._load_error = "joblib not installed"
            return False
        
        model_path = self.model_dir / 'radar_classifier.pkl'
        scaler_path = self.model_dir / 'feature_scaler.pkl'
        
        try:
            if not model_path.exists():
                self._load_error = f"Model not found: {model_path}"
                return False
            
            if not scaler_path.exists():
                self._load_error = f"Scaler not found: {scaler_path}"
                return False
            
            # Load with warnings suppressed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
            
            self.is_ready = True
            self._load_error = None
            return True
            
        except Exception as e:
            self._load_error = str(e)
            return False
    
    def predict(
        self,
        track_data: Dict[str, float],
    ) -> Tuple[str, float]:
        """
        Predict target class from track data.
        
        Args:
            track_data: Dictionary with keys matching FEATURE_ORDER
                - range_km: Target range [km]
                - doppler_hz: Doppler frequency [Hz]
                - snr_db: Signal-to-noise ratio [dB]
                - rcs_est_m2: Estimated RCS [m²]
        
        Returns:
            Tuple of (class_name, confidence)
            Returns ("Unknown", 0.0) if model not available
        """
        if not self.is_ready:
            return ("Unknown", 0.0)
        
        try:
            # Extract features in correct order
            features = np.array([[
                float(track_data.get('range_km', 50.0)),
                float(track_data.get('doppler_hz', 0.0)),
                float(track_data.get('snr_db', 0.0)),
                float(track_data.get('rcs_est_m2', 1.0)),
            ]], dtype=np.float64)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get prediction probabilities
            proba = self.model.predict_proba(features_scaled)[0]
            
            # Get class with highest probability
            class_id = int(np.argmax(proba))
            confidence = float(proba[class_id])
            
            class_name = self.CLASS_NAMES.get(class_id, "Unknown")
            
            return (class_name, confidence)
            
        except Exception as e:
            # Fail gracefully
            return ("Unknown", 0.0)
    
    def predict_with_icon(
        self,
        track_data: Dict[str, float],
    ) -> Tuple[str, str, float]:
        """
        Predict with class icon for UI display.
        
        Returns:
            Tuple of (icon, class_name, confidence)
        """
        class_name, confidence = self.predict(track_data)
        
        # Find icon
        icon = "❓"
        for class_id, name in self.CLASS_NAMES.items():
            if name == class_name:
                icon = self.CLASS_ICONS.get(class_id, "❓")
                break
        
        return (icon, class_name, confidence)
    
    def predict_all_proba(
        self,
        track_data: Dict[str, float],
    ) -> List[Tuple[str, float]]:
        """
        Get probabilities for all classes.
        
        Returns:
            List of (class_name, probability) tuples, sorted by probability
        """
        if not self.is_ready:
            return [("Unknown", 1.0)]
        
        try:
            features = np.array([[
                float(track_data.get('range_km', 50.0)),
                float(track_data.get('doppler_hz', 0.0)),
                float(track_data.get('snr_db', 0.0)),
                float(track_data.get('rcs_est_m2', 1.0)),
            ]], dtype=np.float64)
            
            features_scaled = self.scaler.transform(features)
            proba = self.model.predict_proba(features_scaled)[0]
            
            results = []
            for class_id, prob in enumerate(proba):
                class_name = self.CLASS_NAMES.get(class_id, "Unknown")
                results.append((class_name, float(prob)))
            
            # Sort by probability (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            return results
            
        except Exception:
            return [("Unknown", 1.0)]
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get engine status for debugging.
        
        Returns:
            Status dictionary
        """
        return {
            'is_ready': self.is_ready,
            'model_dir': str(self.model_dir),
            'load_error': self._load_error,
            'model_type': type(self.model).__name__ if self.model else None,
            'n_classes': len(self.CLASS_NAMES),
        }
    
    def reload(self) -> bool:
        """
        Reload models from disk.
        
        Returns:
            True if successful
        """
        self.model = None
        self.scaler = None
        self.is_ready = False
        return self._load_models()
    
    @classmethod
    def reset_singleton(cls) -> None:
        """Reset singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None


def benchmark_inference(n_iterations: int = 1000) -> float:
    """
    Benchmark inference speed.
    
    Args:
        n_iterations: Number of predictions to run
        
    Returns:
        Average time per prediction in milliseconds
    """
    engine = InferenceEngine()
    
    if not engine.is_ready:
        print(f"Engine not ready: {engine._load_error}")
        return -1.0
    
    # Test data
    test_data = {
        'range_km': 50.0,
        'doppler_hz': 1200.0,
        'snr_db': 15.0,
        'rcs_est_m2': 5.0,
    }
    
    # Warm-up
    for _ in range(10):
        engine.predict(test_data)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        engine.predict(test_data)
    elapsed = time.perf_counter() - start
    
    avg_ms = (elapsed / n_iterations) * 1000
    
    print(f"\nInference Benchmark ({n_iterations} iterations):")
    print(f"  Total time: {elapsed*1000:.1f} ms")
    print(f"  Average: {avg_ms:.3f} ms per prediction")
    print(f"  Throughput: {n_iterations/elapsed:.0f} predictions/sec")
    
    if avg_ms < 10:
        print("  ✓ Performance target met (<10ms)")
    else:
        print("  ⚠ Performance target NOT met (>=10ms)")
    
    return avg_ms


def main():
    """Command-line test."""
    engine = InferenceEngine()
    
    print("Inference Engine Status:")
    for key, value in engine.get_status().items():
        print(f"  {key}: {value}")
    
    if engine.is_ready:
        # Test predictions
        test_cases = [
            # Drone-like
            {'range_km': 30, 'doppler_hz': 200, 'snr_db': 5, 'rcs_est_m2': 0.05},
            # Fighter-like
            {'range_km': 80, 'doppler_hz': 8000, 'snr_db': 20, 'rcs_est_m2': 5.0},
            # Missile-like
            {'range_km': 50, 'doppler_hz': 20000, 'snr_db': 12, 'rcs_est_m2': 0.3},
        ]
        
        print("\nTest Predictions:")
        for i, data in enumerate(test_cases):
            icon, name, conf = engine.predict_with_icon(data)
            print(f"  Test {i+1}: {icon} {name} ({conf*100:.1f}%)")
        
        # Benchmark
        benchmark_inference(1000)
    else:
        print("\nModel not available. Run train_classifier.py first.")


if __name__ == "__main__":
    main()
