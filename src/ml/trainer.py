"""
Model Trainer for Radar Target Classification

Trains a RandomForestClassifier on synthetic radar data.

Pipeline:
    1. Load CSV dataset
    2. Feature scaling with StandardScaler
    3. Train/Test split (80/20)
    4. Train RandomForestClassifier
    5. Evaluate: Accuracy, Precision, Recall, F1, Confusion Matrix
    6. Save model and scaler artifacts

Output:
    - models/radar_classifier.pkl
    - models/feature_scaler.pkl
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib

from .dataset_generator import CLASS_DEFINITIONS, TargetClass


class ModelTrainer:
    """
    Trains and evaluates ML models for radar target classification.
    
    Attributes:
        model: Trained classifier
        scaler: Fitted feature scaler
        feature_columns: List of feature column names
        metrics: Dictionary of evaluation metrics
    """
    
    FEATURE_COLUMNS = ['range_km', 'doppler_hz', 'snr_db', 'rcs_est_m2']
    LABEL_COLUMN = 'class_id'
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ):
        """
        Initialize trainer.
        
        Args:
            n_estimators: Number of trees in random forest
            max_depth: Maximum tree depth (None for unlimited)
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.metrics: Dict[str, Any] = {}
        
    def load_data(
        self, 
        csv_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare data from CSV.
        
        Args:
            csv_path: Path to training data CSV
            
        Returns:
            Tuple of (features, labels)
        """
        df = pd.read_csv(csv_path)
        
        # Validate columns
        missing = set(self.FEATURE_COLUMNS + [self.LABEL_COLUMN]) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        X = df[self.FEATURE_COLUMNS].values.astype(np.float64)
        y = df[self.LABEL_COLUMN].values.astype(np.int32)
        
        print(f"Loaded {len(df)} samples from {csv_path}")
        print(f"Features shape: {X.shape}")
        print(f"Classes: {np.unique(y)}")
        
        return X, y
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Train model on data.
        
        Args:
            X: Feature matrix
            y: Label vector
            test_size: Fraction of data for testing
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL TRAINING")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y,
        )
        print(f"\nTrain set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Fit scaler
        print("\nFitting StandardScaler...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print(f"\nTraining RandomForestClassifier (n_estimators={self.n_estimators})...")
        start_time = time.perf_counter()
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
        )
        self.model.fit(X_train_scaled, y_train)
        
        train_time = time.perf_counter() - start_time
        print(f"Training completed in {train_time:.2f}s")
        
        # Evaluate
        print("\nEvaluating on test set...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'train_time_s': train_time,
                'n_train_samples': len(X_train),
                'n_test_samples': len(X_test),
            }
        
        # Print results
        self._print_results(y_test, y_pred)
        
        return self.metrics
    
    def _print_results(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Print formatted evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        print(f"\n  Accuracy:  {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.1f}%)")
        print(f"  Precision: {self.metrics['precision_macro']:.4f}")
        print(f"  Recall:    {self.metrics['recall_macro']:.4f}")
        print(f"  F1 Score:  {self.metrics['f1_macro']:.4f}")
        
        # Confusion Matrix
        print("\nConfusion Matrix:")
        cm = self.metrics['confusion_matrix']
        class_names = [CLASS_DEFINITIONS[TargetClass(i)].name for i in range(len(cm))]
        
        # Header
        print("           ", end="")
        for name in class_names:
            print(f"{name[:8]:>10}", end="")
        print()
        
        # Rows
        for i, row in enumerate(cm):
            print(f"{class_names[i][:10]:>10} ", end="")
            for val in row:
                print(f"{val:>10}", end="")
            print()
        
        # Per-class metrics
        print("\nPer-Class Metrics:")
        print(classification_report(
            y_true, y_pred,
            target_names=class_names,
            digits=3,
        ))
        
        # Feature importances
        if self.model is not None:
            print("\nFeature Importances:")
            for name, importance in zip(self.FEATURE_COLUMNS, self.model.feature_importances_):
                bar = "█" * int(importance * 40)
                print(f"  {name:>12}: {importance:.3f} {bar}")
    
    def save_model(
        self,
        model_dir: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Save trained model and scaler.
        
        Args:
            model_dir: Directory to save models (default: models/)
            
        Returns:
            Tuple of (model_path, scaler_path)
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        if model_dir is None:
            model_dir = Path(__file__).parent.parent.parent / 'models'
        else:
            model_dir = Path(model_dir)
        
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / 'radar_classifier.pkl'
        scaler_path = model_dir / 'feature_scaler.pkl'
        
        # Save artifacts
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Scaler saved to: {scaler_path}")
        
        return str(model_path), str(scaler_path)
    
    def run_pipeline(
        self,
        csv_path: str,
        test_size: float = 0.2,
        save: bool = True,
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            csv_path: Path to training data CSV
            test_size: Fraction of data for testing
            save: Whether to save model artifacts
            
        Returns:
            Dictionary of evaluation metrics
        """
        X, y = self.load_data(csv_path)
        metrics = self.train(X, y, test_size)
        
        if save:
            self.save_model()
        
        return metrics


def main():
    """Command-line entry point."""
    import sys
    
    # Default path
    csv_path = Path(__file__).parent.parent.parent / 'output' / 'training_data.csv'
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"Error: Dataset not found at {csv_path}")
        print("Run dataset_generator.py first to create training data.")
        sys.exit(1)
    
    trainer = ModelTrainer(n_estimators=100)
    metrics = trainer.run_pipeline(str(csv_path))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    if metrics['accuracy'] >= 0.85:
        print(f"\n✓ Target accuracy achieved: {metrics['accuracy']*100:.1f}% >= 85%")
    else:
        print(f"\n⚠ Target accuracy NOT achieved: {metrics['accuracy']*100:.1f}% < 85%")


if __name__ == "__main__":
    main()
