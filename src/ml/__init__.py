"""
RadarSim Machine Learning Module

Provides AI/ML capabilities for target classification based on radar signatures.

Components:
    - DatasetGenerator: Creates synthetic labeled training data
    - ModelTrainer: Trains and evaluates classification models
    - InferenceEngine: Real-time prediction for live simulation

Usage:
    from src.ml import DatasetGenerator, ModelTrainer, InferenceEngine
"""

from .dataset_generator import DatasetGenerator, TargetClass
from .trainer import ModelTrainer
from .inference_engine import InferenceEngine

__all__ = [
    'DatasetGenerator',
    'TargetClass',
    'ModelTrainer',
    'InferenceEngine',
]
