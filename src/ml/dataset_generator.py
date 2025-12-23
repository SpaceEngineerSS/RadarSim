"""
Synthetic Dataset Generator for Radar Target Classification

Generates physically-realistic labeled training data for ML models.

Target Classes:
    - Class 0 (Drone): Low RCS (0.01-0.1 m²), Low Speed (0-100 m/s), Swerling-1
    - Class 1 (Fighter): High RCS (2-10 m²), Medium Speed (200-600 m/s), Steady
    - Class 2 (Missile): Medium RCS (0.1-0.5 m²), High Speed (800-1200 m/s), Swerling-2

Physics:
    - SNR calculated from radar equation
    - RCS_est derived from inverse radar equation with noise
    - Doppler shift from velocity and carrier frequency

References:
    - Skolnik, Introduction to Radar Systems
    - Swerling Target Models (IEEE)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Optional
from pathlib import Path
import os


class TargetClass(IntEnum):
    """Target classification categories."""
    DRONE = 0
    FIGHTER = 1
    MISSILE = 2


@dataclass
class ClassParameters:
    """Parameters defining a target class."""
    class_id: TargetClass
    name: str
    rcs_range: Tuple[float, float]      # m²
    speed_range: Tuple[float, float]    # m/s
    swerling_type: int                  # 0=steady, 1=Swerling-1, 2=Swerling-2
    
    
# Define class characteristics
CLASS_DEFINITIONS = {
    TargetClass.DRONE: ClassParameters(
        class_id=TargetClass.DRONE,
        name="Drone",
        rcs_range=(0.01, 0.1),
        speed_range=(0, 100),
        swerling_type=1,  # Fluctuating
    ),
    TargetClass.FIGHTER: ClassParameters(
        class_id=TargetClass.FIGHTER,
        name="Fighter Jet",
        rcs_range=(2.0, 10.0),
        speed_range=(200, 600),
        swerling_type=0,  # Steady
    ),
    TargetClass.MISSILE: ClassParameters(
        class_id=TargetClass.MISSILE,
        name="Missile",
        rcs_range=(0.1, 0.5),
        speed_range=(800, 1200),
        swerling_type=2,  # Swerling-2 (scan-to-scan)
    ),
}


class DatasetGenerator:
    """
    Generates synthetic radar target data for ML training.
    
    Uses physics-based calculations to create realistic feature distributions
    that reflect actual radar measurement characteristics.
    
    Attributes:
        radar_frequency_hz: Carrier frequency for Doppler calculation
        radar_power_w: Transmit power
        antenna_gain_db: Antenna gain
        noise_figure_db: Receiver noise figure
    """
    
    # Physical constants
    SPEED_OF_LIGHT = 3e8  # m/s
    BOLTZMANN = 1.38e-23  # J/K
    
    def __init__(
        self,
        radar_frequency_hz: float = 10e9,
        radar_power_w: float = 100e3,
        antenna_gain_db: float = 30.0,
        noise_figure_db: float = 4.0,
        bandwidth_hz: float = 1e6,
        seed: Optional[int] = None,
    ):
        """
        Initialize generator with radar parameters.
        
        Args:
            radar_frequency_hz: Operating frequency [Hz]
            radar_power_w: Transmit power [W]
            antenna_gain_db: Antenna gain [dB]
            noise_figure_db: Receiver noise figure [dB]
            bandwidth_hz: Receiver bandwidth [Hz]
            seed: Random seed for reproducibility
        """
        self.frequency = radar_frequency_hz
        self.power = radar_power_w
        self.gain_db = antenna_gain_db
        self.nf_db = noise_figure_db
        self.bandwidth = bandwidth_hz
        
        # Derived parameters
        self.wavelength = self.SPEED_OF_LIGHT / self.frequency
        self.gain_linear = 10 ** (antenna_gain_db / 10)
        self.noise_figure = 10 ** (noise_figure_db / 10)
        
        # Noise power (kT0BF)
        self.noise_power = (
            self.BOLTZMANN * 290 * self.bandwidth * self.noise_figure
        )
        
        if seed is not None:
            np.random.seed(seed)
    
    def _calculate_snr(self, rcs: float, range_m: float) -> float:
        """
        Calculate SNR using radar equation.
        
        SNR = (Pt * G² * λ² * σ) / ((4π)³ * R⁴ * kT₀BF)
        
        Args:
            rcs: Radar cross section [m²]
            range_m: Target range [m]
            
        Returns:
            SNR in dB
        """
        numerator = (
            self.power * 
            (self.gain_linear ** 2) * 
            (self.wavelength ** 2) * 
            rcs
        )
        denominator = (
            ((4 * np.pi) ** 3) * 
            (range_m ** 4) * 
            self.noise_power
        )
        
        snr_linear = numerator / denominator
        return 10 * np.log10(max(snr_linear, 1e-10))
    
    def _calculate_doppler(self, velocity_mps: float) -> float:
        """
        Calculate Doppler frequency shift.
        
        fd = 2 * v * f / c
        
        Args:
            velocity_mps: Radial velocity [m/s]
            
        Returns:
            Doppler shift [Hz]
        """
        return 2 * velocity_mps * self.frequency / self.SPEED_OF_LIGHT
    
    def _estimate_rcs_from_snr(
        self, 
        snr_db: float, 
        range_m: float,
        noise_std: float = 0.3
    ) -> float:
        """
        Inverse radar equation to estimate RCS from measured SNR.
        
        This simulates what a real radar would measure - an estimated RCS
        derived from the return signal strength with measurement noise.
        
        Args:
            snr_db: Measured SNR [dB]
            range_m: Target range [m]
            noise_std: Multiplicative noise standard deviation
            
        Returns:
            Estimated RCS [m²] with noise
        """
        snr_linear = 10 ** (snr_db / 10)
        
        # Invert radar equation for RCS
        numerator = (
            snr_linear * 
            ((4 * np.pi) ** 3) * 
            (range_m ** 4) * 
            self.noise_power
        )
        denominator = (
            self.power * 
            (self.gain_linear ** 2) * 
            (self.wavelength ** 2)
        )
        
        rcs_true = numerator / denominator
        
        # Add multiplicative noise (log-normal)
        noise_factor = np.exp(np.random.normal(0, noise_std))
        rcs_estimated = rcs_true * noise_factor
        
        return max(rcs_estimated, 0.001)  # Minimum 0.001 m²
    
    def _apply_swerling_fluctuation(
        self, 
        snr_db: float, 
        swerling_type: int
    ) -> float:
        """
        Apply Swerling model RCS fluctuation to SNR.
        
        Args:
            snr_db: Base SNR [dB]
            swerling_type: 0=steady, 1=Swerling-1, 2=Swerling-2
            
        Returns:
            Fluctuated SNR [dB]
        """
        if swerling_type == 0:
            # Steady target (small noise)
            return snr_db + np.random.normal(0, 0.5)
        
        elif swerling_type == 1:
            # Swerling-1: Exponential distribution (high variance)
            # Chi-squared with 2 DOF
            fluctuation = np.random.exponential(1.0)
            return snr_db + 10 * np.log10(max(fluctuation, 0.01))
        
        elif swerling_type == 2:
            # Swerling-2: Chi-squared with 4 DOF (less variance than Sw1)
            fluctuation = np.random.gamma(2, 0.5)
            return snr_db + 10 * np.log10(max(fluctuation, 0.01))
        
        return snr_db
    
    def generate_sample(
        self, 
        target_class: TargetClass
    ) -> dict:
        """
        Generate a single labeled sample.
        
        Args:
            target_class: Target class to generate
            
        Returns:
            Dictionary with features and label
        """
        params = CLASS_DEFINITIONS[target_class]
        
        # Random parameters within class bounds
        rcs_true = np.random.uniform(*params.rcs_range)
        speed = np.random.uniform(*params.speed_range)
        range_km = np.random.uniform(10, 150)
        range_m = range_km * 1000
        
        # Calculate base SNR
        snr_base = self._calculate_snr(rcs_true, range_m)
        
        # Apply Swerling fluctuation
        snr_measured = self._apply_swerling_fluctuation(
            snr_base, params.swerling_type
        )
        
        # Add additional measurement noise
        snr_measured += np.random.normal(0, 1.0)
        
        # Calculate Doppler
        doppler = self._calculate_doppler(speed)
        # Add Doppler measurement noise
        doppler += np.random.normal(0, 50)
        
        # Estimate RCS from measured SNR (inverse problem)
        rcs_estimated = self._estimate_rcs_from_snr(snr_measured, range_m)
        
        return {
            'range_km': range_km,
            'doppler_hz': doppler,
            'snr_db': snr_measured,
            'rcs_est_m2': rcs_estimated,
            'class_id': int(target_class),
        }
    
    def generate(
        self,
        samples_per_class: int = 1000,
        output_path: Optional[str] = None,
        shuffle: bool = True,
    ) -> pd.DataFrame:
        """
        Generate complete training dataset.
        
        Args:
            samples_per_class: Number of samples per target class
            output_path: Path to save CSV (default: output/training_data.csv)
            shuffle: Whether to shuffle the dataset
            
        Returns:
            DataFrame with all samples
        """
        samples = []
        
        for target_class in TargetClass:
            print(f"Generating {samples_per_class} samples for {CLASS_DEFINITIONS[target_class].name}...")
            
            for _ in range(samples_per_class):
                sample = self.generate_sample(target_class)
                samples.append(sample)
        
        # Create DataFrame
        df = pd.DataFrame(samples)
        
        if shuffle:
            df = df.sample(frac=1.0).reset_index(drop=True)
        
        # Save to CSV
        if output_path is None:
            output_dir = Path(__file__).parent.parent.parent / 'output'
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / 'training_data.csv'
        
        df.to_csv(output_path, index=False)
        print(f"\n✓ Dataset saved to: {output_path}")
        print(f"  Total samples: {len(df)}")
        print(f"  Features: {list(df.columns)}")
        
        # Print class distribution
        print("\nClass Distribution:")
        for class_id, count in df['class_id'].value_counts().sort_index().items():
            class_name = CLASS_DEFINITIONS[TargetClass(class_id)].name
            print(f"  {class_id}: {class_name} - {count} samples")
        
        return df


def main():
    """Command-line entry point."""
    generator = DatasetGenerator(seed=42)
    df = generator.generate(samples_per_class=1000)
    
    # Print sample statistics
    print("\nFeature Statistics:")
    print(df.describe().round(2))


if __name__ == "__main__":
    main()
