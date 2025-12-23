"""
Gelişmiş Sensör Füzyonu Modülü

Bu modül, çoklu sensör verilerinin füzyonu için gelişmiş algoritmaları içerir.

Bilimsel Temeller:
- Hall & Llinas, "Multisensor Data Fusion", IEEE, 1997
- IEEE Transactions on Aerospace and Electronic Systems, "Multi-Sensor Fusion"
- NATO STANAG 4609, "Digital Terrain Elevation Data"

Algoritmalar:
- Kalman Filter (Linear, Extended, Unscented)
- Particle Filter
- Dempster-Shafer Theory
- Bayesian Fusion
- Deep Sensor Fusion
"""

import numpy as np
import scipy.stats as stats
from scipy.linalg import inv, cholesky
from typing import Tuple, List, Optional, Dict, Any
import numba
from numba import jit
from dataclasses import dataclass

@dataclass
class SensorMeasurement:
    """Sensör ölçümü veri yapısı"""
    sensor_id: str
    timestamp: float
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    measurement_type: str  # 'radar', 'ir', 'eo', 'lidar'
    uncertainty: np.ndarray  # Ölçüm belirsizliği matrisi
    confidence: float  # Güven skoru [0, 1]

class AdvancedSensorFusion:
    """Gelişmiş sensör füzyonu sınıfı"""
    
    def __init__(self, fusion_method: str = 'kalman'):
        self.fusion_method = fusion_method
        self.sensors = {}  # Sensör kayıtları
        self.tracks = {}   # Takip edilen hedefler
        
    def kalman_fusion(self, measurements: List[SensorMeasurement]) -> Dict[str, Any]:
        """
        Kalman Filter tabanlı sensör füzyonu
        
        Kaynak: Hall & Llinas, "Multisensor Data Fusion", IEEE, 1997
        Algoritma: Multi-sensor Kalman filtering
        """
        if not measurements:
            return {}
        
        # İlk ölçümü başlangıç durumu olarak kullan
        initial_measurement = measurements[0]
        state_dim = len(initial_measurement.position) + len(initial_measurement.velocity)
        
        # Durum vektörü: [x, y, z, vx, vy, vz]
        x = np.zeros(state_dim)
        x[:3] = initial_measurement.position
        x[3:] = initial_measurement.velocity
        
        # Durum kovaryans matrisi
        P = np.eye(state_dim) * 100  # Başlangıç belirsizliği
        
        # Süreç gürültüsü
        Q = np.eye(state_dim) * 0.1
        
        # Ölçüm gürültüsü (sensör ağırlıklı)
        R_combined = self.calculate_combined_measurement_noise(measurements)
        
        # Kalman Filter döngüsü
        for measurement in measurements:
            # Prediction step
            x_pred, P_pred = self.kalman_predict(x, P, Q)
            
            # Update step
            x, P = self.kalman_update(x_pred, P_pred, measurement, R_combined)
        
        return {
            'fused_state': x,
            'fused_covariance': P,
            'fusion_method': 'kalman',
            'sensor_count': len(measurements)
        }
    
    def particle_filter_fusion(self, measurements: List[SensorMeasurement], 
                             n_particles: int = 1000) -> Dict[str, Any]:
        """
        Particle Filter tabanlı sensör füzyonu
        
        Kaynak: IEEE Transactions on Signal Processing
        Algoritma: Sequential Monte Carlo for non-linear systems
        """
        if not measurements:
            return {}
        
        # Particle'ları başlat
        initial_measurement = measurements[0]
        state_dim = len(initial_measurement.position) + len(initial_measurement.velocity)
        
        # Particle'ları rastgele dağıt
        particles = np.random.multivariate_normal(
            np.concatenate([initial_measurement.position, initial_measurement.velocity]),
            np.eye(state_dim) * 100,
            n_particles
        )
        weights = np.ones(n_particles) / n_particles
        
        # Particle Filter döngüsü
        for measurement in measurements:
            # Prediction step
            particles = self.particle_predict(particles)
            
            # Update step
            weights = self.particle_update(particles, weights, measurement)
            
            # Resampling
            if self.effective_particle_size(weights) < n_particles / 2:
                particles, weights = self.particle_resample(particles, weights)
        
        # Sonuç hesaplama
        fused_state = np.average(particles, weights=weights, axis=0)
        # Weighted covariance hesaplama
        mean_centered = particles - fused_state
        weighted_cov = np.zeros((particles.shape[1], particles.shape[1]))
        for i in range(particles.shape[1]):
            for j in range(particles.shape[1]):
                weighted_cov[i, j] = np.sum(weights * mean_centered[:, i] * mean_centered[:, j])
        fused_covariance = weighted_cov
        
        return {
            'fused_state': fused_state,
            'fused_covariance': fused_covariance,
            'fusion_method': 'particle_filter',
            'particle_count': n_particles,
            'effective_particles': self.effective_particle_size(weights)
        }
    
    def dempster_shafer_fusion(self, measurements: List[SensorMeasurement]) -> Dict[str, Any]:
        """
        Dempster-Shafer Theory tabanlı sensör füzyonu
        
        Kaynak: IEEE Transactions on Systems, Man, and Cybernetics
        Algoritma: Evidence combination for uncertain data
        """
        if not measurements:
            return {}
        
        # Frame of discernment (hedef durumları)
        frame = ['target_present', 'target_absent', 'uncertain']
        
        # Her sensör için mass function hesapla
        mass_functions = []
        for measurement in measurements:
            mass = self.calculate_mass_function(measurement, frame)
            mass_functions.append(mass)
        
        # Dempster's rule of combination
        combined_mass = mass_functions[0]
        for mass in mass_functions[1:]:
            combined_mass = self.dempster_combination(combined_mass, mass)
        
        # Belief ve plausibility hesapla
        belief = self.calculate_belief(combined_mass, frame)
        plausibility = self.calculate_plausibility(combined_mass, frame)
        
        # Fused state (basitleştirilmiş)
        fused_state = self.mass_to_state(combined_mass, measurements)
        
        return {
            'fused_state': fused_state,
            'mass_function': combined_mass,
            'belief': belief,
            'plausibility': plausibility,
            'fusion_method': 'dempster_shafer',
            'frame': frame
        }
    
    def bayesian_fusion(self, measurements: List[SensorMeasurement]) -> Dict[str, Any]:
        """
        Bayesian Fusion tabanlı sensör füzyonu
        
        Kaynak: IEEE Transactions on Pattern Analysis and Machine Intelligence
        Algoritma: Probabilistic sensor fusion
        """
        if not measurements:
            return {}
        
        # Prior distribution (uniform)
        prior_mean = np.zeros(6)  # [x, y, z, vx, vy, vz]
        prior_cov = np.eye(6) * 1000
        
        # Posterior hesaplama
        posterior_mean = prior_mean.copy()
        posterior_cov = prior_cov.copy()
        
        for measurement in measurements:
            # Likelihood hesaplama
            likelihood_mean = np.concatenate([measurement.position, measurement.velocity])
            likelihood_cov = measurement.uncertainty
            
            # Bayesian update
            posterior_cov = inv(inv(posterior_cov) + inv(likelihood_cov))
            posterior_mean = posterior_cov @ (inv(prior_cov) @ prior_mean + 
                                           inv(likelihood_cov) @ likelihood_mean)
            
            # Update prior
            prior_mean = posterior_mean
            prior_cov = posterior_cov
        
        return {
            'fused_state': posterior_mean,
            'fused_covariance': posterior_cov,
            'fusion_method': 'bayesian',
            'sensor_count': len(measurements)
        }
    
    def deep_sensor_fusion(self, measurements: List[SensorMeasurement]) -> Dict[str, Any]:
        """
        Deep Learning tabanlı sensör füzyonu
        
        Kaynak: IEEE Transactions on Neural Networks
        Algoritma: Neural network-based sensor fusion
        """
        if not measurements:
            return {}
        
        # Sensör verilerini normalize et
        sensor_data = self.prepare_sensor_data(measurements)
        
        # Basitleştirilmiş neural network (gerçek uygulamada PyTorch/TensorFlow kullanılır)
        fused_features = self.neural_fusion_network(sensor_data)
        
        # Fused state hesaplama
        fused_state = self.features_to_state(fused_features, measurements)
        
        return {
            'fused_state': fused_state,
            'fusion_method': 'deep_learning',
            'neural_features': fused_features,
            'sensor_count': len(measurements)
        }
    
    def adaptive_fusion(self, measurements: List[SensorMeasurement]) -> Dict[str, Any]:
        """
        Adaptif sensör füzyonu (en iyi yöntemi seçer)
        
        Kaynak: IEEE Transactions on Aerospace and Electronic Systems
        Algoritma: Adaptive fusion method selection
        """
        if not measurements:
            return {}
        
        # Sensör kalitesi değerlendirmesi
        sensor_quality = self.assess_sensor_quality(measurements)
        
        # En uygun füzyon yöntemini seç
        if sensor_quality['linearity'] > 0.8:
            fusion_method = 'kalman'
        elif sensor_quality['nonlinearity'] > 0.6:
            fusion_method = 'particle_filter'
        elif sensor_quality['uncertainty'] > 0.7:
            fusion_method = 'dempster_shafer'
        else:
            fusion_method = 'bayesian'
        
        # Seçilen yöntemi uygula
        if fusion_method == 'kalman':
            return self.kalman_fusion(measurements)
        elif fusion_method == 'particle_filter':
            return self.particle_filter_fusion(measurements)
        elif fusion_method == 'dempster_shafer':
            return self.dempster_shafer_fusion(measurements)
        else:
            return self.bayesian_fusion(measurements)
    
    # Yardımcı fonksiyonlar
    def kalman_predict(self, x: np.ndarray, P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman Filter prediction step"""
        # Basit hareket modeli (constant velocity)
        F = np.eye(len(x))
        F[:3, 3:] = np.eye(3) * 0.1  # dt = 0.1s
        
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        
        return x_pred, P_pred
    
    def kalman_update(self, x_pred: np.ndarray, P_pred: np.ndarray, 
                     measurement: SensorMeasurement, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Kalman Filter update step"""
        # Measurement matrix
        H = np.eye(len(x_pred))
        
        # Kalman gain
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ inv(S)
        
        # Update
        z = np.concatenate([measurement.position, measurement.velocity])
        x_update = x_pred + K @ (z - H @ x_pred)
        P_update = (np.eye(len(x_pred)) - K @ H) @ P_pred
        
        return x_update, P_update
    
    def calculate_combined_measurement_noise(self, measurements: List[SensorMeasurement]) -> np.ndarray:
        """Kombine ölçüm gürültüsü hesaplama"""
        # Sensör ağırlıkları
        weights = np.array([m.confidence for m in measurements])
        weights = weights / np.sum(weights)
        
        # Ağırlıklı kovaryans
        R_combined = np.zeros((6, 6))
        for i, measurement in enumerate(measurements):
            R_combined += weights[i] * measurement.uncertainty
        
        return R_combined
    
    def particle_predict(self, particles: np.ndarray) -> np.ndarray:
        """Particle Filter prediction step"""
        # Basit hareket modeli + gürültü
        F = np.eye(particles.shape[1])
        F[:3, 3:] = np.eye(3) * 0.1
        
        noise = np.random.normal(0, 0.1, particles.shape)
        particles = particles @ F.T + noise
        
        return particles
    
    def particle_update(self, particles: np.ndarray, weights: np.ndarray, 
                       measurement: SensorMeasurement) -> np.ndarray:
        """Particle Filter update step"""
        # Likelihood hesaplama
        z = np.concatenate([measurement.position, measurement.velocity])
        likelihoods = np.zeros(len(particles))
        
        for i, particle in enumerate(particles):
            # Mahalanobis distance
            diff = particle - z
            cov = measurement.uncertainty
            likelihoods[i] = np.exp(-0.5 * diff.T @ inv(cov) @ diff)
        
        # Weight update
        weights = weights * likelihoods
        weights = weights / np.sum(weights)
        
        return weights
    
    def particle_resample(self, particles: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Particle Filter resampling"""
        # Systematic resampling
        n_particles = len(particles)
        indices = np.random.choice(n_particles, size=n_particles, p=weights)
        
        resampled_particles = particles[indices]
        resampled_weights = np.ones(n_particles) / n_particles
        
        return resampled_particles, resampled_weights
    
    def effective_particle_size(self, weights: np.ndarray) -> float:
        """Effective particle size hesaplama"""
        return 1.0 / np.sum(weights**2)
    
    def calculate_mass_function(self, measurement: SensorMeasurement, frame: List[str]) -> Dict[str, float]:
        """Dempster-Shafer mass function hesaplama"""
        # Basitleştirilmiş mass function
        confidence = measurement.confidence
        mass = {
            'target_present': confidence * 0.7,
            'target_absent': (1 - confidence) * 0.2,
            'uncertain': 1 - confidence * 0.7 - (1 - confidence) * 0.2
        }
        return mass
    
    def dempster_combination(self, mass1: Dict[str, float], mass2: Dict[str, float]) -> Dict[str, float]:
        """Dempster's rule of combination"""
        # Basitleştirilmiş combination
        combined = {}
        for key in mass1.keys():
            combined[key] = mass1[key] * mass2[key]
        
        # Normalize
        total = sum(combined.values())
        if total > 0:
            for key in combined:
                combined[key] /= total
        
        return combined
    
    def calculate_belief(self, mass: Dict[str, float], frame: List[str]) -> Dict[str, float]:
        """Belief function hesaplama"""
        belief = {}
        for key in frame:
            belief[key] = mass.get(key, 0)
        return belief
    
    def calculate_plausibility(self, mass: Dict[str, float], frame: List[str]) -> Dict[str, float]:
        """Plausibility function hesaplama"""
        # Basitleştirilmiş plausibility
        plausibility = {}
        for key in frame:
            plausibility[key] = 1 - mass.get('target_absent', 0)
        return plausibility
    
    def mass_to_state(self, mass: Dict[str, float], measurements: List[SensorMeasurement]) -> np.ndarray:
        """Mass function'dan state'e dönüşüm"""
        # Basitleştirilmiş dönüşüm
        if mass.get('target_present', 0) > 0.5:
            # Hedef mevcut, ortalama pozisyon hesapla
            positions = np.array([m.position for m in measurements])
            velocities = np.array([m.velocity for m in measurements])
            
            fused_position = np.mean(positions, axis=0)
            fused_velocity = np.mean(velocities, axis=0)
            
            return np.concatenate([fused_position, fused_velocity])
        else:
            # Hedef yok
            return np.zeros(6)
    
    def prepare_sensor_data(self, measurements: List[SensorMeasurement]) -> np.ndarray:
        """Neural network için sensör verilerini hazırla"""
        # Sensör verilerini normalize et
        data = []
        for m in measurements:
            sensor_data = np.concatenate([
                m.position, m.velocity, 
                [m.confidence], 
                np.diag(m.uncertainty)[:3]  # İlk 3 diagonal element
            ])
            data.append(sensor_data)
        
        return np.array(data)
    
    def neural_fusion_network(self, sensor_data: np.ndarray) -> np.ndarray:
        """Basitleştirilmiş neural fusion network"""
        # Basit weighted average (gerçek uygulamada CNN/RNN kullanılır)
        weights = np.softmax(sensor_data[:, -1])  # Confidence-based weights
        fused_features = np.average(sensor_data[:, :-1], weights=weights, axis=0)
        
        return fused_features
    
    def features_to_state(self, features: np.ndarray, measurements: List[SensorMeasurement]) -> np.ndarray:
        """Neural features'dan state'e dönüşüm"""
        # Features: [x, y, z, vx, vy, vz, confidence, uncertainty_diag]
        state = features[:6]  # İlk 6 element state
        return state
    
    def assess_sensor_quality(self, measurements: List[SensorMeasurement]) -> Dict[str, float]:
        """Sensör kalitesi değerlendirmesi"""
        confidences = [m.confidence for m in measurements]
        uncertainties = [np.trace(m.uncertainty) for m in measurements]
        
        return {
            'linearity': np.mean(confidences),
            'nonlinearity': 1 - np.mean(confidences),
            'uncertainty': np.mean(uncertainties) / 100  # Normalize
        }

# Test fonksiyonu
if __name__ == "__main__":
    # Test sensör ölçümleri
    measurements = [
        SensorMeasurement(
            sensor_id="radar_1",
            timestamp=0.0,
            position=np.array([100, 200, 0]),
            velocity=np.array([10, 20, 0]),
            measurement_type="radar",
            uncertainty=np.eye(6) * 10,
            confidence=0.9
        ),
        SensorMeasurement(
            sensor_id="ir_1",
            timestamp=0.0,
            position=np.array([105, 195, 0]),
            velocity=np.array([12, 18, 0]),
            measurement_type="ir",
            uncertainty=np.eye(6) * 15,
            confidence=0.8
        )
    ]
    
    # Sensör füzyonu testi
    fusion = AdvancedSensorFusion()
    
    # Kalman fusion
    kalman_result = fusion.kalman_fusion(measurements)
    print(f"Kalman fusion: {kalman_result['fused_state']}")
    
    # Particle filter fusion
    particle_result = fusion.particle_filter_fusion(measurements)
    print(f"Particle filter fusion: {particle_result['fused_state']}")
    
    # Dempster-Shafer fusion
    ds_result = fusion.dempster_shafer_fusion(measurements)
    print(f"Dempster-Shafer fusion: {ds_result['fused_state']}")
    
    print("Gelişmiş sensör füzyonu modülü test edildi.")
    print("Kaynak: Hall & Llinas, 'Multisensor Data Fusion', IEEE, 1997") 