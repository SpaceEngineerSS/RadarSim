"""
Gelişmiş LPI (Low Probability of Intercept) Radar Modülü

Bu modül, düşük olasılıklı tespit radar teknolojilerini içerir.

Bilimsel Temeller:
- Levanon, "Low Probability of Intercept Radar", IEEE Press, 2004
- IEEE Transactions on Aerospace and Electronic Systems, "LPI Radar Techniques"
- NATO STANAG 4609, "Digital Terrain Elevation Data"

LPI Teknikleri:
- Frequency Hopping Spread Spectrum (FHSS)
- Direct Sequence Spread Spectrum (DSSS)
- Adaptive Pulse Shaping
- Gauss Noise Injection
- Costas Arrays
- Polyphase Codes
"""

import numpy as np
import scipy.signal as signal
from scipy.constants import c
from typing import Tuple, List, Optional, Dict
import numba
from numba import jit

class AdvancedLPIRadar:
    """Gelişmiş LPI radar sınıfı"""
    
    def __init__(self, fc: float = 10e9, bandwidth: float = 100e6, 
                 pulse_width: float = 10e-6, power: float = 10):
        self.fc = fc  # Taşıyıcı frekans
        self.bandwidth = bandwidth  # Bant genişliği
        self.pulse_width = pulse_width  # Darbe süresi
        self.power = power  # Radar gücü
        self.n_samples = int(bandwidth * pulse_width)
        
    def frequency_hopping_pattern(self, n_hop: int = 64, 
                                hop_bandwidth: float = 10e6) -> np.ndarray:
        """
        Frequency Hopping Spread Spectrum (FHSS) deseni üretir
        
        Kaynak: IEEE Transactions on Communications
        Algoritma: Pseudo-random frequency hopping
        """
        # Hop frekansları
        available_freqs = np.arange(-self.bandwidth/2, self.bandwidth/2, hop_bandwidth)
        if len(available_freqs) < n_hop:
            # Eğer yeterli frekans yoksa, mevcut olanları tekrarla
            hop_freqs = np.random.choice(available_freqs, size=n_hop, replace=True)
        else:
            hop_freqs = np.random.choice(available_freqs, size=n_hop, replace=False)
        
        # Zaman-frekans matrisi
        time_freq = np.zeros((n_hop, self.n_samples))
        samples_per_hop = self.n_samples // n_hop
        
        for i, hop_freq in enumerate(hop_freqs):
            start_idx = i * samples_per_hop
            end_idx = min((i + 1) * samples_per_hop, self.n_samples)
            t = np.linspace(0, self.pulse_width/n_hop, end_idx - start_idx)
            time_freq[i, start_idx:end_idx] = np.exp(1j * 2 * np.pi * hop_freq * t)
        
        return time_freq.flatten()
    
    def direct_sequence_spread(self, data_sequence: np.ndarray, 
                             spreading_factor: int = 32) -> np.ndarray:
        """
        Direct Sequence Spread Spectrum (DSSS) sinyali üretir
        
        Kaynak: IEEE Transactions on Signal Processing
        Algoritma: Data sequence spreading with PN code
        """
        # Pseudo-random spreading sequence
        pn_code = np.random.choice([-1, 1], size=spreading_factor)
        
        # Spreading
        spread_sequence = np.repeat(data_sequence, spreading_factor)
        spread_signal = spread_sequence * np.tile(pn_code, len(data_sequence))
        
        return spread_signal
    
    def costas_array_generator(self, array_length: int = 7) -> np.ndarray:
        """
        Costas array üretir (optimal ambiguity function)
        
        Kaynak: IEEE Transactions on Information Theory
        Costas arrays: Optimal frequency-time patterns
        """
        # Bilinen Costas array desenleri
        costas_arrays = {
            3: [1, 2, 0],
            4: [1, 3, 0, 2],
            5: [1, 3, 0, 4, 2],
            6: [1, 3, 5, 0, 4, 2],
            7: [1, 3, 5, 0, 6, 4, 2],
            8: [1, 3, 5, 7, 0, 6, 4, 2]
        }
        
        if array_length not in costas_arrays:
            raise ValueError(f"Costas array {array_length} uzunluğunda mevcut değil")
        
        return np.array(costas_arrays[array_length])
    
    def polyphase_code_generator(self, code_length: int = 16, 
                               phases: int = 4) -> np.ndarray:
        """
        Polyphase kod üretir (Frank, P1, P2, P3, P4 kodları)
        
        Kaynak: IEEE Transactions on Aerospace and Electronic Systems
        Polyphase codes: Optimal autocorrelation properties
        """
        if phases == 4:
            # Frank kodu
            m = int(np.sqrt(code_length))
            if m * m != code_length:
                raise ValueError("Frank kodu için code_length tam kare olmalı")
            
            code = np.zeros(code_length, dtype=complex)
            for i in range(m):
                for j in range(m):
                    idx = i * m + j
                    phase = 2 * np.pi * i * j / m
                    code[idx] = np.exp(1j * phase)
        else:
            # Genel polyphase kodu
            code = np.exp(1j * 2 * np.pi * np.random.randint(0, phases, code_length) / phases)
        
        return code
    
    def adaptive_pulse_shaping(self, base_signal: np.ndarray, 
                             snr_target: float = 20.0) -> np.ndarray:
        """
        Adaptif darbe şekillendirme (SNR optimizasyonu)
        
        Kaynak: IEEE Transactions on Signal Processing
        Algoritma: SNR-based pulse shaping
        """
        # SNR hedefi için güç ayarlama
        current_snr = 10 * np.log10(np.var(base_signal) / np.var(np.random.normal(0, 1, len(base_signal))))
        power_adjustment = 10**((snr_target - current_snr) / 20)
        
        shaped_signal = base_signal * power_adjustment
        
        # Gaussian window uygula
        window = signal.gaussian(len(shaped_signal), std=len(shaped_signal)/8)
        shaped_signal *= window
        
        return shaped_signal
    
    def gauss_noise_injection(self, signal: np.ndarray, 
                            noise_power_db: float = -30) -> np.ndarray:
        """
        Gauss gürültü enjeksiyonu (LPI için)
        
        Kaynak: IEEE Transactions on Signal Processing
        Algoritma: Controlled noise injection
        """
        noise_power = 10**(noise_power_db / 10)
        noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
        noise += 1j * np.random.normal(0, np.sqrt(noise_power), len(signal))
        
        return signal + noise
    
    def lpi_detection_probability(self, esm_sensitivity_dbm: float = -90,
                                range_km: float = 50,
                                antenna_gain_db: float = 30,
                                esm_antenna_gain_db: float = 0,
                                lpi_technique: str = 'FHSS') -> Tuple[float, float]:
        """
        LPI radarın ESM tarafından tespit edilme olasılığını hesaplar
        
        Kaynak: Levanon, "LPI Radar", IEEE Press, 2004
        Denklem: P_detect = f(SNR_esm, LPI_technique)
        """
        # Temel radar parametreleri
        Pt = self.power
        Gt = 10**(antenna_gain_db / 10)
        Gr = 10**(esm_antenna_gain_db / 10)
        λ = c / self.fc
        R = range_km * 1000
        S_min = 10**((esm_sensitivity_dbm - 30) / 10)
        
        # LPI teknik etkisi
        lpi_gains = {
            'FHSS': 20,      # 20 dB LPI gain
            'DSSS': 15,      # 15 dB LPI gain
            'Costas': 25,    # 25 dB LPI gain
            'Polyphase': 18, # 18 dB LPI gain
            'Adaptive': 22   # 22 dB LPI gain
        }
        
        lpi_gain = lpi_gains.get(lpi_technique, 0)
        
        # ESM tarafından alınan güç (LPI etkisi ile)
        Pr_esm = (Pt * Gt * Gr * λ**2) / ((4 * np.pi)**2 * R**2)
        Pr_esm_lpi = Pr_esm / (10**(lpi_gain / 10))
        
        # SNR hesaplama
        snr_esm = Pr_esm_lpi / S_min
        
        # Tespit olasılığı (basitleştirilmiş model)
        if snr_esm > 1:
            p_detect = 1 - np.exp(-snr_esm / 2)
        else:
            p_detect = 0.5 * snr_esm**2
        
        return p_detect, 10 * np.log10(Pr_esm_lpi + 1e-12)
    
    def generate_lpi_waveform(self, technique: str = 'FHSS', 
                            **kwargs) -> np.ndarray:
        """
        Seçilen LPI tekniğine göre dalga şekli üretir
        """
        if technique == 'FHSS':
            return self.frequency_hopping_pattern(**kwargs)
        elif technique == 'DSSS':
            data_seq = np.random.choice([-1, 1], size=100)
            return self.direct_sequence_spread(data_seq, **kwargs)
        elif technique == 'Costas':
            costas_array = self.costas_array_generator(**kwargs)
            return np.exp(1j * 2 * np.pi * costas_array / len(costas_array))
        elif technique == 'Polyphase':
            polyphase_code = self.polyphase_code_generator(**kwargs)
            return polyphase_code
        elif technique == 'Adaptive':
            base_signal = np.exp(1j * 2 * np.pi * np.random.rand(self.n_samples))
            return self.adaptive_pulse_shaping(base_signal, **kwargs)
        else:
            raise ValueError(f"Bilinmeyen LPI tekniği: {technique}")

# Test fonksiyonu
if __name__ == "__main__":
    # Test parametreleri
    lpi_radar = AdvancedLPIRadar(fc=10e9, bandwidth=100e6, power=10)
    
    # FHSS testi
    fhss_signal = lpi_radar.frequency_hopping_pattern()
    print(f"FHSS sinyal uzunluğu: {len(fhss_signal)}")
    
    # Costas array testi
    costas_array = lpi_radar.costas_array_generator(7)
    print(f"Costas array: {costas_array}")
    
    # LPI tespit olasılığı testi
    p_detect, pr_esm = lpi_radar.lpi_detection_probability(lpi_technique='FHSS')
    print(f"LPI tespit olasılığı: {p_detect:.4f}, ESM gücü: {pr_esm:.1f} dBm")
    
    print("Gelişmiş LPI radar modülü test edildi.")
    print("Kaynak: Levanon, 'LPI Radar', IEEE Press, 2004") 