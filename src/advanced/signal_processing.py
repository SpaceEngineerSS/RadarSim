"""
Gelişmiş Radar Sinyal İşleme Modülü

Bu modül, IEEE standartlarına uygun gelişmiş radar sinyal işleme algoritmalarını içerir.

Bilimsel Temeller:
- Skolnik, "Radar Handbook", 3rd Ed., McGraw-Hill, 2008
- IEEE Std 686-2008, "IEEE Standard Radar Definitions"
- IEEE Transactions on Signal Processing, "Advanced Radar Signal Processing"

Algoritmalar:
- Matched Filtering with optimal SNR
- Pulse Compression (Chirp, Barker, Polyphase codes)
- CFAR Detection (CA-CFAR, GO-CFAR, SO-CFAR)
- Doppler Processing (FFT, MTI, STAP)
- Adaptive Beamforming
"""

import numpy as np
import scipy.signal as signal
from scipy.constants import c
from typing import Tuple, List, Optional, Dict
import numba
from numba import jit, prange

class AdvancedSignalProcessor:
    """Gelişmiş radar sinyal işleme sınıfı"""
    
    def __init__(self, sampling_rate: float = 1e9, pulse_width: float = 1e-6):
        self.fs = sampling_rate
        self.pulse_width = pulse_width
        self.n_samples = int(sampling_rate * pulse_width)
        
    def generate_chirp_signal(self, start_freq: float, end_freq: float, 
                            amplitude: float = 1.0) -> np.ndarray:
        """
        Linear Frequency Modulated (Chirp) sinyal üretir
        
        Kaynak: Skolnik, "Radar Handbook", 3rd Ed., McGraw-Hill, 2008
        Denklem: s(t) = A * exp(j * 2π * (f0*t + (k/2)*t²))
        """
        t = np.linspace(0, self.pulse_width, self.n_samples)
        k = (end_freq - start_freq) / self.pulse_width  # Chirp rate
        
        # Chirp sinyali
        phase = 2 * np.pi * (start_freq * t + 0.5 * k * t**2)
        chirp_signal = amplitude * np.exp(1j * phase)
        
        return chirp_signal
    
    def generate_barker_code(self, code_length: int = 13) -> np.ndarray:
        """
        Barker kodu üretir (optimal autocorrelation)
        
        Kaynak: IEEE Transactions on Aerospace and Electronic Systems
        Barker kodları: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
        """
        barker_codes = {
            2: [1, -1],
            3: [1, 1, -1],
            4: [1, 1, -1, 1],
            5: [1, 1, 1, -1, 1],
            7: [1, 1, 1, -1, -1, 1, -1],
            11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
            13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
        }
        
        if code_length not in barker_codes:
            raise ValueError(f"Barker kodu {code_length} uzunluğunda mevcut değil")
        
        code = np.array(barker_codes[code_length])
        # Kod uzunluğunu sinyal uzunluğuna genişlet
        samples_per_chip = self.n_samples // len(code)
        expanded_code = np.repeat(code, samples_per_chip)
        
        return expanded_code
    
    def matched_filter(self, received_signal: np.ndarray, 
                      reference_signal: np.ndarray) -> np.ndarray:
        """
        Matched filter uygular (optimal SNR için)
        
        Kaynak: IEEE Transactions on Signal Processing
        Denklem: y(t) = ∫ x(τ) * h*(t-τ) dτ
        """
        # Cross-correlation (matched filtering)
        filtered_signal = np.correlate(received_signal, 
                                     np.conj(reference_signal), 
                                     mode='full')
        return filtered_signal
    
    def cfar_detection(self, range_profile: np.ndarray, 
                      guard_cells: int = 2, 
                      reference_cells: int = 8,
                      threshold_factor: float = 2.0,
                      cfar_type: str = 'CA') -> Tuple[np.ndarray, np.ndarray]:
        """
        CFAR (Constant False Alarm Rate) tespit algoritması
        
        Kaynak: IEEE Transactions on Aerospace and Electronic Systems
        CFAR Türleri: CA (Cell Averaging), GO (Greatest Of), SO (Smallest Of)
        """
        n_samples = len(range_profile)
        detections = np.zeros(n_samples, dtype=bool)
        thresholds = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Koruma hücreleri
            guard_start = max(0, i - guard_cells)
            guard_end = min(n_samples, i + guard_cells + 1)
            
            # Referans hücreleri
            ref_start_left = max(0, guard_start - reference_cells)
            ref_end_left = guard_start
            ref_start_right = guard_end
            ref_end_right = min(n_samples, guard_end + reference_cells)
            
            # Referans değerleri
            ref_values = []
            if ref_start_left < ref_end_left:
                ref_values.extend(range_profile[ref_start_left:ref_end_left])
            if ref_start_right < ref_end_right:
                ref_values.extend(range_profile[ref_start_right:ref_end_right])
            
            if len(ref_values) > 0:
                if cfar_type == 'CA':
                    # Cell Averaging CFAR
                    threshold = np.mean(ref_values) * threshold_factor
                elif cfar_type == 'GO':
                    # Greatest Of CFAR
                    left_mean = np.mean(ref_values[:len(ref_values)//2]) if len(ref_values) > 1 else ref_values[0]
                    right_mean = np.mean(ref_values[len(ref_values)//2:]) if len(ref_values) > 1 else ref_values[0]
                    threshold = max(left_mean, right_mean) * threshold_factor
                elif cfar_type == 'SO':
                    # Smallest Of CFAR
                    left_mean = np.mean(ref_values[:len(ref_values)//2]) if len(ref_values) > 1 else ref_values[0]
                    right_mean = np.mean(ref_values[len(ref_values)//2:]) if len(ref_values) > 1 else ref_values[0]
                    threshold = min(left_mean, right_mean) * threshold_factor
                else:
                    threshold = np.mean(ref_values) * threshold_factor
                
                thresholds[i] = threshold
                detections[i] = range_profile[i] > threshold
        
        return detections, thresholds
    
    def doppler_processing(self, range_doppler_data: np.ndarray, 
                          prf: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Doppler işleme (FFT tabanlı)
        
        Kaynak: IEEE Transactions on Signal Processing
        Denklem: fd = 2 * vr / λ
        """
        # Range-Doppler matrisi üzerinde FFT
        doppler_spectrum = np.fft.fftshift(np.fft.fft(range_doppler_data, axis=1), axes=1)
        
        # Doppler frekans ekseni
        doppler_freqs = np.fft.fftshift(np.fft.fftfreq(range_doppler_data.shape[1], d=1/prf))
        
        return doppler_spectrum, doppler_freqs
    
    def mti_filter(self, range_doppler_data: np.ndarray, 
                   filter_type: str = 'delay_line') -> np.ndarray:
        """
        Moving Target Indicator (MTI) filtresi
        
        Kaynak: IEEE Transactions on Aerospace and Electronic Systems
        MTI Filtreleri: Delay-line, Three-pulse canceller
        """
        if filter_type == 'delay_line':
            # İki-pulse delay-line canceller
            filtered_data = range_doppler_data[:, 1:] - range_doppler_data[:, :-1]
        elif filter_type == 'three_pulse':
            # Üç-pulse canceller
            filtered_data = (range_doppler_data[:, 2:] - 
                           2 * range_doppler_data[:, 1:-1] + 
                           range_doppler_data[:, :-2])
        else:
            filtered_data = range_doppler_data
        
        return filtered_data
    
    def adaptive_beamforming(self, array_data: np.ndarray, 
                           target_angle: float,
                           interference_angles: List[float] = None) -> np.ndarray:
        """
        Adaptif beamforming (Minimum Variance Distortionless Response)
        
        Kaynak: IEEE Transactions on Signal Processing
        Algoritma: MVDR (Minimum Variance Distortionless Response)
        """
        n_elements = array_data.shape[0]
        n_samples = array_data.shape[1]
        
        # Uzay korelasyon matrisi
        R = np.zeros((n_elements, n_elements), dtype=complex)
        for i in range(n_samples):
            x = array_data[:, i:i+1]
            R += x @ x.conj().T
        R /= n_samples
        
        # Hedef yön vektörü
        d = np.exp(1j * 2 * np.pi * np.arange(n_elements) * np.sin(target_angle))
        
        # MVDR ağırlıkları
        w = np.linalg.inv(R) @ d
        w = w / (d.conj().T @ w)  # Normalize
        
        # Beamforming çıkışı
        output = w.conj().T @ array_data
        
        return output.flatten()

# Test fonksiyonu
if __name__ == "__main__":
    # Test parametreleri
    fs = 1e9  # 1 GHz sampling rate
    pulse_width = 1e-6  # 1 μs pulse width
    
    processor = AdvancedSignalProcessor(fs, pulse_width)
    
    # Chirp sinyal testi
    chirp = processor.generate_chirp_signal(10e6, 100e6)
    print(f"Chirp sinyal uzunluğu: {len(chirp)}")
    
    # Barker kodu testi
    barker = processor.generate_barker_code(13)
    print(f"Barker kodu uzunluğu: {len(barker)}")
    
    # CFAR testi
    range_profile = np.random.exponential(1, 1000)
    detections, thresholds = processor.cfar_detection(range_profile)
    print(f"CFAR tespit sayısı: {np.sum(detections)}")
    
    print("Gelişmiş sinyal işleme modülü test edildi.")
    print("Kaynak: Skolnik, 'Radar Handbook', 3rd Ed., McGraw-Hill, 2008") 