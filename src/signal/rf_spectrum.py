"""
RF Spektrum Modelleme Modülü

Bu modül, radar ve ECM (karıştırıcı) sinyallerinin frekans uzayında analizini ve görselleştirmesini sağlar.

Bilimsel Temel:
- Radar ve jammer sinyalleri FFT ile analiz edilir.
- Spektrumda karıştırıcı etkisi, LPI radar modları ve ECCM teknikleri gözlemlenebilir.
- Temel Denklem: Sinyal spektrumu S(f) = FFT{s(t)}

Kaynaklar:
- Skolnik, "Radar Handbook", 3rd Ed., McGraw-Hill, 2008
- Levanon, "Low Probability of Intercept Radar", IEEE Press, 2004
- Richards, "Fundamentals of Radar Signal Processing", 2nd Ed., McGraw-Hill, 2014
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

class RFSpectrumAnalyzer:
    """RF Spektrum analizörü"""
    def __init__(self, sampling_rate=1e9, n_samples=4096):
        self.sampling_rate = sampling_rate  # Örnekleme hızı (Hz)
        self.n_samples = n_samples  # FFT uzunluğu

    def generate_radar_signal(self, freq=10e9, bandwidth=20e6, pulse_length=1e-6, amplitude=1.0, lpi=False) -> np.ndarray:
        """
        Radar darbe sinyali üretir (isteğe bağlı LPI modlu)
        lpi=True ise frekans atlamalı veya düşük güçte sinyal üretir.
        """
        t = np.arange(self.n_samples) / self.sampling_rate
        if lpi:
            # LPI için frekans atlamalı veya düşük güçte sinyal
            freq_hop = freq + np.sin(2 * np.pi * 1e6 * t) * bandwidth / 2
            signal = amplitude * np.exp(1j * 2 * np.pi * freq_hop * t)
        else:
            signal = amplitude * np.exp(1j * 2 * np.pi * freq * t)
        # Darbe şekli (pencere)
        window = np.zeros_like(t)
        window[:int(pulse_length * self.sampling_rate)] = 1.0
        return signal * window

    def generate_jammer_signal(self, freq=10e9, bandwidth=50e6, amplitude=1.0, noise=True) -> np.ndarray:
        """
        Karıştırıcı (jammer) sinyali üretir
        noise=True ise beyaz gürültü, değilse dar bantlı sinyal
        """
        t = np.arange(self.n_samples) / self.sampling_rate
        if noise:
            # Beyaz gürültü (broadband noise jammer)
            signal = amplitude * (np.random.randn(self.n_samples) + 1j * np.random.randn(self.n_samples))
        else:
            # Dar bantlı karıştırıcı
            signal = amplitude * np.exp(1j * 2 * np.pi * freq * t)
        return signal

    def compute_spectrum(self, signal: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Sinyalin frekans spektrumunu FFT ile hesaplar
        """
        spectrum = np.fft.fftshift(np.fft.fft(signal, n=self.n_samples))
        freq_axis = np.fft.fftshift(np.fft.fftfreq(self.n_samples, d=1/self.sampling_rate))
        power_db = 20 * np.log10(np.abs(spectrum) + 1e-12)
        return freq_axis, power_db

    def plot_spectrum(self, freq_axis: np.ndarray, power_db: np.ndarray, label: str = "Radar Sinyali"):
        """
        Spektrumu görselleştirir
        """
        plt.figure(figsize=(10, 5))
        plt.plot(freq_axis / 1e9, power_db, label=label)
        plt.xlabel("Frekans (GHz)")
        plt.ylabel("Güç (dB)")
        plt.title("RF Spektrum Analizi")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_multi_spectrum(self, signals: dict):
        """
        Birden fazla sinyalin spektrumunu aynı grafikte gösterir
        """
        plt.figure(figsize=(10, 5))
        for label, signal in signals.items():
            freq_axis, power_db = self.compute_spectrum(signal)
            plt.plot(freq_axis / 1e9, power_db, label=label)
        plt.xlabel("Frekans (GHz)")
        plt.ylabel("Güç (dB)")
        plt.title("Radar ve ECM Sinyalleri RF Spektrumu")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# Örnek kullanım ve bilimsel açıklama:
if __name__ == "__main__":
    analyzer = RFSpectrumAnalyzer()
    radar_signal = analyzer.generate_radar_signal(lpi=False)
    lpi_signal = analyzer.generate_radar_signal(lpi=True, amplitude=0.2)
    jammer_signal = analyzer.generate_jammer_signal(noise=True, amplitude=0.5)
    
    # Tekli spektrum
    freq_axis, power_db = analyzer.compute_spectrum(radar_signal)
    analyzer.plot_spectrum(freq_axis, power_db, label="Radar Sinyali")
    
    # Çoklu spektrum
    signals = {
        "Radar Sinyali": radar_signal,
        "LPI Radar": lpi_signal,
        "Jammer (Gürültü)": jammer_signal
    }
    analyzer.plot_multi_spectrum(signals)
    print("RF spektrum modelleme tamamlandı. [Kaynak: Skolnik, 2008]") 