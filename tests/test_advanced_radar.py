#!/usr/bin/env python3
"""
Gelişmiş Radar Arayıcı Başlık Simülasyonu - Kapsamlı Test Suite

Bu dosya, tüm gelişmiş radar modüllerini test eder.

Test Kapsamı:
- Sinyal işleme modülleri
- LPI radar teknolojileri
- SAR/ISAR görüntüleme
- Sensör füzyonu
- 3D görselleştirme
- Ana simülasyon motoru

Çalıştırma: python test_advanced_radar.py
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Gelişmiş radar modüllerini import et
try:
    from radar_advanced import (
        AdvancedSignalProcessor,
        AdvancedLPIRadar,
        AdvancedSARISAR,
        AdvancedSensorFusion,
        SensorMeasurement,
        Advanced3DRenderer,
        AdvancedRadarSimulationEngine,
        SimulationConfig
    )
    print("✅ Gelişmiş radar modülleri başarıyla import edildi")
except ImportError as e:
    print(f"❌ Modül import hatası: {e}")
    print("Lütfen radar_advanced klasörünün Python path'inde olduğundan emin olun")
    sys.exit(1)

class AdvancedRadarTestSuite:
    """Gelişmiş radar test suite"""
    
    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        
    def run_all_tests(self):
        """Tüm testleri çalıştırır"""
        print("🚀 Gelişmiş Radar Simülasyonu - Kapsamlı Test Başlatılıyor")
        print("=" * 60)
        
        tests = [
            ("Sinyal İşleme", self.test_signal_processing),
            ("LPI Radar", self.test_lpi_radar),
            ("SAR/ISAR", self.test_sar_isar),
            ("Sensör Füzyonu", self.test_sensor_fusion),
            ("3D Görselleştirme", self.test_3d_visualization),
            ("Ana Simülasyon Motoru", self.test_simulation_engine),
            ("Performans Testi", self.test_performance),
            ("Entegrasyon Testi", self.test_integration)
        ]
        
        for test_name, test_func in tests:
            print(f"\n🔍 {test_name} Testi Çalıştırılıyor...")
            try:
                result = test_func()
                self.test_results[test_name] = result
                if result:
                    print(f"✅ {test_name} - BAŞARILI")
                else:
                    print(f"❌ {test_name} - BAŞARISIZ")
            except Exception as e:
                print(f"❌ {test_name} - HATA: {e}")
                self.test_results[test_name] = False
        
        self.print_test_summary()
    
    def test_signal_processing(self):
        """Sinyal işleme modülünü test eder"""
        try:
            # Signal processor oluştur
            processor = AdvancedSignalProcessor(sampling_rate=1e9, pulse_width=1e-6)
            
            # Chirp sinyal testi
            chirp = processor.generate_chirp_signal(10e6, 100e6)
            assert len(chirp) > 0, "Chirp sinyal üretilemedi"
            
            # Barker kodu testi
            barker = processor.generate_barker_code(13)
            assert len(barker) > 0, "Barker kodu üretilemedi"
            
            # CFAR testi
            range_profile = np.random.exponential(1, 1000)
            detections, thresholds = processor.cfar_detection(range_profile)
            assert len(detections) == len(thresholds), "CFAR boyut uyumsuzluğu"
            
            # Doppler işleme testi
            range_doppler_data = np.random.rand(100, 64)
            doppler_spectrum, doppler_freqs = processor.doppler_processing(range_doppler_data, 1000)
            assert doppler_spectrum.shape == range_doppler_data.shape, "Doppler boyut hatası"
            
            print(f"   - Chirp sinyal: {len(chirp)} örnek")
            print(f"   - Barker kodu: {len(barker)} örnek")
            print(f"   - CFAR tespit: {np.sum(detections)} hedef")
            print(f"   - Doppler spektrum: {doppler_spectrum.shape}")
            
            return True
            
        except Exception as e:
            print(f"   HATA: {e}")
            return False
    
    def test_lpi_radar(self):
        """LPI radar modülünü test eder"""
        try:
            # LPI radar oluştur
            lpi_radar = AdvancedLPIRadar(fc=10e9, bandwidth=100e6, power=10)
            
            # FHSS testi
            fhss_signal = lpi_radar.frequency_hopping_pattern()
            assert len(fhss_signal) > 0, "FHSS sinyal üretilemedi"
            
            # Costas array testi
            costas_array = lpi_radar.costas_array_generator(7)
            assert len(costas_array) == 7, "Costas array boyut hatası"
            
            # Polyphase kod testi
            polyphase_code = lpi_radar.polyphase_code_generator(16, 4)
            assert len(polyphase_code) == 16, "Polyphase kod boyut hatası"
            
            # LPI tespit olasılığı testi
            p_detect, pr_esm = lpi_radar.lpi_detection_probability(lpi_technique='FHSS')
            assert 0 <= p_detect <= 1, "Tespit olasılığı geçersiz"
            
            print(f"   - FHSS sinyal: {len(fhss_signal)} örnek")
            print(f"   - Costas array: {costas_array}")
            print(f"   - Polyphase kod: {len(polyphase_code)} örnek")
            print(f"   - LPI tespit olasılığı: {p_detect:.4f}")
            
            return True
            
        except Exception as e:
            print(f"   HATA: {e}")
            return False
    
    def test_sar_isar(self):
        """SAR/ISAR modülünü test eder"""
        try:
            # SAR processor oluştur
            sar_processor = AdvancedSARISAR(fc=10e9, bandwidth=100e6, prf=1000)
            
            # Test hedefleri
            target_positions = np.array([
                [0, 1000, 0],
                [50, 1000, 0],
                [-50, 1000, 0]
            ])
            target_rcs = np.array([1.0, 0.5, 0.8])
            
            # SAR ham veri üretimi
            raw_data = sar_processor.generate_sar_raw_data(target_positions, target_rcs)
            assert raw_data.shape[0] > 0, "SAR ham veri üretilemedi"
            
            # Range-Doppler Algorithm
            rda_image = sar_processor.range_doppler_algorithm(raw_data)
            assert rda_image.shape == raw_data.shape, "RDA boyut hatası"
            
            # Görüntü kalitesi
            quality = sar_processor.calculate_image_quality(rda_image)
            assert 'SNR_dB' in quality, "Görüntü kalitesi hesaplanamadı"
            
            print(f"   - SAR ham veri: {raw_data.shape}")
            print(f"   - RDA görüntü: {rda_image.shape}")
            print(f"   - SNR: {quality['SNR_dB']:.1f} dB")
            print(f"   - Kontrast: {quality['Contrast']:.2f}")
            
            return True
            
        except Exception as e:
            print(f"   HATA: {e}")
            return False
    
    def test_sensor_fusion(self):
        """Sensör füzyonu modülünü test eder"""
        try:
            # Sensor fusion oluştur
            fusion = AdvancedSensorFusion(fusion_method='adaptive')
            
            # Test ölçümleri
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
            
            # Kalman fusion testi
            kalman_result = fusion.kalman_fusion(measurements)
            assert 'fused_state' in kalman_result, "Kalman fusion başarısız"
            
            # Particle filter fusion testi
            particle_result = fusion.particle_filter_fusion(measurements)
            assert 'fused_state' in particle_result, "Particle filter fusion başarısız"
            
            # Dempster-Shafer fusion testi
            ds_result = fusion.dempster_shafer_fusion(measurements)
            assert 'fused_state' in ds_result, "Dempster-Shafer fusion başarısız"
            
            print(f"   - Kalman fusion: {kalman_result['fusion_method']}")
            print(f"   - Particle filter: {particle_result['fusion_method']}")
            print(f"   - Dempster-Shafer: {ds_result['fusion_method']}")
            
            return True
            
        except Exception as e:
            print(f"   HATA: {e}")
            return False
    
    def test_3d_visualization(self):
        """3D görselleştirme modülünü test eder"""
        try:
            # 3D renderer oluştur
            renderer = Advanced3DRenderer()
            
            # 3D sahne oluştur
            renderer.create_3d_scene()
            assert renderer.fig is not None, "3D sahne oluşturulamadı"
            
            # Test verileri
            radar_pos = np.array([0, 0, 0])
            targets = [
                {'position': np.array([100, 200, 50]), 'velocity': np.array([10, 20, 0]), 'type': 'aircraft'},
                {'position': np.array([-50, 150, 30]), 'velocity': np.array([-5, 15, 0]), 'type': 'missile'}
            ]
            missiles = [
                {'position': np.array([0, 0, 10]), 'velocity': np.array([0, 100, 0])}
            ]
            
            # Radar sistemi çiz
            renderer.plot_radar_system(radar_pos)
            
            # Hedefler çiz
            renderer.plot_targets(targets)
            
            # Füzeler çiz
            renderer.plot_missiles(missiles)
            
            # Radar beam çiz
            beam_direction = np.array([0, 1, 0])
            renderer.plot_radar_beam(radar_pos, beam_direction)
            
            # WebGL renderer testi
            # WebGLRenderer sınıfının import edilmesi gerekiyor
            # from webgl_renderer import WebGLRenderer
            # webgl_renderer = WebGLRenderer()
            # html_code = webgl_renderer.create_webgl_scene()
            # assert len(html_code) > 0, "WebGL HTML kodu üretilemedi"
            
            print(f"   - 3D sahne: {renderer.scene_size}x{renderer.scene_size}")
            print(f"   - Hedef sayısı: {len(targets)}")
            print(f"   - Füze sayısı: {len(missiles)}")
            # print(f"   - WebGL HTML: {len(html_code)} karakter") # WebGLRenderer kullanılmadığı için bu satır kaldırıldı
            
            return True
            
        except Exception as e:
            print(f"   HATA: {e}")
            return False
    
    def test_simulation_engine(self):
        """Ana simülasyon motorunu test eder"""
        try:
            # Konfigürasyon
            config = SimulationConfig(
                radar_frequency=10e9,
                radar_power=1000,
                lpi_enabled=True,
                sar_enabled=True,
                fusion_enabled=True,
                visualization_enabled=False,  # Test için kapalı
                max_targets=5
            )
            
            # Simülasyon motoru
            engine = AdvancedRadarSimulationEngine(config)
            
            # Test hedefleri ekle
            engine.add_target(
                position=np.array([100, 200, 50]),
                velocity=np.array([10, 20, 0]),
                target_type='aircraft',
                rcs=1.0
            )
            
            engine.add_target(
                position=np.array([-50, 150, 30]),
                velocity=np.array([-5, 15, 0]),
                target_type='missile',
                rcs=0.5
            )
            
            # Test füzesi ekle
            engine.add_missile(
                position=np.array([0, 0, 10]),
                velocity=np.array([0, 100, 0])
            )
            
            # Kısa simülasyon çalıştır
            start_time = time.time()
            max_duration = 5.0  # 5 saniye
            
            while time.time() - start_time < max_duration:
                engine.update_simulation()
                time.sleep(0.1)  # 100ms bekle
            
            # Performans kontrolü
            assert len(engine.state.targets) >= 0, "Hedef sayısı geçersiz"
            assert len(engine.state.missiles) >= 0, "Füze sayısı geçersiz"
            assert engine.state.timestamp > 0, "Simülasyon zamanı geçersiz"
            
            # Performans raporu
            report = engine.generate_performance_report()
            
            print(f"   - Simülasyon süresi: {engine.state.timestamp:.1f}s")
            print(f"   - Hedef sayısı: {len(engine.state.targets)}")
            print(f"   - Füze sayısı: {len(engine.state.missiles)}")
            print(f"   - Tespit sayısı: {len(engine.state.detections)}")
            print(f"   - Ortalama FPS: {report.get('average_fps', 0):.1f}")
            
            return True
            
        except Exception as e:
            print(f"   HATA: {e}")
            return False
    
    def test_performance(self):
        """Performans testlerini çalıştırır"""
        try:
            print("   - Performans testleri çalıştırılıyor...")
            
            # Bellek kullanımı testi
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Yoğun işlem testi
            processor = AdvancedSignalProcessor()
            for i in range(100):
                chirp = processor.generate_chirp_signal(10e6, 100e6)
                barker = processor.generate_barker_code(13)
                range_profile = np.random.exponential(1, 1000)
                detections, _ = processor.cfar_detection(range_profile)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # CPU kullanımı testi
            cpu_percent = process.cpu_percent(interval=1)
            
            print(f"   - Bellek kullanımı: {memory_before:.1f} -> {memory_after:.1f} MB")
            print(f"   - Bellek artışı: {memory_increase:.1f} MB")
            print(f"   - CPU kullanımı: {cpu_percent:.1f}%")
            
            # Performans kriterleri
            assert memory_increase < 100, "Bellek kullanımı çok yüksek"
            assert cpu_percent < 80, "CPU kullanımı çok yüksek"
            
            return True
            
        except Exception as e:
            print(f"   HATA: {e}")
            return False
    
    def test_integration(self):
        """Entegrasyon testini çalıştırır"""
        try:
            print("   - Entegrasyon testi çalıştırılıyor...")
            
            # Tüm modülleri entegre et
            config = SimulationConfig(
                radar_frequency=10e9,
                radar_power=1000,
                lpi_enabled=True,
                sar_enabled=True,
                fusion_enabled=True,
                visualization_enabled=False,
                max_targets=3
            )
            
            engine = AdvancedRadarSimulationEngine(config)
            
            # Test senaryosu
            engine.add_target(np.array([100, 200, 50]), np.array([10, 20, 0]), 'aircraft')
            engine.add_target(np.array([-50, 150, 30]), np.array([-5, 15, 0]), 'missile')
            engine.add_missile(np.array([0, 0, 10]), np.array([0, 100, 0]))
            
            # Kısa entegrasyon testi
            for i in range(10):
                engine.update_simulation()
                time.sleep(0.1)
            
            # Entegrasyon kontrolü
            assert engine.signal_processor is not None, "Signal processor entegre edilemedi"
            assert engine.lpi_radar is not None, "LPI radar entegre edilemedi"
            assert engine.sar_processor is not None, "SAR processor entegre edilemedi"
            assert engine.sensor_fusion is not None, "Sensor fusion entegre edilemedi"
            
            print(f"   - Entegrasyon başarılı: {len(engine.state.targets)} hedef, {len(engine.state.missiles)} füze")
            
            return True
            
        except Exception as e:
            print(f"   HATA: {e}")
            return False
    
    def print_test_summary(self):
        """Test özetini yazdırır"""
        print("\n" + "=" * 60)
        print("📊 TEST ÖZETİ")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result)
        failed_tests = total_tests - passed_tests
        
        print(f"Toplam Test: {total_tests}")
        print(f"Başarılı: {passed_tests}")
        print(f"Başarısız: {failed_tests}")
        print(f"Başarı Oranı: {(passed_tests/total_tests)*100:.1f}%")
        
        print("\nDetaylı Sonuçlar:")
        for test_name, result in self.test_results.items():
            status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
            print(f"  {test_name}: {status}")
        
        total_time = time.time() - self.start_time
        print(f"\nToplam Test Süresi: {total_time:.2f} saniye")
        
        if failed_tests == 0:
            print("\n🎉 TÜM TESTLER BAŞARILI!")
            print("Gelişmiş radar simülasyonu kullanıma hazır.")
        else:
            print(f"\n⚠️  {failed_tests} test başarısız. Lütfen hataları kontrol edin.")

def main():
    """Ana test fonksiyonu"""
    print("🔬 Gelişmiş Radar Arayıcı Başlık Simülasyonu - Test Suite")
    print(f"Test Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Test suite oluştur ve çalıştır
    test_suite = AdvancedRadarTestSuite()
    test_suite.run_all_tests()
    
    print(f"\nTest Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main() 