#!/usr/bin/env python3
"""
Gelişmiş Radar Modülleri - Tek Tek Test
Her modülü ayrı ayrı test eder
"""

import numpy as np
import time
from datetime import datetime

def test_signal_processing():
    """Sinyal işleme modülünü test eder"""
    print("🔍 Sinyal İşleme Modülü Test Ediliyor...")
    try:
        from radar_advanced.signal_processing import AdvancedSignalProcessor
        
        processor = AdvancedSignalProcessor()
        
        # Chirp sinyal testi
        chirp = processor.generate_chirp_signal(10e6, 100e6)
        print(f"   - Chirp sinyal: {len(chirp)} örnek")
        
        # Barker kodu testi
        barker = processor.generate_barker_code(13)
        print(f"   - Barker kodu: {len(barker)} örnek")
        
        # CFAR tespit testi
        range_profile = np.random.exponential(1, 1000)
        detections, thresholds = processor.cfar_detection(range_profile)
        print(f"   - CFAR tespit: {np.sum(detections)} hedef")
        
        # Doppler işleme testi
        doppler_data = np.random.randn(100, 64) + 1j * np.random.randn(100, 64)
        doppler_spectrum, doppler_freqs = processor.doppler_processing(doppler_data, 1000)
        print(f"   - Doppler spektrum: {doppler_spectrum.shape}")
        
        print("✅ Sinyal İşleme Modülü BAŞARILI")
        return True
        
    except Exception as e:
        print(f"❌ Sinyal İşleme Modülü BAŞARISIZ: {e}")
        return False

def test_lpi_radar():
    """LPI radar modülünü test eder"""
    print("🔍 LPI Radar Modülü Test Ediliyor...")
    try:
        from radar_advanced.lpi_advanced import AdvancedLPIRadar
        
        lpi = AdvancedLPIRadar()
        
        # FHSS testi
        fhss = lpi.frequency_hopping_pattern(64, 10e6)
        print(f"   - FHSS sinyal: {len(fhss)} örnek")
        
        # Costas array testi
        costas = lpi.costas_array_generator(7)
        print(f"   - Costas array: {costas}")
        
        # Polyphase kod testi
        polyphase = lpi.polyphase_code_generator(16, 4)
        print(f"   - Polyphase kod: {len(polyphase)} örnek")
        
        # LPI tespit olasılığı testi
        prob, range_km = lpi.lpi_detection_probability()
        print(f"   - LPI tespit olasılığı: {prob:.4f}")
        print(f"   - Tespit menzili: {range_km:.1f} km")
        
        print("✅ LPI Radar Modülü BAŞARILI")
        return True
        
    except Exception as e:
        print(f"❌ LPI Radar Modülü BAŞARISIZ: {e}")
        return False

def test_sar_isar():
    """SAR/ISAR modülünü test eder"""
    print("🔍 SAR/ISAR Modülü Test Ediliyor...")
    try:
        from radar_advanced.sar_isar_advanced import AdvancedSARISAR
        
        sar = AdvancedSARISAR()
        
        # Test hedefleri
        target_pos = np.array([[100, 200, 50], [-50, 150, 30]])
        target_rcs = np.array([1.0, 0.5])
        
        # SAR ham veri üretimi
        raw_data = sar.generate_sar_raw_data(target_pos, target_rcs)
        print(f"   - SAR ham veri: {raw_data.shape}")
        
        # RDA algoritması
        rda_image = sar.range_doppler_algorithm(raw_data)
        print(f"   - RDA görüntü: {rda_image.shape}")
        
        # Backprojection algoritması
        target_area = (-200, 200, -200, 200)
        bpa_image = sar.backprojection_algorithm(raw_data, target_area)
        print(f"   - BPA görüntü: {bpa_image.shape}")
        
        # Görüntü kalitesi
        quality = sar.calculate_image_quality(rda_image)
        print(f"   - SNR: {quality['snr']:.1f} dB")
        print(f"   - Kontrast: {quality['contrast']:.2f}")
        
        print("✅ SAR/ISAR Modülü BAŞARILI")
        return True
        
    except Exception as e:
        print(f"❌ SAR/ISAR Modülü BAŞARISIZ: {e}")
        return False

def test_sensor_fusion():
    """Sensör füzyonu modülünü test eder"""
    print("🔍 Sensör Füzyonu Modülü Test Ediliyor...")
    try:
        from radar_advanced.sensor_fusion_advanced import AdvancedSensorFusion, SensorMeasurement
        
        fusion = AdvancedSensorFusion()
        
        # Test ölçümleri
        measurements = [
            SensorMeasurement(
                sensor_id="radar_1",
                timestamp=0.0,
                position=np.array([100, 200, 50]),
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
        print(f"   - Kalman fusion: {kalman_result['fusion_method']}")
        
        # Particle filter fusion testi
        particle_result = fusion.particle_filter_fusion(measurements)
        print(f"   - Particle filter: {particle_result['fusion_method']}")
        
        # Dempster-Shafer fusion testi
        ds_result = fusion.dempster_shafer_fusion(measurements)
        print(f"   - Dempster-Shafer: {ds_result['fusion_method']}")
        
        # Adaptive fusion testi
        adaptive_result = fusion.adaptive_fusion(measurements)
        print(f"   - Adaptive fusion: {adaptive_result['fusion_method']}")
        
        print("✅ Sensör Füzyonu Modülü BAŞARILI")
        return True
        
    except Exception as e:
        print(f"❌ Sensör Füzyonu Modülü BAŞARISIZ: {e}")
        return False

def test_3d_visualization():
    """3D görselleştirme modülünü test eder"""
    print("🔍 3D Görselleştirme Modülü Test Ediliyor...")
    try:
        from radar_advanced.webgl_renderer import Advanced3DRenderer
        
        renderer = Advanced3DRenderer()
        
        # 3D sahne oluşturma
        renderer.create_3d_scene()
        print(f"   - 3D sahne: {renderer.scene_size}x{renderer.scene_size}")
        
        # Test verileri
        radar_pos = np.array([0, 0, 0])
        targets = [
            {'position': np.array([100, 200, 50]), 'velocity': np.array([10, 20, 0]), 'type': 'aircraft'},
            {'position': np.array([-50, 150, 30]), 'velocity': np.array([-5, 15, 0]), 'type': 'missile'}
        ]
        missiles = [
            {'position': np.array([0, 0, 10]), 'velocity': np.array([0, 100, 0])}
        ]
        
        # Radar sistemi çizimi
        renderer.plot_radar_system(radar_pos)
        print("   - Radar sistemi çizildi")
        
        # Hedefler çizimi
        renderer.plot_targets(targets)
        print(f"   - {len(targets)} hedef çizildi")
        
        # Füzeler çizimi
        renderer.plot_missiles(missiles)
        print(f"   - {len(missiles)} füze çizildi")
        
        # Radar beam çizimi
        beam_direction = np.array([0, 1, 0])
        renderer.plot_radar_beam(radar_pos, beam_direction)
        print("   - Radar beam çizildi")
        
        print("✅ 3D Görselleştirme Modülü BAŞARILI")
        return True
        
    except Exception as e:
        print(f"❌ 3D Görselleştirme Modülü BAŞARISIZ: {e}")
        return False

def test_simulation_engine():
    """Ana simülasyon motorunu test eder"""
    print("🔍 Ana Simülasyon Motoru Test Ediliyor...")
    try:
        from radar_advanced.simulation_engine import AdvancedRadarSimulationEngine, SimulationConfig
        
        # Konfigürasyon
        config = SimulationConfig(
            radar_frequency=10e9,
            radar_power=1000,
            lpi_enabled=True,
            sar_enabled=True,
            fusion_enabled=True,
            visualization_enabled=False,
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
        max_duration = 3.0  # 3 saniye
        
        while time.time() - start_time < max_duration:
            engine.update_simulation()
            time.sleep(0.1)
        
        # Performans kontrolü
        print(f"   - Simülasyon süresi: {engine.state.timestamp:.1f}s")
        print(f"   - Hedef sayısı: {len(engine.state.targets)}")
        print(f"   - Füze sayısı: {len(engine.state.missiles)}")
        print(f"   - Tespit sayısı: {len(engine.state.detections)}")
        
        # Performans raporu
        report = engine.generate_performance_report()
        print(f"   - Ortalama FPS: {report.get('average_fps', 0):.1f}")
        
        print("✅ Ana Simülasyon Motoru BAŞARILI")
        return True
        
    except Exception as e:
        print(f"❌ Ana Simülasyon Motoru BAŞARISIZ: {e}")
        return False

def main():
    """Ana test fonksiyonu"""
    print("🚀 Gelişmiş Radar Modülleri - Tek Tek Test")
    print("=" * 50)
    print(f"Test Başlangıç Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    test_results = {}
    
    # Her modülü test et
    test_results['signal_processing'] = test_signal_processing()
    print()
    
    test_results['lpi_radar'] = test_lpi_radar()
    print()
    
    test_results['sar_isar'] = test_sar_isar()
    print()
    
    test_results['sensor_fusion'] = test_sensor_fusion()
    print()
    
    test_results['3d_visualization'] = test_3d_visualization()
    print()
    
    test_results['simulation_engine'] = test_simulation_engine()
    print()
    
    # Sonuçları özetle
    print("=" * 50)
    print("📊 TEST ÖZETİ")
    print("=" * 50)
    
    total_tests = len(test_results)
    successful_tests = sum(test_results.values())
    success_rate = (successful_tests / total_tests) * 100
    
    print(f"Toplam Test: {total_tests}")
    print(f"Başarılı: {successful_tests}")
    print(f"Başarısız: {total_tests - successful_tests}")
    print(f"Başarı Oranı: {success_rate:.1f}%")
    print()
    
    print("Detaylı Sonuçlar:")
    for module, result in test_results.items():
        status = "✅ BAŞARILI" if result else "❌ BAŞARISIZ"
        print(f"  {module.replace('_', ' ').title()}: {status}")
    
    print()
    print(f"Test Bitiş Zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_rate == 100:
        print("🎉 TÜM MODÜLLER BAŞARILI!")
    else:
        print("⚠️  Bazı modüller başarısız. Lütfen hataları kontrol edin.")

if __name__ == "__main__":
    main() 