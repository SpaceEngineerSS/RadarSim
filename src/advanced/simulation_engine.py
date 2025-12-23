"""
Gelişmiş Radar Simülasyon Motoru

Bu modül, tüm radar simülasyon bileşenlerini koordine eden ana motoru içerir.

Bilimsel Temeller:
- IEEE Transactions on Aerospace and Electronic Systems
- NATO STANAG 4609, "Digital Terrain Elevation Data"
- Skolnik, "Radar Handbook", 3rd Ed., McGraw-Hill, 2008

Özellikler:
- Gerçek zamanlı simülasyon motoru
- Modüler bileşen entegrasyonu
- Performans optimizasyonu
- Veri kaydı ve analiz
- Çoklu senaryo desteği
"""

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from .lpi_radar import AdvancedLPIRadar
from .sar_isar import AdvancedSARISAR
from .sensor_fusion import AdvancedSensorFusion, SensorMeasurement

# Gelişmiş modüller
from .signal_processing import AdvancedSignalProcessor
from .webgl_renderer import Advanced3DRenderer


@dataclass
class SimulationConfig:
    """Simülasyon konfigürasyonu"""

    # Radar parametreleri
    radar_frequency: float = 10e9  # Hz
    radar_power: float = 1000  # W
    radar_bandwidth: float = 100e6  # Hz
    pulse_width: float = 1e-6  # s
    prf: float = 1000  # Hz

    # LPI parametreleri
    lpi_enabled: bool = True
    lpi_technique: str = "FHSS"

    # SAR/ISAR parametreleri
    sar_enabled: bool = True
    platform_velocity: float = 100  # m/s
    synthetic_aperture: float = 100  # m

    # Sensör füzyonu
    fusion_enabled: bool = True
    fusion_method: str = "adaptive"

    # Görselleştirme
    visualization_enabled: bool = True
    real_time_plotting: bool = True

    # Performans
    max_targets: int = 100
    simulation_fps: int = 30
    update_interval: float = 1.0 / 30.0  # s


@dataclass
class SimulationState:
    """Simülasyon durumu"""

    timestamp: float = 0.0
    radar_position: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    targets: List[Dict[str, Any]] = field(default_factory=list)
    missiles: List[Dict[str, Any]] = field(default_factory=list)
    detections: List[Dict[str, Any]] = field(default_factory=list)
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    ecm_active: bool = False
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class AdvancedRadarSimulationEngine:
    """Gelişmiş radar simülasyon motoru"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.state = SimulationState()
        self.running = False
        self.paused = False

        # Gelişmiş modüller
        self.signal_processor = AdvancedSignalProcessor(
            sampling_rate=config.radar_bandwidth, pulse_width=config.pulse_width
        )

        self.lpi_radar = AdvancedLPIRadar(
            fc=config.radar_frequency, bandwidth=config.radar_bandwidth, power=config.radar_power
        )

        self.sar_processor = AdvancedSARISAR(
            fc=config.radar_frequency,
            bandwidth=config.radar_bandwidth,
            prf=config.prf,
            platform_velocity=config.platform_velocity,
            synthetic_aperture=config.synthetic_aperture,
        )

        self.sensor_fusion = AdvancedSensorFusion(fusion_method=config.fusion_method)

        self.renderer = Advanced3DRenderer() if config.visualization_enabled else None

        # Performans izleme
        self.performance_log = []
        self.start_time = time.time()  # Başlangıç zamanını hemen ayarla

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def start_simulation(self) -> None:
        """Simülasyonu başlatır"""
        self.running = True
        self.start_time = time.time()
        self.logger.info("Gelişmiş radar simülasyonu başlatıldı")

        # Ana simülasyon döngüsü
        if self.config.visualization_enabled:
            self.renderer.create_3d_scene()

        while self.running:
            if not self.paused:
                self.update_simulation()

            # Performans kontrolü
            self.update_performance_metrics()

            # Görselleştirme güncelleme
            if self.config.visualization_enabled and self.config.real_time_plotting:
                self.update_visualization()

            # FPS kontrolü
            time.sleep(self.config.update_interval)

    def update_simulation(self) -> None:
        """Simülasyon durumunu günceller"""
        # Zaman güncelleme
        if self.start_time is not None:
            self.state.timestamp = time.time() - self.start_time
        else:
            self.state.timestamp = 0.0

        # Hedef hareketi
        self.update_targets()

        # Radar işlemleri
        self.process_radar()

        # Sensör füzyonu
        if self.config.fusion_enabled:
            self.process_sensor_fusion()

        # SAR/ISAR işleme
        if self.config.sar_enabled:
            self.process_sar_isar()

        # ECM/ECCM
        self.process_ecm_eccm()

        # Füze güdümü
        self.update_missiles()

        # Çarpışma kontrolü
        self.check_intercepts()

    def update_targets(self) -> None:
        """Hedefleri günceller"""
        for target in self.state.targets:
            # Hedef hareketi (basit constant velocity model)
            dt = self.config.update_interval
            # Dtype casting hatasını önlemek için float64'e dönüştür
            position = np.array(target["position"], dtype=np.float64)
            velocity = np.array(target["velocity"], dtype=np.float64)
            target["position"] = position + velocity * dt

            # RCS değişkenliği (basitleştirilmiş)
            target["rcs"] *= 1 + 0.01 * np.random.normal(0, 1)
            target["rcs"] = max(0.1, target["rcs"])  # Minimum RCS

            # Hedef izi güncelleme
            if "trajectory" not in target:
                target["trajectory"] = []
            target["trajectory"].append(target["position"].copy())

            # İz uzunluğunu sınırla
            if len(target["trajectory"]) > 100:
                target["trajectory"] = target["trajectory"][-100:]

    def process_radar(self) -> None:
        """Radar işlemlerini gerçekleştirir"""
        # Radar beam yönü (basitleştirilmiş tarama)
        beam_angle = (self.state.timestamp * 30) % 360  # 30°/s tarama
        beam_direction = np.array(
            [np.cos(np.radians(beam_angle)), np.sin(np.radians(beam_angle)), 0]
        )

        # LPI dalga şekli
        if self.config.lpi_enabled:
            lpi_waveform = self.lpi_radar.generate_lpi_waveform(technique=self.config.lpi_technique)
        else:
            # Standart chirp sinyal
            lpi_waveform = self.signal_processor.generate_chirp_signal(
                start_freq=self.config.radar_frequency - self.config.radar_bandwidth / 2,
                end_freq=self.config.radar_frequency + self.config.radar_bandwidth / 2,
            )

        # Hedef tespiti
        detections = []
        for target in self.state.targets:
            # Menzil hesaplama
            range_to_target = np.linalg.norm(target["position"] - self.state.radar_position)

            # Radar denklemi
            received_power = self.calculate_received_power(
                target["rcs"], range_to_target, lpi_waveform
            )

            # Tespit eşiği
            if received_power > self.calculate_detection_threshold():
                detection = {
                    "target_id": target.get("id", "unknown"),
                    "position": target["position"].copy(),
                    "velocity": target["velocity"].copy(),
                    "range": range_to_target,
                    "received_power": received_power,
                    "timestamp": self.state.timestamp,
                }
                detections.append(detection)

        self.state.detections = detections

    def process_sensor_fusion(self) -> None:
        """Sensör füzyonu işlemlerini gerçekleştirir"""
        if not self.state.detections:
            return

        # Sensör ölçümlerini hazırla
        measurements = []
        for detection in self.state.detections:
            measurement = SensorMeasurement(
                sensor_id="radar_main",
                timestamp=detection["timestamp"],
                position=detection["position"],
                velocity=detection["velocity"],
                measurement_type="radar",
                uncertainty=np.eye(6) * 10,  # Basitleştirilmiş belirsizlik
                confidence=0.9,
            )
            measurements.append(measurement)

        # Sensör füzyonu uygula
        fusion_result = self.sensor_fusion.adaptive_fusion(measurements)

        # Füzyon sonuçlarını takip sistemine entegre et
        if fusion_result:
            self.update_tracking_system(fusion_result)

    def process_sar_isar(self) -> None:
        """SAR/ISAR işlemlerini gerçekleştirir"""
        if len(self.state.targets) == 0:
            return

        # Hedef pozisyonları ve RCS değerleri
        target_positions = np.array([t["position"] for t in self.state.targets])
        target_rcs = np.array([t["rcs"] for t in self.state.targets])

        # SAR ham veri üretimi
        raw_data = self.sar_processor.generate_sar_raw_data(target_positions, target_rcs)

        # SAR görüntü işleme
        sar_image = self.sar_processor.range_doppler_algorithm(raw_data)

        # Görüntü kalitesi analizi
        image_quality = self.sar_processor.calculate_image_quality(sar_image)

        # SAR sonuçlarını kaydet
        self.state.performance_metrics["sar_snr"] = image_quality["SNR_dB"]
        self.state.performance_metrics["sar_contrast"] = image_quality["Contrast"]

    def process_ecm_eccm(self) -> None:
        """ECM/ECCM işlemlerini gerçekleştirir"""
        # ECM aktivasyonu (rastgele)
        if np.random.random() < 0.1:  # %10 olasılık
            self.state.ecm_active = True
            self.logger.info("ECM aktivasyonu tespit edildi")

        # ECCM karşı önlemleri
        if self.state.ecm_active:
            # Frekans atlama
            new_frequency = self.config.radar_frequency + np.random.normal(0, 10e6)
            self.config.radar_frequency = new_frequency

            # LPI modu değiştirme
            if np.random.random() < 0.5:
                self.config.lpi_technique = np.random.choice(["FHSS", "DSSS", "Costas"])

    def update_missiles(self) -> None:
        """Füzeleri günceller"""
        for missile in self.state.missiles:
            # Füze hareketi
            dt = self.config.update_interval
            missile["position"] += missile["velocity"] * dt

            # Güdüm sistemi (basitleştirilmiş)
            if self.state.targets:
                # En yakın hedefi seç
                target = min(
                    self.state.targets,
                    key=lambda t: np.linalg.norm(t["position"] - missile["position"]),
                )

                # Proportional Navigation
                los_vector = target["position"] - missile["position"]
                los_vector = los_vector / np.linalg.norm(los_vector)

                # Güdüm komutu
                guidance_gain = 3.0
                missile_speed = np.linalg.norm(missile["velocity"])
                guidance_command = guidance_gain * missile_speed * los_vector

                # Hız güncelleme
                missile["velocity"] += guidance_command * dt

    def check_intercepts(self) -> None:
        """Çarpışma kontrolü yapar"""
        for missile in self.state.missiles:
            for target in self.state.targets:
                distance = np.linalg.norm(missile["position"] - target["position"])
                if distance < 10:  # 10m çarpışma mesafesi
                    self.logger.info(f"Çarpışma tespit edildi: {target.get('id', 'unknown')}")
                    # Hedef ve füze kaldır
                    if target in self.state.targets:
                        self.state.targets.remove(target)
                    if missile in self.state.missiles:
                        self.state.missiles.remove(missile)
                    break

    def update_tracking_system(self, fusion_result: Dict[str, Any]) -> None:
        """Takip sistemini günceller"""
        # Basitleştirilmiş takip sistemi
        fused_state = fusion_result.get("fused_state", None)
        if fused_state is not None:
            track = {
                "id": f"track_{len(self.state.tracks)}",
                "position": fused_state[:3],
                "velocity": fused_state[3:6],
                "timestamp": self.state.timestamp,
                "fusion_method": fusion_result.get("fusion_method", "unknown"),
            }
            self.state.tracks.append(track)

            # Eski takipleri temizle
            current_time = self.state.timestamp
            self.state.tracks = [
                t for t in self.state.tracks if current_time - t["timestamp"] < 10.0
            ]

    def update_visualization(self) -> None:
        """Görselleştirmeyi günceller"""
        if self.renderer is None:
            return

        # Sahneyi temizle
        self.renderer.ax.clear()

        # Radar sistemi
        self.renderer.plot_radar_system(self.state.radar_position)

        # Hedefler
        targets_data = []
        for target in self.state.targets:
            target_data = {
                "position": target["position"],
                "velocity": target["velocity"],
                "type": target.get("type", "unknown"),
            }
            targets_data.append(target_data)

        if targets_data:
            self.renderer.plot_targets(targets_data)

        # Füzeler
        missiles_data = []
        for missile in self.state.missiles:
            missile_data = {"position": missile["position"], "velocity": missile["velocity"]}
            missiles_data.append(missile_data)

        if missiles_data:
            self.renderer.plot_missiles(missiles_data)

        # Radar beam
        beam_angle = (self.state.timestamp * 30) % 360
        beam_direction = np.array(
            [np.cos(np.radians(beam_angle)), np.sin(np.radians(beam_angle)), 0]
        )
        self.renderer.plot_radar_beam(self.state.radar_position, beam_direction)

        # Sahne ayarları
        self.renderer.ax.set_xlabel("X (m)")
        self.renderer.ax.set_ylabel("Y (m)")
        self.renderer.ax.set_zlabel("Z (m)")
        self.renderer.ax.set_title(f"Radar Simulation - Time: {self.state.timestamp:.1f}s")

        plt.pause(0.001)

    def update_performance_metrics(self) -> None:
        """Performans metriklerini günceller"""
        current_time = time.time()

        # FPS hesaplama
        if len(self.performance_log) > 0:
            fps = 1.0 / (current_time - self.performance_log[-1]["timestamp"])
        else:
            fps = 0

        # Performans metrikleri
        metrics = {
            "timestamp": current_time,
            "fps": fps,
            "target_count": len(self.state.targets),
            "detection_count": len(self.state.detections),
            "track_count": len(self.state.tracks),
            "missile_count": len(self.state.missiles),
            "ecm_active": self.state.ecm_active,
            "memory_usage": self.get_memory_usage(),
        }

        self.performance_log.append(metrics)

        # Log boyutunu sınırla
        if len(self.performance_log) > 1000:
            self.performance_log = self.performance_log[-1000:]

    def calculate_received_power(
        self, rcs: float, range_distance: float, waveform: np.ndarray
    ) -> float:
        """Alınan güç hesaplama"""
        # Radar denklemi
        Pt = self.config.radar_power
        Gt = 30  # dB antenna gain
        Gr = 30  # dB receiver gain
        λ = 3e8 / self.config.radar_frequency
        R = range_distance

        # Temel radar denklemi
        Pr = (Pt * Gt * Gr * λ**2 * rcs) / ((4 * np.pi) ** 3 * R**4)

        # LPI etkisi
        if self.config.lpi_enabled:
            Pr *= 0.1  # LPI güç azaltması

        # Atmosferik zayıflama (basitleştirilmiş)
        atmospheric_loss = np.exp(-0.01 * R / 1000)  # dB/km
        Pr *= atmospheric_loss

        return Pr

    def calculate_detection_threshold(self) -> float:
        """Tespit eşiği hesaplama"""
        # Basitleştirilmiş tespit eşiği
        noise_power = 1e-12  # W
        snr_threshold = 10  # dB
        return noise_power * (10 ** (snr_threshold / 10))

    def get_memory_usage(self) -> float:
        """Bellek kullanımı (basitleştirilmiş)"""
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB

    def add_target(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        target_type: str = "aircraft",
        rcs: float = 1.0,
    ) -> None:
        """Hedef ekler"""
        # Dtype casting hatasını önlemek için float64'e dönüştür
        position_float = np.array(position, dtype=np.float64)
        velocity_float = np.array(velocity, dtype=np.float64)

        target = {
            "id": f"target_{len(self.state.targets)}",
            "position": position_float,
            "velocity": velocity_float,
            "type": target_type,
            "rcs": rcs,
            "trajectory": [],
        }
        self.state.targets.append(target)

    def add_missile(self, position: np.ndarray, velocity: np.ndarray) -> None:
        """Füze ekler"""
        # Dtype casting hatasını önlemek için float64'e dönüştür
        position_float = np.array(position, dtype=np.float64)
        velocity_float = np.array(velocity, dtype=np.float64)

        missile = {
            "id": f"missile_{len(self.state.missiles)}",
            "position": position_float,
            "velocity": velocity_float,
        }
        self.state.missiles.append(missile)

    def pause_simulation(self) -> None:
        """Simülasyonu duraklatır"""
        self.paused = True
        self.logger.info("Simülasyon duraklatıldı")

    def resume_simulation(self) -> None:
        """Simülasyonu devam ettirir"""
        self.paused = False
        self.logger.info("Simülasyon devam ettirildi")

    def stop_simulation(self) -> None:
        """Simülasyonu durdurur"""
        self.running = False
        self.logger.info("Simülasyon durduruldu")

        # Performans raporu
        self.generate_performance_report()

    def generate_performance_report(self) -> Dict[str, Any]:
        """Performans raporu oluşturur"""
        if not self.performance_log:
            return {}

        # İstatistikler
        fps_values = [log["fps"] for log in self.performance_log if log["fps"] > 0]
        target_counts = [log["target_count"] for log in self.performance_log]
        detection_counts = [log["detection_count"] for log in self.performance_log]

        report = {
            "simulation_duration": self.state.timestamp,
            "average_fps": np.mean(fps_values) if fps_values else 0,
            "max_fps": np.max(fps_values) if fps_values else 0,
            "average_targets": np.mean(target_counts),
            "total_detections": sum(detection_counts),
            "ecm_activations": sum(1 for log in self.performance_log if log["ecm_active"]),
            "memory_usage_mb": np.mean([log["memory_usage"] for log in self.performance_log]),
        }

        self.logger.info(f"Performans raporu: {report}")
        return report

    def save_simulation_data(self, filename: str = None) -> None:
        """Simülasyon verilerini kaydeder"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"radar_simulation_{timestamp}.json"

        data = {
            "config": self.config.__dict__,
            "performance_log": self.performance_log,
            "final_state": {
                "timestamp": self.state.timestamp,
                "target_count": len(self.state.targets),
                "detection_count": len(self.state.detections),
                "track_count": len(self.state.tracks),
            },
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2, default=str)

        self.logger.info(f"Simülasyon verileri {filename} dosyasına kaydedildi")


# Test fonksiyonu
if __name__ == "__main__":
    # Test konfigürasyonu
    config = SimulationConfig(
        radar_frequency=10e9,
        radar_power=1000,
        lpi_enabled=True,
        sar_enabled=True,
        fusion_enabled=True,
        visualization_enabled=True,
        max_targets=10,
    )

    # Simülasyon motoru
    engine = AdvancedRadarSimulationEngine(config)

    # Test hedefleri ekle
    engine.add_target(
        position=np.array([100, 200, 50]),
        velocity=np.array([10, 20, 0]),
        target_type="aircraft",
        rcs=1.0,
    )

    engine.add_target(
        position=np.array([-50, 150, 30]),
        velocity=np.array([-5, 15, 0]),
        target_type="missile",
        rcs=0.5,
    )

    # Test füzesi ekle
    engine.add_missile(position=np.array([0, 0, 10]), velocity=np.array([0, 100, 0]))

    print("Gelişmiş radar simülasyon motoru test edildi.")
    print("Simülasyon başlatılıyor...")

    # Simülasyonu başlat (kısa süre için)
    try:
        engine.start_simulation()
    except KeyboardInterrupt:
        engine.stop_simulation()
        print("Simülasyon kullanıcı tarafından durduruldu.")
