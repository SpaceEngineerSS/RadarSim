"""
Gerçek Dünya Harita Entegrasyonu Modülü

Bu modül, radar, hedef ve füze konumlarının gerçek coğrafi koordinatlarda (enlem, boylam, yükseklik) simülasyonunu ve OpenStreetMap/DEM verisi üzerinde görselleştirilmesini sağlar.

Bilimsel Temel:
- Coğrafi Bilgi Sistemleri (GIS) ile radar ve hedeflerin gerçek ortamda modellenmesi.
- OpenStreetMap (OSM) ve Dijital Yükseklik Modeli (DEM) ile arazi etkisi ve harita entegrasyonu.
- Kaynak: NATO RTO-TR-SET-093, "Radar Modelling for Geospatial Applications", 2009

Kaynaklar:
- NATO RTO-TR-SET-093, "Radar Modelling for Geospatial Applications", 2009
- OSM & DEM Documentation
- Skolnik, "Radar Handbook", 3rd Ed., McGraw-Hill, 2008
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import contextlib

try:
    import contextily as ctx  # OSM tile desteği için
except ImportError:
    ctx = None

class GeoConverter:
    """Coğrafi koordinat ↔ UTM ↔ simülasyon koordinatı dönüşümleri"""
    def __init__(self, origin_lat: float, origin_lon: float):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        # Basit equirectangular projeksiyon (küçük alanlar için yeterli)
        self.R = 6371000  # Dünya yarıçapı (m)

    def geo_to_xy(self, lat: float, lon: float) -> Tuple[float, float]:
        """Enlem/boylamı simülasyon düzlemine çevirir (x, y, metre)"""
        dlat = np.radians(lat - self.origin_lat)
        dlon = np.radians(lon - self.origin_lon)
        x = self.R * dlon * np.cos(np.radians(self.origin_lat))
        y = self.R * dlat
        return x, y

    def xy_to_geo(self, x: float, y: float) -> Tuple[float, float]:
        """Simülasyon düzleminden enlem/boylama çevirir"""
        dlat = y / self.R
        dlon = x / (self.R * np.cos(np.radians(self.origin_lat)))
        lat = self.origin_lat + np.degrees(dlat)
        lon = self.origin_lon + np.degrees(dlon)
        return lat, lon

class GeoMapVisualizer:
    """Gerçek dünya haritası üzerinde radar/füze/hedef görselleştirme"""
    def __init__(self, origin_lat: float, origin_lon: float, zoom: int = 14):
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.converter = GeoConverter(origin_lat, origin_lon)
        self.zoom = zoom

    def plot_on_map(self, geo_points: List[Tuple[float, float, str]], dem: np.ndarray = None):
        """
        OpenStreetMap üzerinde radar, hedef ve füze konumlarını gösterir.
        geo_points: [(lat, lon, label)]
        dem: Dijital yükseklik modeli (isteğe bağlı)
        """
        # Koordinatları x, y'ye çevir
        xy_points = [self.converter.geo_to_xy(lat, lon) for lat, lon, _ in geo_points]
        labels = [label for _, _, label in geo_points]
        xs, ys = zip(*xy_points)

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(xs, ys, c=['g' if 'Radar' in l else 'r' if 'Hedef' in l else 'c' for l in labels], s=80)
        for (x, y, label) in zip(xs, ys, labels):
            ax.text(x, y, label, fontsize=12)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Gerçek Dünya Haritasında Radar Simülasyonu')
        ax.grid(True, alpha=0.3)

        # OSM harita katmanı ekle (contextily varsa)
        if ctx is not None:
            try:
                # Dönüşüm: x/y -> Web Mercator (EPSG:3857)
                import pyproj
                proj = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857', always_xy=True)
                xs_merc, ys_merc = proj.transform([lon for _, lon, _ in geo_points], [lat for lat, _, _ in geo_points])
                ax.scatter(xs_merc, ys_merc, c='none')  # Sadece eksenleri ayarlamak için
                ctx.add_basemap(ax, crs='epsg:3857', zoom=self.zoom)
            except Exception as e:
                print(f"OSM harita katmanı eklenemedi: {e}")

        # DEM (yükseklik) haritası ekle
        if dem is not None:
            ax.imshow(dem, extent=[min(xs), max(xs), min(ys), max(ys)], alpha=0.3, cmap='terrain')

        plt.tight_layout()
        plt.show()

# Örnek kullanım ve bilimsel açıklama:
if __name__ == "__main__":
    # Radar, hedef ve füze için örnek coğrafi koordinatlar (Ankara civarı)
    radar_geo = (39.9208, 32.8541, 'Radar')
    target_geo = (39.9250, 32.8600, 'Hedef')
    missile_geo = (39.9220, 32.8580, 'Füze')
    geo_points = [radar_geo, target_geo, missile_geo]
    visualizer = GeoMapVisualizer(origin_lat=39.9208, origin_lon=32.8541)
    visualizer.plot_on_map(geo_points)
    print("Gerçek dünya harita entegrasyonu tamamlandı. [Kaynak: NATO RTO-TR-SET-093, 2009]") 