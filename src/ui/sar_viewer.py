"""
SAR Image Viewer

Dialog window for displaying Synthetic Aperture Radar (SAR) images.

Features:
    - 2D heatmap display (Range vs Cross-Range)
    - Contrast/brightness adjustment
    - Image quality metrics (SNR, Contrast, Resolution)
    - Export to PNG capability

Reference: Oliver & Quegan, "Understanding SAR Images", IEEE Press, 1998
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QFileDialog, QGroupBox, QGridLayout, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np
from typing import Optional, Dict, Any, List

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

# Import real SAR physics engine (Phase 25)
try:
    from src.advanced.sar_isar import AdvancedSARISAR
    SAR_PHYSICS_AVAILABLE = True
except ImportError:
    SAR_PHYSICS_AVAILABLE = False
    AdvancedSARISAR = None


class SARViewer(QDialog):
    """
    SAR Image Viewer Dialog.
    
    Displays SAR/ISAR imagery with contrast controls and quality metrics.
    
    Attributes:
        image_data: Current 2D numpy array being displayed
        image_item: PyQtGraph ImageItem for rendering
    """
    
    def __init__(self, parent: Optional[QWidget] = None):
        """
        Initialize SAR Viewer.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.setWindowTitle("SAR Image Viewer")
        self.setMinimumSize(800, 600)
        self.setModal(False)  # Non-blocking dialog
        
        # State
        self.image_data: Optional[np.ndarray] = None
        self.quality_metrics: Dict[str, float] = {}
        
        # Real simulation data (set externally via set_simulation_data)
        self._targets: List = []
        self._radar_pos: Optional[np.ndarray] = None
        self._radar_freq_hz: float = 10e9
        self._radar_bandwidth: float = 100e6
        
        # Apply dark theme
        self._apply_theme()
        
        # Setup UI
        self._setup_ui()
        
        # Generate demo image on open (for testing)
        self._generate_demo_image()
    
    def _apply_theme(self) -> None:
        """Apply dark radar-style theme."""
        self.setStyleSheet("""
            QDialog {
                background-color: #0a1510;
                color: #00dd66;
            }
            QGroupBox {
                border: 1px solid #00aa55;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
                color: #00dd66;
                font-family: 'Consolas', monospace;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #00dd66;
                font-family: 'Consolas', monospace;
            }
            QPushButton {
                background-color: #002815;
                color: #00ff88;
                border: 1px solid #00aa55;
                padding: 8px 16px;
                font-family: 'Consolas', monospace;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #003825;
            }
            QPushButton:pressed {
                background-color: #001510;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00aa55;
                height: 6px;
                background: #001510;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #00ff88;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
        """)
    
    def _setup_ui(self) -> None:
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("📡 SAR/ISAR IMAGE VIEWER")
        header.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet("color: #00ff88; padding: 10px;")
        layout.addWidget(header)
        
        # Main content area
        content_layout = QHBoxLayout()
        
        # Left: Image display
        image_group = QGroupBox("SAR Image (Range vs Cross-Range)")
        image_layout = QVBoxLayout(image_group)
        
        if PYQTGRAPH_AVAILABLE:
            # Create pyqtgraph ImageView
            pg.setConfigOptions(antialias=True)
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground(QColor(5, 15, 10))
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setLabel('bottom', 'Cross-Range', units='m')
            self.plot_widget.setLabel('left', 'Range', units='m')
            
            # Image item
            self.image_item = pg.ImageItem()
            self.plot_widget.addItem(self.image_item)
            
            # Colorbar
            self.colorbar = pg.ColorBarItem(
                values=(0, 1),
                colorMap=pg.colormap.get('viridis'),
                orientation='right'
            )
            self.colorbar.setImageItem(self.image_item)
            
            image_layout.addWidget(self.plot_widget)
        else:
            # Fallback if pyqtgraph not available
            fallback_label = QLabel("PyQtGraph not available.\nInstall with: pip install pyqtgraph")
            fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_layout.addWidget(fallback_label)
            self.image_item = None
        
        content_layout.addWidget(image_group, stretch=3)
        
        # Right: Controls and metrics
        right_panel = QVBoxLayout()
        
        # Quality Metrics
        metrics_group = QGroupBox("Image Quality Metrics")
        metrics_layout = QGridLayout(metrics_group)
        
        self.snr_label = QLabel("SNR: -- dB")
        self.contrast_label = QLabel("Contrast: --")
        self.resolution_label = QLabel("Resolution: -- m")
        self.peak_label = QLabel("Peak: -- dB")
        
        metrics_layout.addWidget(self.snr_label, 0, 0)
        metrics_layout.addWidget(self.contrast_label, 1, 0)
        metrics_layout.addWidget(self.resolution_label, 2, 0)
        metrics_layout.addWidget(self.peak_label, 3, 0)
        
        right_panel.addWidget(metrics_group)
        
        # Contrast Control
        contrast_group = QGroupBox("Display Controls")
        contrast_layout = QVBoxLayout(contrast_group)
        
        contrast_layout.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(1, 100)
        self.contrast_slider.setValue(50)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        contrast_layout.addWidget(self.contrast_slider)
        
        contrast_layout.addWidget(QLabel("Brightness:"))
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(0, 100)
        self.brightness_slider.setValue(50)
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        contrast_layout.addWidget(self.brightness_slider)
        
        right_panel.addWidget(contrast_group)
        
        # Algorithm Info
        algo_group = QGroupBox("Algorithm Info")
        algo_layout = QVBoxLayout(algo_group)
        
        self.algo_label = QLabel("Algorithm: Range-Doppler")
        self.bandwidth_label = QLabel("Bandwidth: 100 MHz")
        self.aperture_label = QLabel("Aperture: 100 m")
        
        algo_layout.addWidget(self.algo_label)
        algo_layout.addWidget(self.bandwidth_label)
        algo_layout.addWidget(self.aperture_label)
        
        right_panel.addWidget(algo_group)
        
        # Stretch
        right_panel.addStretch()
        
        content_layout.addLayout(right_panel, stretch=1)
        layout.addLayout(content_layout)
        
        # Bottom buttons
        button_layout = QHBoxLayout()
        
        self.generate_btn = QPushButton("🔄 Generate New Image")
        self.generate_btn.clicked.connect(self.generate_image)
        button_layout.addWidget(self.generate_btn)
        
        self.export_btn = QPushButton("💾 Export PNG")
        self.export_btn.clicked.connect(self._export_image)
        button_layout.addWidget(self.export_btn)
        
        self.close_btn = QPushButton("✕ Close")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.close_btn)
        
        layout.addLayout(button_layout)
    
    def update_image(self, data: np.ndarray, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Update the displayed SAR image.
        
        Args:
            data: 2D numpy array of SAR image (complex magnitude)
            metrics: Optional quality metrics dictionary
        """
        if data is None or not PYQTGRAPH_AVAILABLE:
            return
        
        self.image_data = data
        
        # Convert to dB scale for display
        data_db = 20 * np.log10(np.abs(data) + 1e-10)
        data_db = np.clip(data_db, -60, 0)  # Clip to reasonable range
        
        # Normalize to 0-1
        data_norm = (data_db + 60) / 60
        
        # Update image
        self.image_item.setImage(data_norm.T, autoLevels=False)
        self.image_item.setLevels([0, 1])
        
        # Update metrics
        if metrics:
            self.quality_metrics = metrics
            self._update_metrics_display()
    
    def _update_metrics_display(self) -> None:
        """Update the metrics labels."""
        snr = self.quality_metrics.get('SNR_dB', 0)
        contrast = self.quality_metrics.get('Contrast', 0)
        resolution = self.quality_metrics.get('Resolution_m', 0)
        peak = self.quality_metrics.get('Peak_dB', 0)
        
        self.snr_label.setText(f"SNR: {snr:.1f} dB")
        self.contrast_label.setText(f"Contrast: {contrast:.2f}")
        self.resolution_label.setText(f"Resolution: {resolution:.1f} m")
        self.peak_label.setText(f"Peak: {peak:.1f} dB")
    
    def _on_contrast_changed(self, value: int) -> None:
        """Handle contrast slider change."""
        if self.image_item is None:
            return
        
        # Map slider value to level range
        max_level = value / 100.0
        self.image_item.setLevels([0, max_level])
    
    def _on_brightness_changed(self, value: int) -> None:
        """Handle brightness slider change."""
        # Brightness adjustment is simulated via level offset
        pass  # Simplified for now
    
    def set_simulation_data(
        self,
        targets: List,
        radar_pos: np.ndarray,
        radar_freq_hz: float = 10e9,
        radar_bandwidth: float = 100e6
    ) -> None:
        """
        Set simulation data for real SAR imaging.
        
        Call this method to enable physics-based SAR image generation
        instead of demo/mock data.
        
        Args:
            targets: List of Target objects from SimulationEngine
            radar_pos: Radar position [x, y, z] in meters
            radar_freq_hz: Radar carrier frequency [Hz]
            radar_bandwidth: Radar bandwidth [Hz]
        """
        self._targets = targets
        self._radar_pos = np.asarray(radar_pos)
        self._radar_freq_hz = radar_freq_hz
        self._radar_bandwidth = radar_bandwidth
    
    def _generate_sar_from_targets(self) -> bool:
        """
        Generate SAR image using real physics engine.
        
        Uses AdvancedSARISAR to create phase history from target positions
        and apply Range-Doppler Algorithm for image formation.
        
        Reference: Cumming & Wong, "Digital Processing of SAR Data", Artech House
        
        Returns:
            True if successful, False if fallback to demo is needed
        """
        if not SAR_PHYSICS_AVAILABLE or not PYQTGRAPH_AVAILABLE:
            return False
        
        if len(self._targets) == 0:
            return False  # No targets, use demo
        
        try:
            # Extract target positions and RCS values
            target_positions = []
            target_rcs = []
            
            for target in self._targets:
                pos = target.position
                target_positions.append([pos[0], pos[1], pos[2] if len(pos) > 2 else 0.0])
                target_rcs.append(target.rcs_mean if hasattr(target, 'rcs_mean') else 1.0)
            
            target_positions = np.array(target_positions)
            target_rcs = np.array(target_rcs)
            
            # Initialize SAR processor with radar parameters
            sar_processor = AdvancedSARISAR(
                fc=self._radar_freq_hz,
                bandwidth=self._radar_bandwidth,
                prf=1000,  # 1 kHz PRF
                platform_velocity=100,  # 100 m/s synthetic aperture platform
                synthetic_aperture=100  # 100 m aperture
            )
            
            # Generate raw SAR data (phase history)
            raw_data = sar_processor.generate_sar_raw_data(
                target_positions, target_rcs,
                range_samples=512, azimuth_samples=256
            )
            
            # Apply Range-Doppler Algorithm for image formation
            sar_image = sar_processor.range_doppler_algorithm(raw_data)
            
            # Calculate image quality metrics
            metrics = sar_processor.calculate_image_quality(sar_image)
            metrics['Scene'] = f'{len(self._targets)} Targets'
            metrics['Algorithm'] = 'Range-Doppler'
            
            # Update algorithm info labels
            self.algo_label.setText(f"Algorithm: Range-Doppler")
            self.bandwidth_label.setText(f"Bandwidth: {self._radar_bandwidth/1e6:.0f} MHz")
            self.aperture_label.setText(f"Targets: {len(self._targets)}")
            
            # Display the image
            self.update_image(np.abs(sar_image), metrics)
            
            return True
            
        except Exception as e:
            print(f"[SAR] Real imaging failed: {e}, falling back to demo")
            return False
    
    def generate_image(self) -> None:
        """
        Generate SAR image - tries real physics first, falls back to demo.
        
        This is the main entry point called by the Generate button.
        """
        # Try real SAR physics if targets are available
        if self._targets and self._generate_sar_from_targets():
            return  # Success with real physics
        
        # Fallback to demo image
        self._generate_demo_image()
    
    def _generate_demo_image(self) -> None:
        """Generate a realistic demo SAR image for testing."""
        if not PYQTGRAPH_AVAILABLE:
            return
        
        # Image size (Range x Cross-Range)
        range_bins = 512
        azimuth_bins = 512
        
        # Initialize image array
        image = np.zeros((range_bins, azimuth_bins), dtype=np.float64)
        
        # ═══ 1. TERRAIN BACKGROUND (Clutter) ═══
        # Create multi-scale Perlin-like noise for terrain
        for scale in [64, 32, 16, 8]:
            noise = np.random.randn(range_bins // scale, azimuth_bins // scale)
            noise = np.kron(noise, np.ones((scale, scale)))[:range_bins, :azimuth_bins]
            image += noise * (0.3 / scale * 8)
        
        # Add Rayleigh-distributed speckle (SAR characteristic)
        speckle = np.random.rayleigh(0.15, (range_bins, azimuth_bins))
        image = np.abs(image) + speckle
        
        # ═══ 2. ROADS (Linear Features) ═══
        # Horizontal road
        road_y = 200
        image[road_y-3:road_y+3, 50:450] = 0.1  # Roads are dark (smooth)
        
        # Diagonal road
        for i in range(300):
            x = 100 + i
            y = 350 - i // 2
            if 0 <= y < range_bins and 0 <= x < azimuth_bins:
                image[y-2:y+2, x-1:x+1] = 0.1
        
        # ═══ 3. BUILDINGS (Strong Reflectors) ═══
        buildings = [
            # (range, azimuth, range_size, az_size, intensity)
            (120, 150, 20, 25, 3.0),   # Large building
            (130, 250, 15, 20, 2.5),   # Medium building
            (125, 350, 12, 15, 2.0),   # Small building
            (280, 180, 25, 30, 3.5),   # Industrial
            (300, 300, 18, 22, 2.8),   # Warehouse
            (350, 120, 10, 12, 1.8),   # House
            (360, 200, 8, 10, 1.5),    # House
            (370, 280, 10, 12, 1.7),   # House
        ]
        
        for r, a, rs, az, intensity in buildings:
            # Building footprint
            image[r:r+rs, a:a+az] = intensity
            
            # Double-bounce (bright line at near-range edge)
            image[r-2:r, a:a+az] = intensity * 1.5
            
            # Shadow (dark behind building)
            shadow_len = rs + 5
            image[r+rs:r+rs+shadow_len, a:a+az] = 0.05
        
        # ═══ 4. VEHICLES (Point Targets) ═══
        vehicles = [
            (180, 160, 2.0),   # Truck
            (185, 165, 1.5),   # Car
            (250, 200, 2.5),   # Tank
            (255, 208, 2.3),   # APC
            (320, 350, 1.8),   # Vehicle
            (400, 100, 1.0),   # Small car
            (410, 300, 1.2),   # Motorbike
        ]
        
        for r, a, intensity in vehicles:
            # Point spread function (2D Gaussian approximation)
            sigma = 2.5
            for dr in range(-6, 7):
                for da in range(-6, 7):
                    rr, aa = r + dr, a + da
                    if 0 <= rr < range_bins and 0 <= aa < azimuth_bins:
                        psf = intensity * np.exp(-(dr**2 + da**2) / (2 * sigma**2))
                        image[rr, aa] += psf
        
        # ═══ 5. AIRCRAFT ON RUNWAY ═══
        aircraft_r, aircraft_a = 450, 256
        # Fuselage
        for dr in range(-15, 16):
            for da in range(-3, 4):
                rr, aa = aircraft_r + dr, aircraft_a + da
                if 0 <= rr < range_bins and 0 <= aa < azimuth_bins:
                    image[rr, aa] = 4.0
        # Wings
        for dr in range(-2, 3):
            for da in range(-25, 26):
                rr, aa = aircraft_r + dr, aircraft_a + da
                if 0 <= rr < range_bins and 0 <= aa < azimuth_bins:
                    image[rr, aa] = 3.5
        
        # ═══ 6. WATER BODY (Very Dark) ═══
        water_center = (80, 400)
        for r in range(range_bins):
            for a in range(azimuth_bins):
                dist = np.sqrt((r - water_center[0])**2 + (a - water_center[1])**2)
                if dist < 40:
                    image[r, a] = 0.02 + np.random.rayleigh(0.01)
        
        # ═══ 7. FOREST AREA (Moderate Backscatter) ═══
        for r in range(380, 500):
            for a in range(350, 500):
                if r < range_bins and a < azimuth_bins:
                    image[r, a] = 0.8 + np.random.rayleigh(0.3)
        
        # Apply log compression for display (typical SAR processing)
        image = np.maximum(image, 1e-10)
        image_db = 10 * np.log10(image)
        
        # Normalize for display
        image_display = image_db - np.min(image_db)
        image_display = image_display / np.max(image_display)
        
        # Calculate metrics
        signal_power = np.max(image**2)
        noise_power = np.var(speckle)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        # Contrast ratio (buildings vs terrain)
        building_region = image[120:145, 150:180]
        terrain_region = image[200:230, 50:100]
        contrast = np.mean(building_region) / (np.mean(terrain_region) + 1e-10)
        
        metrics = {
            'SNR_dB': snr_db,
            'Contrast': contrast,
            'Resolution_m': 1.0,  # 1 meter resolution
            'Peak_dB': 10 * np.log10(np.max(image)),
            'Scene': 'Urban Airfield',
            'Range': f'{range_bins} bins',
            'Azimuth': f'{azimuth_bins} bins',
        }
        
        self.update_image(image_display, metrics)
    
    def _export_image(self) -> None:
        """Export current image to PNG file."""
        if self.image_data is None:
            return
        
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Export SAR Image",
            "sar_image.png",
            "PNG Files (*.png);;All Files (*)"
        )
        
        if filepath:
            try:
                import matplotlib.pyplot as plt
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Plot image
                data_db = 20 * np.log10(np.abs(self.image_data) + 1e-10)
                im = ax.imshow(data_db.T, cmap='viridis', aspect='auto',
                              origin='lower', vmin=-60, vmax=0)
                
                ax.set_xlabel('Cross-Range (m)')
                ax.set_ylabel('Range (m)')
                ax.set_title('SAR Image')
                
                plt.colorbar(im, ax=ax, label='dB')
                
                # Save
                plt.savefig(filepath, dpi=150, bbox_inches='tight',
                           facecolor='#0a1510')
                plt.close()
                
                print(f"[SAR] Image exported to: {filepath}")
                
            except Exception as e:
                print(f"[SAR] Export failed: {e}")


# Need to import QColor for the plot widget
from PyQt6.QtGui import QColor
