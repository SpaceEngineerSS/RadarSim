"""
SNR Histogram Widget

Displays real-time distribution of Signal-to-Noise Ratio values
from detected targets.

Helps visualize:
- Strong targets (high SNR, right side)
- Weak targets near noise floor (low SNR, left side)
- Detection threshold effectiveness
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np
from collections import deque

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False


class SNRHistogramWidget(QWidget):
    """
    SNR Distribution Histogram Widget.
    
    Shows the frequency distribution of SNR values from detections,
    helping operators understand target signal strength distribution.
    """
    
    def __init__(self, parent=None, history_size: int = 500):
        super().__init__(parent)
        
        self.setMinimumSize(600, 450)
        
        # SNR history buffer
        self.snr_history = deque(maxlen=history_size)
        self.history_size = history_size
        
        # Histogram parameters
        self.bin_count = 30
        self.snr_min = -10
        self.snr_max = 50
        
        # Stats
        self.total_detections = 0
        
        # Apply theme
        self._apply_theme()
        
        # Setup UI
        self._setup_ui()
        
        # Generate demo data
        self._generate_demo_data()
    
    def _apply_theme(self):
        """Apply dark radar theme."""
        self.setStyleSheet("""
            QWidget {
                background-color: #0a1510;
                color: #00dd66;
                font-family: 'Consolas', monospace;
            }
            QGroupBox {
                border: 1px solid #00aa55;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #003322;
                border: 1px solid #00aa55;
                color: #00ff88;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #004433;
            }
        """)
    
    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("📊 SNR DISTRIBUTION (Detection Strength)")
        title.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #00ff88; padding: 5px;")
        layout.addWidget(title)
        
        # Main content
        content = QHBoxLayout()
        
        # Left: Stats panel
        stats = self._create_stats_panel()
        content.addWidget(stats, stretch=1)
        
        # Right: Histogram
        hist = self._create_histogram()
        content.addWidget(hist, stretch=2)
        
        layout.addLayout(content)
    
    def _create_stats_panel(self) -> QGroupBox:
        """Create statistics panel."""
        group = QGroupBox("Statistics")
        layout = QGridLayout(group)
        layout.setSpacing(10)
        
        row = 0
        
        # Total detections
        layout.addWidget(QLabel("Total Detections:"), row, 0)
        self.total_label = QLabel("0")
        self.total_label.setStyleSheet("color: #00ffaa; font-weight: bold;")
        layout.addWidget(self.total_label, row, 1)
        row += 1
        
        # Mean SNR
        layout.addWidget(QLabel("Mean SNR:"), row, 0)
        self.mean_label = QLabel("-- dB")
        self.mean_label.setStyleSheet("color: #00ffaa; font-weight: bold;")
        layout.addWidget(self.mean_label, row, 1)
        row += 1
        
        # Median SNR
        layout.addWidget(QLabel("Median SNR:"), row, 0)
        self.median_label = QLabel("-- dB")
        layout.addWidget(self.median_label, row, 1)
        row += 1
        
        # Std Dev
        layout.addWidget(QLabel("Std Dev:"), row, 0)
        self.std_label = QLabel("-- dB")
        layout.addWidget(self.std_label, row, 1)
        row += 1
        
        # Min/Max
        layout.addWidget(QLabel("Min/Max:"), row, 0)
        self.minmax_label = QLabel("-- / -- dB")
        layout.addWidget(self.minmax_label, row, 1)
        row += 1
        
        # Separator
        layout.addWidget(QLabel(""), row, 0)
        row += 1
        
        # Threshold zones
        zone_title = QLabel("Detection Zones:")
        zone_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(zone_title, row, 0, 1, 2)
        row += 1
        
        # Weak detections
        layout.addWidget(QLabel("⬤ Weak (<10 dB):"), row, 0)
        self.weak_label = QLabel("0%")
        self.weak_label.setStyleSheet("color: #ff5555;")
        layout.addWidget(self.weak_label, row, 1)
        row += 1
        
        # Moderate
        layout.addWidget(QLabel("⬤ Moderate (10-20 dB):"), row, 0)
        self.mod_label = QLabel("0%")
        self.mod_label.setStyleSheet("color: #ffaa00;")
        layout.addWidget(self.mod_label, row, 1)
        row += 1
        
        # Strong
        layout.addWidget(QLabel("⬤ Strong (>20 dB):"), row, 0)
        self.strong_label = QLabel("0%")
        self.strong_label.setStyleSheet("color: #00ff88;")
        layout.addWidget(self.strong_label, row, 1)
        row += 1
        
        # Clear button
        layout.addWidget(QLabel(""), row, 0)
        row += 1
        
        clear_btn = QPushButton("Clear History")
        clear_btn.clicked.connect(self._clear_history)
        layout.addWidget(clear_btn, row, 0, 1, 2)
        
        layout.setRowStretch(row + 1, 1)
        
        return group
    
    def _create_histogram(self) -> QGroupBox:
        """Create histogram plot."""
        group = QGroupBox("SNR Histogram")
        layout = QVBoxLayout(group)
        
        if PYQTGRAPH_AVAILABLE:
            pg.setConfigOptions(antialias=True)
            
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground('#051008')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setLabel('bottom', 'SNR', units='dB')
            self.plot_widget.setLabel('left', 'Count')
            
            # Create bar graph item
            self.bar_graph = pg.BarGraphItem(
                x=[],
                height=[],
                width=1.5,
                brush='#00aa55'
            )
            self.plot_widget.addItem(self.bar_graph)
            
            # Add threshold lines
            # Detection threshold (~13 dB typical)
            self.threshold_line = pg.InfiniteLine(
                pos=13,
                angle=90,
                pen=pg.mkPen('#ffff00', width=2, style=Qt.PenStyle.DashLine)
            )
            self.plot_widget.addItem(self.threshold_line)
            
            # Add zone shading
            # Weak zone (red)
            weak_region = pg.LinearRegionItem(
                values=[self.snr_min, 10],
                brush=pg.mkBrush(255, 50, 50, 30),
                movable=False
            )
            self.plot_widget.addItem(weak_region)
            
            # Strong zone (green)
            strong_region = pg.LinearRegionItem(
                values=[20, self.snr_max],
                brush=pg.mkBrush(0, 255, 100, 30),
                movable=False
            )
            self.plot_widget.addItem(strong_region)
            
            layout.addWidget(self.plot_widget)
        else:
            fallback = QLabel("PyQtGraph not available")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(fallback)
        
        return group
    
    def add_snr_value(self, snr_db: float):
        """Add a new SNR value to the histogram."""
        self.snr_history.append(snr_db)
        self.total_detections += 1
        self._update_display()
    
    def add_snr_values(self, snr_values: list):
        """Add multiple SNR values."""
        for snr in snr_values:
            self.snr_history.append(snr)
        self.total_detections += len(snr_values)
        self._update_display()
    
    def _generate_demo_data(self):
        """Generate demo data for display."""
        # Simulate typical radar SNR distribution
        # Mix of weak and strong targets
        weak = np.random.normal(8, 3, 50)
        moderate = np.random.normal(16, 4, 100)
        strong = np.random.normal(28, 5, 30)
        
        demo_data = np.concatenate([weak, moderate, strong])
        self.snr_history.extend(demo_data)
        self.total_detections = len(demo_data)
        
        self._update_display()
    
    def _update_display(self):
        """Update histogram and statistics."""
        if len(self.snr_history) == 0:
            return
        
        snr_array = np.array(self.snr_history)
        
        # Update statistics
        self.total_label.setText(str(self.total_detections))
        self.mean_label.setText(f"{np.mean(snr_array):.1f} dB")
        self.median_label.setText(f"{np.median(snr_array):.1f} dB")
        self.std_label.setText(f"{np.std(snr_array):.1f} dB")
        self.minmax_label.setText(f"{np.min(snr_array):.1f} / {np.max(snr_array):.1f} dB")
        
        # Calculate zone percentages
        weak_pct = np.sum(snr_array < 10) / len(snr_array) * 100
        strong_pct = np.sum(snr_array > 20) / len(snr_array) * 100
        mod_pct = 100 - weak_pct - strong_pct
        
        self.weak_label.setText(f"{weak_pct:.1f}%")
        self.mod_label.setText(f"{mod_pct:.1f}%")
        self.strong_label.setText(f"{strong_pct:.1f}%")
        
        # Update histogram
        if PYQTGRAPH_AVAILABLE:
            hist, bin_edges = np.histogram(
                snr_array,
                bins=self.bin_count,
                range=(self.snr_min, self.snr_max)
            )
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            self.bar_graph.setOpts(x=bin_centers, height=hist, width=1.5)
    
    def _clear_history(self):
        """Clear SNR history."""
        self.snr_history.clear()
        self.total_detections = 0
        self._update_display()
        
        # Reset histogram
        if PYQTGRAPH_AVAILABLE:
            self.bar_graph.setOpts(x=[], height=[], width=1.5)


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    widget = SNRHistogramWidget()
    widget.show()
    sys.exit(app.exec())
