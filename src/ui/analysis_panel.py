"""
Analysis Panel Widget

Post-flight analysis dashboard with pyqtgraph charts:
- SNR vs Time plot
- Detection Statistics
- Jamming Impact analysis

Integrates with ReplayLoader to display historical data.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QGroupBox, QComboBox, QFrame
)
from PyQt6.QtCore import Qt
from typing import Optional, Dict, Any

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.io.replay_loader import ReplayLoader


class AnalysisPanel(QWidget):
    """
    Post-flight analysis dashboard.
    
    Displays SNR history, detection statistics, and detection rate
    charts based on data from ReplayLoader.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.loader: Optional[ReplayLoader] = None
        self._setup_ui()
    
    def _setup_ui(self):
        """Create UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Header
        header = QLabel("POST-FLIGHT ANALYSIS")
        header.setStyleSheet("""
            QLabel {
                color: #ff8800;
                font-family: 'Consolas', monospace;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: rgba(50, 30, 0, 200);
                border: 1px solid #aa5500;
            }
        """)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Statistics row
        stats_layout = QHBoxLayout()
        
        self.stat_duration = self._create_stat_box("DURATION", "0:00")
        self.stat_detections = self._create_stat_box("DETECTIONS", "0")
        self.stat_rate = self._create_stat_box("DET. RATE", "0%")
        self.stat_mean_snr = self._create_stat_box("AVG SNR", "0 dB")
        self.stat_max_snr = self._create_stat_box("MAX SNR", "0 dB")
        
        stats_layout.addWidget(self.stat_duration)
        stats_layout.addWidget(self.stat_detections)
        stats_layout.addWidget(self.stat_rate)
        stats_layout.addWidget(self.stat_mean_snr)
        stats_layout.addWidget(self.stat_max_snr)
        
        layout.addLayout(stats_layout)
        
        # Target selector
        selector_layout = QHBoxLayout()
        selector_label = QLabel("TARGET:")
        selector_label.setStyleSheet("color: #888888; font-size: 11px;")
        selector_layout.addWidget(selector_label)
        
        self.target_combo = QComboBox()
        self.target_combo.addItem("All Targets", -1)
        self.target_combo.currentIndexChanged.connect(self._on_target_selected)
        self._style_combo(self.target_combo)
        selector_layout.addWidget(self.target_combo)
        selector_layout.addStretch()
        
        layout.addLayout(selector_layout)
        
        # SNR Plot
        if PYQTGRAPH_AVAILABLE:
            self.snr_plot = pg.PlotWidget()
            self.snr_plot.setBackground('#0a1510')
            self.snr_plot.setTitle("SNR vs Time", color='#ff8800', size='12pt')
            self.snr_plot.setLabel('left', 'SNR', units='dB', color='#888888')
            self.snr_plot.setLabel('bottom', 'Time', units='s', color='#888888')
            self.snr_plot.showGrid(x=True, y=True, alpha=0.3)
            self.snr_plot.setMinimumHeight(200)
            
            # Detection threshold line (13 dB typical)
            self.threshold_line = pg.InfiniteLine(
                pos=13.0, angle=0,
                pen=pg.mkPen(color=(255, 100, 100), width=2, style=Qt.PenStyle.DashLine)
            )
            self.snr_plot.addItem(self.threshold_line)
            
            # SNR curve
            self.snr_curve = self.snr_plot.plot(
                pen=pg.mkPen(color=(0, 255, 100), width=2)
            )
            
            # Current time marker
            self.time_marker = pg.InfiniteLine(
                pos=0, angle=90,
                pen=pg.mkPen(color=(255, 200, 0), width=2)
            )
            self.snr_plot.addItem(self.time_marker)
            
            layout.addWidget(self.snr_plot)
        else:
            no_plot = QLabel("pyqtgraph not available")
            no_plot.setStyleSheet("color: #ff5555;")
            layout.addWidget(no_plot)
        
        # Detection Rate Plot
        if PYQTGRAPH_AVAILABLE:
            self.det_plot = pg.PlotWidget()
            self.det_plot.setBackground('#0a1510')
            self.det_plot.setTitle("Detection Rate (Rolling)", color='#00aaff', size='12pt')
            self.det_plot.setLabel('left', 'Rate', units='%', color='#888888')
            self.det_plot.setLabel('bottom', 'Time', units='s', color='#888888')
            self.det_plot.showGrid(x=True, y=True, alpha=0.3)
            self.det_plot.setMinimumHeight(150)
            
            self.det_curve = self.det_plot.plot(
                pen=pg.mkPen(color=(0, 170, 255), width=2)
            )
            
            layout.addWidget(self.det_plot)
    
    def _create_stat_box(self, label: str, value: str) -> QFrame:
        """Create a statistics display box."""
        box = QFrame()
        box.setStyleSheet("""
            QFrame {
                background-color: rgba(20, 20, 20, 200);
                border: 1px solid #333333;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(2)
        
        label_widget = QLabel(label)
        label_widget.setStyleSheet("color: #666666; font-size: 10px;")
        label_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        value_widget = QLabel(value)
        value_widget.setObjectName("value")
        value_widget.setStyleSheet("""
            QLabel {
                color: #00ff88;
                font-family: 'Consolas', monospace;
                font-size: 16px;
                font-weight: bold;
            }
        """)
        value_widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(label_widget)
        layout.addWidget(value_widget)
        
        return box
    
    def _update_stat(self, stat_widget: QFrame, value: str):
        """Update a stat box value."""
        value_label = stat_widget.findChild(QLabel, "value")
        if value_label:
            value_label.setText(value)
    
    def _style_combo(self, combo: QComboBox):
        """Apply dark theme styling to combo box."""
        combo.setStyleSheet("""
            QComboBox {
                color: #00ff88;
                background-color: #1a1a1a;
                border: 1px solid #333333;
                padding: 4px 8px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                border-radius: 3px;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #00aa55;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a1a;
                color: #00ff88;
                selection-background-color: #003322;
            }
        """)
    
    def set_loader(self, loader: ReplayLoader):
        """Set the replay loader and populate data."""
        self.loader = loader
        
        if loader is None:
            self.clear()
            return
        
        # Populate target selector
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItem("All Targets", -1)
        
        if loader.metadata:
            for target_id in loader.metadata.target_ids:
                self.target_combo.addItem(f"Target {target_id}", target_id)
        
        self.target_combo.blockSignals(False)
        
        # Update statistics
        stats = loader.get_detection_stats()
        self._update_stat(self.stat_duration, self._format_time(stats['duration']))
        self._update_stat(self.stat_detections, str(stats['total_detections']))
        self._update_stat(self.stat_rate, f"{stats['detection_rate']*100:.1f}%")
        self._update_stat(self.stat_mean_snr, f"{stats['mean_snr']:.1f} dB")
        self._update_stat(self.stat_max_snr, f"{stats['max_snr']:.1f} dB")
        
        # Update charts
        self._update_charts()
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as M:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    
    def _update_charts(self, target_id: int = None):
        """Update all charts with current data."""
        if not PYQTGRAPH_AVAILABLE or self.loader is None:
            return
        
        # Get SNR history
        times, snr_values = self.loader.get_snr_history(target_id)
        
        if len(times) > 0:
            self.snr_curve.setData(times, snr_values)
        else:
            self.snr_curve.setData([], [])
        
        # Calculate rolling detection rate
        if self.loader._measurements and len(self.loader._measurements['time']) > 0:
            times_all = self.loader._measurements['time']
            detected = self.loader._measurements['detected'].astype(float)
            
            # Rolling average with window of 10 samples
            window = min(10, len(detected))
            if window > 0:
                kernel = np.ones(window) / window
                rolling_rate = np.convolve(detected, kernel, mode='valid') * 100
                times_rolling = times_all[window-1:]
                
                if len(times_rolling) > 0:
                    self.det_curve.setData(times_rolling, rolling_rate)
    
    def set_current_time(self, t: float):
        """Update the current time marker."""
        if PYQTGRAPH_AVAILABLE and hasattr(self, 'time_marker'):
            self.time_marker.setValue(t)
    
    def _on_target_selected(self, index: int):
        """Handle target selection change."""
        target_id = self.target_combo.currentData()
        if target_id == -1:
            target_id = None
        self._update_charts(target_id)
    
    def clear(self):
        """Clear all data."""
        if PYQTGRAPH_AVAILABLE:
            self.snr_curve.setData([], [])
            self.det_curve.setData([], [])
        
        self._update_stat(self.stat_duration, "0:00")
        self._update_stat(self.stat_detections, "0")
        self._update_stat(self.stat_rate, "0%")
        self._update_stat(self.stat_mean_snr, "0 dB")
        self._update_stat(self.stat_max_snr, "0 dB")
        
        self.target_combo.clear()
        self.target_combo.addItem("All Targets", -1)
