"""
A-Scope (Amplitude Scope) Display

Amplitude vs Range diagnostic oscilloscope view.

Shows:
    - Real-time signal power vs range
    - CFAR threshold overlay
    - Detection markers
    - CFAR cell visualization on hover (Phase 23)

Scientific Value: Visualize target returns breaking the noise floor.

Reference: Richards, "Fundamentals of Radar Signal Processing"
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QFont, QColor
import pyqtgraph as pg
import numpy as np
from typing import Dict, List, Any


class AScope(QWidget):
    """
    A-Scope (Amplitude Scope) Display.
    
    Shows signal amplitude/power vs range for a selected azimuth sector.
    
    Features:
        - Real-time signal power curve
        - CFAR threshold overlay
        - Target detection markers
        - CFAR cell hover visualization (Phase 23)
    """
    
    def __init__(
        self,
        max_range_km: float = 150.0,
        parent: QWidget = None
    ):
        """
        Initialize A-scope.
        
        Args:
            max_range_km: Maximum display range [km]
            parent: Parent widget
        """
        super().__init__(parent)
        self.max_range_km = max_range_km
        
        # State
        self._current_azimuth = 0.0
        self._targets = []
        
        # CFAR parameters (Phase 23)
        self._cfar_guard_cells = 2
        self._cfar_reference_cells = 8
        self._hover_range_idx = -1  # Current hover index
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI components."""
        self.setMinimumHeight(200)
        self.setStyleSheet("background-color: rgb(10, 20, 15);")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        # Header with azimuth selector
        header_layout = QHBoxLayout()
        
        header_label = QLabel("A-SCOPE: SIGNAL vs RANGE")
        header_label.setStyleSheet("""
            QLabel {
                color: #00dd66;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        header_layout.addWidget(header_label)
        
        header_layout.addStretch()
        
        # Azimuth display
        self.azimuth_label = QLabel("AZ: 0°")
        self.azimuth_label.setStyleSheet("""
            QLabel {
                color: #00aa44;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        header_layout.addWidget(self.azimuth_label)
        
        layout.addLayout(header_layout)
        
        # Create plot widget
        pg.setConfigOptions(antialias=True)
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(QColor(10, 20, 15))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Axis labels
        self.plot_widget.setLabel('left', 'Power', units='dB')
        self.plot_widget.setLabel('bottom', 'Range', units='km')
        
        # Set styling
        self.plot_widget.getAxis('left').setPen(pg.mkPen(color=(0, 150, 75)))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen(color=(0, 150, 75)))
        self.plot_widget.getAxis('left').setTextPen(pg.mkPen(color=(0, 150, 75)))
        self.plot_widget.getAxis('bottom').setTextPen(pg.mkPen(color=(0, 150, 75)))
        
        # Set range
        self.plot_widget.setXRange(0, self.max_range_km)
        self.plot_widget.setYRange(-30, 50)
        
        layout.addWidget(self.plot_widget)
        
        # Create plot items
        # Noise floor
        self.noise_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=(80, 80, 80), width=1)
        )
        self.plot_widget.addItem(self.noise_curve)
        
        # CFAR threshold
        self.threshold_curve = pg.PlotCurveItem(
            pen=pg.mkPen(color=(200, 100, 0), width=2, style=Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(self.threshold_curve)
        
        # Signal power at targets
        self.signal_scatter = pg.ScatterPlotItem(
            size=15,
            symbol='t',  # Triangle up
            pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 255, 100, 200)
        )
        self.plot_widget.addItem(self.signal_scatter)
        
        # ═══ PHASE 23: CFAR CELL VISUALIZATION ═══
        # Cell Under Test (CUT) marker
        self.cut_marker = pg.InfiniteLine(
            pos=0, angle=90,
            pen=pg.mkPen(color=(255, 50, 50), width=2)
        )
        self.cut_marker.setVisible(False)
        self.plot_widget.addItem(self.cut_marker)
        
        # Guard cells region (grey)
        self.guard_region = pg.LinearRegionItem(
            values=[0, 1],
            brush=pg.mkBrush(100, 100, 100, 60),
            movable=False
        )
        self.guard_region.setVisible(False)
        self.plot_widget.addItem(self.guard_region)
        
        # Reference cells region - left (green)
        self.ref_region_left = pg.LinearRegionItem(
            values=[0, 1],
            brush=pg.mkBrush(0, 200, 0, 40),
            movable=False
        )
        self.ref_region_left.setVisible(False)
        self.plot_widget.addItem(self.ref_region_left)
        
        # Reference cells region - right (green)
        self.ref_region_right = pg.LinearRegionItem(
            values=[0, 1],
            brush=pg.mkBrush(0, 200, 0, 40),
            movable=False
        )
        self.ref_region_right.setVisible(False)
        self.plot_widget.addItem(self.ref_region_right)
        
        # Connect mouse move
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        
        # Legend
        self._add_legend()
        
        # CFAR info label
        self.cfar_label = QLabel("Hover over plot to see CFAR cells")
        self.cfar_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
        """)
        layout.addWidget(self.cfar_label)
    
    def _add_legend(self):
        """Add custom legend."""
        legend = pg.LegendItem(offset=(70, 30))
        legend.setParentItem(self.plot_widget.plotItem)
        
        # Create dummy items for legend
        noise_item = pg.PlotDataItem(pen=pg.mkPen(color=(80, 80, 80), width=1))
        threshold_item = pg.PlotDataItem(pen=pg.mkPen(color=(200, 100, 0), width=2, style=Qt.PenStyle.DashLine))
        
        legend.addItem(noise_item, 'Noise Floor')
        legend.addItem(threshold_item, 'CFAR Threshold')
    
    def _on_mouse_moved(self, pos):
        """Handle mouse movement for CFAR cell visualization."""
        if self.plot_widget.sceneBoundingRect().contains(pos):
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
            range_km = mouse_point.x()
            
            if 0 < range_km < self.max_range_km:
                self._show_cfar_cells(range_km)
            else:
                self._hide_cfar_cells()
        else:
            self._hide_cfar_cells()
    
    def _show_cfar_cells(self, range_km: float):
        """Show CFAR cells at given range."""
        # Cell width in km (based on resolution)
        cell_width_km = self.max_range_km / 200
        
        guard_width = self._cfar_guard_cells * cell_width_km
        ref_width = self._cfar_reference_cells * cell_width_km
        
        # CUT position
        self.cut_marker.setValue(range_km)
        self.cut_marker.setVisible(True)
        
        # Guard cells (around CUT)
        self.guard_region.setRegion([range_km - guard_width, range_km + guard_width])
        self.guard_region.setVisible(True)
        
        # Reference cells (outside guard)
        left_start = range_km - guard_width - ref_width
        left_end = range_km - guard_width
        self.ref_region_left.setRegion([max(0, left_start), left_end])
        self.ref_region_left.setVisible(True)
        
        right_start = range_km + guard_width
        right_end = range_km + guard_width + ref_width
        self.ref_region_right.setRegion([right_start, min(self.max_range_km, right_end)])
        self.ref_region_right.setVisible(True)
        
        # Update label
        self.cfar_label.setText(
            f"CFAR at {range_km:.1f} km | "
            f"CUT: Red | Guard: {self._cfar_guard_cells*2} cells | "
            f"Ref: {self._cfar_reference_cells*2} cells"
        )
    
    def _hide_cfar_cells(self):
        """Hide CFAR cell visualization."""
        self.cut_marker.setVisible(False)
        self.guard_region.setVisible(False)
        self.ref_region_left.setVisible(False)
        self.ref_region_right.setVisible(False)
        self.cfar_label.setText("Hover over plot to see CFAR cells")
    
    @pyqtSlot(dict)
    def update_display(self, state: Dict[str, Any]):
        """
        Update display with new simulation state.
        
        Args:
            state: State dictionary from SimulationWorker
        """
        # Get radar azimuth
        radar_data = state.get('radar', {})
        self._current_azimuth = radar_data.get('antenna_azimuth_deg', 0)
        self.azimuth_label.setText(f"AZ: {self._current_azimuth:.1f}°")
        
        # Get targets
        targets = state.get('targets', [])
        self._targets = targets
        
        # Update curves
        self._update_noise_floor()
        self._update_threshold()
        self._update_targets()
    
    def _update_noise_floor(self):
        """Update noise floor curve."""
        # Simulate noise floor across range
        n_points = 200
        ranges = np.linspace(0.1, self.max_range_km, n_points)
        
        # Noise floor with 1/R^4 falloff from near range
        # and some random variation
        base_noise = -10 + np.random.randn(n_points) * 2
        
        self.noise_curve.setData(ranges, base_noise)
    
    def _update_threshold(self):
        """Update CFAR threshold line."""
        n_points = 200
        ranges = np.linspace(0.1, self.max_range_km, n_points)
        
        # CFAR threshold (above noise floor by Pfa factor)
        threshold = 3 + np.random.randn(n_points) * 0.5  # ~13 dB above noise
        
        self.threshold_curve.setData(ranges, threshold)
    
    def _update_targets(self):
        """Update target markers."""
        if not self._targets:
            self.signal_scatter.setData([], [])
            return
        
        ranges = []
        snrs = []
        brushes = []
        
        for target in self._targets:
            range_km = target['range_km']
            snr = target['snr_db']
            is_detected = target['is_detected']
            
            if range_km <= self.max_range_km:
                ranges.append(range_km)
                snrs.append(snr)
                
                if is_detected:
                    brushes.append(pg.mkBrush(0, 255, 100, 220))
                else:
                    brushes.append(pg.mkBrush(255, 100, 0, 150))
        
        if ranges:
            self.signal_scatter.setData(ranges, snrs, brush=brushes)
        else:
            self.signal_scatter.setData([], [])

