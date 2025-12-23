"""
B-Scope Radar Display (Range vs Azimuth Cartesian)

Scientific Basis:
    B-Scope is used for AESA (Active Electronically Scanned Array) and
    fighter radar displays. Unlike PPI (polar), B-Scope shows:
    - X-Axis: Azimuth (degrees from boresight)
    - Y-Axis: Range (km from radar)
    
    AESA radars do NOT have a rotating sweep line - they use electronic
    beam steering to update all targets within the FOV simultaneously.

Reference:
    Stimson, "Introduction to Airborne Radar", Chapter 8
    AN/APG-77 (F-22), AN/APG-81 (F-35) display specifications
"""

import time
import numpy as np
from typing import List, Dict, Optional
from collections import deque
from dataclasses import dataclass

import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont


@dataclass
class BScopeBlip:
    """Target blip data for B-Scope display."""
    target_id: int
    azimuth_deg: float
    range_km: float
    intensity: float
    creation_time: float
    is_detected: bool
    snr_db: float
    name: str = ""


class BScope(QWidget):
    """
    B-Scope Radar Display (Cartesian: Azimuth vs Range).
    
    Used for AESA/Fighter radars. Unlike PPI, this display shows
    targets on a rectangular grid with:
    - X-Axis: Azimuth (degrees, typically ±60°)
    - Y-Axis: Range (km, 0 to max_range)
    
    Features:
        - No rotating sweep line (AESA uses electronic beam steering)
        - Raster bar effect for visual feedback
        - Target symbols with SNR-based intensity
        - Fading persistence for historical detections
    """
    
    # Signal for target selection
    target_selected = None  # Will be connected if needed
    
    def __init__(self, max_range_km: float = 200.0, 
                 azimuth_limits: tuple = (-60.0, 60.0),
                 parent: QWidget = None):
        """
        Initialize B-Scope display.
        
        Args:
            max_range_km: Maximum display range [km]
            azimuth_limits: (min_az, max_az) in degrees
            parent: Parent widget
        """
        super().__init__(parent)
        self.max_range_km = max_range_km
        self.azimuth_limits = azimuth_limits
        
        # State
        self.blips: List[BScopeBlip] = []
        self.blip_history: deque = deque(maxlen=200)
        self.last_update_time = time.time()
        
        # Raster bar position (for visual effect)
        self.raster_position = 0.0  # 0 to 1 (top to bottom)
        self.raster_speed = 2.0  # Scans per second
        
        # Target data storage
        self.target_data: Dict[int, Dict] = {}
        self.selected_target_id: Optional[int] = None
        
        # ECM strobe storage (Phase 23)
        self.ecm_strobes = []  # List of LinearRegionItems
        self.active_jammer_azimuths = []  # Jammer azimuth angles
        
        # Performance limiting
        self._min_update_interval = 1.0 / 20.0  # 20 FPS
        self._last_frame_time = 0.0
        
        self._setup_ui()
        
        # Raster animation timer
        self._raster_timer = QTimer(self)
        self._raster_timer.timeout.connect(self._update_raster)
        self._raster_timer.start(50)  # 20 Hz
    
    def _setup_ui(self):
        """Setup the B-Scope UI components."""
        self.setMinimumSize(500, 400)
        self.setStyleSheet("background-color: rgb(5, 10, 15);")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("B-SCOPE (AESA)")
        header.setStyleSheet("""
            QLabel {
                color: #00aaff;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
                padding: 3px;
                background-color: rgba(0, 20, 40, 200);
            }
        """)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # PyQtGraph plot widget (Cartesian)
        pg.setConfigOptions(antialias=True, background='k')
        
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground((5, 15, 25))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Set axis labels
        self.plot_widget.setLabel('bottom', 'Azimuth', units='°')
        self.plot_widget.setLabel('left', 'Range', units='km')
        
        # Set axis ranges
        self.plot_widget.setXRange(self.azimuth_limits[0], self.azimuth_limits[1])
        self.plot_widget.setYRange(0, self.max_range_km)
        
        # Invert Y axis (0 at bottom, max_range at top)
        self.plot_widget.getPlotItem().invertY(False)
        
        layout.addWidget(self.plot_widget)
        
        # Create scatter for current targets
        self.target_scatter = pg.ScatterPlotItem(
            size=12,
            symbol='s',  # Square symbol for B-Scope
            pen=pg.mkPen(color=(0, 200, 255), width=2),
            brush=pg.mkBrush(0, 150, 200, 180)
        )
        self.plot_widget.addItem(self.target_scatter)
        
        # Create scatter for historical blips (fading)
        self.history_scatter = pg.ScatterPlotItem(
            size=8,
            symbol='s',
            pen=pg.mkPen(None)
        )
        self.plot_widget.addItem(self.history_scatter)
        
        # Raster bar (horizontal line sweeping top to bottom)
        self.raster_line = pg.InfiniteLine(
            pos=self.max_range_km,
            angle=0,  # Horizontal
            pen=pg.mkPen(color=(0, 255, 100, 80), width=3)
        )
        self.plot_widget.addItem(self.raster_line)
        
        # Boresight line (center reference)
        self.boresight_line = pg.InfiniteLine(
            pos=0,
            angle=90,  # Vertical
            pen=pg.mkPen(color=(100, 100, 100, 100), width=1, style=Qt.PenStyle.DashLine)
        )
        self.plot_widget.addItem(self.boresight_line)
        
        # Selected target marker
        self.selected_marker = pg.ScatterPlotItem(
            size=20,
            symbol='o',
            pen=pg.mkPen(color=(255, 200, 0), width=3),
            brush=pg.mkBrush(None)
        )
        self.plot_widget.addItem(self.selected_marker)
        
        # ═══ PHASE 23: ECM STROBE ITEMS ═══
        # Create a pool of strobe items (will be shown/hidden as needed)
        self.strobe_pool = []
        for i in range(5):  # Support up to 5 simultaneous jammers
            strobe = pg.LinearRegionItem(
                values=[0, 3],
                brush=pg.mkBrush(255, 255, 0, 60),  # Yellow semi-transparent
                movable=False,
                orientation='vertical'
            )
            strobe.setVisible(False)
            self.plot_widget.addItem(strobe)
            self.strobe_pool.append(strobe)
        
        # Status bar
        self.status_label = QLabel("RANGE: 200 km | TARGETS: 0 | FOV: ±60°")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #00aaff;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 3px;
                background-color: rgba(0, 20, 40, 200);
            }
        """)
        layout.addWidget(self.status_label)
    
    def _update_raster(self):
        """Update raster bar position (visual effect only)."""
        # Move raster bar from top to bottom
        self.raster_position += self.raster_speed * 0.05  # 50ms interval
        if self.raster_position > 1.0:
            self.raster_position = 0.0
        
        # Map position to range (top = max_range, bottom = 0)
        range_pos = self.max_range_km * (1.0 - self.raster_position)
        self.raster_line.setValue(range_pos)
    
    def update_display(self, state: dict):
        """
        Update B-Scope display with simulation state.
        
        Args:
            state: State dictionary from SimulationWorker
        """
        current_time = time.time()
        
        # Frame rate limiting
        if current_time - self._last_frame_time < self._min_update_interval:
            return
        self._last_frame_time = current_time
        
        # Process targets
        targets = state.get('targets', [])
        current_blips = []
        
        az_list = []
        range_list = []
        brushes = []
        
        for target in targets:
            azimuth_deg = target['azimuth_deg']
            range_km = target['range_km']
            
            # Check if within FOV
            if not (self.azimuth_limits[0] <= azimuth_deg <= self.azimuth_limits[1]):
                continue
            
            if not target['is_detected']:
                continue
            
            # Store target data
            self.target_data[target['id']] = target
            
            # Create blip
            blip = BScopeBlip(
                target_id=target['id'],
                azimuth_deg=azimuth_deg,
                range_km=range_km,
                intensity=1.0,
                creation_time=current_time,
                is_detected=True,
                snr_db=target['snr_db'],
                name=target.get('name', '')
            )
            current_blips.append(blip)
            self.blip_history.append(blip)
            
            # Add to display lists
            az_list.append(azimuth_deg)
            range_list.append(range_km)
            
            # MIL-STD-2525D Affiliation-based color
            name_lower = target.get('name', '').lower()
            snr = target['snr_db']
            
            if 'bandit' in name_lower or 'hostile' in name_lower or 'enemy' in name_lower:
                # HOSTILE - Red
                base_color = (255, 68, 68)
            elif 'friendly' in name_lower or 'allied' in name_lower:
                # FRIENDLY - Cyan
                base_color = (0, 191, 255)
            elif 'neutral' in name_lower or 'civilian' in name_lower:
                # NEUTRAL - Green
                base_color = (0, 255, 0)
            else:
                # UNKNOWN - Yellow
                base_color = (255, 255, 0)
            
            # Intensity modulation based on SNR
            alpha = min(230, max(100, int(180 * min(1.0, snr / 20.0))))
            brushes.append(pg.mkBrush(base_color[0], base_color[1], base_color[2], alpha))
        
        self.blips = current_blips
        
        # Update scatter plot
        if az_list:
            self.target_scatter.setData(az_list, range_list, brush=brushes)
        else:
            self.target_scatter.setData([], [])
        
        # Update historical blips (fading effect)
        self._update_history(current_time)
        
        # Update selected target marker
        self._update_selected_marker()
        
        # Update status
        detection_count = state.get('detection_count', 0)
        total_targets = state.get('total_targets', 0)
        
        self.status_label.setText(
            f"RANGE: {self.max_range_km:.0f} km | "
            f"TARGETS: {total_targets} | "
            f"DETECTIONS: {detection_count} | "
            f"FOV: {self.azimuth_limits[0]:.0f}° to {self.azimuth_limits[1]:.0f}°"
        )
        
        # ═══ PHASE 23: ECM STROBE VISUALIZATION ═══
        self._update_ecm_strobes(state)
        
        self.last_update_time = current_time
    
    def _update_ecm_strobes(self, state: dict):
        """
        Update ECM strobe visualization.
        
        When jammers are active, draw vertical 'noise bars' at their
        azimuth positions to show 'burn-through' effect.
        """
        # Get jammer azimuths from state
        jammer_azimuths = state.get('jammer_azimuths', [])
        
        # Also check targets for active jammers
        for target in state.get('targets', []):
            if target.get('jammer_active', False):
                az = target.get('azimuth_deg', 0)
                if az not in jammer_azimuths:
                    jammer_azimuths.append(az)
        
        # Hide all strobes first
        for strobe in self.strobe_pool:
            strobe.setVisible(False)
        
        # Show strobes for active jammers
        strobe_width = 5.0  # degrees width
        for i, az in enumerate(jammer_azimuths[:len(self.strobe_pool)]):
            strobe = self.strobe_pool[i]
            strobe.setRegion([az - strobe_width/2, az + strobe_width/2])
            strobe.setVisible(True)
    
    def _update_history(self, current_time: float):
        """Update historical blip display with fading effect."""
        decay_time = 3.0  # seconds
        
        az_list = []
        range_list = []
        brushes = []
        
        for blip in self.blip_history:
            age = current_time - blip.creation_time
            if age < decay_time:
                intensity = np.exp(-age / (decay_time / 2))
                if intensity > 0.1:
                    az_list.append(blip.azimuth_deg)
                    range_list.append(blip.range_km)
                    alpha = int(intensity * 100)
                    brushes.append(pg.mkBrush(0, 100, 150, alpha))
        
        if az_list:
            self.history_scatter.setData(az_list, range_list, brush=brushes)
        else:
            self.history_scatter.setData([], [])
    
    def _update_selected_marker(self):
        """Update selected target marker position."""
        if self.selected_target_id and self.selected_target_id in self.target_data:
            target = self.target_data[self.selected_target_id]
            self.selected_marker.setData(
                [target['azimuth_deg']], 
                [target['range_km']]
            )
        else:
            self.selected_marker.setData([], [])
    
    def set_azimuth_limits(self, min_az: float, max_az: float):
        """Set azimuth display limits."""
        self.azimuth_limits = (min_az, max_az)
        self.plot_widget.setXRange(min_az, max_az)
    
    def set_max_range(self, range_km: float):
        """Set maximum display range."""
        self.max_range_km = range_km
        self.plot_widget.setYRange(0, range_km)
    
    def select_target(self, target_id: int):
        """Select a target by ID."""
        self.selected_target_id = target_id
        self._update_selected_marker()
