"""
Plan Position Indicator (PPI) Scope

Polar radar display showing targets in range-azimuth coordinates.

Features:
    - Rotating sweep line synchronized with radar antenna
    - Target blips with phosphor persistence (fade effect)
    - Range rings and azimuth grid lines
    - Military/cyberpunk dark theme aesthetic
    - Interactive target selection (click to select)
    - Jamming strobe visualization

Reference: Skolnik, "Radar Handbook", 3rd Ed., Chapter 7
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QRadialGradient
import pyqtgraph as pg
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque
import time

# MIL-STD-2525 Symbology
try:
    from .symbology import Affiliation, SymbolColors, get_affiliation_from_name
    SYMBOLOGY_AVAILABLE = True
except ImportError:
    SYMBOLOGY_AVAILABLE = False


@dataclass
class TargetBlip:
    """
    Target blip with phosphor persistence data.
    
    Attributes:
        target_id: Unique target identifier
        x: Cartesian X position [m]
        y: Cartesian Y position [m]
        range_km: Range to target [km]
        azimuth_deg: Azimuth angle [degrees]
        intensity: Current phosphor intensity (0-1)
        creation_time: When blip was created
        is_detected: Detection status
        snr_db: Signal-to-noise ratio
        name: Target name (for affiliation detection)
    """
    target_id: int
    x: float
    y: float
    range_km: float
    azimuth_deg: float
    intensity: float
    creation_time: float
    is_detected: bool
    snr_db: float
    name: str = ""


class PPIScope(QWidget):
    """
    Plan Position Indicator (PPI) Radar Display.
    
    Shows targets in polar coordinates with:
    - Rotating sweep line
    - Phosphor persistence effect (blips fade over time)
    - Range rings every 25 km
    - Azimuth markers every 30°
    - Interactive target selection (click on blips)
    - Jamming strobe visualization
    
    Military-grade dark theme.
    """
    
    # Signals
    target_selected = pyqtSignal(int)  # Emits target_id when clicked
    
    # Color scheme (military/cyberpunk) - MIL-STD-2525 compliant
    COLOR_BACKGROUND = QColor(5, 15, 10)
    COLOR_GRID = QColor(0, 60, 30)
    COLOR_SWEEP = QColor(0, 255, 100, 180)
    COLOR_BLIP_DETECTED = QColor(0, 255, 50)
    COLOR_BLIP_MISS = QColor(100, 100, 100)
    COLOR_TEXT = QColor(0, 200, 80)
    COLOR_PHOSPHOR = QColor(0, 255, 100)
    COLOR_SELECTED = QColor(255, 200, 0)  # Yellow for selected target
    COLOR_JAMMING = QColor(255, 50, 50, 120)  # Red for jamming strobes
    
    # MIL-STD-2525D Affiliation colors
    COLOR_HOSTILE = QColor(255, 68, 68)      # Red diamond
    COLOR_FRIENDLY = QColor(0, 191, 255)     # Cyan rectangle
    COLOR_NEUTRAL = QColor(0, 255, 0)        # Green square
    COLOR_UNKNOWN = QColor(255, 255, 0)      # Yellow
    
    def __init__(
        self,
        max_range_km: float = 150.0,
        phosphor_decay_s: float = 5.0,
        parent: QWidget = None
    ):
        """
        Initialize PPI scope.
        
        Args:
            max_range_km: Maximum display range [km]
            phosphor_decay_s: Phosphor decay time constant [s]
            parent: Parent widget
        """
        super().__init__(parent)
        self.max_range_km = max_range_km
        self.phosphor_decay_s = phosphor_decay_s
        
        # State
        self.current_sweep_angle = 0.0  # radians
        self.blips: List[TargetBlip] = []
        self.blip_history: deque = deque(maxlen=500)  # Historical blips for persistence
        self.last_update_time = time.time()
        
        # ═══ PERFORMANCE: Frame rate limiting - REDUCED ═══
        self._min_update_interval = 1.0 / 15.0  # 15 FPS max (was 30)
        self._last_frame_time = 0.0
        
        # Target selection
        self.selected_target_id: Optional[int] = None
        self.target_data: Dict[int, Dict] = {}  # Full target data by ID
        
        # Jamming strobes
        self.jamming_strobes: List[Dict] = []  # [{azimuth_rad, intensity, width}]
        
        # ═══ ADVANCED SCAN MODE SUPPORT ═══
        self.scan_type = "circular"  # "circular", "sector", "stare"
        self.sector_limits = (-60.0, 60.0)  # degrees for sector mode
        self.sweep_direction = 1  # +1 or -1 for sector oscillation
        self.stare_angle_deg = 0.0  # Fixed angle for stare mode
        self.scan_speed_deg_s = 36.0  # degrees per second
        
        # Setup UI
        self._setup_ui()
        
        # Decay timer - SLOWED DOWN for performance
        self._decay_timer = QTimer(self)
        self._decay_timer.timeout.connect(self._decay_phosphor)
        self._decay_timer.start(200)  # 5 Hz decay update (was 20 Hz)
    
    def _setup_ui(self):
        """Setup the UI components."""
        self.setMinimumSize(500, 500)
        self.setStyleSheet(f"background-color: rgb(5, 15, 10);")
        
        # Create pyqtgraph plot widget
        pg.setConfigOptions(antialias=True, background='k')
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Header
        header = QLabel("PPI SCOPE")
        header.setStyleSheet("""
            QLabel {
                color: #00ff64;
                font-family: 'Consolas', monospace;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: rgba(0, 50, 25, 150);
            }
        """)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Create custom plot widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setBackground(QColor(5, 15, 10))
        self.plot_widget.showGrid(False, False)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        
        # Set range
        self.plot_widget.setXRange(-self.max_range_km, self.max_range_km)
        self.plot_widget.setYRange(-self.max_range_km, self.max_range_km)
        
        layout.addWidget(self.plot_widget)
        
        # Draw static elements
        self._draw_grid()
        
        # Create sweep line
        self.sweep_line = pg.PlotCurveItem(
            pen=pg.mkPen(color=(0, 255, 100, 180), width=2)
        )
        self.plot_widget.addItem(self.sweep_line)
        
        # Create scatter plot for blips
        self.blip_scatter = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 255, 50, 200)
        )
        self.plot_widget.addItem(self.blip_scatter)
        
        # Create scatter for historical blips (phosphor)
        self.phosphor_scatter = pg.ScatterPlotItem(
            size=10,
            pen=pg.mkPen(None)
        )
        self.plot_widget.addItem(self.phosphor_scatter)
        
        # ═══ VELOCITY LEADER LINES ═══
        # Lines showing target heading and speed
        self._velocity_leaders: List[pg.PlotCurveItem] = []
        
        # ═══ NOISE OVERLAY (Raw Video Effect) ═══
        # Creates realistic CRT phosphor noise
        self.noise_image = pg.ImageItem()
        self.noise_image.setZValue(-10)  # Behind targets
        self.noise_image.setOpacity(0.3)
        self._noise_data = np.zeros((100, 100), dtype=np.float32)
        self._update_noise()
        self.plot_widget.addItem(self.noise_image)
        
        # Status bar
        self.status_label = QLabel("RANGE: 150 km | TARGETS: 0 | DETECTIONS: 0")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #00cc55;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 3px;
                background-color: rgba(0, 30, 15, 200);
            }
        """)
        layout.addWidget(self.status_label)
        
        # Selected target marker (yellow ring)
        self.selected_marker = pg.ScatterPlotItem(
            size=20,
            symbol='o',
            pen=pg.mkPen(color=(255, 200, 0), width=3),
            brush=pg.mkBrush(None)
        )
        self.plot_widget.addItem(self.selected_marker)
        
        # Initialize strobe items list
        self._strobe_items = []
        
        # ═══ FOV CONE INDICATOR (for SECTOR/STARE modes) ═══
        # Semi-transparent wedge showing radar coverage area
        self.fov_cone = pg.PlotCurveItem(
            pen=pg.mkPen(color=(0, 150, 255, 100), width=2),
            fillLevel=0,
            brush=pg.mkBrush(0, 100, 200, 30)
        )
        self.fov_cone.setVisible(False)  # Hidden by default (CIRCULAR mode)
        self.plot_widget.addItem(self.fov_cone)
        
        # Connect mouse click events (deferred to after scene is ready)
        QTimer.singleShot(100, self._connect_mouse_events)
    
    def _draw_grid(self):
        """Draw range rings and azimuth lines."""
        # Range rings every 25 km
        ring_intervals = [25, 50, 75, 100, 125, 150]
        for r in ring_intervals:
            if r <= self.max_range_km:
                theta = np.linspace(0, 2 * np.pi, 100)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                ring = pg.PlotCurveItem(
                    x, y,
                    pen=pg.mkPen(color=(0, 60, 30, 100), width=1)
                )
                self.plot_widget.addItem(ring)
                
                # Range labels
                label = pg.TextItem(
                    f"{r}km",
                    color=(0, 100, 50),
                    anchor=(0.5, 0.5)
                )
                label.setPos(0, r)
                label.setFont(QFont('Consolas', 8))
                self.plot_widget.addItem(label)
        
        # Azimuth lines every 30°
        for az in range(0, 360, 30):
            az_rad = np.radians(az)
            x = [0, self.max_range_km * np.sin(az_rad)]
            y = [0, self.max_range_km * np.cos(az_rad)]
            line = pg.PlotCurveItem(
                x, y,
                pen=pg.mkPen(color=(0, 50, 25, 80), width=1)
            )
            self.plot_widget.addItem(line)
            
            # Azimuth labels (N, E, S, W and intermediate)
            cardinal = {0: 'N', 90: 'E', 180: 'S', 270: 'W'}
            if az in cardinal:
                text = cardinal[az]
            else:
                text = f"{az}°"
            
            label = pg.TextItem(
                text,
                color=(0, 120, 60),
                anchor=(0.5, 0.5)
            )
            label.setPos(
                (self.max_range_km + 8) * np.sin(az_rad),
                (self.max_range_km + 8) * np.cos(az_rad)
            )
            label.setFont(QFont('Consolas', 9))
            self.plot_widget.addItem(label)
    
    @pyqtSlot(dict)
    def update_display(self, state: Dict[str, Any]):
        """
        Update display with new simulation state.
        
        Throttled to 30 FPS for performance.
        
        Args:
            state: State dictionary from SimulationWorker
        """
        current_time = time.time()
        
        # ═══ PERFORMANCE: Frame rate limiting ═══
        if current_time - self._last_frame_time < self._min_update_interval:
            return  # Skip this frame
        self._last_frame_time = current_time
        
        # ═══ ADVANCED SCAN MODE: Use our own sweep angle calculation ═══
        # Calculate time delta for sweep angle update
        dt = current_time - self.last_update_time if self.last_update_time > 0 else 0.033
        
        # Update sweep angle based on scan mode (CIRCULAR/SECTOR/STARE)
        self.update_sweep_angle(dt)
        
        # Update FOV cone indicator (visible for SECTOR/STARE modes)
        self._update_fov_cone()
        
        # Update sweep line visualization
        self._update_sweep_line()
        
        # Process targets
        targets = state.get('targets', [])
        current_blips = []
        
        for target in targets:
            # Convert polar to Cartesian
            range_km = target['range_km']
            azimuth_rad = target['azimuth_rad']
            
            # Note: Azimuth 0 = North (+Y), 90 = East (+X)
            x = range_km * np.sin(azimuth_rad)
            y = range_km * np.cos(azimuth_rad)
            
            blip = TargetBlip(
                target_id=target['id'],
                x=x,
                y=y,
                range_km=range_km,
                azimuth_deg=target['azimuth_deg'],
                intensity=1.0,
                creation_time=current_time,
                is_detected=target['is_detected'],
                snr_db=target['snr_db'],
                name=target.get('name', '')
            )
            
            if target['is_detected']:
                current_blips.append(blip)
                self.blip_history.append(blip)
        
        self.blips = current_blips
        
        # Update displays
        self._update_blips()
        self._update_phosphor()
        self._update_velocity_leaders(targets)  # Draw heading vectors
        
        # Update status
        detection_count = state.get('detection_count', 0)
        total_targets = state.get('total_targets', 0)
        sim_time = state.get('time', 0)
        
        self.status_label.setText(
            f"RANGE: {self.max_range_km:.0f} km | "
            f"TARGETS: {total_targets} | "
            f"DETECTIONS: {detection_count} | "
            f"TIME: {sim_time:.1f}s"
        )
        
        self.last_update_time = current_time
    
    def _update_sweep_line(self):
        """Update the rotating sweep line."""
        # Convert azimuth to plot coordinates
        # Azimuth 0 = North (+Y), increases clockwise
        x = [0, self.max_range_km * np.sin(self.current_sweep_angle)]
        y = [0, self.max_range_km * np.cos(self.current_sweep_angle)]
        self.sweep_line.setData(x, y)
    
    def _update_blips(self):
        """Update current target blips with MIL-STD-2525 colors."""
        if not self.blips:
            self.blip_scatter.setData([], [])
            self.selected_marker.setData([], [])
            return
        
        x = [b.x for b in self.blips]
        y = [b.y for b in self.blips]
        
        # MIL-STD-2525D Affiliation-based coloring
        brushes = []
        symbols = []
        for blip in self.blips:
            # Determine affiliation from target name
            name_lower = blip.name.lower() if blip.name else ''
            
            if 'bandit' in name_lower or 'hostile' in name_lower or 'enemy' in name_lower:
                # HOSTILE - Red Diamond
                color = (255, 68, 68, 230)
                symbol = 'd'  # diamond
            elif 'friendly' in name_lower or 'allied' in name_lower or 'blue' in name_lower:
                # FRIENDLY - Cyan Rectangle  
                color = (0, 191, 255, 230)
                symbol = 's'  # square
            elif 'neutral' in name_lower or 'civilian' in name_lower:
                # NEUTRAL - Green Square
                color = (0, 255, 0, 200)
                symbol = 's'
            else:
                # UNKNOWN - Yellow (default)
                color = (255, 255, 0, 200)
                symbol = 'o'  # circle
            
            # Intensity modulation based on SNR
            alpha = min(255, max(100, int(color[3] * min(1.0, blip.snr_db / 20.0))))
            brushes.append(pg.mkBrush(color[0], color[1], color[2], alpha))
            symbols.append(symbol)
        
        self.blip_scatter.setData(x, y, brush=brushes, symbol=symbols)
        
        # Update selected target marker
        if self.selected_target_id is not None:
            for blip in self.blips:
                if blip.target_id == self.selected_target_id:
                    self.selected_marker.setData([blip.x], [blip.y])
                    break
            else:
                # Selected target not in current blips
                self.selected_marker.setData([], [])
    
    def _update_phosphor(self):
        """Update phosphor persistence display."""
        current_time = time.time()
        
        # Filter and fade historical blips
        valid_blips = []
        x_list = []
        y_list = []
        brushes = []
        
        for blip in self.blip_history:
            age = current_time - blip.creation_time
            if age < self.phosphor_decay_s:
                # Calculate fade (exponential decay)
                intensity = np.exp(-age / (self.phosphor_decay_s / 3))
                
                if intensity > 0.05:
                    valid_blips.append(blip)
                    x_list.append(blip.x)
                    y_list.append(blip.y)
                    
                    alpha = int(intensity * 150)
                    brushes.append(pg.mkBrush(0, 180, 80, alpha))
        
        if x_list:
            self.phosphor_scatter.setData(x_list, y_list, brush=brushes)
        else:
            self.phosphor_scatter.setData([], [])
    
    def _decay_phosphor(self):
        """Timer callback to update phosphor decay."""
        self._update_phosphor()
        self._update_noise()  # Also update noise
    
    def _update_noise(self):
        """Update radar noise overlay for realistic CRT effect."""
        # Generate Gaussian noise
        size = 100
        noise = np.random.normal(0, 0.15, (size, size)).astype(np.float32)
        
        # Create radial mask (more noise at edges where SNR is lower)
        center = size // 2
        y, x = np.ogrid[:size, :size]
        radial_dist = np.sqrt((x - center)**2 + (y - center)**2) / center
        
        # Circular mask with edge boost
        circular_mask = radial_dist <= 1.0
        edge_boost = 0.5 + 0.5 * radial_dist  # More noise at edges
        
        noise = noise * circular_mask * edge_boost
        noise = np.clip(noise, 0, 1)
        
        # Create green phosphor colormap
        rgba = np.zeros((size, size, 4), dtype=np.uint8)
        rgba[:, :, 1] = (noise * 100).astype(np.uint8)  # Green channel
        rgba[:, :, 3] = (noise * 60).astype(np.uint8)   # Alpha
        
        self.noise_image.setImage(rgba)
        
        # Scale to display range
        r = self.max_range_km
        self.noise_image.setRect(-r, -r, r * 2, r * 2)
    
    def _update_velocity_leaders(self, targets: List[Dict]):
        """Draw velocity leader lines from targets."""
        # Remove old leaders
        for leader in self._velocity_leaders:
            self.plot_widget.removeItem(leader)
        self._velocity_leaders = []
        
        for target in targets:
            if not target.get('is_detected', False):
                continue
            
            # Get target position and velocity
            range_km = target.get('range_km', 0)
            azimuth_rad = target.get('azimuth_rad', 0)
            velocity = target.get('velocity_mps', 0)
            heading_rad = target.get('heading_rad', 0)
            
            if velocity < 10:  # Skip slow targets
                continue
            
            # Target position
            x = range_km * np.sin(azimuth_rad)
            y = range_km * np.cos(azimuth_rad)
            
            # Leader endpoint (scaled by speed)
            leader_len = min(velocity * 0.03, 15)  # Max 15 km leader
            end_x = x + leader_len * np.sin(heading_rad)
            end_y = y + leader_len * np.cos(heading_rad)
            
            # Get color based on affiliation
            name = target.get('name', '').lower()
            if 'bandit' in name or 'hostile' in name:
                color = (255, 68, 68, 150)
            elif 'friendly' in name:
                color = (0, 191, 255, 150)
            else:
                color = (255, 255, 0, 150)
            
            leader = pg.PlotCurveItem(
                [x, end_x], [y, end_y],
                pen=pg.mkPen(color=color, width=1, style=Qt.PenStyle.DashLine)
            )
            self.plot_widget.addItem(leader)
            self._velocity_leaders.append(leader)

    def set_max_range(self, range_km: float):
        """Set maximum display range."""
        self.max_range_km = range_km
        self.plot_widget.setXRange(-range_km, range_km)
        self.plot_widget.setYRange(-range_km, range_km)
    
    def set_scan_mode(self, scan_type: str, sector_limits: tuple = None, 
                      scan_speed_deg_s: float = 36.0, stare_angle_deg: float = 0.0):
        """
        Set radar scan mode for visualization.
        
        Args:
            scan_type: "circular", "sector", or "stare"
            sector_limits: (min_az, max_az) in degrees for sector mode
            scan_speed_deg_s: Antenna scan speed in degrees/second
            stare_angle_deg: Fixed azimuth for stare mode
        """
        self.scan_type = scan_type.lower()
        self.scan_speed_deg_s = scan_speed_deg_s
        self.stare_angle_deg = stare_angle_deg
        
        if sector_limits is not None:
            self.sector_limits = sector_limits
        
        # Reset sweep direction for sector mode
        if self.scan_type == "sector":
            self.sweep_direction = 1
            # Start at sector center
            center = (self.sector_limits[0] + self.sector_limits[1]) / 2
            self.current_sweep_angle = np.radians(center)
        elif self.scan_type == "stare":
            self.current_sweep_angle = np.radians(stare_angle_deg)
        
        print(f"[SCAN] Mode={scan_type}, Speed={scan_speed_deg_s}°/s, Limits={sector_limits}")
    
    def update_sweep_angle(self, dt: float):
        """
        Update sweep angle based on current scan mode.
        
        Called each update cycle to advance the antenna position.
        
        Args:
            dt: Time delta in seconds
            
        Scan Modes:
            CIRCULAR: Continuous 360° rotation
            SECTOR: Oscillating wiper between limits
            STARE: Fixed azimuth (no movement)
        """
        if self.scan_type == "circular":
            # Continuous rotation
            self.current_sweep_angle += np.radians(self.scan_speed_deg_s * dt)
            self.current_sweep_angle = self.current_sweep_angle % (2 * np.pi)
            
        elif self.scan_type == "sector":
            # Oscillating wiper pattern
            current_deg = np.degrees(self.current_sweep_angle)
            new_deg = current_deg + self.scan_speed_deg_s * dt * self.sweep_direction
            
            # Check limits and reverse direction
            if new_deg >= self.sector_limits[1]:
                new_deg = self.sector_limits[1]
                self.sweep_direction = -1
            elif new_deg <= self.sector_limits[0]:
                new_deg = self.sector_limits[0]
                self.sweep_direction = 1
            
            self.current_sweep_angle = np.radians(new_deg)
            
        elif self.scan_type == "stare":
            # Fixed azimuth - no movement
            self.current_sweep_angle = np.radians(self.stare_angle_deg)
    
    def _update_fov_cone(self):
        """
        Update FOV cone indicator for SECTOR/STARE modes.
        
        Draws a semi-transparent wedge showing the radar's coverage area.
        Hidden for CIRCULAR mode (full 360° coverage).
        """
        if self.scan_type == "circular":
            self.fov_cone.setVisible(False)
            return
        
        # Get sector limits
        if self.scan_type == "stare":
            # For STARE, show a narrow cone around stare angle
            beamwidth = 5.0  # degrees - narrow FOV
            min_az = self.stare_angle_deg - beamwidth / 2
            max_az = self.stare_angle_deg + beamwidth / 2
        else:  # sector
            min_az, max_az = self.sector_limits
        
        # Create wedge shape (sector of a circle)
        # Start at origin, arc along the sector, return to origin
        num_points = 30
        angles = np.linspace(np.radians(min_az), np.radians(max_az), num_points)
        
        # Note: Azimuth 0 = North (+Y), 90 = East (+X)
        x_arc = self.max_range_km * np.sin(angles)
        y_arc = self.max_range_km * np.cos(angles)
        
        # Build wedge: origin → arc → origin
        x_wedge = np.concatenate([[0], x_arc, [0]])
        y_wedge = np.concatenate([[0], y_arc, [0]])
        
        self.fov_cone.setData(x_wedge, y_wedge)
        self.fov_cone.setVisible(True)
    
    def wheelEvent(self, event):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()
        if delta > 0:
            # Zoom in
            self.max_range_km = max(25, self.max_range_km - 25)
        else:
            # Zoom out
            self.max_range_km = min(500, self.max_range_km + 25)
        
        self.set_max_range(self.max_range_km)
        event.accept()
    
    def _handle_mouse_click(self, event):
        """
        Handle mouse click for target selection.
        
        Finds the nearest target to click position and emits target_selected signal.
        Selection threshold: 5% of current range (e.g., 7.5 km at 150 km range).
        """
        pos = event.scenePos()
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)
        click_x = mouse_point.x()
        click_y = mouse_point.y()
        
        selection_threshold = self.max_range_km * 0.05  # 5% of range
        
        # Find nearest target
        nearest_id = None
        nearest_dist = float('inf')
        
        for blip in self.blips:
            dist = np.sqrt((blip.x - click_x)**2 + (blip.y - click_y)**2)
            if dist < nearest_dist and dist < selection_threshold:
                nearest_dist = dist
                nearest_id = blip.target_id
        
        if nearest_id is not None:
            self.selected_target_id = nearest_id
            self.target_selected.emit(nearest_id)
            self._update_blips()  # Refresh to show selection
    
    def get_selected_target_data(self) -> Optional[Dict]:
        """
        Get full data for the currently selected target.
        
        Returns:
            Target data dictionary or None if no selection
        """
        if self.selected_target_id is not None:
            return self.target_data.get(self.selected_target_id)
        return None
    
    def draw_jamming_strobes(self, jammers: List[Dict]):
        """
        Draw jamming strobes on the PPI display.
        
        Jamming creates radial noise lines (strobes) at the jammer's bearing.
        
        Args:
            jammers: List of jammer dicts with:
                - azimuth_rad: Bearing to jammer
                - intensity: Jamming intensity (0-1)
                - width_deg: Strobe angular width
        
        Reference: Adamy, "EW 101", Chapter 6
        """
        self.jamming_strobes = jammers
        self._draw_strobes()
    
    def _draw_strobes(self):
        """Internal method to draw jamming strobe lines."""
        # Remove old strobe items
        if hasattr(self, '_strobe_items'):
            for item in self._strobe_items:
                self.plot_widget.removeItem(item)
        self._strobe_items = []
        
        for jammer in self.jamming_strobes:
            az_rad = jammer.get('azimuth_rad', 0)
            intensity = jammer.get('intensity', 0.5)
            width_deg = jammer.get('width_deg', 10)
            
            # Create strobe as a filled polygon (wedge)
            n_points = 20
            width_rad = np.radians(width_deg)
            
            # Create wedge outline
            angles = np.linspace(az_rad - width_rad/2, az_rad + width_rad/2, n_points)
            x_points = [0] + [self.max_range_km * np.sin(a) for a in angles] + [0]
            y_points = [0] + [self.max_range_km * np.cos(a) for a in angles] + [0]
            
            # Red noise strobe with varying alpha based on intensity
            alpha = int(intensity * 100)
            strobe = pg.PlotCurveItem(
                x_points, y_points,
                pen=pg.mkPen(color=(255, 50, 50, alpha), width=1),
                fillLevel=0,
                fillBrush=pg.mkBrush(255, 30, 30, alpha // 2)
            )
            self.plot_widget.addItem(strobe)
            self._strobe_items.append(strobe)
    
    def _connect_mouse_events(self):
        """Connect mouse click events to handler."""
        self.plot_widget.scene().sigMouseClicked.connect(self._handle_mouse_click)
