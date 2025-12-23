"""
Range-Height Indicator (RHI) Scope

Displays radar data in Range vs Altitude format, showing terrain profile
and target elevation for improved situational awareness.

Scientific Basis:
    - RHI is used for elevation scanning in meteorological and military radars
    - Shows vertical cross-section along a selected azimuth bearing
    - Essential for terrain-following radar and altitude verification

Reference: Stimson, "Introduction to Airborne Radar", Chapter 10
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import pyqtgraph as pg
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class RHIBlip:
    """Target blip data for RHI display."""
    target_id: int
    range_km: float
    altitude_m: float
    intensity: float
    creation_time: float
    is_detected: bool
    is_masked: bool = False  # Terrain masked


class RHIScope(QWidget):
    """
    Range-Height Indicator (RHI) Scope.
    
    Displays targets in Range vs Altitude format with terrain profile overlay.
    
    Features:
        - X-axis: Range (0 to max_range_km)
        - Y-axis: Altitude (0 to max_altitude_m)
        - Terrain profile cross-section at selected azimuth
        - Target markers showing actual altitude
        - Terrain-masked targets shown differently
        
    Reference: Stimson, "Introduction to Airborne Radar", Chapter 10
    """
    
    def __init__(
        self, 
        max_range_km: float = 150.0,
        max_altitude_m: float = 15000.0,
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize RHI scope.
        
        Args:
            max_range_km: Maximum display range [km]
            max_altitude_m: Maximum display altitude [m]
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.max_range_km = max_range_km
        self.max_altitude_m = max_altitude_m
        self.current_azimuth_deg: float = 0.0
        
        # Target history for persistence
        self.blip_history: deque = deque(maxlen=100)
        self.current_blips: Dict[int, RHIBlip] = {}
        
        # Terrain profile data
        self.terrain_ranges: Optional[np.ndarray] = None
        self.terrain_elevations: Optional[np.ndarray] = None
        
        # Radar position (for horizon calculation)
        self.radar_altitude_m: float = 100.0
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the RHI scope UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("RHI SCOPE")
        title.setStyleSheet("""
            QLabel {
                color: #00ccff;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        header_layout.addWidget(title)
        
        self.azimuth_label = QLabel("AZ: 000°")
        self.azimuth_label.setStyleSheet("""
            QLabel {
                color: #00ff88;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        header_layout.addWidget(self.azimuth_label)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        # Create PyQtGraph plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground('#0a1510')
        self.plot_widget.setAspectLocked(False)
        
        # Configure axes
        self.plot_widget.setLabel('bottom', 'Range', units='km')
        self.plot_widget.setLabel('left', 'Altitude', units='m')
        self.plot_widget.setXRange(0, self.max_range_km)
        self.plot_widget.setYRange(0, self.max_altitude_m)
        
        # Grid styling
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        # Style axes
        axis_style = {'color': '#00aa55', 'font-size': '10pt'}
        self.plot_widget.getAxis('bottom').setStyle(tickFont=QFont('Consolas', 9))
        self.plot_widget.getAxis('left').setStyle(tickFont=QFont('Consolas', 9))
        self.plot_widget.getAxis('bottom').setPen(pg.mkPen('#00aa55', width=1))
        self.plot_widget.getAxis('left').setPen(pg.mkPen('#00aa55', width=1))
        self.plot_widget.getAxis('bottom').setTextPen('#00aa55')
        self.plot_widget.getAxis('left').setTextPen('#00aa55')
        
        layout.addWidget(self.plot_widget, stretch=1)
        
        # Create plot items
        self._create_plot_items()
        
        # Draw horizon line
        self._draw_horizon()
    
    def _create_plot_items(self) -> None:
        """Create the plot items for terrain and targets."""
        # Create initial empty curves for terrain fill
        initial_x = np.array([0, self.max_range_km])
        initial_y = np.zeros(2)
        self._terrain_curve1 = pg.PlotCurveItem(x=initial_x, y=initial_y)
        self._terrain_curve2 = pg.PlotCurveItem(x=initial_x, y=initial_y)
        
        # Terrain fill (mountain silhouette)
        self.terrain_fill = pg.FillBetweenItem(
            curve1=self._terrain_curve1,
            curve2=self._terrain_curve2,
            brush=pg.mkBrush('#2a3530')  # Dark terrain color
        )
        self.plot_widget.addItem(self.terrain_fill)
        
        # Terrain outline
        self.terrain_curve = pg.PlotCurveItem(
            pen=pg.mkPen('#4a6560', width=2)
        )
        self.plot_widget.addItem(self.terrain_curve)
        
        # Target scatter plot (detected)
        self.target_scatter = pg.ScatterPlotItem(
            size=12, 
            pen=pg.mkPen('#00ff88', width=2),
            brush=pg.mkBrush('#00ff8880')
        )
        self.plot_widget.addItem(self.target_scatter)
        
        # Masked target scatter (behind terrain)
        self.masked_scatter = pg.ScatterPlotItem(
            size=10, 
            pen=pg.mkPen('#ff555580', width=1),
            brush=pg.mkBrush('#ff555540'),
            symbol='x'
        )
        self.plot_widget.addItem(self.masked_scatter)
        
        # Radar position marker
        self.radar_marker = pg.ScatterPlotItem(
            pos=[(0, self.radar_altitude_m)],
            size=15,
            pen=pg.mkPen('#ffcc00', width=2),
            brush=pg.mkBrush('#ffcc0080'),
            symbol='t'  # Triangle
        )
        self.plot_widget.addItem(self.radar_marker)
        
        # Selected target highlight
        self.selected_marker = pg.TargetItem(
            pos=(0, 0),
            size=20,
            pen=pg.mkPen('#ffcc00', width=2)
        )
        self.selected_marker.setVisible(False)
        self.plot_widget.addItem(self.selected_marker)
    
    def _draw_horizon(self) -> None:
        """Draw radar horizon line."""
        # Horizon line (simplified - actual horizon depends on target altitude)
        from src.physics.constants import EARTH_RADIUS_EFFECTIVE
        
        # Calculate horizon for various altitudes
        horizon_x = []
        horizon_y = []
        
        for alt in np.linspace(0, self.max_altitude_m, 50):
            # Range to horizon for target at this altitude
            r_radar = np.sqrt(2 * EARTH_RADIUS_EFFECTIVE * self.radar_altitude_m)
            r_target = np.sqrt(2 * EARTH_RADIUS_EFFECTIVE * alt)
            horizon_range_m = r_radar + r_target
            horizon_km = horizon_range_m / 1000
            
            if horizon_km <= self.max_range_km:
                horizon_x.append(horizon_km)
                horizon_y.append(alt)
        
        if horizon_x:
            self.horizon_curve = pg.PlotCurveItem(
                x=horizon_x, y=horizon_y,
                pen=pg.mkPen('#ff880060', width=1, style=Qt.PenStyle.DashLine)
            )
            self.plot_widget.addItem(self.horizon_curve)
    
    def set_azimuth(self, azimuth_deg: float) -> None:
        """
        Set current azimuth for RHI slice.
        
        Args:
            azimuth_deg: Azimuth angle in degrees (0 = North)
        """
        self.current_azimuth_deg = azimuth_deg
        self.azimuth_label.setText(f"AZ: {azimuth_deg:03.0f}°")
    
    def set_terrain_profile(
        self, 
        ranges_km: np.ndarray, 
        elevations_m: np.ndarray
    ) -> None:
        """
        Set terrain profile for current azimuth.
        
        Args:
            ranges_km: Array of range values [km]
            elevations_m: Array of terrain heights [m]
        """
        self.terrain_ranges = ranges_km
        self.terrain_elevations = elevations_m
        
        # Update terrain curve
        self.terrain_curve.setData(x=ranges_km, y=elevations_m)
        
        # Update terrain fill
        # Create a curve that fills down to zero
        fill_x = np.concatenate([[ranges_km[0]], ranges_km, [ranges_km[-1]]])
        fill_y_top = np.concatenate([[0], elevations_m, [0]])
        fill_y_bottom = np.zeros_like(fill_x)
        
        curve1 = pg.PlotCurveItem(x=fill_x, y=fill_y_top)
        curve2 = pg.PlotCurveItem(x=fill_x, y=fill_y_bottom)
        self.terrain_fill.setCurves(curve1, curve2)
    
    def set_radar_altitude(self, altitude_m: float) -> None:
        """
        Set radar altitude.
        
        Args:
            altitude_m: Radar altitude [m]
        """
        self.radar_altitude_m = altitude_m
        self.radar_marker.setData(pos=[(0, altitude_m)])
    
    def update_display(self, state: Dict[str, Any]) -> None:
        """
        Update RHI display with simulation state.
        
        Args:
            state: State dictionary containing targets and terrain data
        """
        targets = state.get('targets', [])
        current_time = state.get('time', 0)
        
        # Filter targets near current azimuth (±10° sector)
        detected_ranges = []
        detected_alts = []
        masked_ranges = []
        masked_alts = []
        
        for target in targets:
            # Check if target is in RHI azimuth slice
            target_az = target.get('azimuth_deg', 0)
            az_diff = abs(target_az - self.current_azimuth_deg)
            if az_diff > 180:
                az_diff = 360 - az_diff
            
            # Only show targets within ±15° of current azimuth
            if az_diff <= 15:
                range_km = target.get('range_km', 0)
                altitude_m = target.get('altitude_m', target.get('z', 0))
                is_masked = target.get('terrain_masked', False)
                is_detected = target.get('is_detected', True)
                
                if is_masked:
                    masked_ranges.append(range_km)
                    masked_alts.append(altitude_m)
                elif is_detected:
                    detected_ranges.append(range_km)
                    detected_alts.append(altitude_m)
        
        # Update scatter plots
        if detected_ranges:
            self.target_scatter.setData(
                x=detected_ranges, 
                y=detected_alts
            )
        else:
            self.target_scatter.setData(x=[], y=[])
        
        if masked_ranges:
            self.masked_scatter.setData(
                x=masked_ranges,
                y=masked_alts
            )
        else:
            self.masked_scatter.setData(x=[], y=[])
        
        # Update terrain profile if provided
        terrain_data = state.get('terrain_profile', None)
        if terrain_data:
            self.set_terrain_profile(
                terrain_data['ranges_km'],
                terrain_data['elevations_m']
            )
    
    def set_max_range(self, range_km: float) -> None:
        """Set maximum display range."""
        self.max_range_km = range_km
        self.plot_widget.setXRange(0, range_km)
    
    def set_max_altitude(self, altitude_m: float) -> None:
        """Set maximum display altitude."""
        self.max_altitude_m = altitude_m
        self.plot_widget.setYRange(0, altitude_m)
    
    def highlight_target(self, range_km: float, altitude_m: float) -> None:
        """
        Highlight a selected target.
        
        Args:
            range_km: Target range [km]
            altitude_m: Target altitude [m]
        """
        self.selected_marker.setPos(range_km, altitude_m)
        self.selected_marker.setVisible(True)
    
    def clear_highlight(self) -> None:
        """Clear target highlight."""
        self.selected_marker.setVisible(False)
    
    def set_terrain(self, terrain_map) -> None:
        """
        Set terrain from TerrainMap object.
        
        Generates a profile along north bearing (azimuth 0°).
        
        Args:
            terrain_map: TerrainMap instance from physics module
        """
        if terrain_map is None:
            return
        
        # Generate profile along north bearing (azimuth = 0)
        num_points = 100
        ranges_km = np.linspace(0, self.max_range_km, num_points)
        elevations_m = []
        
        for r_km in ranges_km:
            # Convert range to meters, sample at y = range (north)
            r_m = r_km * 1000
            try:
                elev = terrain_map.get_elevation(0, r_m)  # x=0, y=range (north)
                elevations_m.append(elev)
            except Exception:
                elevations_m.append(0)
        
        # Set the profile
        self.set_terrain_profile(ranges_km, np.array(elevations_m))
        print(f"[RHI] Terrain profile set: {num_points} points")

