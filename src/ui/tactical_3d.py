"""
3D Tactical Map Visualization

OpenGL-based 3D tactical display showing terrain, targets, and radar coverage
in a rotatable, zoomable view for enhanced situational awareness.

Features:
    - Terrain surface rendering from TerrainMap
    - 3D target positions with MIL-STD-2525 colors
    - Radar beam/sector volume visualization
    - Orbit camera controls (rotate/pan/zoom)

Reference:
    - Stimson, "Introduction to Airborne Radar", Chapter 12 (Displays)
"""

import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QPushButton
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from typing import Optional, Dict, Any, List, Tuple

# Try to import OpenGL components
try:
    import pyqtgraph.opengl as gl
    from pyqtgraph.opengl import GLViewWidget, GLSurfacePlotItem, GLScatterPlotItem, GLLinePlotItem, GLMeshItem
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("[WARNING] PyQtGraph OpenGL not available - 3D view disabled")


class TacticalMap3D(QWidget):
    """
    3D Tactical Map using PyQtGraph OpenGL.
    
    Displays:
        - Terrain surface (from TerrainMap)
        - Target positions as colored spheres
        - Radar position and beam cone
        - Track history trails
        
    Controls:
        - Left mouse: Rotate view
        - Middle mouse: Pan view
        - Scroll: Zoom in/out
    """
    
    def __init__(
        self,
        max_range_km: float = 150.0,
        max_altitude_km: float = 15.0,
        parent: Optional[QWidget] = None
    ) -> None:
        """
        Initialize 3D tactical map.
        
        Args:
            max_range_km: Maximum display range [km]
            max_altitude_km: Maximum altitude [km]
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.max_range_km = max_range_km
        self.max_altitude_km = max_altitude_km
        
        # Terrain data
        self.terrain_data: Optional[np.ndarray] = None
        self.terrain_item: Optional[GLSurfacePlotItem] = None
        
        # Target visualization
        self.target_scatter: Optional[GLScatterPlotItem] = None
        self.track_lines: Dict[int, GLLinePlotItem] = {}
        
        # Radar visualization
        self.radar_marker: Optional[GLScatterPlotItem] = None
        self.beam_mesh: Optional[GLMeshItem] = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the 3D view UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)
        
        # Header
        header_layout = QHBoxLayout()
        
        title = QLabel("3D TACTICAL MAP")
        title.setStyleSheet("""
            QLabel {
                color: #00ccff;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
            }
        """)
        header_layout.addWidget(title)
        
        # View controls
        self.reset_btn = QPushButton("RESET VIEW")
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #002815;
                color: #00dd66;
                border: 1px solid #00dd66;
                padding: 3px 8px;
                font-family: 'Consolas', monospace;
                font-size: 10px;
            }
            QPushButton:hover {
                background-color: #003825;
            }
        """)
        self.reset_btn.clicked.connect(self._reset_view)
        header_layout.addWidget(self.reset_btn)
        
        header_layout.addStretch()
        layout.addLayout(header_layout)
        
        if OPENGL_AVAILABLE:
            # Create OpenGL view
            self.gl_view = GLViewWidget()
            self.gl_view.setBackgroundColor('#0a1510')
            
            # Set initial camera position
            self.gl_view.setCameraPosition(distance=300, elevation=30, azimuth=45)
            
            layout.addWidget(self.gl_view, stretch=1)
            
            # Initialize scene elements
            self._create_grid()
            self._create_axes()
            self._create_target_scatter()
            self._create_radar_marker()
        else:
            # Fallback message
            fallback = QLabel("OpenGL not available\nInstall: pip install PyOpenGL")
            fallback.setStyleSheet("""
                QLabel {
                    color: #ff6666;
                    font-family: 'Consolas', monospace;
                    font-size: 14px;
                    padding: 50px;
                }
            """)
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(fallback, stretch=1)
    
    def _create_grid(self) -> None:
        """Create ground grid."""
        if not OPENGL_AVAILABLE:
            return
        
        # Create a grid at Z=0
        grid = gl.GLGridItem()
        grid.setSize(self.max_range_km * 2, self.max_range_km * 2)
        grid.setSpacing(25, 25)  # 25 km grid spacing
        grid.setColor((0.2, 0.4, 0.3, 0.5))
        self.gl_view.addItem(grid)
    
    def _create_axes(self) -> None:
        """Create coordinate axes."""
        if not OPENGL_AVAILABLE:
            return
        
        axis_len = 50  # km
        
        # X-axis (East) - Red
        x_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [axis_len, 0, 0]]),
            color=(1, 0.3, 0.3, 0.8),
            width=2
        )
        self.gl_view.addItem(x_axis)
        
        # Y-axis (North) - Green
        y_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, axis_len, 0]]),
            color=(0.3, 1, 0.3, 0.8),
            width=2
        )
        self.gl_view.addItem(y_axis)
        
        # Z-axis (Up) - Blue
        z_axis = gl.GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, axis_len / 5]]),
            color=(0.3, 0.5, 1, 0.8),
            width=2
        )
        self.gl_view.addItem(z_axis)
    
    def _create_target_scatter(self) -> None:
        """Create target scatter plot item."""
        if not OPENGL_AVAILABLE:
            return
        
        self.target_scatter = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3)),
            size=10,
            color=(1, 0.3, 0.3, 0.9),
            pxMode=True
        )
        self.gl_view.addItem(self.target_scatter)
    
    def _create_radar_marker(self) -> None:
        """Create radar position marker."""
        if not OPENGL_AVAILABLE:
            return
        
        self.radar_marker = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0.1]]),  # Small elevation above ground
            size=20,
            color=(0, 1, 0.5, 1.0),
            pxMode=True
        )
        self.gl_view.addItem(self.radar_marker)
    
    def set_terrain(
        self, 
        x_grid: np.ndarray, 
        y_grid: np.ndarray, 
        elevation: np.ndarray
    ) -> None:
        """
        Set terrain surface data.
        
        Args:
            x_grid: X coordinates [km] (meshgrid format)
            y_grid: Y coordinates [km] (meshgrid format)
            elevation: Elevation values [km] (2D array)
        """
        if not OPENGL_AVAILABLE:
            return
        
        # Remove existing terrain
        if self.terrain_item is not None:
            self.gl_view.removeItem(self.terrain_item)
        
        # Create surface plot
        # Convert to the format expected by GLSurfacePlotItem
        self.terrain_item = gl.GLSurfacePlotItem(
            x=x_grid[0, :],  # 1D array of x values
            y=y_grid[:, 0],  # 1D array of y values
            z=elevation,
            shader='shaded',
            color=(0.2, 0.4, 0.3, 0.8)
        )
        self.gl_view.addItem(self.terrain_item)
    
    def set_terrain_from_map(
        self, 
        terrain_map,
        resolution: int = 50
    ) -> None:
        """
        Generate terrain surface from TerrainMap object.
        
        Args:
            terrain_map: TerrainMap instance from physics module
            resolution: Grid resolution (points per axis)
        """
        if not OPENGL_AVAILABLE or terrain_map is None:
            return
        
        # Generate grid covering the display area
        x_range = np.linspace(-self.max_range_km, self.max_range_km, resolution)
        y_range = np.linspace(-self.max_range_km, self.max_range_km, resolution)
        
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        elevation = np.zeros_like(x_grid)
        
        # Sample terrain elevation (convert km to m for terrain query)
        for i in range(resolution):
            for j in range(resolution):
                x_m = x_grid[i, j] * 1000
                y_m = y_grid[i, j] * 1000
                elevation[i, j] = terrain_map.get_elevation(x_m, y_m) / 1000  # Convert m to km
        
        self.set_terrain(x_grid, y_grid, elevation)
    
    def update_display(self, state: Dict[str, Any]) -> None:
        """
        Update 3D display with simulation state.
        
        Args:
            state: State dictionary containing targets, radar position, etc.
        """
        if not OPENGL_AVAILABLE:
            return
        
        targets = state.get('targets', [])
        
        if not targets:
            self.target_scatter.setData(pos=np.zeros((1, 3)), size=0)
            return
        
        # Build target positions and colors
        positions = []
        colors = []
        sizes = []
        
        for target in targets:
            # Get position (convert m to km)
            x_km = target.get('x', target.get('position', [0, 0, 0])[0] if isinstance(target.get('position'), list) else 0) / 1000
            y_km = target.get('y', target.get('position', [0, 0, 0])[1] if isinstance(target.get('position'), list) else 0) / 1000
            z_km = target.get('altitude_m', target.get('z', 0)) / 1000
            
            # Handle position arrays
            if 'position' in target and isinstance(target['position'], (list, np.ndarray)):
                pos = target['position']
                x_km = pos[0] / 1000
                y_km = pos[1] / 1000
                z_km = pos[2] / 1000 if len(pos) > 2 else 0
            
            positions.append([x_km, y_km, z_km])
            
            # Color based on detection/affiliation
            is_detected = target.get('is_detected', True)
            is_false = target.get('is_false_target', False)
            name = target.get('name', '').lower()
            
            if not is_detected:
                colors.append((0.5, 0.5, 0.5, 0.3))  # Gray for undetected
            elif is_false:
                colors.append((1, 0.5, 0, 0.8))  # Orange for false targets
            elif 'bandit' in name or 'hostile' in name:
                colors.append((1, 0.3, 0.3, 0.9))  # Red for hostile
            elif 'friendly' in name:
                colors.append((0, 0.7, 1, 0.9))  # Cyan for friendly
            else:
                colors.append((1, 1, 0, 0.9))  # Yellow for unknown
            
            # Size based on RCS or detection
            rcs = target.get('rcs_m2', 5.0)
            base_size = 8 + min(rcs, 20)
            sizes.append(base_size)
        
        # Update scatter plot
        if positions:
            pos_array = np.array(positions)
            color_array = np.array(colors)
            size_array = np.array(sizes)
            
            self.target_scatter.setData(
                pos=pos_array,
                size=size_array,
                color=color_array
            )
    
    def set_radar_position(self, x_km: float, y_km: float, z_km: float = 0.1) -> None:
        """Set radar position marker."""
        if not OPENGL_AVAILABLE or self.radar_marker is None:
            return
        
        self.radar_marker.setData(pos=np.array([[x_km, y_km, z_km]]))
    
    def _reset_view(self) -> None:
        """Reset camera to default position."""
        if OPENGL_AVAILABLE:
            self.gl_view.setCameraPosition(distance=300, elevation=30, azimuth=45)
    
    def set_camera_position(
        self, 
        distance: float = 300, 
        elevation: float = 30, 
        azimuth: float = 45
    ) -> None:
        """
        Set camera position.
        
        Args:
            distance: Distance from center [km]
            elevation: Elevation angle [degrees]
            azimuth: Azimuth angle [degrees]
        """
        if OPENGL_AVAILABLE:
            self.gl_view.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)


def create_demo_terrain() -> np.ndarray:
    """
    Create demo terrain data for testing.
    
    Returns:
        Tuple of (x_grid, y_grid, elevation) in km
    """
    resolution = 50
    x = np.linspace(-150, 150, resolution)
    y = np.linspace(-150, 150, resolution)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Create some simple peaks
    def gaussian_peak(x, y, cx, cy, height, width):
        return height * np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * width**2))
    
    elevation = np.zeros_like(x_grid)
    
    # Add mountain peaks
    elevation += gaussian_peak(x_grid, y_grid, 50, 60, 3, 20)
    elevation += gaussian_peak(x_grid, y_grid, 45, 55, 2.5, 15)
    elevation += gaussian_peak(x_grid, y_grid, 55, 65, 2.8, 18)
    elevation += gaussian_peak(x_grid, y_grid, -30, 40, 1.5, 25)
    
    # Add some noise
    elevation += 0.2 * np.random.random(elevation.shape)
    
    return x_grid, y_grid, elevation
