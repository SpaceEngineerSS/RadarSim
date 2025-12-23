"""
Raw Video Noise Effects

Adds realistic CRT/phosphor display effects to radar scopes including:
- Gaussian noise overlay (scaled by 1/SNR)
- Phosphor glow effects
- Scan line artifacts

Scientific Basis:
    Real radar displays show noise from:
    - Receiver thermal noise
    - Atmospheric returns
    - Internal electronics
    
    The noise floor is visible as "grass" on A-scopes and
    as speckle on PPI displays.

Reference:
    Skolnik, "Radar Handbook", 3rd Ed., Chapter 2.6 (Receiver Noise)
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import QTimer
from typing import Optional, Tuple
import pyqtgraph as pg


class NoiseOverlay:
    """
    Noise overlay generator for radar displays.
    
    Creates realistic noise effects that scale with:
    - Receiver noise figure
    - Radar type (mechanical = more noise, AESA = cleaner)
    - Display gain setting
    
    Usage:
        overlay = NoiseOverlay(width=400, height=400)
        image_item = overlay.create_image_item()
        plot_widget.addItem(image_item)
        
        # In update loop:
        overlay.update_noise(snr_db=15.0, radar_type="mechanical")
    """
    
    def __init__(
        self,
        width: int = 400,
        height: int = 400,
        base_intensity: float = 0.15
    ) -> None:
        """
        Initialize noise overlay.
        
        Args:
            width: Overlay width in pixels
            height: Overlay height in pixels
            base_intensity: Base noise intensity (0-1)
        """
        self.width = width
        self.height = height
        self.base_intensity = base_intensity
        
        # Noise buffer
        self.noise_buffer = np.zeros((height, width), dtype=np.float32)
        
        # Color map (green phosphor)
        self.colormap = self._create_phosphor_colormap()
        
        # Image item for PyQtGraph
        self.image_item: Optional[pg.ImageItem] = None
        
        # Noise parameters by radar type
        self.noise_profiles = {
            'mechanical': {
                'intensity': 0.25,
                'speckle': 0.4,
                'scan_lines': True,
            },
            'aesa': {
                'intensity': 0.08,
                'speckle': 0.15,
                'scan_lines': False,
            },
            'default': {
                'intensity': 0.15,
                'speckle': 0.25,
                'scan_lines': False,
            }
        }
    
    def _create_phosphor_colormap(self) -> pg.ColorMap:
        """Create green phosphor-like colormap."""
        # Black to green with slight yellow at high intensity
        positions = [0.0, 0.3, 0.6, 1.0]
        colors = [
            (0, 0, 0, 0),           # Black (transparent)
            (0, 40, 20, 50),        # Dark green
            (0, 120, 60, 100),      # Medium green
            (50, 200, 80, 150),     # Bright green
        ]
        return pg.ColorMap(positions, colors)
    
    def create_image_item(self) -> pg.ImageItem:
        """
        Create PyQtGraph ImageItem for overlay.
        
        Returns:
            Configured ImageItem for noise display
        """
        self.image_item = pg.ImageItem()
        self.image_item.setZValue(100)  # Above other items
        self.image_item.setOpacity(0.6)
        
        # Set initial noise
        self._generate_noise()
        self._update_image()
        
        return self.image_item
    
    def _generate_noise(
        self,
        intensity: float = 0.15,
        speckle: float = 0.25,
        scan_lines: bool = False
    ) -> None:
        """
        Generate noise pattern.
        
        Args:
            intensity: Overall noise intensity (0-1)
            speckle: Speckle noise intensity
            scan_lines: Add horizontal scan lines
        """
        # Base Gaussian noise
        noise = np.random.normal(0, intensity, (self.height, self.width))
        
        # Add speckle (multiplicative noise)
        if speckle > 0:
            speckle_noise = np.random.exponential(speckle, (self.height, self.width))
            noise += speckle_noise * 0.3
        
        # Add scan lines for old mechanical radars
        if scan_lines:
            for y in range(0, self.height, 3):
                noise[y, :] *= 0.7
        
        # Clip and normalize
        self.noise_buffer = np.clip(noise, 0, 1).astype(np.float32)
    
    def _update_image(self) -> None:
        """Update the image item with current noise buffer."""
        if self.image_item is None:
            return
        
        # Convert to RGBA using colormap
        rgba = self.colormap.map(self.noise_buffer, mode='byte')
        self.image_item.setImage(rgba)
    
    def update_noise(
        self,
        snr_db: float = 20.0,
        radar_type: str = 'default',
        gain: float = 1.0
    ) -> None:
        """
        Update noise based on current radar state.
        
        Args:
            snr_db: Average SNR [dB] - lower = more noise visible
            radar_type: 'mechanical', 'aesa', or 'default'
            gain: Display gain multiplier
        """
        # Get noise profile
        profile = self.noise_profiles.get(radar_type, self.noise_profiles['default'])
        
        # Scale intensity by inverse SNR
        # At high SNR (>30 dB), noise is barely visible
        # At low SNR (<10 dB), noise dominates
        snr_factor = max(0.1, min(1.0, 20.0 / max(snr_db, 1.0)))
        
        intensity = profile['intensity'] * snr_factor * gain
        speckle = profile['speckle'] * snr_factor
        scan_lines = profile['scan_lines']
        
        self._generate_noise(intensity, speckle, scan_lines)
        self._update_image()
    
    def set_bounds(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float
    ) -> None:
        """
        Set the display bounds for the overlay.
        
        Args:
            x_min, x_max: X-axis bounds
            y_min, y_max: Y-axis bounds
        """
        if self.image_item is None:
            return
        
        # Scale and position the image
        width = x_max - x_min
        height = y_max - y_min
        
        self.image_item.setRect(x_min, y_min, width, height)


class RadialNoiseOverlay(NoiseOverlay):
    """
    Radial noise pattern for PPI displays.
    
    Creates noise that's more intense near the center (close range)
    and fades toward the edges, simulating the 1/R⁴ power falloff.
    """
    
    def __init__(
        self,
        diameter: int = 400,
        base_intensity: float = 0.15
    ) -> None:
        """
        Initialize radial noise overlay.
        
        Args:
            diameter: Display diameter in pixels
            base_intensity: Base noise intensity
        """
        super().__init__(diameter, diameter, base_intensity)
        
        # Create radial distance mask
        center = diameter // 2
        y, x = np.ogrid[:diameter, :diameter]
        self.radial_mask = np.sqrt((x - center)**2 + (y - center)**2) / center
        self.radial_mask = 1 - np.clip(self.radial_mask, 0, 1)  # Inverse: bright at center
    
    def _generate_noise(
        self,
        intensity: float = 0.15,
        speckle: float = 0.25,
        scan_lines: bool = False
    ) -> None:
        """Generate radial noise pattern."""
        # Base Gaussian noise
        noise = np.random.normal(0, intensity, (self.height, self.width))
        
        # Add speckle
        if speckle > 0:
            speckle_noise = np.random.exponential(speckle, (self.height, self.width))
            noise += speckle_noise * 0.2
        
        # Apply radial mask (more noise at edges where SNR is lower)
        # Actually, in radar displays, noise is more visible at far ranges
        # where signal power drops as R⁴
        edge_boost = 1 - self.radial_mask * 0.5  # More noise at edges
        noise *= edge_boost
        
        # Circular mask
        circular_mask = self.radial_mask > 0.02
        noise *= circular_mask
        
        self.noise_buffer = np.clip(noise, 0, 1).astype(np.float32)


class ScanLineEffect:
    """
    Animated scan line effect for the sweep radar look.
    
    Creates a fading line that rotates like a radar sweep,
    leaving a phosphor persistence trail.
    """
    
    def __init__(
        self,
        diameter: int = 400,
        persistence_frames: int = 30
    ) -> None:
        """
        Initialize scan line effect.
        
        Args:
            diameter: Display diameter
            persistence_frames: Frames for phosphor decay
        """
        self.diameter = diameter
        self.persistence = persistence_frames
        
        # Create persistence buffer
        self.buffer = np.zeros((diameter, diameter), dtype=np.float32)
        
        # Current sweep angle
        self.sweep_angle_deg = 0.0
        
        # Decay rate per frame
        self.decay_rate = 1.0 / persistence_frames
    
    def update(self, angle_deg: float) -> np.ndarray:
        """
        Update scan effect with current sweep angle.
        
        Args:
            angle_deg: Current sweep angle [degrees]
            
        Returns:
            Updated intensity buffer
        """
        # Decay existing buffer
        self.buffer *= (1 - self.decay_rate)
        
        # Draw new sweep line
        center = self.diameter // 2
        angle_rad = np.radians(angle_deg)
        
        # Calculate line endpoint
        end_x = int(center + np.sin(angle_rad) * center * 0.95)
        end_y = int(center - np.cos(angle_rad) * center * 0.95)
        
        # Draw line using Bresenham (simplified)
        dx = abs(end_x - center)
        dy = abs(end_y - center)
        steps = max(dx, dy)
        
        if steps > 0:
            x_step = (end_x - center) / steps
            y_step = (end_y - center) / steps
            
            for i in range(steps):
                x = int(center + x_step * i)
                y = int(center + y_step * i)
                if 0 <= x < self.diameter and 0 <= y < self.diameter:
                    self.buffer[y, x] = 1.0
        
        self.sweep_angle_deg = angle_deg
        return self.buffer
