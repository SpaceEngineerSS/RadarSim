"""
Terrain Model and Line-of-Sight Physics

Implements procedural terrain generation and line-of-sight (LOS) calculations
for realistic radar occlusion by mountains and terrain features.

Features:
    - Procedural terrain using multi-octave noise (Perlin-like)
    - Raymarching LOS algorithm with terrain intersection
    - Earth curvature and refraction effects (4/3 Earth model)
    - Terrain profile extraction for RHI scope

References:
    - Skolnik, "Radar Handbook", 3rd Ed., Chapter 2.12 (Radar Horizon)
    - IEEE Std 686-2008: Radar Definitions
    - ITU-R P.526: Propagation by diffraction
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
import numba

from .constants import EARTH_RADIUS_EFFECTIVE


# =============================================================================
# PROCEDURAL TERRAIN GENERATION
# =============================================================================

@numba.jit(nopython=True, cache=True)
def _noise_2d(x: float, y: float, seed: int = 0) -> float:
    """
    Simple 2D noise function using hash-based pseudo-random.
    
    This is a simplified noise function that produces smooth, reproducible
    terrain-like patterns without requiring external libraries.
    
    Args:
        x: X coordinate
        y: Y coordinate
        seed: Random seed for reproducibility
        
    Returns:
        Noise value in range [-1, 1]
    """
    # Integer coordinates
    xi = int(np.floor(x))
    yi = int(np.floor(y))
    
    # Fractional part
    xf = x - xi
    yf = y - yi
    
    # Smoothstep interpolation
    u = xf * xf * (3 - 2 * xf)
    v = yf * yf * (3 - 2 * yf)
    
    # Hash function for pseudo-random corners
    def hash_2d(ix: int, iy: int) -> float:
        h = (ix * 374761393 + iy * 668265263 + seed) ^ (seed * 1013904223)
        h = ((h >> 13) ^ h) * 1274126177
        return ((h & 0x7fffffff) / 0x7fffffff) * 2 - 1
    
    # Sample four corners
    n00 = hash_2d(xi, yi)
    n10 = hash_2d(xi + 1, yi)
    n01 = hash_2d(xi, yi + 1)
    n11 = hash_2d(xi + 1, yi + 1)
    
    # Bilinear interpolation
    nx0 = n00 * (1 - u) + n10 * u
    nx1 = n01 * (1 - u) + n11 * u
    
    return nx0 * (1 - v) + nx1 * v


@numba.jit(nopython=True, cache=True)
def _fractal_noise(x: float, y: float, octaves: int = 4, 
                   persistence: float = 0.5, lacunarity: float = 2.0,
                   seed: int = 12345) -> float:
    """
    Multi-octave fractal noise for realistic terrain.
    
    Combines multiple noise frequencies to create natural-looking terrain
    with both large-scale features (mountains) and small-scale detail (ridges).
    
    Args:
        x: X coordinate
        y: Y coordinate
        octaves: Number of noise layers (detail levels)
        persistence: Amplitude reduction per octave (0.5 = halve each time)
        lacunarity: Frequency increase per octave (2.0 = double each time)
        seed: Random seed
        
    Returns:
        Combined noise value in range [-1, 1]
    """
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_value = 0.0
    
    for _ in range(octaves):
        total += _noise_2d(x * frequency, y * frequency, seed) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    
    return total / max_value


@numba.jit(nopython=True, cache=True)
def _check_los_raycast(
    radar_x: float, radar_y: float, radar_z: float,
    target_x: float, target_y: float, target_z: float,
    terrain_scale: float, terrain_height_scale: float,
    num_steps: int, seed: int
) -> Tuple[bool, float, float, float]:
    """
    JIT-compiled raymarching LOS check.
    
    Steps along the line from radar to target, checking if the ray
    intersects the terrain at any point.
    
    Args:
        radar_x, radar_y, radar_z: Radar position [m]
        target_x, target_y, target_z: Target position [m]
        terrain_scale: Horizontal scale of terrain features [m]
        terrain_height_scale: Maximum terrain height [m]
        num_steps: Number of ray steps
        seed: Terrain seed
        
    Returns:
        Tuple of (is_visible, block_range, block_x, block_y)
        - is_visible: True if LOS is clear
        - block_range: Range to blocking terrain (if blocked)
        - block_x, block_y: Position of blocking terrain
    """
    # Direction vector
    dx = target_x - radar_x
    dy = target_y - radar_y
    dz = target_z - radar_z
    
    total_dist = np.sqrt(dx*dx + dy*dy + dz*dz)
    if total_dist < 1.0:
        return True, 0.0, 0.0, 0.0
    
    # Normalize
    dx /= total_dist
    dy /= total_dist
    dz /= total_dist
    
    step_size = total_dist / num_steps
    
    for i in range(1, num_steps):
        # Current position along ray
        t = i * step_size
        px = radar_x + dx * t
        py = radar_y + dy * t
        pz = radar_z + dz * t
        
        # Earth curvature correction (4/3 Earth model)
        # Height loss due to curvature: h = d² / (2 * Re)
        earth_curve = (t * t) / (2.0 * EARTH_RADIUS_EFFECTIVE)
        effective_height = pz - earth_curve
        
        # Get terrain height at this point
        noise_val = _fractal_noise(
            px / terrain_scale, 
            py / terrain_scale, 
            octaves=4, 
            persistence=0.5, 
            lacunarity=2.0,
            seed=seed
        )
        # Map noise [-1, 1] to terrain height [0, max_height]
        terrain_height = (noise_val + 1.0) * 0.5 * terrain_height_scale
        
        # Check if ray is below terrain
        if effective_height < terrain_height:
            return False, t, px, py
    
    return True, 0.0, 0.0, 0.0


# =============================================================================
# TERRAIN MAP CLASS
# =============================================================================

@dataclass
class TerrainConfig:
    """
    Terrain generation configuration.
    
    Attributes:
        seed: Random seed for reproducible terrain
        scale: Horizontal scale of terrain features [m] (larger = broader mountains)
        max_height: Maximum terrain height [m]
        octaves: Noise detail levels
        persistence: Amplitude decay per octave
        lacunarity: Frequency increase per octave
    """
    seed: int = 12345
    scale: float = 50000.0  # 50 km feature scale
    max_height: float = 2000.0  # 2 km maximum elevation
    octaves: int = 4
    persistence: float = 0.5
    lacunarity: float = 2.0
    
    # Mountain peaks for specific scenarios
    mountain_peaks: List[Tuple[float, float, float]] = field(default_factory=list)


class TerrainMap:
    """
    Terrain model with procedural generation and LOS physics.
    
    Provides:
    - Procedural terrain height at any (x, y) coordinate
    - Line-of-sight checking with terrain occlusion
    - Terrain profile extraction for visualization
    - Earth curvature effects
    
    Reference: Skolnik, "Radar Handbook", 3rd Ed., Chapter 2.12
    """
    
    def __init__(self, config: Optional[TerrainConfig] = None) -> None:
        """
        Initialize terrain map.
        
        Args:
            config: Terrain configuration (uses defaults if None)
        """
        self.config = config or TerrainConfig()
        
        # Cache for performance
        self._elevation_cache: dict = {}
        self._cache_max_size = 10000
    
    def get_elevation(self, x: float, y: float) -> float:
        """
        Get terrain elevation at coordinates.
        
        Combines procedural noise with explicit mountain peaks
        for scenario-specific terrain features.
        
        Args:
            x: X coordinate [m] (East)
            y: Y coordinate [m] (North)
            
        Returns:
            Terrain elevation [m] above sea level
        """
        # Check cache
        cache_key = (int(x / 100), int(y / 100))  # 100m resolution cache
        if cache_key in self._elevation_cache:
            return self._elevation_cache[cache_key]
        
        # Base procedural terrain
        noise_val = _fractal_noise(
            x / self.config.scale,
            y / self.config.scale,
            octaves=self.config.octaves,
            persistence=self.config.persistence,
            lacunarity=self.config.lacunarity,
            seed=self.config.seed
        )
        
        # Map noise [-1, 1] to height [0, max_height]
        base_height = (noise_val + 1.0) * 0.5 * self.config.max_height
        
        # Add explicit mountain peaks
        peak_contribution = 0.0
        for px, py, peak_height in self.config.mountain_peaks:
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            # Gaussian peak falloff (10 km base radius)
            peak_radius = 10000.0
            peak_contribution += peak_height * np.exp(-(dist**2) / (2 * peak_radius**2))
        
        total_height = base_height + peak_contribution
        
        # Cache result
        if len(self._elevation_cache) < self._cache_max_size:
            self._elevation_cache[cache_key] = total_height
        
        return total_height
    
    def check_line_of_sight(
        self, 
        radar_pos: np.ndarray, 
        target_pos: np.ndarray,
        num_steps: int = 100
    ) -> Tuple[bool, Optional[float], Optional[Tuple[float, float]]]:
        """
        Check if line-of-sight is clear between radar and target.
        
        Uses raymarching algorithm with terrain intersection testing
        and Earth curvature correction (4/3 Earth model).
        
        Args:
            radar_pos: Radar position [x, y, z] in meters
            target_pos: Target position [x, y, z] in meters
            num_steps: Number of ray steps (higher = more accurate)
            
        Returns:
            Tuple of (is_visible, block_range_m, block_position):
            - is_visible: True if LOS is clear
            - block_range_m: Distance to blocking terrain (None if visible)
            - block_position: (x, y) of blocking point (None if visible)
            
        Reference: Skolnik, Chapter 2.12 - Radar Horizon
        """
        radar_pos = np.asarray(radar_pos, dtype=np.float64)
        target_pos = np.asarray(target_pos, dtype=np.float64)
        
        # Ensure 3D
        if len(radar_pos) == 2:
            radar_pos = np.array([radar_pos[0], radar_pos[1], 0.0])
        if len(target_pos) == 2:
            target_pos = np.array([target_pos[0], target_pos[1], 0.0])
        
        is_visible, block_range, block_x, block_y = _check_los_raycast(
            radar_pos[0], radar_pos[1], radar_pos[2],
            target_pos[0], target_pos[1], target_pos[2],
            self.config.scale, self.config.max_height,
            num_steps, self.config.seed
        )
        
        if is_visible:
            return True, None, None
        else:
            return False, block_range, (block_x, block_y)
    
    def get_terrain_profile(
        self, 
        radar_pos: np.ndarray,
        azimuth_rad: float,
        max_range_m: float,
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get terrain elevation profile along a radial from radar.
        
        Used for RHI scope visualization.
        
        Args:
            radar_pos: Radar position [x, y, z] in meters
            azimuth_rad: Azimuth angle [rad] (0 = North, clockwise)
            max_range_m: Maximum range to sample [m]
            num_points: Number of sample points
            
        Returns:
            Tuple of (ranges, elevations):
            - ranges: Array of range values [m]
            - elevations: Array of terrain heights [m]
        """
        ranges = np.linspace(0, max_range_m, num_points)
        elevations = np.zeros(num_points)
        
        # Direction vector
        dx = np.sin(azimuth_rad)
        dy = np.cos(azimuth_rad)
        
        for i, r in enumerate(ranges):
            x = radar_pos[0] + dx * r
            y = radar_pos[1] + dy * r
            elevations[i] = self.get_elevation(x, y)
        
        return ranges, elevations
    
    def get_horizon_range(self, radar_altitude_m: float, target_altitude_m: float = 0.0) -> float:
        """
        Calculate radar horizon range using 4/3 Earth model.
        
        R_horizon = sqrt(2 * Re * h_radar) + sqrt(2 * Re * h_target)
        
        Args:
            radar_altitude_m: Radar altitude above surface [m]
            target_altitude_m: Target altitude above surface [m]
            
        Returns:
            Maximum line-of-sight range [m]
            
        Reference: Skolnik, Eq. 2.76
        """
        # 4/3 Earth radius for standard atmospheric refraction
        Re = EARTH_RADIUS_EFFECTIVE
        
        radar_horizon = np.sqrt(2 * Re * max(0, radar_altitude_m))
        target_horizon = np.sqrt(2 * Re * max(0, target_altitude_m))
        
        return radar_horizon + target_horizon
    
    def clear_cache(self) -> None:
        """Clear elevation cache to free memory."""
        self._elevation_cache.clear()


# =============================================================================
# PRESET TERRAIN CONFIGURATIONS
# =============================================================================

def create_flat_terrain() -> TerrainMap:
    """Create flat terrain (no elevation)."""
    config = TerrainConfig(max_height=0.0)
    return TerrainMap(config)


def create_mountainous_terrain(seed: int = 42) -> TerrainMap:
    """
    Create challenging mountainous terrain.
    
    Features 2-3 km peaks with realistic ridgelines.
    """
    config = TerrainConfig(
        seed=seed,
        scale=30000.0,  # 30 km feature scale
        max_height=3000.0,  # 3 km max
        octaves=5,
        persistence=0.55,
        lacunarity=2.1
    )
    return TerrainMap(config)


def create_ambush_terrain() -> TerrainMap:
    """
    Create terrain for mountain ambush scenario.
    
    High mountain range between radar and target spawn area.
    """
    config = TerrainConfig(
        seed=54321,
        scale=25000.0,
        max_height=1500.0,
        octaves=4,
        persistence=0.5,
        lacunarity=2.0,
        # Add explicit mountain peaks for the ambush
        mountain_peaks=[
            (50000.0, 60000.0, 2500.0),   # 2.5 km peak at 50km E, 60km N
            (45000.0, 55000.0, 2000.0),   # 2 km peak
            (55000.0, 65000.0, 2200.0),   # 2.2 km peak
        ]
    )
    return TerrainMap(config)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_los_physics() -> dict:
    """
    Validate line-of-sight calculations.
    
    Test cases:
    1. High-altitude target over mountains: Should be visible
    2. Low-altitude target behind mountain: Should be blocked
    3. Flat terrain: All targets visible
    
    Returns:
        Dict with validation results
    """
    # Create test terrain
    terrain = create_ambush_terrain()
    
    # Radar at origin, sea level
    radar_pos = np.array([0.0, 0.0, 100.0])  # 100m altitude
    
    # Test 1: High-flying target (10 km altitude)
    high_target = np.array([50000.0, 60000.0, 10000.0])
    visible_high, _, _ = terrain.check_line_of_sight(radar_pos, high_target)
    
    # Test 2: Low target behind mountain (500m altitude)
    low_target = np.array([60000.0, 70000.0, 500.0])
    visible_low, block_range, block_pos = terrain.check_line_of_sight(radar_pos, low_target)
    
    # Test 3: Horizon calculation
    horizon_range = terrain.get_horizon_range(100.0, 100.0)
    
    # Flat terrain test
    flat_terrain = create_flat_terrain()
    visible_flat, _, _ = flat_terrain.check_line_of_sight(radar_pos, low_target)
    
    return {
        'test_cases': {
            'high_altitude_over_mountains': {
                'target_altitude_m': 10000,
                'is_visible': visible_high,
                'expected': True,
            },
            'low_altitude_behind_mountain': {
                'target_altitude_m': 500,
                'is_visible': visible_low,
                'block_range_km': block_range / 1000 if block_range else None,
            },
            'flat_terrain_always_visible': {
                'is_visible': visible_flat,
                'expected': True,
            },
        },
        'horizon_range_km': horizon_range / 1000,
        'validation': {
            'high_target_correct': visible_high == True,
            'flat_terrain_correct': visible_flat == True,
        }
    }
