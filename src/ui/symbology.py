"""
MIL-STD-2525 Military Symbology Generator

Provides standardized military symbols for radar displays following
MIL-STD-2525D (Joint Military Symbology) and NATO APP-6D standards.

Symbol Types:
    - HOSTILE: Red Diamond (♦)
    - FRIENDLY: Cyan/Blue Rectangle (■)
    - NEUTRAL: Green Square
    - UNKNOWN: Yellow Circle/Clover

Features:
    - Vector velocity leaders
    - Track history trails
    - Classification labels
    - Altitude indicators

Reference:
    - MIL-STD-2525D: Joint Military Symbology
    - NATO STANAG 2019 (APP-6D)
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF, QFont, QPainterPath
from PyQt6.QtCore import Qt, QPointF
import pyqtgraph as pg


class Affiliation(Enum):
    """Target affiliation categories (MIL-STD-2525D)."""
    HOSTILE = "hostile"      # Red - Enemy
    FRIENDLY = "friendly"    # Blue/Cyan - Allied
    NEUTRAL = "neutral"      # Green - Non-combatant
    UNKNOWN = "unknown"      # Yellow - Unidentified


class SymbolType(Enum):
    """Symbol frame types based on battle dimension."""
    AIR = "air"              # Aircraft, missiles
    GROUND = "ground"        # Vehicles, troops
    SURFACE = "surface"      # Ships
    SUBSURFACE = "subsurface"  # Submarines


@dataclass
class TargetSymbol:
    """
    Target symbol data for rendering.
    
    Attributes:
        x, y: Screen coordinates
        affiliation: HOSTILE/FRIENDLY/NEUTRAL/UNKNOWN
        symbol_type: AIR/GROUND/SURFACE/SUBSURFACE
        heading_deg: Movement direction [degrees]
        speed_mps: Speed for velocity leader [m/s]
        altitude_m: Altitude for label
        track_id: Target ID for label
        classification: Optional classification string (e.g., "F-16")
        is_selected: Highlight when selected
        snr_db: Signal strength for intensity
    """
    x: float
    y: float
    affiliation: Affiliation = Affiliation.UNKNOWN
    symbol_type: SymbolType = SymbolType.AIR
    heading_deg: float = 0.0
    speed_mps: float = 0.0
    altitude_m: float = 0.0
    track_id: int = 0
    classification: str = ""
    is_selected: bool = False
    snr_db: float = 20.0


# =============================================================================
# COLOR DEFINITIONS (MIL-STD-2525D / NATO APP-6D)
# =============================================================================

class SymbolColors:
    """Standard military symbology colors."""
    
    # Primary affiliation colors
    HOSTILE = QColor(255, 68, 68)       # Red (#FF4444)
    FRIENDLY = QColor(0, 191, 255)      # Cyan (#00BFFF)
    NEUTRAL = QColor(0, 255, 0)         # Green (#00FF00)
    UNKNOWN = QColor(255, 255, 0)       # Yellow (#FFFF00)
    
    # Glow/highlight variants (for selection)
    HOSTILE_GLOW = QColor(255, 100, 100, 100)
    FRIENDLY_GLOW = QColor(100, 200, 255, 100)
    NEUTRAL_GLOW = QColor(100, 255, 100, 100)
    UNKNOWN_GLOW = QColor(255, 255, 100, 100)
    
    # Track history (faded)
    HOSTILE_TRAIL = QColor(255, 68, 68, 80)
    FRIENDLY_TRAIL = QColor(0, 191, 255, 80)
    NEUTRAL_TRAIL = QColor(0, 255, 0, 80)
    UNKNOWN_TRAIL = QColor(255, 255, 0, 80)
    
    # Velocity leader
    VELOCITY_LEADER = QColor(255, 255, 255, 180)
    
    @classmethod
    def get_color(cls, affiliation: Affiliation) -> QColor:
        """Get primary color for affiliation."""
        return {
            Affiliation.HOSTILE: cls.HOSTILE,
            Affiliation.FRIENDLY: cls.FRIENDLY,
            Affiliation.NEUTRAL: cls.NEUTRAL,
            Affiliation.UNKNOWN: cls.UNKNOWN,
        }.get(affiliation, cls.UNKNOWN)
    
    @classmethod
    def get_trail_color(cls, affiliation: Affiliation) -> QColor:
        """Get trail color for affiliation."""
        return {
            Affiliation.HOSTILE: cls.HOSTILE_TRAIL,
            Affiliation.FRIENDLY: cls.FRIENDLY_TRAIL,
            Affiliation.NEUTRAL: cls.NEUTRAL_TRAIL,
            Affiliation.UNKNOWN: cls.UNKNOWN_TRAIL,
        }.get(affiliation, cls.UNKNOWN_TRAIL)


# =============================================================================
# SYMBOL GENERATOR
# =============================================================================

class SymbolGenerator:
    """
    MIL-STD-2525D compliant symbol generator.
    
    Generates military standard symbols for radar displays with:
    - Affiliation-based frame shapes
    - Velocity leader lines
    - Track ID labels
    - Altitude indicators
    """
    
    # Symbol sizes
    SYMBOL_SIZE = 12  # Base symbol size in pixels
    LEADER_SCALE = 0.1  # Velocity leader scale (pixels per m/s)
    LEADER_MAX = 60  # Maximum leader length
    
    def __init__(self) -> None:
        """Initialize symbol generator."""
        self.font = QFont("Consolas", 8)
        self.font_bold = QFont("Consolas", 9, QFont.Weight.Bold)
    
    def create_hostile_symbol(self, size: int = SYMBOL_SIZE) -> QPolygonF:
        """
        Create HOSTILE diamond shape (♦).
        
        MIL-STD-2525D: Diamond frame for hostile forces.
        """
        half = size / 2
        return QPolygonF([
            QPointF(0, -half),      # Top
            QPointF(half, 0),       # Right
            QPointF(0, half),       # Bottom
            QPointF(-half, 0),      # Left
        ])
    
    def create_friendly_symbol(self, size: int = SYMBOL_SIZE) -> QPolygonF:
        """
        Create FRIENDLY rectangle shape (■).
        
        MIL-STD-2525D: Rectangle frame for friendly forces.
        """
        half = size / 2
        return QPolygonF([
            QPointF(-half, -half * 0.7),   # Top-left
            QPointF(half, -half * 0.7),    # Top-right
            QPointF(half, half * 0.7),     # Bottom-right
            QPointF(-half, half * 0.7),    # Bottom-left
        ])
    
    def create_neutral_symbol(self, size: int = SYMBOL_SIZE) -> QPolygonF:
        """
        Create NEUTRAL square shape.
        
        MIL-STD-2525D: Square frame for neutral entities.
        """
        half = size / 2
        return QPolygonF([
            QPointF(-half, -half),
            QPointF(half, -half),
            QPointF(half, half),
            QPointF(-half, half),
        ])
    
    def create_unknown_symbol(self, size: int = SYMBOL_SIZE) -> QPolygonF:
        """
        Create UNKNOWN quatrefoil/clover shape.
        
        MIL-STD-2525D: Quatrefoil frame for unknown entities.
        For simplicity, using a circle approximation.
        """
        # Create octagon as quatrefoil approximation
        half = size / 2
        d = half * 0.4  # Corner cut distance
        return QPolygonF([
            QPointF(-half + d, -half),
            QPointF(half - d, -half),
            QPointF(half, -half + d),
            QPointF(half, half - d),
            QPointF(half - d, half),
            QPointF(-half + d, half),
            QPointF(-half, half - d),
            QPointF(-half, -half + d),
        ])
    
    def get_symbol_polygon(
        self, 
        affiliation: Affiliation, 
        size: int = SYMBOL_SIZE
    ) -> QPolygonF:
        """Get appropriate polygon for affiliation."""
        return {
            Affiliation.HOSTILE: self.create_hostile_symbol(size),
            Affiliation.FRIENDLY: self.create_friendly_symbol(size),
            Affiliation.NEUTRAL: self.create_neutral_symbol(size),
            Affiliation.UNKNOWN: self.create_unknown_symbol(size),
        }.get(affiliation, self.create_unknown_symbol(size))
    
    def draw_symbol(
        self, 
        painter: QPainter, 
        symbol: TargetSymbol,
        show_leader: bool = True,
        show_label: bool = True,
        show_altitude: bool = True
    ) -> None:
        """
        Draw a complete MIL-STD-2525D symbol with annotations.
        
        Args:
            painter: QPainter to draw on
            symbol: TargetSymbol data
            show_leader: Draw velocity leader line
            show_label: Draw track ID label
            show_altitude: Draw altitude indicator
        """
        painter.save()
        painter.translate(symbol.x, symbol.y)
        
        color = SymbolColors.get_color(symbol.affiliation)
        
        # Selection glow effect
        if symbol.is_selected:
            glow_pen = QPen(color, 4)
            glow_pen.setColor(QColor(color.red(), color.green(), color.blue(), 100))
            painter.setPen(glow_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            poly = self.get_symbol_polygon(symbol.affiliation, self.SYMBOL_SIZE + 6)
            painter.drawPolygon(poly)
        
        # Velocity leader line
        if show_leader and symbol.speed_mps > 5:
            leader_len = min(symbol.speed_mps * self.LEADER_SCALE, self.LEADER_MAX)
            heading_rad = np.radians(symbol.heading_deg)
            end_x = leader_len * np.sin(heading_rad)
            end_y = -leader_len * np.cos(heading_rad)  # Negative Y is up
            
            leader_pen = QPen(SymbolColors.VELOCITY_LEADER, 1)
            leader_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(leader_pen)
            painter.drawLine(QPointF(0, 0), QPointF(end_x, end_y))
        
        # Main symbol
        pen = QPen(color, 2)
        painter.setPen(pen)
        
        # Intensity based on SNR (brighter = higher SNR)
        alpha = min(255, max(80, int(150 + symbol.snr_db * 3)))
        fill_color = QColor(color.red(), color.green(), color.blue(), alpha // 2)
        painter.setBrush(QBrush(fill_color))
        
        poly = self.get_symbol_polygon(symbol.affiliation, self.SYMBOL_SIZE)
        painter.drawPolygon(poly)
        
        # Track ID label (above symbol)
        if show_label and symbol.track_id > 0:
            painter.setFont(self.font_bold if symbol.is_selected else self.font)
            painter.setPen(color)
            label = f"T{symbol.track_id:02d}"
            painter.drawText(QPointF(-12, -self.SYMBOL_SIZE - 3), label)
        
        # Altitude indicator (below symbol)
        if show_altitude and symbol.altitude_m > 0:
            painter.setFont(self.font)
            painter.setPen(QColor(200, 200, 200, 180))
            alt_label = f"{symbol.altitude_m/1000:.1f}km" if symbol.altitude_m >= 1000 else f"{symbol.altitude_m:.0f}m"
            painter.drawText(QPointF(-15, self.SYMBOL_SIZE + 12), alt_label)
        
        # Classification label (right of symbol)
        if symbol.classification:
            painter.setFont(self.font)
            painter.setPen(QColor(180, 180, 180, 200))
            painter.drawText(QPointF(self.SYMBOL_SIZE + 4, 4), symbol.classification)
        
        painter.restore()


# =============================================================================
# PYQTGRAPH INTEGRATION
# =============================================================================

def create_symbol_scatter_data(
    targets: List[Dict[str, Any]],
    affiliation_map: Optional[Dict[int, Affiliation]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], List[QColor]]:
    """
    Convert target list to PyQtGraph scatter plot data with MIL-STD symbols.
    
    Args:
        targets: List of target dictionaries with 'x', 'y', 'id', etc.
        affiliation_map: Optional mapping of target_id to Affiliation
        
    Returns:
        Tuple of (x_coords, y_coords, symbol_list, color_list)
    """
    if not targets:
        return np.array([]), np.array([]), [], []
    
    x_coords = []
    y_coords = []
    symbols = []
    colors = []
    
    # Symbol mapping for PyQtGraph
    SYMBOL_MAP = {
        Affiliation.HOSTILE: 'd',     # Diamond
        Affiliation.FRIENDLY: 's',    # Square
        Affiliation.NEUTRAL: 's',     # Square
        Affiliation.UNKNOWN: 'o',     # Circle
    }
    
    affiliation_map = affiliation_map or {}
    
    for target in targets:
        x_coords.append(target.get('x', 0))
        y_coords.append(target.get('y', 0))
        
        target_id = target.get('id', 0)
        affiliation = affiliation_map.get(target_id, Affiliation.UNKNOWN)
        
        symbols.append(SYMBOL_MAP.get(affiliation, 'o'))
        colors.append(SymbolColors.get_color(affiliation))
    
    return np.array(x_coords), np.array(y_coords), symbols, colors


def get_affiliation_from_name(name: str) -> Affiliation:
    """
    Infer affiliation from target name using common military terms.
    
    Args:
        name: Target name string
        
    Returns:
        Inferred Affiliation
    """
    name_lower = name.lower()
    
    # Hostile indicators
    if any(term in name_lower for term in ['bandit', 'hostile', 'bogey', 'enemy', 'red']):
        return Affiliation.HOSTILE
    
    # Friendly indicators
    if any(term in name_lower for term in ['friendly', 'allied', 'blue', 'own']):
        return Affiliation.FRIENDLY
    
    # Neutral indicators
    if any(term in name_lower for term in ['neutral', 'civilian', 'commercial']):
        return Affiliation.NEUTRAL
    
    return Affiliation.UNKNOWN
