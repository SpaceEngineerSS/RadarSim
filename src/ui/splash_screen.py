"""
Splash Screen for RadarSim

Professional loading screen displayed during application startup.

Features:
    - Dark theme matching main application
    - Progress bar with stage indicators
    - Version and branding display
"""

from PyQt6.QtWidgets import QSplashScreen, QProgressBar, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont, QLinearGradient


class RadarSimSplash(QSplashScreen):
    """
    Professional splash screen for RadarSim.
    
    Displays loading progress with stage messages.
    """
    
    VERSION = "1.0.0"
    
    STAGES = [
        "Initializing Core Systems...",
        "Loading Physics Engine...",
        "Configuring Signal Processing...",
        "Preparing User Interface...",
        "Ready"
    ]
    
    def __init__(self) -> None:
        """Initialize splash screen."""
        # Create splash image
        pixmap = self._create_splash_pixmap()
        super().__init__(pixmap)
        
        self.setWindowFlags(
            Qt.WindowType.SplashScreen | 
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        
        # Progress tracking
        self._current_stage = 0
        self._progress = 0
        
    def _create_splash_pixmap(self) -> QPixmap:
        """Create the splash screen background."""
        width, height = 500, 300
        pixmap = QPixmap(width, height)
        
        # Create painter
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Dark gradient background
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, QColor(5, 20, 15))
        gradient.setColorAt(1, QColor(10, 35, 25))
        painter.fillRect(0, 0, width, height, gradient)
        
        # Border
        painter.setPen(QColor(0, 170, 85))
        painter.drawRect(0, 0, width - 1, height - 1)
        
        # Title
        painter.setPen(QColor(0, 255, 100))
        title_font = QFont('Consolas', 32, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(0, 50, width, 60, Qt.AlignmentFlag.AlignCenter, "RadarSim")
        
        # Subtitle
        painter.setPen(QColor(0, 200, 100, 180))
        sub_font = QFont('Consolas', 11)
        painter.setFont(sub_font)
        painter.drawText(0, 100, width, 30, Qt.AlignmentFlag.AlignCenter, 
                        "Professional Radar Simulation Engine")
        
        # Version
        painter.setPen(QColor(0, 150, 80, 150))
        ver_font = QFont('Consolas', 9)
        painter.setFont(ver_font)
        painter.drawText(0, height - 35, width, 20, Qt.AlignmentFlag.AlignCenter,
                        f"Version {self.VERSION}")
        
        # Decorative radar rings
        painter.setPen(QColor(0, 100, 50, 50))
        center_x, center_y = width // 2, 180
        for r in [40, 60, 80]:
            painter.drawEllipse(center_x - r, center_y - r, r * 2, r * 2)
        
        painter.end()
        return pixmap
    
    def set_progress(self, stage: int, message: str = None) -> None:
        """
        Update progress display.
        
        Args:
            stage: Stage index (0-4)
            message: Optional override message
        """
        self._current_stage = stage
        self._progress = int((stage / len(self.STAGES)) * 100)
        
        msg = message or self.STAGES[min(stage, len(self.STAGES) - 1)]
        self.showMessage(
            f"  {msg}",
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignLeft,
            QColor(0, 200, 100)
        )
        
        # Force repaint
        self.repaint()
    
    def advance_stage(self) -> None:
        """Advance to next loading stage."""
        self._current_stage += 1
        if self._current_stage < len(self.STAGES):
            self.set_progress(self._current_stage)


def show_splash_with_progress(app, callback, delay_ms: int = 400) -> RadarSimSplash:
    """
    Show splash screen and advance through stages.
    
    Args:
        app: QApplication instance
        callback: Function to call when loading complete
        delay_ms: Delay between stages in milliseconds
        
    Returns:
        RadarSimSplash instance
    """
    splash = RadarSimSplash()
    splash.show()
    splash.set_progress(0)
    
    def advance():
        if splash._current_stage < len(RadarSimSplash.STAGES) - 1:
            splash.advance_stage()
            QTimer.singleShot(delay_ms, advance)
        else:
            # Loading complete
            callback()
            splash.close()
    
    QTimer.singleShot(delay_ms, advance)
    return splash
