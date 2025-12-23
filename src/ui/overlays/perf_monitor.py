"""
Performance Monitor Overlay

Displays real-time performance metrics (FPS, target count, memory)
as a semi-transparent overlay on radar scopes.
"""

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import time

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PerformanceOverlay(QWidget):
    """
    Semi-transparent performance metrics overlay.
    
    Shows:
        - FPS (Frames Per Second)
        - Target count
        - Memory usage (if psutil available)
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Make overlay transparent and overlay on parent
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        
        # Performance tracking
        self.last_frame_time = time.perf_counter()
        self.frame_times = []
        self.max_frame_samples = 30
        
        # Target count
        self.target_count = 0
        self.detection_count = 0
        
        # Setup UI
        self._setup_ui()
        
        # Update timer (2 Hz)
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_display)
        self.update_timer.start(500)
    
    def _setup_ui(self):
        """Setup the overlay UI."""
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(0, 20, 10, 180);
                border: 1px solid #00aa55;
                border-radius: 5px;
            }
            QLabel {
                color: #00ff88;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                background: transparent;
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        
        # FPS
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        layout.addWidget(self.fps_label)
        
        # Target count
        self.target_label = QLabel("TGT: -- / --")
        layout.addWidget(self.target_label)
        
        # Memory (if available)
        if PSUTIL_AVAILABLE:
            self.mem_label = QLabel("MEM: -- MB")
            layout.addWidget(self.mem_label)
        else:
            self.mem_label = None
        
        # Adjust size
        self.adjustSize()
        self.setFixedSize(self.sizeHint())
    
    def record_frame(self):
        """Record a frame for FPS calculation."""
        current_time = time.perf_counter()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        self.frame_times.append(dt)
        if len(self.frame_times) > self.max_frame_samples:
            self.frame_times.pop(0)
    
    def update_counts(self, target_count: int, detection_count: int = 0):
        """Update target and detection counts."""
        self.target_count = target_count
        self.detection_count = detection_count
    
    def _update_display(self):
        """Update the display labels."""
        # Calculate FPS
        if self.frame_times:
            avg_dt = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_dt if avg_dt > 0 else 0
            self.fps_label.setText(f"FPS: {fps:.1f}")
            
            # Color based on FPS
            if fps >= 25:
                color = "#00ff88"  # Green
            elif fps >= 15:
                color = "#ffaa00"  # Yellow
            else:
                color = "#ff5555"  # Red
            self.fps_label.setStyleSheet(f"color: {color}; background: transparent;")
        
        # Update target count
        self.target_label.setText(f"TGT: {self.target_count} | DET: {self.detection_count}")
        
        # Update memory
        if PSUTIL_AVAILABLE and self.mem_label:
            try:
                process = psutil.Process()
                mem_mb = process.memory_info().rss / (1024 * 1024)
                self.mem_label.setText(f"MEM: {mem_mb:.0f} MB")
            except Exception:
                pass
    
    def position_at(self, x: int, y: int):
        """Position the overlay at specific coordinates."""
        self.move(x, y)
    
    def position_top_left(self, margin: int = 10):
        """Position overlay at parent's top-left corner."""
        self.move(margin, margin)
    
    def position_top_right(self, margin: int = 10):
        """Position overlay at parent's top-right corner."""
        if self.parent():
            x = self.parent().width() - self.width() - margin
            self.move(x, margin)


class PerformanceMonitor:
    """
    Performance monitoring singleton for the application.
    
    Tracks FPS and other metrics across the application.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.last_frame_time = time.perf_counter()
        self.frame_count = 0
        self.fps = 0.0
        self.update_interval = 1.0  # seconds
        self._initialized = True
    
    def tick(self):
        """Call this once per frame."""
        current_time = time.perf_counter()
        self.frame_count += 1
        
        elapsed = current_time - self.last_frame_time
        if elapsed >= self.update_interval:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.last_frame_time = current_time
    
    def get_fps(self) -> float:
        """Get current FPS."""
        return self.fps
