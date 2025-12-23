"""
Playback Panel Widget

Video player-style controls for replay mode:
- Load File button
- Play/Pause button
- Stop button (reset to start)
- Timeline slider with scrubbing
- Speed control (0.5x, 1x, 2x, 5x)
- Current/Total time display

Signals:
    file_load_requested: Emitted when Load button clicked
    time_changed(float): Emitted when slider moves
    play_state_changed(bool): Emitted when play/pause toggled
    speed_changed(float): Emitted when speed control changes
"""

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton,
    QSlider, QLabel, QComboBox, QFrame, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from typing import Optional


class PlaybackPanel(QWidget):
    """
    Video player-style playback controls for replay mode.
    
    Provides timeline scrubbing, play/pause, and speed control
    with dark military theme matching the rest of the UI.
    """
    
    # Signals
    file_load_requested = pyqtSignal(str)  # filepath
    time_changed = pyqtSignal(float)       # time in seconds
    play_state_changed = pyqtSignal(bool)  # is_playing
    speed_changed = pyqtSignal(float)      # speed multiplier
    stop_requested = pyqtSignal()          # stop and reset
    
    # Speed options
    SPEEDS = [0.25, 0.5, 1.0, 2.0, 5.0]
    
    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        
        self.duration = 0.0
        self.current_time = 0.0
        self.is_playing = False
        self.speed = 1.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_timer_tick)
        self._slider_pressed = False
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Create UI components with dark military theme."""
        self.setFixedHeight(60)
        self.setStyleSheet("""
            QWidget {
                background-color: #0a1510;
            }
        """)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(10)
        
        # Mode label
        self.mode_label = QLabel("REPLAY")
        self.mode_label.setStyleSheet("""
            QLabel {
                color: #ff8800;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
                padding: 3px 8px;
                background-color: rgba(100, 50, 0, 150);
                border: 1px solid #ff8800;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.mode_label)
        
        # Load button
        self.load_btn = QPushButton("📂 LOAD")
        self._style_button(self.load_btn, primary=False)
        self.load_btn.clicked.connect(self._on_load_clicked)
        layout.addWidget(self.load_btn)
        
        # Separator
        layout.addWidget(self._create_separator())
        
        # Play/Pause button
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self._style_button(self.play_btn, primary=True)
        self.play_btn.clicked.connect(self._on_play_pause)
        layout.addWidget(self.play_btn)
        
        # Stop button
        self.stop_btn = QPushButton("■")
        self.stop_btn.setFixedWidth(40)
        self._style_button(self.stop_btn, primary=False)
        self.stop_btn.clicked.connect(self._on_stop)
        layout.addWidget(self.stop_btn)
        
        # Timeline slider
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 1000)  # 0-1000 for smooth scrubbing
        self.timeline.setValue(0)
        self.timeline.sliderMoved.connect(self._on_slider_moved)
        self.timeline.sliderPressed.connect(self._on_slider_pressed)
        self.timeline.sliderReleased.connect(self._on_slider_released)
        self._style_slider(self.timeline)
        layout.addWidget(self.timeline, stretch=1)
        
        # Time display
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("""
            QLabel {
                color: #00ff88;
                font-family: 'Consolas', monospace;
                font-size: 13px;
                font-weight: bold;
                min-width: 100px;
            }
        """)
        layout.addWidget(self.time_label)
        
        # Speed selector
        speed_label = QLabel("SPEED:")
        speed_label.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(speed_label)
        
        self.speed_combo = QComboBox()
        for speed in self.SPEEDS:
            self.speed_combo.addItem(f"{speed}x", speed)
        self.speed_combo.setCurrentIndex(2)  # Default 1.0x
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self._style_combo(self.speed_combo)
        layout.addWidget(self.speed_combo)
    
    def _create_separator(self) -> QFrame:
        """Create a vertical separator line."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet("color: #333333;")
        return sep
    
    def _style_button(self, btn: QPushButton, primary: bool = False):
        """Apply dark theme styling to button."""
        if primary:
            btn.setStyleSheet("""
                QPushButton {
                    color: #00ff88;
                    background-color: #003322;
                    border: 1px solid #00aa55;
                    padding: 6px 12px;
                    font-family: 'Consolas', monospace;
                    font-size: 14px;
                    font-weight: bold;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #004433;
                }
                QPushButton:pressed {
                    background-color: #00aa55;
                    color: #001100;
                }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton {
                    color: #aaaaaa;
                    background-color: #222222;
                    border: 1px solid #444444;
                    padding: 6px 12px;
                    font-family: 'Consolas', monospace;
                    font-size: 12px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #333333;
                    color: #ffffff;
                }
                QPushButton:pressed {
                    background-color: #444444;
                }
            """)
    
    def _style_slider(self, slider: QSlider):
        """Apply dark theme styling to slider."""
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #333333;
                height: 8px;
                background: #1a1a1a;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #ff8800;
                border: 1px solid #aa5500;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:hover {
                background: #ffaa00;
            }
            QSlider::sub-page:horizontal {
                background: #ff6600;
                border-radius: 4px;
            }
        """)
    
    def _style_combo(self, combo: QComboBox):
        """Apply dark theme styling to combo box."""
        combo.setStyleSheet("""
            QComboBox {
                color: #00ff88;
                background-color: #1a1a1a;
                border: 1px solid #333333;
                padding: 4px 8px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                border-radius: 3px;
                min-width: 60px;
            }
            QComboBox:hover {
                border-color: #00aa55;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1a1a;
                color: #00ff88;
                selection-background-color: #003322;
            }
        """)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _update_time_display(self):
        """Update the time label."""
        current = self._format_time(self.current_time)
        total = self._format_time(self.duration)
        self.time_label.setText(f"{current} / {total}")
    
    # Public methods
    
    def set_duration(self, duration: float):
        """Set total duration of recording."""
        self.duration = duration
        self._update_time_display()
    
    def set_time(self, t: float):
        """Set current time without emitting signal."""
        self.current_time = min(t, self.duration)
        if not self._slider_pressed and self.duration > 0:
            slider_value = int((self.current_time / self.duration) * 1000)
            self.timeline.blockSignals(True)
            self.timeline.setValue(slider_value)
            self.timeline.blockSignals(False)
        self._update_time_display()
    
    def set_enabled(self, enabled: bool):
        """Enable or disable playback controls."""
        self.play_btn.setEnabled(enabled)
        self.stop_btn.setEnabled(enabled)
        self.timeline.setEnabled(enabled)
        self.speed_combo.setEnabled(enabled)
    
    # Event handlers
    
    def _on_load_clicked(self):
        """Handle Load button click."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Recording",
            "output",
            "HDF5 Files (*.h5);;All Files (*)"
        )
        if filepath:
            self.file_load_requested.emit(filepath)
    
    def _on_play_pause(self):
        """Handle play/pause button."""
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_btn.setText("❚❚")
            interval = int(33 / self.speed)  # ~30 FPS scaled by speed
            self._timer.start(interval)
        else:
            self.play_btn.setText("▶")
            self._timer.stop()
        
        self.play_state_changed.emit(self.is_playing)
    
    def _on_stop(self):
        """Handle stop button - reset to start."""
        self.is_playing = False
        self.play_btn.setText("▶")
        self._timer.stop()
        self.current_time = 0.0
        self.set_time(0.0)
        self.stop_requested.emit()
        self.time_changed.emit(0.0)
    
    def _on_timer_tick(self):
        """Advance playback time."""
        dt = 0.033 * self.speed  # ~30ms * speed
        self.current_time += dt
        
        if self.current_time >= self.duration:
            self.current_time = self.duration
            self._on_stop()  # Auto-stop at end
            return
        
        self.set_time(self.current_time)
        self.time_changed.emit(self.current_time)
    
    def _on_slider_moved(self, value: int):
        """Handle slider drag."""
        if self.duration > 0:
            self.current_time = (value / 1000) * self.duration
            self._update_time_display()
            self.time_changed.emit(self.current_time)
    
    def _on_slider_pressed(self):
        """Pause playback while scrubbing."""
        self._slider_pressed = True
        self._was_playing = self.is_playing
        if self.is_playing:
            self._timer.stop()
    
    def _on_slider_released(self):
        """Resume playback after scrubbing."""
        self._slider_pressed = False
        if hasattr(self, '_was_playing') and self._was_playing:
            self._timer.start(int(33 / self.speed))
    
    def _on_speed_changed(self, index: int):
        """Handle speed selection."""
        self.speed = self.SPEEDS[index]
        if self.is_playing:
            self._timer.setInterval(int(33 / self.speed))
        self.speed_changed.emit(self.speed)
    
    def stop(self):
        """Stop playback (called externally)."""
        self._timer.stop()
        self.is_playing = False
        self.play_btn.setText("▶")
