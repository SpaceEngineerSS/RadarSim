"""
Radar Controls Panel

Extracted from main_window.py for modular architecture.
Provides radar parameter adjustments including frequency, power, range, and simulation speed.

Architecture: Single-responsibility component for radar control UI.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QPushButton, QGroupBox, QComboBox
)
from PyQt6.QtCore import Qt
from typing import Optional, Callable, Any


class ControlPanel(QWidget):
    """
    Control panel with radar parameter adjustments.
    
    Provides knobs for:
        - Radar frequency
        - Transmit power
        - Display range
        - Simulation speed
        - Radar architecture presets
    """
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        # Callbacks for value changes
        self._freq_callback: Optional[Callable[[float], None]] = None
        self._power_callback: Optional[Callable[[float], None]] = None
        self._range_callback: Optional[Callable[[float], None]] = None
        self._speed_callback: Optional[Callable[[float], None]] = None
        self._arch_callback: Optional[Callable[[Any], None]] = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup control panel UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Header
        header = QLabel("RADAR CONTROLS")
        header.setStyleSheet("""
            QLabel {
                color: #00ff88;
                font-family: 'Consolas', monospace;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: rgba(0, 40, 20, 200);
                border: 1px solid #00aa55;
            }
        """)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # ═══ RADAR ARCHITECTURE DROPDOWN ═══
        arch_group = self._create_control_group("RADAR ARCHITECTURE")
        arch_layout = QVBoxLayout(arch_group)
        
        self.arch_combo = QComboBox()
        self.arch_combo.addItems([
            "Surveillance (S-Band)",
            "Early Warning (UHF)",
            "Fighter AESA (X-Band)",
            "Missile Seeker (Ka-Band)",
            "Naval Surface (S-Band)"
        ])
        self.arch_combo.setStyleSheet("""
            QComboBox {
                background-color: rgba(0, 40, 20, 200);
                color: #00ff88;
                border: 1px solid #00aa55;
                padding: 5px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: rgb(10, 30, 20);
                color: #00ff88;
                selection-background-color: rgb(0, 80, 40);
            }
        """)
        self.arch_combo.currentTextChanged.connect(self._on_arch_changed)
        arch_layout.addWidget(self.arch_combo)
        
        # Info label showing derived physics
        self.arch_info_label = QLabel("λ=10.0cm | G=37.5dB")
        self.arch_info_label.setStyleSheet("color: #888888; font-size: 10px;")
        self.arch_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        arch_layout.addWidget(self.arch_info_label)
        
        layout.addWidget(arch_group)
        
        # Frequency control
        freq_group = self._create_control_group("FREQUENCY")
        freq_layout = QVBoxLayout(freq_group)
        
        self.freq_label = QLabel("10.0 GHz")
        self.freq_label.setStyleSheet("color: #00dd66; font-size: 16px; font-weight: bold;")
        self.freq_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        freq_layout.addWidget(self.freq_label)
        
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(1, 40)  # 1-40 GHz
        self.freq_slider.setValue(10)
        self.freq_slider.valueChanged.connect(self._on_freq_changed)
        self._style_slider(self.freq_slider)
        freq_layout.addWidget(self.freq_slider)
        
        layout.addWidget(freq_group)
        
        # Power control
        power_group = self._create_control_group("TX POWER")
        power_layout = QVBoxLayout(power_group)
        
        self.power_label = QLabel("100 kW")
        self.power_label.setStyleSheet("color: #00dd66; font-size: 16px; font-weight: bold;")
        self.power_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        power_layout.addWidget(self.power_label)
        
        self.power_slider = QSlider(Qt.Orientation.Horizontal)
        self.power_slider.setRange(10, 500)  # 10-500 kW
        self.power_slider.setValue(100)
        self.power_slider.valueChanged.connect(self._on_power_changed)
        self._style_slider(self.power_slider)
        power_layout.addWidget(self.power_slider)
        
        layout.addWidget(power_group)
        
        # Range control
        range_group = self._create_control_group("MAX RANGE")
        range_layout = QVBoxLayout(range_group)
        
        self.range_label = QLabel("150 km")
        self.range_label.setStyleSheet("color: #00dd66; font-size: 16px; font-weight: bold;")
        self.range_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        range_layout.addWidget(self.range_label)
        
        self.range_slider = QSlider(Qt.Orientation.Horizontal)
        self.range_slider.setRange(25, 500)  # 25-500 km
        self.range_slider.setValue(150)
        self.range_slider.setSingleStep(25)
        self.range_slider.valueChanged.connect(self._on_range_changed)
        self._style_slider(self.range_slider)
        range_layout.addWidget(self.range_slider)
        
        layout.addWidget(range_group)
        
        # Simulation speed
        speed_group = self._create_control_group("SIM SPEED")
        speed_layout = QVBoxLayout(speed_group)
        
        self.speed_label = QLabel("1.0x")
        self.speed_label.setStyleSheet("color: #00dd66; font-size: 16px; font-weight: bold;")
        self.speed_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        speed_layout.addWidget(self.speed_label)
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 50)  # 0.1x to 5x (divide by 10)
        self.speed_slider.setValue(10)  # 1.0x
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        self._style_slider(self.speed_slider)
        speed_layout.addWidget(self.speed_slider)
        
        layout.addWidget(speed_group)
        
        # Playback controls
        playback_group = self._create_control_group("PLAYBACK")
        playback_layout = QHBoxLayout(playback_group)
        
        self.play_btn = QPushButton("▶ PLAY")
        self.play_btn.setCheckable(True)
        self.play_btn.setChecked(True)
        self._style_button(self.play_btn)
        playback_layout.addWidget(self.play_btn)
        
        self.reset_btn = QPushButton("⟲ RESET")
        self._style_button(self.reset_btn)
        playback_layout.addWidget(self.reset_btn)
        
        layout.addWidget(playback_group)
        
        layout.addStretch()
    
    def _create_control_group(self, title: str) -> QGroupBox:
        """Create a styled control group."""
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                color: #00aa66;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                font-weight: bold;
                border: 1px solid #006633;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        return group
    
    def _style_slider(self, slider: QSlider) -> None:
        """Apply dark theme styling to slider."""
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #005533;
                height: 8px;
                background: #002211;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00ff88;
                border: 1px solid #00aa55;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #00ffaa;
            }
        """)
    
    def _style_button(self, button: QPushButton) -> None:
        """Apply dark theme styling to button."""
        button.setStyleSheet("""
            QPushButton {
                color: #00ff88;
                background-color: #003322;
                border: 1px solid #00aa55;
                padding: 8px 15px;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #004433;
            }
            QPushButton:pressed {
                background-color: #00aa55;
                color: #001100;
            }
            QPushButton:checked {
                background-color: #00aa55;
                color: #001100;
            }
        """)
    
    def _on_freq_changed(self, value: int) -> None:
        """Handle frequency slider change."""
        self.freq_label.setText(f"{value:.1f} GHz")
        if self._freq_callback:
            self._freq_callback(value * 1e9)  # Convert GHz to Hz
    
    def _on_power_changed(self, value: int) -> None:
        """Handle power slider change."""
        self.power_label.setText(f"{value} kW")
        if self._power_callback:
            self._power_callback(value * 1000)  # Convert kW to W
    
    def _on_range_changed(self, value: int) -> None:
        """Handle range slider change."""
        self.range_label.setText(f"{value} km")
        if self._range_callback:
            self._range_callback(value)
    
    def _on_speed_changed(self, value: int) -> None:
        """Handle speed slider change."""
        speed = value / 10.0
        self.speed_label.setText(f"{speed:.1f}x")
        if self._speed_callback:
            self._speed_callback(speed)
    
    def set_freq_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for frequency changes."""
        self._freq_callback = callback
    
    def set_power_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for power changes."""
        self._power_callback = callback
    
    def set_range_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for range changes."""
        self._range_callback = callback
    
    def set_speed_callback(self, callback: Callable[[float], None]) -> None:
        """Set callback for speed changes."""
        self._speed_callback = callback
    
    def set_arch_callback(self, callback: Callable[[Any], None]) -> None:
        """Set callback for architecture preset changes."""
        self._arch_callback = callback
    
    def update_from_radar(self, radar: Any) -> None:
        """
        Update control panel sliders from radar parameters.
        
        Called when a new scenario is loaded to sync UI with radar config.
        
        Args:
            radar: Radar object with frequency_hz and power_watts attributes
        """
        try:
            # Update frequency slider (convert Hz to GHz)
            freq_ghz = int(radar.frequency_hz / 1e9)
            freq_ghz = max(1, min(40, freq_ghz))  # Clamp to slider range
            self.freq_slider.setValue(freq_ghz)
            self.freq_label.setText(f"{freq_ghz} GHz")
            
            # Update power slider (convert W to kW)
            power_kw = int(radar.power_watts / 1e3)
            power_kw = max(10, min(500, power_kw))  # Clamp to slider range
            self.power_slider.setValue(power_kw)
            self.power_label.setText(f"{power_kw} kW")
            
            print(f"[CONTROLS] Synced: {freq_ghz} GHz, {power_kw} kW")
        except Exception as e:
            print(f"[CONTROLS] Sync failed: {e}")
    
    def _on_arch_changed(self, preset_name: str) -> None:
        """Handle radar architecture preset change."""
        try:
            # Import here to avoid circular imports
            from src.physics.radar_equation import get_preset
            
            preset = get_preset(preset_name)
            if preset:
                # Update info label with derived physics
                wavelength_cm = preset.wavelength_m * 100
                self.arch_info_label.setText(
                    f"λ={wavelength_cm:.1f}cm | G={preset.gain_db:.1f}dB"
                )
                
                # Update sliders to match preset
                freq_ghz = int(preset.frequency_hz / 1e9)
                self.freq_slider.setValue(max(1, min(40, freq_ghz)))
                self.freq_label.setText(f"{freq_ghz} GHz")
                
                power_kw = int(preset.peak_power_watts / 1e3)
                self.power_slider.setValue(max(10, min(500, power_kw)))
                self.power_label.setText(f"{power_kw} kW")
                
                # Notify callback
                if self._arch_callback:
                    self._arch_callback(preset)
                    
                print(f"[ARCH] Applied: {preset_name} → λ={wavelength_cm:.1f}cm, G={preset.gain_db:.1f}dB")
        except Exception as e:
            print(f"[ARCH] Failed to apply preset: {e}")
