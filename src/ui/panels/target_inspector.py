"""
Target Inspector Panel

Extracted from main_window.py for modular architecture.
Displays detailed information about selected targets including AI classification and ECM controls.

Architecture: Single-responsibility component for target inspection UI.
"""

import time
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QPushButton, QFrame, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import Optional, Dict, Any

# AI/ML Classification
try:
    from src.ml.inference_engine import InferenceEngine
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False


class TargetInspector(QWidget):
    """
    Target Inspector Panel.
    
    Displays detailed information about the selected target:
        - Name/ID
        - Range (km)
        - Azimuth (deg)
        - Velocity (m/s)
        - RCS (m²)
        - SNR (dB)
        - Probability of Detection (Pd)
        - AI Classification
        - ECM Controls
    """
    
    # Signal emitted when ECM state changes (active, ecm_type, target_id)
    ecm_changed = pyqtSignal(bool, str, int)
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        
        self._current_target_id: Optional[int] = None
        self._ecm_active: bool = False
        
        # AI Classification throttling
        self._last_ai_classification_time: float = 0.0
        self._ai_throttle_interval: float = 5.0  # Run AI every 5 seconds max
        
        # Initialize inference engine
        self._inference_engine: Optional[Any] = None
        if ML_AVAILABLE:
            try:
                self._inference_engine = InferenceEngine()
            except Exception:
                pass
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Setup the inspector UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        
        # Header
        header = QLabel("TARGET INSPECTOR")
        header.setStyleSheet("""
            QLabel {
                color: #ffcc00;
                font-family: 'Consolas', monospace;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: rgba(50, 40, 0, 200);
                border: 1px solid #aa8800;
            }
        """)
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)
        
        # Create info labels
        self.labels: Dict[str, QLabel] = {}
        fields = [
            ("ID", "---"),
            ("TYPE", "---"),
            ("RANGE", "--- km"),
            ("AZIMUTH", "--- °"),
            ("VELOCITY", "--- m/s"),
            ("RCS", "--- m²"),
            ("SNR", "--- dB"),
            ("Pd", "---%"),
        ]
        
        for name, default in fields:
            row = QFrame()
            row.setStyleSheet("""
                QFrame {
                    background-color: rgba(30, 25, 0, 150);
                    border: 1px solid #665500;
                    border-radius: 3px;
                    padding: 2px;
                }
            """)
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(5, 3, 5, 3)
            
            name_label = QLabel(name)
            name_label.setStyleSheet("color: #aa8800; font-size: 11px;")
            row_layout.addWidget(name_label)
            
            value_label = QLabel(default)
            value_label.setStyleSheet("color: #ffcc00; font-size: 12px; font-weight: bold;")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            row_layout.addWidget(value_label)
            
            layout.addWidget(row)
            self.labels[name] = value_label
        
        # Jammer indicator
        self.jammer_label = QLabel("🔇 NO JAMMER")
        self.jammer_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                padding: 5px;
                background-color: rgba(30, 30, 30, 200);
                border: 1px solid #555555;
            }
        """)
        self.jammer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.jammer_label)
        
        # ═══════════════════════════════════════════════════════════
        # ECM CONTROL PANEL
        # ═══════════════════════════════════════════════════════════
        self._setup_ecm_controls(layout)
        
        # AI Classification Section
        self._setup_ai_classification(layout)
        
        layout.addStretch()
    
    def _setup_ecm_controls(self, layout: QVBoxLayout) -> None:
        """Setup ECM control section."""
        ecm_header = QLabel("ECM CONTROLS")
        ecm_header.setStyleSheet("""
            QLabel {
                color: #ff6600;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                margin-top: 10px;
                background-color: rgba(60, 30, 0, 200);
                border: 1px solid #aa5500;
            }
        """)
        ecm_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ecm_header)
        
        # ECM Toggle Button
        self.ecm_toggle_btn = QPushButton("🔇 JAMMER OFF")
        self.ecm_toggle_btn.setCheckable(True)
        self.ecm_toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #888888;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
                padding: 8px;
                border: 2px solid #555555;
                border-radius: 5px;
            }
            QPushButton:checked {
                background-color: #aa3300;
                color: #ffffff;
                border: 2px solid #ff5500;
            }
            QPushButton:hover {
                border: 2px solid #ff8800;
            }
        """)
        self.ecm_toggle_btn.clicked.connect(self._on_ecm_toggle)
        layout.addWidget(self.ecm_toggle_btn)
        
        # ECM Type Selector
        ecm_type_layout = QHBoxLayout()
        ecm_type_label = QLabel("TYPE:")
        ecm_type_label.setStyleSheet("color: #aa5500; font-size: 11px;")
        ecm_type_layout.addWidget(ecm_type_label)
        
        self.ecm_type_combo = QComboBox()
        self.ecm_type_combo.addItems([
            "Noise Barrage",
            "Noise Spot",
            "DRFM Repeater",
            "Chaff Cloud",
            "Decoy"
        ])
        self.ecm_type_combo.setStyleSheet("""
            QComboBox {
                background-color: #2a2a2a;
                color: #ff8800;
                font-size: 11px;
                padding: 3px;
                border: 1px solid #aa5500;
            }
            QComboBox::drop-down {
                border: none;
            }
        """)
        ecm_type_layout.addWidget(self.ecm_type_combo)
        layout.addLayout(ecm_type_layout)
        
        # ECM Power Slider
        power_layout = QHBoxLayout()
        power_label = QLabel("POWER:")
        power_label.setStyleSheet("color: #aa5500; font-size: 11px;")
        power_layout.addWidget(power_label)
        
        self.ecm_power_slider = QSlider(Qt.Orientation.Horizontal)
        self.ecm_power_slider.setMinimum(100)
        self.ecm_power_slider.setMaximum(5000)
        self.ecm_power_slider.setValue(500)
        self.ecm_power_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #333333;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #ff6600;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }
            QSlider::sub-page:horizontal {
                background: #aa3300;
                border-radius: 3px;
            }
        """)
        self.ecm_power_slider.valueChanged.connect(self._on_ecm_power_changed)
        power_layout.addWidget(self.ecm_power_slider)
        
        self.ecm_power_label = QLabel("500 W")
        self.ecm_power_label.setStyleSheet("color: #ff8800; font-size: 11px; font-weight: bold;")
        self.ecm_power_label.setMinimumWidth(50)
        power_layout.addWidget(self.ecm_power_label)
        layout.addLayout(power_layout)
    
    def _setup_ai_classification(self, layout: QVBoxLayout) -> None:
        """Setup AI classification section."""
        ai_header = QLabel("AI CLASSIFICATION")
        ai_header.setStyleSheet("""
            QLabel {
                color: #00ccff;
                font-family: 'Consolas', monospace;
                font-size: 12px;
                font-weight: bold;
                padding: 5px;
                margin-top: 10px;
                background-color: rgba(0, 40, 60, 200);
                border: 1px solid #0088aa;
            }
        """)
        ai_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(ai_header)
        
        # AI Icon + Class Label
        self.ai_icon_label = QLabel("❓")
        self.ai_icon_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                padding: 5px;
            }
        """)
        self.ai_icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.ai_icon_label)
        
        self.ai_class_label = QLabel("UNKNOWN")
        self.ai_class_label.setStyleSheet("""
            QLabel {
                color: #00ccff;
                font-family: 'Consolas', monospace;
                font-size: 14px;
                font-weight: bold;
                padding: 3px;
            }
        """)
        self.ai_class_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.ai_class_label)
        
        self.ai_confidence_label = QLabel("Confidence: ---")
        self.ai_confidence_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 3px;
            }
        """)
        self.ai_confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.ai_confidence_label)
    
    def update_target(self, target_data: Optional[Dict[str, Any]]) -> None:
        """
        Update inspector with target data.
        
        Args:
            target_data: Dictionary with target information
        """
        if target_data is None:
            self._clear()
            return
        
        self._current_target_id = target_data.get('id', None)
        
        self.labels["ID"].setText(str(target_data.get('id', '---')))
        self.labels["TYPE"].setText(str(target_data.get('type', 'Unknown')))
        self.labels["RANGE"].setText(f"{target_data.get('range_km', 0):.1f} km")
        self.labels["AZIMUTH"].setText(f"{target_data.get('azimuth_deg', 0):.1f}°")
        self.labels["VELOCITY"].setText(f"{target_data.get('velocity_mps', 0):.0f} m/s")
        self.labels["RCS"].setText(f"{target_data.get('rcs_m2', 0):.1f} m²")
        self.labels["SNR"].setText(f"{target_data.get('snr_db', 0):.1f} dB")
        
        # Calculate Pd from SNR (simplified Swerling 1 approximation)
        snr_db = target_data.get('snr_db', 0)
        pd = self._calculate_pd(snr_db)
        self.labels["Pd"].setText(f"{pd*100:.0f}%")
        
        # Jammer status - USE USER'S TOGGLE STATE, not target_data
        self._update_jammer_display()
        
        # ═══ SMART AI CLASSIFICATION (THROTTLED) ═══
        current_time = time.time()
        target_changed = (target_data.get('id') != self._current_target_id)
        time_elapsed = (current_time - self._last_ai_classification_time) >= self._ai_throttle_interval
        
        if target_changed or time_elapsed:
            self._update_ai_classification(target_data)
            self._last_ai_classification_time = current_time
    
    def _update_jammer_display(self) -> None:
        """Update jammer display based on current ECM state."""
        if self._ecm_active:
            self.jammer_label.setText("⚠ JAMMER ACTIVE")
            self.jammer_label.setStyleSheet("""
                QLabel {
                    color: #ff5555;
                    font-family: 'Consolas', monospace;
                    font-size: 12px;
                    padding: 5px;
                    background-color: rgba(50, 0, 0, 200);
                    border: 1px solid #aa0000;
                }
            """)
        else:
            self.jammer_label.setText("🔇 NO JAMMER")
            self.jammer_label.setStyleSheet("""
                QLabel {
                    color: #888888;
                    font-family: 'Consolas', monospace;
                    font-size: 12px;
                    padding: 5px;
                    background-color: rgba(30, 30, 30, 200);
                    border: 1px solid #555555;
                }
            """)
    
    def _calculate_pd(self, snr_db: float) -> float:
        """
        Calculate probability of detection from SNR.
        
        Uses simplified Swerling 1 model approximation.
        
        Reference: Skolnik, "Radar Handbook", Chapter 2
        
        Args:
            snr_db: Signal-to-noise ratio [dB]
            
        Returns:
            Probability of detection (0-1)
        """
        # Simplified sigmoid approximation
        # At SNR=13 dB, Pd ≈ 0.5 for Pfa=1e-6
        threshold_db = 13.0
        return 1.0 / (1.0 + np.exp(-(snr_db - threshold_db) / 3))
    
    def _clear(self) -> None:
        """Clear all fields."""
        for label in self.labels.values():
            label.setText("---")
        self.jammer_label.setText("🔇 NO TARGET")
        self.ai_icon_label.setText("❓")
        self.ai_class_label.setText("UNKNOWN")
        self.ai_confidence_label.setText("Confidence: ---")
        self.ai_confidence_label.setStyleSheet("""
            QLabel {
                color: #888888;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 3px;
            }
        """)
    
    def _update_ai_classification(self, target_data: Dict[str, Any]) -> None:
        """
        Run AI classification on target data.
        
        Uses InferenceEngine to predict target type from radar features.
        """
        if self._inference_engine is None or not self._inference_engine.is_ready:
            self.ai_icon_label.setText("⚠")
            self.ai_class_label.setText("ML DISABLED")
            self.ai_confidence_label.setText("Model not loaded")
            return
        
        # Extract features for inference
        velocity_mps = target_data.get('velocity_mps', 0)
        frequency_hz = 10e9  # Default X-band
        doppler_hz = 2 * velocity_mps * frequency_hz / 3e8
        
        track_data = {
            'range_km': target_data.get('range_km', 50.0),
            'doppler_hz': doppler_hz,
            'snr_db': target_data.get('snr_db', 0),
            'rcs_est_m2': target_data.get('rcs_m2', 1.0),
        }
        
        # Run inference
        icon, class_name, confidence = self._inference_engine.predict_with_icon(track_data)
        
        # Update display
        self.ai_icon_label.setText(icon)
        self.ai_class_label.setText(class_name.upper())
        self.ai_confidence_label.setText(f"Confidence: {confidence*100:.1f}%")
        
        # Color-code confidence
        if confidence >= 0.8:
            color = "#00ff88"  # Green - high confidence
        elif confidence >= 0.5:
            color = "#ffcc00"  # Yellow - medium confidence
        else:
            color = "#ff5555"  # Red - low confidence
        
        self.ai_confidence_label.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-family: 'Consolas', monospace;
                font-size: 11px;
                font-weight: bold;
                padding: 3px;
            }}
        """)
    
    def _on_ecm_toggle(self, checked: bool) -> None:
        """Handle ECM toggle button click."""
        self._ecm_active = checked
        
        if checked:
            self.ecm_toggle_btn.setText("📡 JAMMER ON")
        else:
            self.ecm_toggle_btn.setText("🔇 JAMMER OFF")
        
        self._update_jammer_display()
        
        # Update the selected target's jammer state
        if self._current_target_id is not None:
            self._apply_ecm_to_target()
    
    def _on_ecm_power_changed(self, value: int) -> None:
        """Handle ECM power slider change."""
        if value >= 1000:
            self.ecm_power_label.setText(f"{value/1000:.1f} kW")
        else:
            self.ecm_power_label.setText(f"{value} W")
    
    def _apply_ecm_to_target(self) -> None:
        """Apply ECM settings and emit signal to simulation engine."""
        ecm_type = self.ecm_type_combo.currentText()
        ecm_power = self.ecm_power_slider.value()
        
        print(f"[ECM] Target {self._current_target_id}: "
              f"Active={self._ecm_active}, Type={ecm_type}, Power={ecm_power}W")
        
        # Emit signal for MainWindow to forward to simulation thread
        if self._current_target_id is not None:
            self.ecm_changed.emit(self._ecm_active, ecm_type, self._current_target_id)
    
    def get_ecm_state(self) -> Dict[str, Any]:
        """Get current ECM configuration."""
        return {
            'active': self._ecm_active,
            'type': self.ecm_type_combo.currentText(),
            'power_watts': self.ecm_power_slider.value(),
            'target_id': self._current_target_id
        }
