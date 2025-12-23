"""
Ambiguity Analysis Plot

Visualizes the Range-Velocity trade-off for pulsed radar systems.

Key Relationships:
    - Max Unambiguous Range: R_un = c / (2 * PRF)
    - Max Unambiguous Velocity: V_un = λ * PRF / 4
    - These are inversely related: High PRF = Long-range ambiguity, good velocity
    
Reference:
    - Skolnik, "Radar Handbook", 3rd Ed., Chapter 4 (MTI & Pulse Doppler)
    - Richards, "Fundamentals of Radar Signal Processing", Chapter 6
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QGroupBox, QGridLayout, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
import numpy as np

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

# Physical constants
SPEED_OF_LIGHT = 299792458.0  # m/s


class AmbiguityPlot(QWidget):
    """
    Radar Ambiguity Analysis Widget.
    
    Shows the trade-off between maximum unambiguous range and velocity
    as a function of PRF and frequency.
    
    Features:
        - Interactive PRF slider
        - Frequency slider
        - Real-time R_max / V_max display
        - Blind speed zones visualization
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumSize(600, 500)
        
        # Default parameters
        self.prf_hz = 1000  # 1 kHz
        self.frequency_ghz = 10.0  # X-band
        self.pulse_width_us = 1.0  # 1 microsecond
        
        # Apply dark theme
        self._apply_theme()
        
        # Setup UI
        self._setup_ui()
        
        # Initial calculation
        self._update_display()
    
    def _apply_theme(self):
        """Apply dark radar theme."""
        self.setStyleSheet("""
            QWidget {
                background-color: #0a1510;
                color: #00dd66;
                font-family: 'Consolas', monospace;
            }
            QGroupBox {
                border: 1px solid #00aa55;
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #00dd66;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00aa55;
                height: 8px;
                background: #001510;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00ff88;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::sub-page:horizontal {
                background: #004422;
                border-radius: 4px;
            }
        """)
    
    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        
        # Title
        title = QLabel("📊 AMBIGUITY ANALYSIS (PRF Trade-Off)")
        title.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #00ff88; padding: 10px;")
        layout.addWidget(title)
        
        # Main content
        content = QHBoxLayout()
        
        # Left: Controls
        controls = self._create_controls()
        content.addWidget(controls, stretch=1)
        
        # Right: Plots
        plots = self._create_plots()
        content.addWidget(plots, stretch=2)
        
        layout.addLayout(content)
        
        # Bottom: Key equations
        equations = self._create_equations_panel()
        layout.addWidget(equations)
    
    def _create_controls(self) -> QGroupBox:
        """Create control panel with sliders."""
        group = QGroupBox("Radar Parameters")
        layout = QGridLayout(group)
        layout.setSpacing(10)
        
        row = 0
        
        # PRF Slider
        layout.addWidget(QLabel("PRF [Hz]:"), row, 0)
        self.prf_slider = QSlider(Qt.Orientation.Horizontal)
        self.prf_slider.setRange(100, 10000)
        self.prf_slider.setValue(self.prf_hz)
        self.prf_slider.valueChanged.connect(self._on_prf_changed)
        layout.addWidget(self.prf_slider, row, 1)
        self.prf_label = QLabel(f"{self.prf_hz} Hz")
        self.prf_label.setMinimumWidth(80)
        layout.addWidget(self.prf_label, row, 2)
        row += 1
        
        # Frequency Slider
        layout.addWidget(QLabel("Frequency [GHz]:"), row, 0)
        self.freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.freq_slider.setRange(1, 40)  # 1-40 GHz
        self.freq_slider.setValue(int(self.frequency_ghz))
        self.freq_slider.valueChanged.connect(self._on_freq_changed)
        layout.addWidget(self.freq_slider, row, 1)
        self.freq_label = QLabel(f"{self.frequency_ghz:.1f} GHz")
        self.freq_label.setMinimumWidth(80)
        layout.addWidget(self.freq_label, row, 2)
        row += 1
        
        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #00aa55;")
        layout.addWidget(sep, row, 0, 1, 3)
        row += 1
        
        # Results display
        layout.addWidget(QLabel("Max Unambiguous Range:"), row, 0)
        self.r_max_label = QLabel("-- km")
        self.r_max_label.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.r_max_label, row, 1, 1, 2)
        row += 1
        
        layout.addWidget(QLabel("Max Unambiguous Velocity:"), row, 0)
        self.v_max_label = QLabel("-- m/s")
        self.v_max_label.setStyleSheet("color: #ffaa00; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.v_max_label, row, 1, 1, 2)
        row += 1
        
        # Wavelength
        layout.addWidget(QLabel("Wavelength:"), row, 0)
        self.lambda_label = QLabel("-- cm")
        layout.addWidget(self.lambda_label, row, 1, 1, 2)
        row += 1
        
        # Blind speeds
        layout.addWidget(QLabel("1st Blind Speed:"), row, 0)
        self.blind_speed_label = QLabel("-- m/s")
        self.blind_speed_label.setStyleSheet("color: #ff6666;")
        layout.addWidget(self.blind_speed_label, row, 1, 1, 2)
        
        layout.setRowStretch(row + 1, 1)
        
        return group
    
    def _create_plots(self) -> QGroupBox:
        """Create the plot area."""
        group = QGroupBox("Trade-Off Visualization")
        layout = QVBoxLayout(group)
        
        if PYQTGRAPH_AVAILABLE:
            # Range-Velocity trade-off curve
            pg.setConfigOptions(antialias=True)
            
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground(QColor(5, 15, 10))
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setLabel('bottom', 'PRF', units='Hz')
            self.plot_widget.setLabel('left', 'Distance', units='km / (m/s)')
            self.plot_widget.addLegend()
            
            # Plot curves
            self.r_curve = self.plot_widget.plot(
                [], [], 
                pen=pg.mkPen(color='#00ffaa', width=2),
                name='R_max (km)'
            )
            self.v_curve = self.plot_widget.plot(
                [], [], 
                pen=pg.mkPen(color='#ffaa00', width=2),
                name='V_max (m/s)'
            )
            
            # Current operating point
            self.operating_point_r = self.plot_widget.plot(
                [], [],
                pen=None,
                symbol='o',
                symbolSize=12,
                symbolBrush='#00ffaa'
            )
            self.operating_point_v = self.plot_widget.plot(
                [], [],
                pen=None,
                symbol='s',
                symbolSize=12,
                symbolBrush='#ffaa00'
            )
            
            layout.addWidget(self.plot_widget)
            
            # Draw initial curves
            self._draw_curves()
        else:
            fallback = QLabel("PyQtGraph not available.\nInstall with: pip install pyqtgraph")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(fallback)
        
        return group
    
    def _create_equations_panel(self) -> QGroupBox:
        """Create panel showing key equations."""
        group = QGroupBox("Key Relationships")
        layout = QHBoxLayout(group)
        
        eq1 = QLabel("R_max = c / (2·PRF)")
        eq1.setStyleSheet("color: #00ffaa; font-size: 12px;")
        layout.addWidget(eq1)
        
        eq2 = QLabel("V_max = λ·PRF / 4")
        eq2.setStyleSheet("color: #ffaa00; font-size: 12px;")
        layout.addWidget(eq2)
        
        eq3 = QLabel("V_blind = n·λ·PRF / 2")
        eq3.setStyleSheet("color: #ff6666; font-size: 12px;")
        layout.addWidget(eq3)
        
        return group
    
    def _calculate_ambiguity(self, prf_hz: float, freq_ghz: float):
        """
        Calculate ambiguity parameters.
        
        Args:
            prf_hz: Pulse Repetition Frequency [Hz]
            freq_ghz: Radar frequency [GHz]
            
        Returns:
            dict with R_max, V_max, wavelength, blind_speed
        """
        # Wavelength
        wavelength_m = SPEED_OF_LIGHT / (freq_ghz * 1e9)
        
        # Maximum unambiguous range
        # R_un = c / (2 * PRF)
        r_max_m = SPEED_OF_LIGHT / (2 * prf_hz)
        r_max_km = r_max_m / 1000
        
        # Maximum unambiguous velocity (Doppler)
        # V_un = λ * PRF / 4
        v_max_mps = wavelength_m * prf_hz / 4
        
        # First blind speed
        # V_blind = λ * PRF / 2
        blind_speed_mps = wavelength_m * prf_hz / 2
        
        return {
            'r_max_km': r_max_km,
            'v_max_mps': v_max_mps,
            'wavelength_cm': wavelength_m * 100,
            'blind_speed_mps': blind_speed_mps,
        }
    
    def _on_prf_changed(self, value: int):
        """Handle PRF slider change."""
        self.prf_hz = value
        self.prf_label.setText(f"{value} Hz")
        self._update_display()
    
    def _on_freq_changed(self, value: int):
        """Handle frequency slider change."""
        self.frequency_ghz = float(value)
        self.freq_label.setText(f"{value} GHz")
        self._update_display()
    
    def _update_display(self):
        """Update all display elements."""
        result = self._calculate_ambiguity(self.prf_hz, self.frequency_ghz)
        
        # Update labels
        self.r_max_label.setText(f"{result['r_max_km']:.1f} km")
        self.v_max_label.setText(f"{result['v_max_mps']:.1f} m/s")
        self.lambda_label.setText(f"{result['wavelength_cm']:.2f} cm")
        self.blind_speed_label.setText(f"{result['blind_speed_mps']:.1f} m/s")
        
        # Update operating point on plot
        if PYQTGRAPH_AVAILABLE:
            self.operating_point_r.setData([self.prf_hz], [result['r_max_km']])
            self.operating_point_v.setData([self.prf_hz], [result['v_max_mps']])
    
    def _draw_curves(self):
        """Draw PRF vs R_max / V_max curves."""
        if not PYQTGRAPH_AVAILABLE:
            return
        
        prf_range = np.linspace(100, 10000, 200)
        
        # Calculate for current frequency
        r_max_km = []
        v_max_mps = []
        
        for prf in prf_range:
            result = self._calculate_ambiguity(prf, self.frequency_ghz)
            r_max_km.append(result['r_max_km'])
            v_max_mps.append(result['v_max_mps'])
        
        self.r_curve.setData(prf_range, r_max_km)
        self.v_curve.setData(prf_range, v_max_mps)


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    widget = AmbiguityPlot()
    widget.show()
    sys.exit(app.exec())
