"""
ROC Curve Visualization Widget

Displays Receiver Operating Characteristic curves showing the trade-off
between Probability of Detection (Pd) and Probability of False Alarm (Pfa).

Reference: Skolnik, "Radar Handbook", Chapter 2
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QGroupBox, QGridLayout, QComboBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import numpy as np

try:
    import pyqtgraph as pg
    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

try:
    from src.physics.metrics import generate_roc_curves, calculate_pd_swerling
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


class ROCCurveWidget(QWidget):
    """
    ROC Curve Visualization Widget.
    
    Shows Pd vs Pfa curves for different SNR values,
    allowing the user to understand detection trade-offs.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setMinimumSize(600, 450)
        
        # Default parameters
        self.swerling_case = 1
        self.current_snr_db = 13.0
        self.current_pfa = 1e-6
        
        # Apply theme
        self._apply_theme()
        
        # Setup UI
        self._setup_ui()
        
        # Initial plot
        self._update_plot()
    
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
            QComboBox {
                background-color: #001510;
                border: 1px solid #00aa55;
                padding: 5px;
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
        """)
    
    def _setup_ui(self):
        """Setup the UI layout."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Title
        title = QLabel("📈 ROC CURVES (Detection vs False Alarm)")
        title.setFont(QFont("Consolas", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #00ff88; padding: 5px;")
        layout.addWidget(title)
        
        # Main content
        content = QHBoxLayout()
        
        # Left: Controls
        controls = self._create_controls()
        content.addWidget(controls, stretch=1)
        
        # Right: Plot
        plot = self._create_plot()
        content.addWidget(plot, stretch=2)
        
        layout.addLayout(content)
    
    def _create_controls(self) -> QGroupBox:
        """Create control panel."""
        group = QGroupBox("Parameters")
        layout = QGridLayout(group)
        layout.setSpacing(10)
        
        row = 0
        
        # Swerling Model selector
        layout.addWidget(QLabel("Swerling Model:"), row, 0)
        self.swerling_combo = QComboBox()
        self.swerling_combo.addItems([
            "Case 0 (Non-fluctuating)",
            "Case 1 (Slow, Many Scatterers)",
            "Case 2 (Fast, Many Scatterers)",
            "Case 3 (Slow, Dominant)",
            "Case 4 (Fast, Dominant)"
        ])
        self.swerling_combo.setCurrentIndex(1)
        self.swerling_combo.currentIndexChanged.connect(self._on_swerling_changed)
        layout.addWidget(self.swerling_combo, row, 1, 1, 2)
        row += 1
        
        # Current operating SNR
        layout.addWidget(QLabel("Operating SNR [dB]:"), row, 0)
        self.snr_slider = QSlider(Qt.Orientation.Horizontal)
        self.snr_slider.setRange(0, 30)
        self.snr_slider.setValue(13)
        self.snr_slider.valueChanged.connect(self._on_snr_changed)
        layout.addWidget(self.snr_slider, row, 1)
        self.snr_label = QLabel("13 dB")
        layout.addWidget(self.snr_label, row, 2)
        row += 1
        
        # Display current performance
        layout.addWidget(QLabel(""), row, 0)  # Spacer
        row += 1
        
        layout.addWidget(QLabel("Current Pd @ Pfa=1e-6:"), row, 0)
        self.pd_label = QLabel("--")
        self.pd_label.setStyleSheet("color: #00ffaa; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.pd_label, row, 1, 1, 2)
        row += 1
        
        layout.addWidget(QLabel("Required SNR for Pd=0.9:"), row, 0)
        self.req_snr_label = QLabel("--")
        self.req_snr_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
        layout.addWidget(self.req_snr_label, row, 1, 1, 2)
        row += 1
        
        # Legend
        layout.addWidget(QLabel(""), row, 0)
        row += 1
        legend_title = QLabel("Curve Colors:")
        legend_title.setStyleSheet("font-weight: bold;")
        layout.addWidget(legend_title, row, 0, 1, 3)
        row += 1
        
        colors = [
            ("5 dB", "#ff5555"),
            ("10 dB", "#ffaa00"),
            ("13 dB", "#00ff88"),
            ("15 dB", "#00aaff"),
            ("20 dB", "#aa55ff"),
        ]
        for snr_text, color in colors:
            lbl = QLabel(f"⬤ {snr_text}")
            lbl.setStyleSheet(f"color: {color};")
            layout.addWidget(lbl, row, 0, 1, 3)
            row += 1
        
        layout.setRowStretch(row, 1)
        
        return group
    
    def _create_plot(self) -> QGroupBox:
        """Create plot area."""
        group = QGroupBox("ROC Curves")
        layout = QVBoxLayout(group)
        
        if PYQTGRAPH_AVAILABLE:
            pg.setConfigOptions(antialias=True)
            
            self.plot_widget = pg.PlotWidget()
            self.plot_widget.setBackground('#051008')
            self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self.plot_widget.setLabel('bottom', 'Pfa (log scale)')
            self.plot_widget.setLabel('left', 'Pd')
            self.plot_widget.setLogMode(x=True, y=False)
            self.plot_widget.setXRange(-10, -2)
            self.plot_widget.setYRange(0, 1)
            
            # Create curve placeholders
            self.roc_curves = {}
            colors = {
                5: '#ff5555',
                10: '#ffaa00',
                13: '#00ff88',
                15: '#00aaff',
                20: '#aa55ff'
            }
            for snr, color in colors.items():
                self.roc_curves[snr] = self.plot_widget.plot(
                    [], [],
                    pen=pg.mkPen(color=color, width=2),
                    name=f'{snr} dB'
                )
            
            # Operating point marker
            self.operating_point = self.plot_widget.plot(
                [], [],
                pen=None,
                symbol='o',
                symbolSize=15,
                symbolBrush='#ffffff'
            )
            
            # Threshold line
            self.threshold_line = pg.InfiniteLine(
                pos=1e-6,
                angle=90,
                pen=pg.mkPen('#ffff00', width=1, style=Qt.PenStyle.DashLine)
            )
            self.plot_widget.addItem(self.threshold_line)
            
            layout.addWidget(self.plot_widget)
        else:
            fallback = QLabel("PyQtGraph not available")
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(fallback)
        
        return group
    
    def _on_swerling_changed(self, index: int):
        """Handle Swerling model change."""
        self.swerling_case = index
        self._update_plot()
    
    def _on_snr_changed(self, value: int):
        """Handle SNR slider change."""
        self.current_snr_db = float(value)
        self.snr_label.setText(f"{value} dB")
        self._update_metrics()
        self._update_operating_point()
    
    def _update_plot(self):
        """Update ROC curves."""
        if not PYQTGRAPH_AVAILABLE or not METRICS_AVAILABLE:
            return
        
        # Generate curves
        roc_data = generate_roc_curves(
            snr_values_db=[5, 10, 13, 15, 20],
            pfa_range=(1e-10, 1e-2),
            n_points=100,
            swerling_case=self.swerling_case
        )
        
        pfa = roc_data['pfa']
        
        for snr, curve in self.roc_curves.items():
            if snr in roc_data['pd']:
                pd = roc_data['pd'][snr]
                curve.setData(pfa, pd)
        
        self._update_metrics()
        self._update_operating_point()
    
    def _update_metrics(self):
        """Update displayed metrics."""
        if not METRICS_AVAILABLE:
            return
        
        # Calculate Pd at current SNR
        pd = calculate_pd_swerling(
            self.current_snr_db,
            pfa=1e-6,
            swerling_case=self.swerling_case
        )
        self.pd_label.setText(f"{pd:.3f} ({pd*100:.1f}%)")
        
        # Required SNR for Pd=0.9
        from src.physics.metrics import albersheim_snr
        req_snr = albersheim_snr(pd=0.9, pfa=1e-6)
        self.req_snr_label.setText(f"{req_snr:.1f} dB")
    
    def _update_operating_point(self):
        """Update operating point marker."""
        if not PYQTGRAPH_AVAILABLE or not METRICS_AVAILABLE:
            return
        
        pfa = self.current_pfa
        pd = calculate_pd_swerling(
            self.current_snr_db,
            pfa=pfa,
            swerling_case=self.swerling_case
        )
        
        self.operating_point.setData([pfa], [pd])


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    widget = ROCCurveWidget()
    widget.show()
    sys.exit(app.exec())
