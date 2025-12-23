"""
Analysis Window

Standalone window for post-flight analysis.
Opens when loading HDF5 recording files.
"""

from PyQt6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QTabWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from typing import Optional

from .analysis_panel import AnalysisPanel

# Phase 20: Ambiguity Analysis
try:
    from .analysis.ambiguity_plot import AmbiguityPlot
    AMBIGUITY_AVAILABLE = True
except ImportError:
    AMBIGUITY_AVAILABLE = False

# Phase 21: ROC Curves
try:
    from .analysis.roc_curve import ROCCurveWidget
    ROC_AVAILABLE = True
except ImportError:
    ROC_AVAILABLE = False

# Phase 21: SNR Histogram
try:
    from .analysis.snr_histogram import SNRHistogramWidget
    SNR_HIST_AVAILABLE = True
except ImportError:
    SNR_HIST_AVAILABLE = False


class AnalysisWindow(QMainWindow):
    """
    Standalone analysis window for replay mode.
    
    Opens as a separate window when loading HDF5 recordings,
    preventing layout disruption in the main window.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("RadarSim - Post-Flight Analysis")
        self.setMinimumSize(800, 600)
        self.resize(900, 700)
        
        # Dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a1510;
            }
            QTabWidget::pane {
                border: 1px solid #00aa55;
                background-color: #0a1510;
            }
            QTabBar::tab {
                background-color: #001510;
                color: #00aa55;
                padding: 8px 16px;
                border: 1px solid #00aa55;
            }
            QTabBar::tab:selected {
                background-color: #003322;
                color: #00ff88;
            }
        """)
        
        # Central widget with tabs
        central = QWidget()
        self.setCentralWidget(central)
        
        layout = QVBoxLayout(central)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Tab widget for multiple analysis views
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Tab 1: Recording Analysis (existing)
        self.analysis_panel = AnalysisPanel()
        self.tabs.addTab(self.analysis_panel, "📈 Recording Analysis")
        
        # Tab 2: Ambiguity Analysis (Phase 20)
        if AMBIGUITY_AVAILABLE:
            self.ambiguity_plot = AmbiguityPlot()
            self.tabs.addTab(self.ambiguity_plot, "📊 Ambiguity (PRF)")
        else:
            self.ambiguity_plot = None
        
        # Tab 3: ROC Curves (Phase 21)
        if ROC_AVAILABLE:
            self.roc_curve = ROCCurveWidget()
            self.tabs.addTab(self.roc_curve, "📉 ROC Curves")
        else:
            self.roc_curve = None
        
        # Tab 4: SNR Histogram (Phase 21)
        if SNR_HIST_AVAILABLE:
            self.snr_histogram = SNRHistogramWidget()
            self.tabs.addTab(self.snr_histogram, "📊 SNR Stats")
        else:
            self.snr_histogram = None
    
    def set_loader(self, loader):
        """Pass the replay loader to the analysis panel."""
        self.analysis_panel.set_loader(loader)
    
    def set_current_time(self, t: float):
        """Update the time marker."""
        self.analysis_panel.set_current_time(t)
    
    def add_snr_value(self, snr_db: float):
        """Add SNR value to histogram (for real-time updates)."""
        if self.snr_histogram:
            self.snr_histogram.add_snr_value(snr_db)
    
    def add_snr_values(self, snr_values: list):
        """Add multiple SNR values to histogram."""
        if self.snr_histogram:
            self.snr_histogram.add_snr_values(snr_values)
    
    def closeEvent(self, event):
        """Just hide instead of closing."""
        self.hide()
        event.ignore()
