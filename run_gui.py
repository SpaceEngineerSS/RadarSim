#!/usr/bin/env python3
"""
RadarSim - Professional Radar Operator Console

Launch the modern PyQt6-based radar simulation GUI.

Usage:
    python run_gui.py
    
Features:
    - PPI Scope with rotating sweep and afterglow
    - A-Scope signal display
    - Range-Doppler Map
    - Threaded simulation engine (30+ FPS)
    - Scenario loading (YAML)
    - NATO symbology colors

References:
    - IEEE Std 686-2008 (Radar Definitions)
    - MIL-STD-2525D (Symbology)
"""

import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch the RadarSim GUI."""
    print("=" * 60)
    print("RadarSim - Professional Radar Operator Console")
    print("=" * 60)
    print()
    
    # Check dependencies
    try:
        from PyQt6.QtWidgets import QApplication
        print("✓ PyQt6 OK")
    except ImportError:
        print("✗ PyQt6 not installed. Run: pip install PyQt6")
        return 1
    
    try:
        import pyqtgraph
        print("✓ PyQtGraph OK")
    except ImportError:
        print("✗ PyQtGraph not installed. Run: pip install pyqtgraph")
        return 1
    
    try:
        import yaml
        print("✓ PyYAML OK")
    except ImportError:
        print("✗ PyYAML not installed. Run: pip install pyyaml")
        return 1
    
    try:
        import numba
        print("✓ Numba OK")
    except ImportError:
        print("⚠ Numba not installed (optional, for performance)")
    
    try:
        from src.physics import calculate_snr
        print("✓ Physics engine OK")
    except ImportError as e:
        print(f"✗ Physics engine error: {e}")
        return 1
    
    print()
    print("Starting GUI...")
    print("=" * 60)
    
    # Launch new UI
    from src.ui.main_window import MainWindow
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Apply dark theme palette
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(10, 25, 15))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 200, 100))
    palette.setColor(QPalette.ColorRole.Base, QColor(5, 20, 10))
    palette.setColor(QPalette.ColorRole.Text, QColor(0, 200, 100))
    palette.setColor(QPalette.ColorRole.Button, QColor(10, 30, 20))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 200, 100))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 100, 50))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 255, 100))
    app.setPalette(palette)
    
    window = MainWindow()
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
