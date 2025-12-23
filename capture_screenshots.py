"""
Automated Screenshot Capture for RadarSim Documentation

Run this script to automatically capture all required screenshots
for the README and documentation.

Usage:
    python capture_screenshots.py
"""

import sys
import os
import time

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QScreen
import numpy as np


def capture_screenshots():
    """Main function to capture all screenshots."""
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'docs', 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("RadarSim - Automated Screenshot Capture")
    print("=" * 60)
    
    app = QApplication(sys.argv)
    
    # Import after QApplication is created
    from src.ui.main_window import MainWindow
    
    window = MainWindow()
    window.show()
    window.resize(1280, 800)
    
    screenshots_taken = []
    
    def take_screenshot(name: str, widget=None, delay_ms: int = 500):
        """Take a screenshot and save it."""
        def capture():
            target = widget if widget else window
            pixmap = target.grab()
            filepath = os.path.join(output_dir, f"{name}.png")
            pixmap.save(filepath)
            print(f"✓ Saved: {name}.png")
            screenshots_taken.append(name)
        
        QTimer.singleShot(delay_ms, capture)
    
    def capture_sequence():
        """Capture screenshots in sequence."""
        delay = 0
        step = 1500  # 1.5 second between captures
        
        # 1. Main window (PPI Scope)
        print("\n[1/8] Capturing PPI Scope...")
        QTimer.singleShot(delay, lambda: take_screenshot("ppi_scope"))
        delay += step
        
        # 2. Switch to RHI tab
        print("[2/8] Capturing RHI Scope...")
        def capture_rhi():
            if hasattr(window, 'display_tabs') and window.display_tabs.count() > 1:
                window.display_tabs.setCurrentIndex(1)
            take_screenshot("rhi_scope")
        QTimer.singleShot(delay, capture_rhi)
        delay += step
        
        # 3. Switch to 3D Tactical
        print("[3/8] Capturing 3D Tactical...")
        def capture_3d():
            if hasattr(window, 'display_tabs') and window.display_tabs.count() > 2:
                window.display_tabs.setCurrentIndex(2)
            take_screenshot("3d_tactical")
        QTimer.singleShot(delay, capture_3d)
        delay += step
        
        # 4. Back to PPI
        print("[4/8] Switching back to PPI...")
        def back_to_ppi():
            if hasattr(window, 'display_tabs'):
                window.display_tabs.setCurrentIndex(0)
        QTimer.singleShot(delay, back_to_ppi)
        delay += step
        
        # 5. Open Analysis Window and capture tabs
        print("[5/8] Capturing Analysis Window tabs...")
        def capture_analysis():
            if hasattr(window, 'analysis_window'):
                window.analysis_window.show()
                window.analysis_window.resize(800, 600)
                
                # Capture each tab
                QTimer.singleShot(500, lambda: take_screenshot("recording_analysis", window.analysis_window))
                
                def capture_ambiguity():
                    if window.analysis_window.tabs.count() > 1:
                        window.analysis_window.tabs.setCurrentIndex(1)
                    take_screenshot("ambiguity_plot", window.analysis_window)
                QTimer.singleShot(1500, capture_ambiguity)
                
                def capture_roc():
                    if window.analysis_window.tabs.count() > 2:
                        window.analysis_window.tabs.setCurrentIndex(2)
                    take_screenshot("roc_curves", window.analysis_window)
                QTimer.singleShot(2500, capture_roc)
                
                def capture_snr():
                    if window.analysis_window.tabs.count() > 3:
                        window.analysis_window.tabs.setCurrentIndex(3)
                    take_screenshot("snr_histogram", window.analysis_window)
                QTimer.singleShot(3500, capture_snr)
        
        QTimer.singleShot(delay, capture_analysis)
        delay += 5000  # Allow time for analysis captures
        
        # 6. A-Scope capture
        print("[6/8] Capturing A-Scope...")
        def capture_a_scope():
            if hasattr(window, 'a_scope'):
                take_screenshot("a_scope_cfar", window.a_scope)
        QTimer.singleShot(delay, capture_a_scope)
        delay += step
        
        # 7. Open SAR Viewer
        print("[7/8] Capturing SAR Viewer...")
        def capture_sar():
            try:
                from src.ui.sar_viewer import SARViewer
                sar_dialog = SARViewer(window)
                sar_dialog.show()
                sar_dialog.resize(700, 600)
                QTimer.singleShot(1000, lambda: take_screenshot("sar_viewer", sar_dialog))
            except Exception as e:
                print(f"  SAR capture failed: {e}")
        QTimer.singleShot(delay, capture_sar)
        delay += 2000
        
        # 8. Final summary
        def finish():
            print("\n" + "=" * 60)
            print(f"Screenshots captured: {len(screenshots_taken)}")
            print(f"Output directory: {output_dir}")
            print("=" * 60)
            
            # Close after a delay
            QTimer.singleShot(2000, app.quit)
        
        QTimer.singleShot(delay, finish)
    
    # Start capture sequence after window is shown
    QTimer.singleShot(2000, capture_sequence)
    
    return app.exec()


if __name__ == "__main__":
    sys.exit(capture_screenshots())
