"""RadarSim desktop application entry point."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtWidgets import QApplication
        from src.ui.main_window import MainWindow
    except ImportError as exc:
        print("RadarSim GUI dependencies are missing. Install with: pip install 'radarsim[gui]'", file=sys.stderr)
        print(f"Missing dependency: {exc.name}", file=sys.stderr)
        return 1

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("RadarSim")
    app.setOrganizationName("RadarSim")
    app.setStyle("Fusion")

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
    raise SystemExit(main())
