"""RadarSim desktop application entry point."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.cli import add_headless_arguments, run_headless

    parser = argparse.ArgumentParser(description="RadarSim radar simulator")
    parser.add_argument(
        "scenario",
        nargs="?",
        default=None,
        help="scenario YAML or JSON file to load on startup",
    )
    add_headless_arguments(parser)
    parsed_args, remaining_argv = parser.parse_known_args(
        sys.argv[1:] if argv is None else argv
    )
    if parsed_args.headless:
        if parsed_args.scenario is None:
            parser.error("--headless needs a scenario file")
        return run_headless(
            parsed_args.scenario,
            duration_s=parsed_args.duration,
            seed=parsed_args.seed,
            out_path=parsed_args.out,
        )

    try:
        from PySide6.QtGui import QColor, QPalette
        from PySide6.QtWidgets import QApplication
        from src.ui.main_window import MainWindow
    except ImportError as exc:
        print("RadarSim GUI dependencies are missing. Install with: pip install 'radarsim[gui]'", file=sys.stderr)
        print(f"Missing dependency: {exc.name}", file=sys.stderr)
        return 1

    app = QApplication.instance() or QApplication([sys.argv[0]] + remaining_argv)
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

    window = MainWindow(scenario_path=parsed_args.scenario)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
