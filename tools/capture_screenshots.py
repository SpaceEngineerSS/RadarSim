"""
Capture the interface screenshots used in the README and documentation.

    python tools/capture_screenshots.py        # writes docs/assets/screens/*.png

Runs the real application off-screen with fixed seeds, so the pictures are
reproducible and always show the current interface.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings  # noqa: E402

from radarsim.io.scenario_loader import resolve_scenario  # noqa: E402
from radarsim.ui.main_window import MainWindow, ensure_application  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "screens"

# name, scenario, simulated seconds before the shot, tab, selected object, drawer tab
SHOTS = [
    ("ppi", "f16_vs_sa6", 75, 0, "F-16 Bravo", 0),
    ("ppi_naval", "naval_battlegroup", 30, 0, "ASM_1_Brahmos", None),
    ("bscope", "close_air_combat", 20, 1, "Bandit 1", None),
    ("rhi", "mountain_ambush", 60, 2, "Low Flyer", None),
    ("ascope", "ground_clutter_filtering", 12, 3, None, None),
    ("coverage", "basic_tracking", 20, 0, "Target 1", 1),
    ("editor", "basic_tracking", 5, 0, None, None),
]


def spin(app, seconds: float) -> None:
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.004)


def run_to(app, window, sim_seconds: float) -> None:
    window.controller.set_speed(10.0)
    deadline = time.time() + 60
    while time.time() < deadline:
        frame = window.controller.latest
        if frame is not None and frame.time >= sim_seconds:
            break
        spin(app, 0.05)
    window.controller.set_speed(1.0)


def main() -> int:
    language = sys.argv[1] if len(sys.argv) > 1 else "en"
    settings = QSettings("RadarSim", "RadarSim")
    saved_language = settings.value("view/language", "en")
    settings.setValue("view/language", language)
    app = ensure_application()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        for name, scenario, seconds, tab, select, drawer in SHOTS:
            window = MainWindow(resolve_scenario(scenario))
            window.resize(1440, 900)
            window.show()
            window.controller.call(lambda engine: setattr(engine, "rng", __import__("numpy").random.default_rng(3)))
            run_to(app, window, seconds)
            if select:
                target = next(t for t in window.controller.latest.targets if t.name == select)
                window._select(target.target_id)
            window.tabs.setCurrentIndex(tab)
            if drawer is not None:
                window._open_drawer(drawer)
                if drawer == 1:
                    window.coverage.compute()
            if name == "editor":
                window.editor_action.trigger()
                spin(app, 0.3)
                target_id = window.controller.add_object(-30_000.0, 25_000.0)
                window._select(target_id)
            spin(app, 3.0 if drawer == 1 else 1.2)
            suffix = "" if language == "en" else f"_{language}"
            window.grab().save(str(OUT / f"{name}{suffix}.png"))
            print(f"wrote {name}{suffix}.png")
            window.close()
            app.processEvents()
    finally:
        settings.setValue("view/language", saved_language)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
