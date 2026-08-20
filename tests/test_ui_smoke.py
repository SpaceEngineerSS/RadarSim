import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QApplication

from src.physics.metrics import calculate_pd_swerling
from src.ui.main_window import MainWindow
from src.ui.panels.target_inspector import TargetInspector
from src.ui.sar_viewer import SARViewer


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


def test_main_window_constructs_and_closes(application) -> None:
    window = MainWindow()
    window.show()
    application.processEvents()

    assert window.windowTitle() == "RadarSim - Professional Radar Simulation"
    assert window.centralWidget() is not None

    window.close()
    application.processEvents()


def test_target_inspector_uses_engine_detection_model(application) -> None:
    inspector = TargetInspector()
    assert inspector._calculate_pd(8.0) == pytest.approx(
        calculate_pd_swerling(8.0, pfa=1e-6, swerling_case=1, n_pulses=1)
    )
    inspector.close()


def test_sar_viewer_normalizes_image_and_applies_brightness(application) -> None:
    viewer = SARViewer()
    image = np.zeros((32, 32), dtype=complex)
    image[16, 16] = 10.0
    viewer.update_image(
        image,
        {
            "SNR_dB": 20.0,
            "Contrast": 2.0,
            "Range_Resolution_m": 1.5,
            "Azimuth_Resolution_m": 0.5,
            "Peak_dB": 0.0,
        },
    )
    before = viewer.image_item.getLevels()
    viewer.brightness_slider.setValue(80)
    application.processEvents()
    after = viewer.image_item.getLevels()

    assert viewer._image_db.max() == pytest.approx(0.0)
    assert viewer._image_db.min() == pytest.approx(-80.0)
    assert not np.allclose(before, after)
    assert "1.50 × 0.50 m" in viewer.resolution_label.text()
    viewer.close()
