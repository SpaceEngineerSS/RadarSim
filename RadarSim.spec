# PyInstaller build for the RadarSim desktop application.
#
#   pip install -e ".[gui,build]"
#   pyinstaller --noconfirm --clean RadarSim.spec
#
# One-folder build: it starts faster than a one-file build, which has to
# unpack itself to a temporary folder on every launch.

import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

datas = collect_data_files("radarsim")  # scenarios, fonts, icons, translations

hiddenimports = collect_submodules("radarsim") + [
    "pyqtgraph.opengl",
    "OpenGL.platform.win32",
    "OpenGL.platform.glx",
    "OpenGL.platform.darwin",
]

# Large modules the application never imports.
excludes = [
    "tkinter", "matplotlib", "IPython", "jupyter", "notebook", "pytest", "sphinx",
    "docutils", "PyQt5", "PyQt6", "PySide2",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.QtMultimedia", "PySide6.QtQuick",
    "PySide6.QtQml", "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtPdf",
    "PySide6.QtBluetooth", "PySide6.QtPositioning", "PySide6.QtSensors", "PySide6.QtSql",
]

a = Analysis(
    ["run_gui.py"],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="RadarSim",
    console=False,
    icon="radarsim/ui/assets/radarsim.ico",
)

coll = COLLECT(exe, a.binaries, a.datas, name="RadarSim")

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="RadarSim.app",
        icon="radarsim/ui/assets/radarsim.png",
        bundle_identifier="io.github.spaceengineerss.radarsim",
    )
