from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

project = "RadarSim"
copyright = "2026, RadarSim Contributors"
author = "RadarSim Contributors"
release = "3.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]

templates_path: list[str] = []
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = ["PySide6", "pyqtgraph", "OpenGL"]
