Installation
============

RadarSim supports CPython 3.9 through 3.12. Use a virtual environment.

.. code-block:: console

   git clone https://github.com/SpaceEngineerSS/RadarSim.git
   cd RadarSim
   python -m venv .venv
   python -m pip install --upgrade pip
   python -m pip install -e ".[gui]"

Install development and documentation tools with:

.. code-block:: console

   python -m pip install -e ".[gui,dev,docs]"

Run the installed application with ``radarsim``. From a source checkout,
``python run_gui.py`` uses the same entry point.

The core dependency set is NumPy, SciPy, Numba, h5py, and PyYAML. The GUI
extra adds PySide6, pyqtgraph, Matplotlib, and PyOpenGL.
