Quick Start
===========

Launching the GUI
-----------------

.. code-block:: bash

   python run_gui.py

This opens the professional radar operator console with:

* **PPI Scope**: Plan Position Indicator with rotating sweep
* **A-Scope**: Amplitude vs Range display
* **Range-Doppler Map**: 2D FFT visualization

Controls
--------

============== =====================================
Control        Description
============== =====================================
Frequency      Changes ITU-R atmospheric loss
Power          Adjusts transmitted power
Antenna RPM    Changes PPI sweep speed
ECM Toggle     Activates jamming (noise strobes)
============== =====================================

Loading Scenarios
-----------------

Use **File > Load Scenario** to load YAML scenario files from the ``scenarios/`` folder.

Example: ``scenarios/f16_vs_sa6.yaml``

Flight Recording
----------------

Click **STOP** to save the session to HDF5 format in ``output/session_*.h5``.
