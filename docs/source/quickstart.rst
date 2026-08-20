Quick start
===========

Desktop application
-------------------

.. code-block:: console

   radarsim

Load a YAML or JSON file from ``scenarios/``, inspect the radar and
environment fields, and start the simulation. PPI, RHI, A-scope,
range-Doppler, tracking, recording, and imaging views consume the same model
state.

Headless simulation
-------------------

.. code-block:: python

   from src.io.scenario_loader import ScenarioLoader

   loader = ScenarioLoader("scenarios/basic_tracking.json")
   engine = loader.create_simulation_engine()

   for _ in range(100):
       detections = engine.step()

Use a fixed NumPy random seed and preserve the scenario file when results
must be reproduced. Positive radial velocity and Doppler mean motion away
from the radar.
