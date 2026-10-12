# Getting started

## Desktop builds

The release page has ready-to-run builds for Windows, Linux and macOS
(`RadarSim-<version>-<platform>`). Unpack the archive and start `RadarSim`
(`RadarSim.exe` on Windows, `RadarSim.app` on macOS). No Python installation is needed.

## Installing with pip

RadarSim needs Python 3.9 to 3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install "radarsim[gui]"
radarsim
```

Without the `gui` extra you get the simulation library and the command-line runner,
which have no Qt dependency.

From a clone of the repository:

```bash
git clone https://github.com/SpaceEngineerSS/RadarSim.git
cd RadarSim
pip install -e ".[gui,dev]"
python -m radarsim
```

## First run

The application opens with the `basic_tracking` scenario: an X-band surveillance
radar and three targets about 40 km out. Within a few seconds you should see

- the sweep line turning clockwise on the PPI,
- small plots appearing where the beam crosses each target and fading over one
  revolution,
- track symbols (`T1`, `T2`, …) once a target has been detected on three
  consecutive scans,
- entries in the event log at the bottom: first detections, tracks initiated and
  confirmed.

Click a target on the scope or in the list on the left to see its range, SNR and the
outcome of its last look in the inspector. Change any radar parameter on the right
and the next look uses it.

Other scenarios are under **File › Built-in scenarios**. The
[scenario catalogue](scenarios.md) explains what each one is for.

## Without the interface

```bash
radarsim --list                                  # built-in scenarios
radarsim --headless f16_vs_sa6 --duration 120    # summary per object
radarsim --headless naval_battlegroup --out looks.csv --seed 4
```

See [Command line and Python](headless.md) for batch runs and the Python API.
