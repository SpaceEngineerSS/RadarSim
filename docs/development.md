# Development

## Setup

```bash
git clone https://github.com/SpaceEngineerSS/RadarSim.git
cd RadarSim
python -m venv .venv && source .venv/bin/activate
pip install -e ".[gui,dev,docs]"
```

## Checks

```bash
pytest                                   # tests (QT_QPA_PLATFORM=offscreen on headless machines)
ruff check .                             # lint
mypy radarsim --exclude radarsim/ui/     # types of the core package
python tools/i18n_check.py tr            # untranslated interface strings
python benchmarks/engine_benchmark.py    # step and frame cost per scenario
mkdocs serve                             # documentation at http://127.0.0.1:8000
```

## Layout

```text
radarsim/
  physics/      radar equation, propagation, clutter, RCS, ECM, terrain, metrics
  signal/       waveforms, CFAR, pulse-Doppler, antenna patterns, SAR/ISAR
  tracking/     Kalman, EKF, track manager, monopulse
  simulation/   engine, objects, frames, events, analysis, recorder, headless runner,
                multi-radar network
  io/           scenario schema, loader, exporter, replay, reports
  ui/           desktop interface (theme, widgets, views, panels, controller)
  scenarios/    built-in scenarios (package data)
tests/          pytest suite
tools/          figure, screenshot, translation and release helpers
examples/       runnable API examples
benchmarks/     performance scripts
```

Physics, signal, tracking, simulation and io never import Qt; the interface talks to
the engine only through `ui/controller.py` and receives immutable `FrameState`
snapshots from the simulation thread.

## Rules I keep

- Every model states its reference and validity range, and comes with a test against
  something independent of the implementation.
- No unexplained constants: a number that is not a physical constant has a comment
  saying where it comes from.
- Random draws go through the engine's `numpy.random.Generator`, so a seed
  reproduces a run.
- Conventions in [Conventions](conventions.md) are not negotiable.

## Adding a translation

Interface strings are wrapped in `tr("…")`. Add `radarsim/ui/assets/i18n/<code>.json`,
run `python tools/i18n_check.py <code> --write` to fill in the missing keys, translate
them and add the language to `LANGUAGES` in `radarsim/ui/i18n.py`.

## Desktop builds

```bash
pip install -e ".[gui,build]"
pyinstaller --noconfirm --clean RadarSim.spec    # dist/RadarSim/
```

## Releasing

1. Update `version` in `pyproject.toml` and add a section to `CHANGELOG.md`.
2. Tag `vX.Y.Z` and push the tag.
3. The release workflow runs the full CI, builds the wheel and the Windows, Linux and
   macOS applications, publishes them with checksums and the changelog section, and
   deploys this documentation.
