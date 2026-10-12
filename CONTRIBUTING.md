# Contributing

Contributions should keep scientific claims, code, tests, and documentation synchronized.

## Development setup

```bash
git clone https://github.com/SpaceEngineerSS/RadarSim.git
cd RadarSim
python -m venv .venv
python -m pip install -e ".[gui,dev,docs]"
```

Use a topic branch in your fork and open a pull request against `main`. Describe the physical or software requirement, the chosen method, its limitations, and the verification evidence.

## Required checks

```bash
python -m ruff check .
python -m pytest -q
python -m mypy radarsim --exclude radarsim/ui/
python -m bandit -r radarsim -ll
python tools/i18n_check.py tr
mkdocs build --strict
```

For UI changes, run the desktop application, keep the offscreen smoke tests passing and regenerate the screenshots with `python tools/capture_screenshots.py` if the change is visible. New interface strings go through `tr()` and need a Turkish entry.

## Scientific changes

A model change must include a primary publication or standard, SI units, sign conventions, a validity domain, failure behaviour outside that domain, and tests from an analytic identity or published numerical case. Monte Carlo validation needs a fixed seed, trial count, tolerance rationale, and a test that would catch an incorrect distribution or scale.

Do not present example platform names or open-source parameter guesses as measured equipment performance. Do not silently substitute one algorithm for another. If an advertised method is unavailable, return a clear `NotImplementedError` and document it.

## Code style

Use type hints on public functions, focused modules, and docstrings where units or non-obvious contracts matter. Prefer stable linear algebra and explicit validation. Comments should explain a scientific choice, invariant, or unusual constraint; they should not narrate ordinary syntax. Keep UI formulas delegated to the scientific core: `physics`, `signal`, `tracking`, `simulation` and `io` never import Qt.

## Compatibility

Scenario and recording changes need round-trip tests. Breaking public APIs or file formats require a major-version note. New dependencies must be justified and added to `pyproject.toml`.

Contributions are licensed under the repository’s MIT License.
