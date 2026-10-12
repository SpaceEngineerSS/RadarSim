"""Headless command-line scenario runner."""

import csv
import io
from pathlib import Path

from radarsim.__main__ import main
from radarsim.cli import run_headless

SCENARIOS = Path(__file__).parents[1] / "radarsim" / "scenarios"


def test_headless_run_prints_summary_and_writes_csv(tmp_path):
    out = tmp_path / "looks.csv"
    stream = io.StringIO()
    code = run_headless(
        str(SCENARIOS / "basic_tracking.json"), duration_s=30, seed=1,
        out_path=str(out), stream=stream,
    )
    text = stream.getvalue()
    assert code == 0
    assert "Target 1" in text and "confirmed" in text
    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    assert rows and {"target_id", "snr_db", "is_detected"} <= set(rows[0])


def test_main_dispatches_headless_without_qt(capsys):
    assert main(["--headless", str(SCENARIOS / "basic_tracking.json"), "--duration", "5"]) == 0
    assert "Basic Tracking" in capsys.readouterr().out


def test_version_flag_prints_package_version(capsys):
    import pytest

    from radarsim import __version__
    from radarsim.__main__ import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out
