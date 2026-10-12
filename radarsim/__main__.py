"""RadarSim entry point: desktop application, or a headless scenario run."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    import argparse

    from radarsim import __version__
    from radarsim.cli import add_headless_arguments, list_scenarios, run_headless

    parser = argparse.ArgumentParser(prog="radarsim", description="RadarSim radar simulator")
    parser.add_argument("scenario", nargs="?", default=None, help="scenario YAML or JSON file")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    add_headless_arguments(parser)
    args, qt_args = parser.parse_known_args(sys.argv[1:] if argv is None else argv)

    if args.list:
        return list_scenarios()
    if args.scenario is not None and not args.headless:
        from radarsim.io.scenario_loader import resolve_scenario

        try:
            args.scenario = resolve_scenario(args.scenario)
        except FileNotFoundError as exc:
            parser.error(str(exc))
    if args.headless:
        if args.scenario is None:
            parser.error("--headless needs a scenario file")
        return run_headless(args.scenario, duration_s=args.duration, seed=args.seed, out_path=args.out)

    try:
        from radarsim.ui.main_window import MainWindow, ensure_application
    except ImportError as exc:
        print("The desktop interface needs the GUI extras: pip install 'radarsim[gui]'", file=sys.stderr)
        print(f"Missing dependency: {exc.name}", file=sys.stderr)
        return 1

    app = ensure_application([sys.argv[0]] + qt_args)
    window = MainWindow(scenario_path=args.scenario)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
