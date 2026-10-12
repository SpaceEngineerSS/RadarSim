"""
Regenerate the figures used in the documentation.

    python tools/make_figures.py            # writes docs/assets/figures/*.png

Every figure is computed from RadarSim itself, so the documentation shows
what the code does. The style follows the desktop interface palette.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from radarsim.io.scenario_loader import ScenarioLoader, builtin_scenarios  # noqa: E402
from radarsim.physics.metrics import calculate_pd_swerling, cfar_loss_db  # noqa: E402
from radarsim.signal.pulse_doppler import PulseDopplerProcessor  # noqa: E402
from radarsim.simulation.analysis import coverage_curve, monte_carlo_pd, probe_engine  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "figures"

INK = "#1c2126"
MUTED = "#59616a"
GRID = "#d6dadf"
PAPER = "#f7f8f9"
ACCENT = "#c8891f"
SERIES = ["#1c2126", "#c8891f", "#2f7fb0", "#c2443a", "#3f8f45"]


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "font.family": ["IBM Plex Sans", "DejaVu Sans"],
            "font.size": 9.5,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "figure.dpi": 150,
        }
    )
    for font in (Path(__file__).resolve().parents[1] / "radarsim" / "ui" / "assets" / "fonts").glob("*.ttf"):
        matplotlib.font_manager.fontManager.addfont(str(font))


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}.png")


def pd_versus_snr() -> None:
    snr = np.linspace(-5, 30, 300)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for case, colour in zip(range(5), SERIES):
        pd = [calculate_pd_swerling(s, pfa=1e-6, swerling_case=case) for s in snr]
        ax.plot(snr, pd, color=colour, label=f"Swerling {case}")
    ax.set(xlabel="Single-look SNR [dB]", ylabel="Probability of detection", ylim=(0, 1.01),
           title="Detection probability, Pfa = 10⁻⁶, one pulse")
    ax.legend(loc="lower right")
    save(fig, "pd_vs_snr")


def cfar_losses() -> None:
    cells = np.arange(8, 66, 2)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for kind, colour in zip(("CA", "GO", "SO", "OS"), SERIES):
        ax.plot(cells, [cfar_loss_db(kind, int(n), 1e-6) for n in cells], color=colour, label=f"{kind}-CFAR")
    ax.set(xlabel="Reference cells N", ylabel="CFAR loss [dB]", ylim=(0, 9),
           title="CFAR loss against a known-noise detector, Pfa = 10⁻⁶")
    ax.legend()
    save(fig, "cfar_loss")


def coverage_and_monte_carlo() -> None:
    engine = ScenarioLoader(str(next(p for p in builtin_scenarios() if p.stem == "basic_tracking"))).create_simulation_engine()
    probe = probe_engine(engine)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for case, colour in ((0, SERIES[0]), (1, SERIES[1]), (3, SERIES[2])):
        curve = coverage_curve(probe, rcs_m2=1.0, swerling=case, max_range_m=120e3, points=200)
        ax.plot(curve.range_m / 1e3, curve.pd, color=colour, label=f"Swerling {case}")
    result = monte_carlo_pd(probe, np.linspace(20e3, 110e3, 10), looks_per_point=400, rcs_m2=1.0, swerling=1, seed=4)
    km = result.range_m / 1e3
    ax.errorbar(km, result.pd_empirical, yerr=[result.pd_empirical - result.pd_low, result.pd_high - result.pd_empirical],
                fmt="o", ms=3.5, color=INK, ecolor=MUTED, elinewidth=1, capsize=0, label="Monte Carlo, 400 looks (95 %)")
    ax.set(xlabel="Range [km]", ylabel="Probability of detection", ylim=(0, 1.01),
           title="Coverage of the basic_tracking radar for a 1 m² target at 3 km")
    ax.legend(loc="lower left")
    save(fig, "coverage")


def range_doppler() -> None:
    processor = PulseDopplerProcessor(prf_hz=4000.0, n_pulses=64, n_range_bins=1024, bandwidth_hz=5e6,
                                      sample_rate_hz=10e6, pulse_width_s=8e-6, frequency_hz=10e9, window_type="hamming")
    amplitudes = [processor.amplitude_for_output_snr(10 ** (s / 10)) for s in (35, 28, 22)]
    rd = processor.process_cpi(np.array([8000.0, 11500.0, 13000.0]), np.array([-40.0, 25.0, 12.0]),
                               np.array(amplitudes), noise_power=1.0, seed=3)
    data = rd.data_db - np.median(rd.data_db)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    image = ax.imshow(data, aspect="auto", origin="lower", cmap="cividis", vmin=0, vmax=40,
                      extent=(rd.range_axis_m[0] / 1e3, rd.range_axis_m[-1] / 1e3,
                              rd.velocity_axis_mps[0], rd.velocity_axis_mps[-1]))
    ax.grid(False)
    ax.set(xlabel="Range [km]", ylabel="Radial velocity [m/s]", xlim=(6, 15),
           title="Range–Doppler map: matched filter, Hamming window, 64 pulses")
    fig.colorbar(image, ax=ax, label="dB above noise", pad=0.02)
    save(fig, "range_doppler")


def detection_timeline() -> None:
    names = ("naval_battlegroup", "mountain_ambush", "f16_vs_sa6")
    fig, axes = plt.subplots(len(names), 1, figsize=(6.4, 5.2), sharex=False)
    for ax, name in zip(axes, names):
        engine = ScenarioLoader(str(next(p for p in builtin_scenarios() if p.stem == name))).create_simulation_engine()
        engine.seed = 5
        engine.rng = np.random.default_rng(5)
        log = engine.run(120.0)
        targets = {t.target_id: t.name for t in engine.targets}
        for row, (target_id, label) in enumerate(targets.items()):
            looks = [r for r in log.detection_history if r.target_id == target_id]
            hit = [r.time for r in looks if r.is_detected]
            miss = [r.time for r in looks if not r.is_detected]
            ax.plot(miss, [row] * len(miss), "|", color=GRID, ms=8)
            ax.plot(hit, [row] * len(hit), "|", color=ACCENT if "Chaff" in label or "MALD" in label else INK, ms=8)
        ax.set_yticks(range(len(targets)), list(targets.values()), fontsize=8)
        ax.set_title(name.replace("_", " "), fontsize=9.5)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, 120)
    axes[-1].set_xlabel("Simulated time [s]  (black: detection, grey: look without detection)")
    save(fig, "detection_timeline")


def main() -> int:
    style()
    pd_versus_snr()
    cfar_losses()
    coverage_and_monte_carlo()
    range_doppler()
    detection_timeline()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
