"""
A-scope: received power against range along the current beam.

The trace is referred to the thermal noise floor (0 dB). Noise is drawn as a
fresh exponential-power realization each frame (square-law detector output),
targets inside the main lobe appear as compressed-pulse responses at their
SNR, a noise jammer in the beam raises the floor by its jammer-to-noise ratio,
and the detection threshold for the configured Pfa (plus CFAR loss) is drawn
as a line.
"""

import numpy as np
from PySide6.QtCore import Qt

from radarsim.physics.geometry import wrap_to_pi
from radarsim.simulation.frame import FrameState

from ..i18n import tr
from ..theme import palette
from .base import ScopeView, pen

N_BINS = 600


class AScopeView(ScopeView):
    title = "A-Scope"

    def __init__(self, parent=None):
        super().__init__(parent, axes=True)
        p = palette()
        self._rng = np.random.default_rng(0)
        self.plot.setLabel("bottom", tr("Range"), units="km")
        self.plot.setLabel("left", tr("Power above noise"), units="dB")
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.view_box.setRange(yRange=(-12, 45), padding=0)
        self._trace = self.plot.plot(pen=pen(p.echo, 1.0, 0.9))
        self._threshold = self.plot.plot(pen=pen(p.threshold, 1.0, style=Qt.PenStyle.DashLine))
        self._info = self.label("", p.scope_label, 10)
        self.max_range_km = 100.0

    def set_max_range(self, km: float) -> None:
        self.max_range_km = float(km)

    def render(self, frame: FrameState, extrapolate_s: float) -> None:
        radar = frame.radar
        r_max = min(self.max_range_km, radar.unambiguous_range_m / 1000.0)
        ranges = np.linspace(0.2, r_max, N_BINS)
        noise = self._rng.exponential(1.0, N_BINS)

        resolution_km = max(radar.range_resolution_m / 1000.0, (ranges[1] - ranges[0]))
        signal = np.zeros(N_BINS)
        jnr = 0.0
        for target in frame.targets:
            off = abs(wrap_to_pi(target.azimuth - radar.antenna_azimuth))
            if off > radar.beamwidth_az:
                continue
            gain = np.exp(-4 * np.log(2) * (off / radar.beamwidth_az) ** 2) ** 2
            if target.jammer_active and target.jammer_loss_db > 0.0 and "noise" in target.ecm_type:
                jnr += (10 ** (target.jammer_loss_db / 10) - 1.0) * gain
            if np.isfinite(target.snr_db) and not target.masked:
                x = (ranges - target.range_m / 1000.0) / resolution_km
                signal += 10 ** (target.snr_db / 10) * gain * np.sinc(x) ** 2
        power = noise * (1.0 + jnr) + signal
        self._trace.setData(ranges, 10 * np.log10(np.maximum(power, 1e-3)))

        pfa = frame.settings.get("pfa", 1e-6)
        threshold_db = 10 * np.log10(-np.log(pfa)) + frame.settings.get("cfar_loss_db", 0.0)
        floor_db = 10 * np.log10(1.0 + jnr)
        self._threshold.setData([0, r_max], [threshold_db + floor_db] * 2)
        self.view_box.setRange(xRange=(0, r_max), padding=0)
        self._info.setPos(r_max * 0.02, 42)
        self._info.setText(
            f"AZ {np.degrees(radar.antenna_azimuth) % 360:05.1f}   "
            f"Pfa {pfa:.0e}   threshold {threshold_db:.1f} dB"
            + (f"   JNR {10 * np.log10(1 + jnr):.1f} dB" if jnr > 0 else "")
        )
