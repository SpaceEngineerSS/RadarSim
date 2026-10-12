"""
Right panel: radar, processing, environment and electronic-attack settings.

Every control writes straight to the engine through the controller and is
refreshed from the engine when a scenario is loaded, so what is shown is
always what is simulated.
"""

from typing import Any, Dict

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ..i18n import tr
from ..widgets import ChoiceBox, NumberField, Section, ToggleSwitch


def engine_snapshot(engine) -> Dict[str, Any]:
    """Values shown in the settings panel (called on the simulation thread)."""
    radar = engine.radar
    return {
        "frequency_hz": radar.frequency_hz,
        "power_watts": radar.power_watts,
        "prf_hz": radar.prf_hz,
        "pulse_width_s": radar.pulse_width_s,
        "receiver_bandwidth_hz": radar.receiver_bandwidth_hz,
        "antenna_gain_db": radar.antenna_gain_db,
        "beamwidth_az_rad": radar.beamwidth_rad,
        "beamwidth_el_rad": radar.beamwidth_el_rad,
        "scan_rate_rpm": radar.scan_rate_rpm,
        "scan_type": radar.scan_type.value,
        "pfa": engine.probability_false_alarm,
        "pulses": engine.pulses_integrated,
        "cfar": engine.cfar.cfar_type if engine.cfar else None,
        "cfar_cells": engine.cfar.reference_cells if engine.cfar else 16,
        "mti": engine.mti_enabled,
        "monopulse": engine.monopulse_enabled,
        "agility": engine.frequency_agility_enabled,
        "pulse_doppler": engine.pulse_doppler_enabled,
        "clutter": engine.clutter_enabled,
        "terrain_type": engine.terrain_type,
        "sea_state": engine.sea_state,
        "rain": engine.rain_rate_mm_hr,
        "ecm": engine.ecm_active,
        "ecm_type": engine.ecm_type,
        "has_jammers": any(t.has_jammer for t in engine.targets),
    }


class SettingsPanel(QWidget):
    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.c = controller
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        radar = Section(tr("Radar"))
        self.frequency = radar.add_row(tr("Frequency"), NumberField("GHz", 0.1, 100, 3, 1e9, 0.1))
        self.power = radar.add_row(tr("Peak power"), NumberField("kW", 0.001, 10000, 1, 1e3, 10))
        self.prf = radar.add_row(tr("PRF"), NumberField("kHz", 0.05, 500, 3, 1e3, 0.1),
                                 tr("Pulse repetition frequency; sets the unambiguous range c/(2·PRF)."))
        self.pulse = radar.add_row(tr("Pulse width"), NumberField("µs", 0.01, 1000, 2, 1e-6, 0.1))
        self.bandwidth = radar.add_row(tr("Bandwidth"), NumberField("MHz", 0.01, 2000, 2, 1e6, 1),
                                       tr("Receiver bandwidth. τ·B > 1 means pulse compression."))
        self.gain = radar.add_row(tr("Antenna gain"), NumberField("dB", 0, 60, 1, step=0.5))
        self.beam_az = radar.add_row(tr("Beamwidth az"), NumberField("°", 0.1, 30, 2, np.pi / 180, 0.1))
        self.beam_el = radar.add_row(tr("Beamwidth el"), NumberField("°", 0.1, 60, 2, np.pi / 180, 0.1))
        self.scan = radar.add_row(tr("Scan rate"), NumberField("rpm", 0, 120, 1, step=1))
        layout.addWidget(radar)

        processing = Section(tr("Processing"))
        self.pfa = processing.add_row(
            tr("False-alarm rate"),
            ChoiceBox([(f"{v:.0e}", v) for v in (1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)]),
        )
        self.pulses = processing.add_row(tr("Pulses integrated"), NumberField("", 1, 512, 0, step=1))
        self.cfar = processing.add_row(
            tr("CFAR"), ChoiceBox([(tr("Off (known noise)"), None)] + [(f"{t}-CFAR", t) for t in ("CA", "GO", "SO", "OS")]),
            tr("Adaptive threshold; its CFAR loss is applied to every look."),
        )
        self.cfar_cells = processing.add_row(tr("Reference cells"), NumberField("", 4, 128, 0, step=2))
        self.mti = processing.add_row(tr("MTI canceller"), ToggleSwitch(),
                                      tr("Two-pulse canceller: removes slow returns and clutter."))
        self.monopulse = processing.add_row(tr("Monopulse"), ToggleSwitch(),
                                            tr("Sum/difference angle estimate instead of fixed angle noise."))
        self.agility = processing.add_row(tr("Frequency agility"), ToggleSwitch(),
                                          tr("Forces noise jammers to spread over a 10% band."))
        self.pulse_doppler = processing.add_row(tr("Pulse-Doppler map"), ToggleSwitch())
        layout.addWidget(processing)

        environment = Section(tr("Environment"))
        self.clutter = environment.add_row(tr("Surface clutter"), ToggleSwitch())
        self.terrain = environment.add_row(
            tr("Surface"),
            ChoiceBox([(tr(t.capitalize()), t) for t in ("rural", "urban", "forest", "desert", "mountain", "sea")]),
        )
        self.sea_state = environment.add_row(tr("Sea state"), NumberField("", 0, 6, 0, step=1))
        self.rain = environment.add_row(tr("Rain rate"), NumberField("mm/h", 0, 150, 1, step=1))
        layout.addWidget(environment)

        attack = Section(tr("Electronic attack"))
        self.ecm = attack.add_row(tr("Jammers on"), ToggleSwitch())
        self.ecm_type = attack.add_row(
            tr("Technique"),
            ChoiceBox([(tr("As in scenario"), "per_target"), (tr("Barrage noise"), "noise_barrage"),
                       (tr("Spot noise"), "noise_spot"), ("DRFM", "drfm"), (tr("Chaff"), "chaff"),
                       (tr("Decoy"), "decoy")]),
        )
        layout.addWidget(attack)
        layout.addStretch()
        self._connect()

    def _connect(self) -> None:
        c = self.c
        self.frequency.committed.connect(lambda v: c.set_radar(frequency_hz=v))
        self.power.committed.connect(lambda v: c.set_radar(power_watts=v))
        self.prf.committed.connect(lambda v: c.set_radar(prf_hz=v))
        self.pulse.committed.connect(lambda v: c.set_radar(pulse_width_s=v))
        self.bandwidth.committed.connect(lambda v: c.set_radar(receiver_bandwidth_hz=v))
        self.gain.committed.connect(lambda v: c.set_radar(antenna_gain_db=v))
        self.beam_az.committed.connect(lambda v: c.set_radar(beamwidth_az_rad=v))
        self.beam_el.committed.connect(lambda v: c.set_radar(beamwidth_el_rad=v))
        self.scan.committed.connect(lambda v: c.set_radar(scan_rate_rpm=v))
        self.pfa.chosen.connect(lambda v: c.set_engine(probability_false_alarm=v))
        self.pulses.committed.connect(lambda v: c.set_engine(pulses_integrated=int(v)))
        self.cfar.chosen.connect(lambda v: c.set_cfar(v, int(self.cfar_cells.value())))
        self.cfar_cells.committed.connect(lambda v: c.set_cfar(self.cfar.currentData(), int(v)))
        self.mti.toggled.connect(lambda v: c.set_engine(mti_enabled=v))
        self.monopulse.toggled.connect(lambda v: c.set_engine(monopulse_enabled=v))
        self.agility.toggled.connect(lambda v: c.set_engine(frequency_agility_enabled=v))
        self.pulse_doppler.toggled.connect(lambda v: c.set_pulse_doppler(v))
        self.clutter.toggled.connect(lambda v: c.set_engine(clutter_enabled=v))
        self.terrain.chosen.connect(lambda v: c.set_engine(terrain_type=v))
        self.sea_state.committed.connect(lambda v: c.set_engine(sea_state=int(v)))
        self.rain.committed.connect(lambda v: c.set_engine(rain_rate_mm_hr=v))
        self.ecm.toggled.connect(lambda v: c.set_ecm(v, self.ecm_type.currentData()))
        self.ecm_type.chosen.connect(lambda v: c.set_ecm(self.ecm.isChecked(), v))

    def sync(self, values: Dict[str, Any]) -> None:
        """Show engine values without writing them back."""
        for field, key in (
            (self.frequency, "frequency_hz"), (self.power, "power_watts"), (self.prf, "prf_hz"),
            (self.pulse, "pulse_width_s"), (self.bandwidth, "receiver_bandwidth_hz"),
            (self.gain, "antenna_gain_db"), (self.beam_az, "beamwidth_az_rad"),
            (self.beam_el, "beamwidth_el_rad"), (self.scan, "scan_rate_rpm"),
            (self.pulses, "pulses"), (self.cfar_cells, "cfar_cells"),
            (self.sea_state, "sea_state"), (self.rain, "rain"),
        ):
            field.set_si(values[key])
        self.scan.setEnabled(values["scan_type"] == "circular")
        for switch, key in (
            (self.mti, "mti"), (self.monopulse, "monopulse"), (self.agility, "agility"),
            (self.pulse_doppler, "pulse_doppler"), (self.clutter, "clutter"), (self.ecm, "ecm"),
        ):
            switch.set_checked_silent(bool(values[key]))
        self.pfa.set_data_silent(values["pfa"])
        self.cfar.set_data_silent(values["cfar"])
        self.terrain.set_data_silent(values["terrain_type"])
        self.ecm_type.set_data_silent(values["ecm_type"])
