"""Inspector: live readout of the selected object, editable in editor mode."""

from typing import Optional

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QPushButton, QVBoxLayout, QWidget

from radarsim.simulation.frame import FrameState, TargetFrame

from ..i18n import tr
from ..theme import SPACE, palette
from ..widgets import ChoiceBox, NumberField, Section, ValueGrid, caption

READOUT = (
    "Range", "Azimuth", "Elevation", "Altitude", "Speed", "Heading",
    "Radial velocity", "RCS", "SNR (last look)", "Pd (last look)",
    "Clutter loss", "Jamming loss", "Status",
)


class Inspector(QWidget):
    edited = Signal(int, dict)  # target id, changed fields (SI units)
    delete_requested = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.section = Section(tr("Selection"))
        layout.addWidget(self.section)
        self._empty = caption(tr("Select an object on the scope or in the list."))
        self._empty.setWordWrap(True)
        self.section.add_widget(self._empty)
        self._name = caption("", "title")
        self.section.add_widget(self._name)
        self._grid = ValueGrid([tr(text) for text in READOUT])
        self.section.add_widget(self._grid)
        self._editor = self._build_editor()
        self.section.add_widget(self._editor)
        self._target_id: Optional[int] = None
        self._editing = False
        self._show(False)

    def _build_editor(self) -> QWidget:
        box = QWidget()
        form = QVBoxLayout(box)
        form.setContentsMargins(0, SPACE.s, 0, 0)
        form.setSpacing(SPACE.s)
        self.edit_name = QLineEdit()
        self.edit_affiliation = ChoiceBox([(tr(a.capitalize()), a) for a in ("hostile", "friend", "neutral", "unknown")])
        self.edit_rcs = NumberField("m²", 1e-4, 1e5, 3)
        self.edit_swerling = ChoiceBox([(f"Swerling {i}", i) for i in range(5)])
        self.edit_altitude = NumberField("m", 0, 30000, 0, step=100)
        self.edit_speed = NumberField("m/s", 0, 3000, 0, step=10)
        self.edit_heading = NumberField("°", 0, 360, 0, step=5)
        self.edit_turn = NumberField("°/s", -30, 30, 1, step=0.5)
        rows = [
            (tr("Name"), self.edit_name), (tr("Affiliation"), self.edit_affiliation),
            (tr("RCS"), self.edit_rcs), (tr("Fluctuation"), self.edit_swerling),
            (tr("Altitude"), self.edit_altitude), (tr("Speed"), self.edit_speed),
            (tr("Heading"), self.edit_heading), (tr("Turn rate"), self.edit_turn),
        ]
        for label, widget in rows:
            row = QHBoxLayout()
            row.addWidget(caption(label), 1)
            row.addWidget(widget, 1)
            form.addLayout(row)
        delete = QPushButton(tr("Delete object"))
        delete.setProperty("variant", "ghost")
        delete.clicked.connect(lambda: self._target_id is not None and self.delete_requested.emit(self._target_id))
        form.addWidget(delete)

        self.edit_name.editingFinished.connect(lambda: self._emit(name=self.edit_name.text()))
        self.edit_affiliation.chosen.connect(lambda v: self._emit(affiliation=v))
        self.edit_rcs.committed.connect(lambda v: self._emit(rcs_m2=v))
        self.edit_swerling.chosen.connect(lambda v: self._emit(swerling=v))
        self.edit_altitude.committed.connect(lambda v: self._emit(altitude_m=v))
        self.edit_speed.committed.connect(lambda v: self._emit(speed_mps=v))
        self.edit_heading.committed.connect(lambda v: self._emit(heading_deg=v))
        self.edit_turn.committed.connect(lambda v: self._emit(turn_rate_deg_s=v))
        return box

    def _emit(self, **fields) -> None:
        if self._target_id is not None:
            self.edited.emit(self._target_id, fields)

    def set_editing(self, editing: bool) -> None:
        self._editing = editing
        self._show(self._target_id is not None)

    def _show(self, has_target: bool) -> None:
        self._empty.setVisible(not has_target)
        self._name.setVisible(has_target)
        self._grid.setVisible(has_target and not self._editing)
        self._editor.setVisible(has_target and self._editing)

    def set_target(self, target_id: Optional[int]) -> None:
        self._target_id = target_id
        self._grid.clear()
        self._show(target_id is not None)

    def update_frame(self, frame: FrameState) -> None:
        if self._target_id is None:
            return
        target = frame.target(self._target_id)
        if target is None:
            self.set_target(None)
            return
        self._name.setText(f"{target.name} · {tr(target.affiliation)} {tr(target.target_type)}")
        if self._editing:
            self._fill_editor(target)
            return
        p = palette()
        g = self._grid
        g.set(tr("Range"), f"{target.range_m / 1000:8.2f} km")
        g.set(tr("Azimuth"), f"{np.degrees(target.azimuth) % 360:6.1f}°")
        g.set(tr("Elevation"), f"{np.degrees(target.elevation):6.2f}°")
        g.set(tr("Altitude"), f"{target.altitude:8.0f} m")
        g.set(tr("Speed"), f"{target.speed:6.0f} m/s")
        g.set(tr("Heading"), f"{np.degrees(target.heading) % 360:6.1f}°")
        g.set(tr("Radial velocity"), f"{target.radial_velocity:+7.1f} m/s")
        g.set(tr("RCS"), f"{target.rcs_m2:g} m²")
        snr = "–" if not np.isfinite(target.snr_db) or target.masked else f"{target.snr_db:6.1f} dB"
        g.set(tr("SNR (last look)"), snr)
        g.set(tr("Pd (last look)"), f"{target.pd:5.2f}", p.ok if target.pd >= 0.9 else None)
        g.set(tr("Clutter loss"), f"{target.clutter_loss_db:5.1f} dB")
        g.set(tr("Jamming loss"), f"{target.jammer_loss_db:5.1f} dB", p.hostile if target.jammer_loss_db > 10 else None)
        if target.masked:
            status, colour = tr("hidden by terrain"), p.text_faint
        elif not np.isfinite(target.last_look_time):
            status, colour = tr("not looked at yet"), p.text_faint
        else:
            ago = max(frame.time - target.last_look_time, 0.0)
            outcome = tr("detected") if target.last_look_detected else tr("missed")
            status = f"{outcome} · {ago:.0f} s"
            colour = None if target.last_look_detected else p.accent
        g.set(tr("Status"), status, colour)

    def _fill_editor(self, target: TargetFrame) -> None:
        if not self.edit_name.hasFocus():
            self.edit_name.setText(target.name)
        self.edit_affiliation.set_data_silent(target.affiliation)
        for field, value in (
            (self.edit_rcs, target.rcs_m2),
            (self.edit_altitude, target.altitude),
            (self.edit_speed, target.speed),
            (self.edit_heading, np.degrees(target.heading) % 360),
        ):
            if not field.hasFocus():
                field.set_si(value)
