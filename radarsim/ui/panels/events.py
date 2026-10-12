"""Event log: what happened and when, with a category filter and export."""

from typing import List

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from radarsim.simulation.events import EventKind, SimEvent

from ..i18n import tr
from ..theme import SPACE, palette
from ..theme.fonts import mono_font
from ..widgets import ChoiceBox

MAX_ROWS = 2000

CATEGORIES = {
    "all": None,
    "tracks": {EventKind.TRACK_INITIATED, EventKind.TRACK_CONFIRMED, EventKind.TRACK_LOST},
    "detections": {EventKind.FIRST_DETECTION, EventKind.DEPLOYED},
    "ew": {EventKind.JAMMED, EventKind.BURN_THROUGH},
    "terrain": {EventKind.MASKED, EventKind.UNMASKED},
}


def _colour(kind: EventKind) -> str:
    p = palette()
    return {
        EventKind.TRACK_CONFIRMED: p.ok,
        EventKind.TRACK_LOST: p.accent,
        EventKind.JAMMED: p.hostile,
        EventKind.BURN_THROUGH: p.ok,
        EventKind.MASKED: p.text_faint,
    }.get(kind, p.text)


class EventLog(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACE.m, SPACE.xs, SPACE.m, 0)
        bar = QHBoxLayout()
        self.filter = ChoiceBox(
            [(tr("All events"), "all"), (tr("Tracks"), "tracks"), (tr("Detections"), "detections"),
             (tr("Electronic attack"), "ew"), (tr("Terrain"), "terrain")]
        )
        self.filter.chosen.connect(lambda _: self._refill())
        bar.addWidget(self.filter)
        bar.addStretch()
        layout.addLayout(bar)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels([tr("Time"), tr("Event"), tr("Detail")])
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22)
        layout.addWidget(self.table)
        self.events: List[SimEvent] = []

    def clear(self) -> None:
        self.events.clear()
        self.table.setRowCount(0)

    def add(self, events) -> None:
        if not events:
            return
        self.events.extend(events)
        if len(self.events) > MAX_ROWS:
            self.events = self.events[-MAX_ROWS:]
            self._refill()
            return
        allowed = CATEGORIES[self.filter.currentData()]
        for event in events:
            if allowed is None or event.kind in allowed:
                self._append(event)
        self.table.scrollToBottom()

    def _refill(self) -> None:
        self.table.setRowCount(0)
        allowed = CATEGORIES[self.filter.currentData()]
        for event in self.events:
            if allowed is None or event.kind in allowed:
                self._append(event)
        self.table.scrollToBottom()

    def _append(self, event: SimEvent) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        cells = (
            f"{int(event.time // 60):02d}:{event.time % 60:05.2f}",
            tr(event.kind.value.replace("_", " ")),
            event.message,
        )
        for column, text in enumerate(cells):
            item = QTableWidgetItem(text)
            if column == 0:
                item.setFont(mono_font(11))
                item.setForeground(QColor(palette().text_muted))
            elif column == 1:
                item.setForeground(QColor(_colour(event.kind)))
            item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            self.table.setItem(row, column, item)
