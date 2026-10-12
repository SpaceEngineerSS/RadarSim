"""Left panel: scenario summary and the list of objects in the scene."""

from typing import Dict, Optional

from PySide6.QtCore import QPointF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from src.simulation.frame import FrameState

from ..i18n import tr
from ..theme import SPACE, affiliation_colour, palette
from ..theme.fonts import mono_font
from ..widgets import caption

_GLYPHS: Dict[str, QIcon] = {}


def _glyph(colour: str, affiliation: str, target_type: str) -> QIcon:
    key = f"{colour}-{affiliation}-{target_type}"
    if key in _GLYPHS:
        return _GLYPHS[key]
    pixmap = QPixmap(QSize(28, 28))
    pixmap.setDevicePixelRatio(2.0)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    pen = painter.pen()
    pen.setColor(QColor(colour))
    pen.setWidthF(1.4)
    painter.setPen(pen)
    if target_type in ("chaff", "decoy"):
        painter.drawLine(3, 3, 11, 11)
        painter.drawLine(11, 3, 3, 11)
    elif affiliation == "friend":
        painter.drawEllipse(2, 2, 10, 10)
    elif affiliation == "hostile":
        painter.drawPolygon(
            QPolygonF([QPointF(7, 1.5), QPointF(12.5, 7), QPointF(7, 12.5), QPointF(1.5, 7)])
        )
    elif affiliation == "neutral":
        painter.drawRect(2, 2, 10, 10)
    else:
        painter.drawEllipse(4, 1, 6, 6)
        painter.drawEllipse(4, 7, 6, 6)
        painter.drawEllipse(1, 4, 6, 6)
        painter.drawEllipse(7, 4, 6, 6)
    painter.end()
    _GLYPHS[key] = QIcon(pixmap)
    return _GLYPHS[key]


class ObjectsPanel(QFrame):
    target_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setProperty("rule", "right")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, SPACE.m, 0, 0)
        layout.setSpacing(SPACE.xs)

        self._title = QLabel("–")
        self._title.setProperty("role", "title")
        self._title.setWordWrap(True)
        self._description = caption("")
        self._description.setWordWrap(True)
        for widget in (self._title, self._description):
            widget.setContentsMargins(SPACE.lg, 0, SPACE.lg, 0)
            layout.addWidget(widget)

        header = caption(tr("OBJECTS"), "section")
        header.setContentsMargins(SPACE.lg, SPACE.m, SPACE.lg, 0)
        layout.addWidget(header)
        self._list = QTreeWidget()
        self._list.setColumnCount(2)
        self._list.setHeaderHidden(True)
        self._list.setRootIsDecorated(False)
        self._list.setIndentation(0)
        self._list.setIconSize(QSize(14, 14))
        self._list.setUniformRowHeights(True)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list.header().setStretchLastSection(False)
        self._list.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._list.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._list.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self._list, 1)
        self._items: Dict[int, QTreeWidgetItem] = {}
        self._summary = caption("")
        self._summary.setWordWrap(True)
        self._summary.setContentsMargins(SPACE.lg, SPACE.s, SPACE.lg, SPACE.m)
        layout.addWidget(self._summary)
        self._muted = False

    def set_scenario(self, name: str, description: str) -> None:
        self._title.setText(name)
        self._description.setText(description)
        self._list.clear()
        self._items.clear()

    def update_frame(self, frame: FrameState) -> None:
        p = palette()
        present = set()
        for target in frame.targets:
            present.add(target.target_id)
            item = self._items.get(target.target_id)
            if item is None:
                item = QTreeWidgetItem([target.name, ""])
                item.setData(0, Qt.ItemDataRole.UserRole, target.target_id)
                item.setIcon(0, _glyph(affiliation_colour(p, target.affiliation, target.target_type),
                                       target.affiliation, target.target_type))
                item.setTextAlignment(1, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._list.addTopLevelItem(item)
                self._items[target.target_id] = item
            state = (
                tr("masked") if target.masked
                else tr("jammed") if target.jammer_loss_db > 10
                else f"{target.range_m / 1000:5.1f} km"
            )
            if item.text(0) != target.name:
                item.setText(0, target.name)
            if item.text(1) != state:
                item.setText(1, state)
                item.setFont(1, mono_font(11))
            item.setForeground(0, QColor(p.text_faint if target.masked else p.text))
            item.setForeground(1, QColor(p.text_muted))
        for target_id in [t for t in self._items if t not in present]:
            item = self._items.pop(target_id)
            self._list.takeTopLevelItem(self._list.indexOfTopLevelItem(item))
        confirmed = sum(1 for t in frame.tracks if t.status == "confirmed")
        self._summary.setText(
            f"{len(frame.targets)} {tr('objects')} · {confirmed} {tr('confirmed tracks')} · "
            f"{frame.detections}/{frame.looks} {tr('detections')}"
        )

    def select(self, target_id: Optional[int]) -> None:
        self._muted = True
        self._list.clearSelection()
        item = self._items.get(target_id) if target_id is not None else None
        if item is not None:
            item.setSelected(True)
            self._list.scrollToItem(item)
        self._muted = False

    def _on_selection(self) -> None:
        if self._muted:
            return
        items = self._list.selectedItems()
        if items:
            self.target_selected.emit(int(items[0].data(0, Qt.ItemDataRole.UserRole)))
