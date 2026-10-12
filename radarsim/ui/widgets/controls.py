"""Form controls shared by the panels."""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PySide6.QtCore import Property, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..motion import tween
from ..theme import MOTION, SPACE, palette
from ..theme.icons import icon


def caption(text: str, role: str = "caption") -> QLabel:
    label = QLabel(text)
    label.setProperty("role", role)
    return label


def separator() -> QFrame:
    line = QFrame()
    line.setObjectName("Separator")
    return line


class Section(QWidget):
    """Titled group with a collapsible body (animated height)."""

    def __init__(self, title: str, collapsed: bool = False, parent: Optional[QWidget] = None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(SPACE.lg, 0, SPACE.lg, SPACE.s)
        outer.setSpacing(0)

        self._header = QToolButton()
        self._header.setText(title.upper())
        self._header.setCheckable(True)
        self._header.setChecked(not collapsed)
        self._header.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._header.setStyleSheet(
            f"QToolButton {{ color: {palette().text_muted}; font-size: 11px; font-weight: 600;"
            " letter-spacing: 0.6px; border: none; padding: 8px 0 6px 0; }"
            f"QToolButton:hover {{ color: {palette().text}; background: transparent; }}"
            "QToolButton:checked { background: transparent; border: none; }"
        )
        self._header.setIconSize(QSize(14, 14))
        self._header.toggled.connect(self._toggle)
        outer.addWidget(self._header)

        self.body = QWidget()
        self.form = QGridLayout(self.body)
        self.form.setContentsMargins(0, 0, 0, SPACE.xs)
        self.form.setHorizontalSpacing(SPACE.m)
        self.form.setVerticalSpacing(SPACE.s)
        self.form.setColumnStretch(0, 1)
        outer.addWidget(self.body)
        self._rows = 0
        self._update_icon(not collapsed)
        if collapsed:
            self.body.setMaximumHeight(0)

    def add_row(self, label: str, widget: QWidget, hint: str = "") -> QWidget:
        text = caption(label)
        if hint:
            text.setToolTip(hint)
            widget.setToolTip(hint)
        self.form.addWidget(text, self._rows, 0)
        self.form.addWidget(widget, self._rows, 1)
        self._rows += 1
        return widget

    def add_widget(self, widget: QWidget) -> QWidget:
        self.form.addWidget(widget, self._rows, 0, 1, 2)
        self._rows += 1
        return widget

    def _update_icon(self, expanded: bool) -> None:
        self._header.setIcon(icon("chevron-down" if expanded else "chevron-right", 14))

    def _toggle(self, expanded: bool) -> None:
        self._update_icon(expanded)
        target = self.body.sizeHint().height() if expanded else 0
        start = self.body.height() if self.body.maximumHeight() > 0 else 0

        def apply(value):
            self.body.setMaximumHeight(int(value))

        animation = tween(self, start, target, apply, MOTION.normal)
        if expanded:
            animation.finished.connect(lambda: self.body.setMaximumHeight(16_777_215))


class NumberField(QDoubleSpinBox):
    """
    Numeric entry shown in display units, reporting SI values.

    ``scale`` converts display units to SI (e.g. 1e9 for GHz -> Hz). The value
    is committed on Enter / focus loss, not on every keystroke.
    """

    committed = Signal(float)

    def __init__(
        self,
        unit: str,
        minimum: float,
        maximum: float,
        decimals: int = 2,
        scale: float = 1.0,
        step: float = 0.0,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self._scale = scale
        self.setRange(minimum, maximum)
        self.setDecimals(decimals)
        self.setSingleStep(step or 10 ** -decimals * 10)
        self.setSuffix(f" {unit}" if unit else "")
        self.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
        self.setKeyboardTracking(False)
        self.setMinimumWidth(104)
        self.valueChanged.connect(lambda value: self.committed.emit(value * self._scale))

    def set_si(self, value: float) -> None:
        """Show an SI value without emitting ``committed``."""
        self.blockSignals(True)
        self.setValue(value / self._scale)
        self.blockSignals(False)


class ToggleSwitch(QAbstractButton):
    """Two-state switch with an animated knob."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(30, 16)
        self._position = 0.0
        self.toggled.connect(self._animate)

    def _get_position(self) -> float:
        return self._position

    def _set_position(self, value: float) -> None:
        self._position = value
        self.update()

    position = Property(float, _get_position, _set_position)

    def _animate(self, checked: bool) -> None:
        tween(self, self._position, 1.0 if checked else 0.0, self._set_position, MOTION.fast)

    def set_checked_silent(self, checked: bool) -> None:
        self.blockSignals(True)
        self.setChecked(checked)
        self.blockSignals(False)
        self._set_position(1.0 if checked else 0.0)

    def sizeHint(self) -> QSize:
        return QSize(30, 16)

    def paintEvent(self, _event) -> None:
        p = palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        track = QColor(p.line_strong)
        on = QColor(p.accent)
        mix = QColor(
            int(track.red() + (on.red() - track.red()) * self._position),
            int(track.green() + (on.green() - track.green()) * self._position),
            int(track.blue() + (on.blue() - track.blue()) * self._position),
        )
        if not self.isEnabled():
            mix = QColor(p.line)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(mix)
        painter.drawRoundedRect(QRectF(0, 3, 30, 10), 5, 5)
        knob_x = 1 + self._position * 14
        painter.setBrush(QColor(p.text if self.isEnabled() else p.text_faint))
        painter.drawEllipse(QRectF(knob_x, 1, 14, 14))


class ChoiceBox(QComboBox):
    """Combo box whose items carry data; ``chosen`` emits the data."""

    chosen = Signal(object)

    def __init__(self, items: Iterable[Tuple[str, object]], parent: Optional[QWidget] = None):
        super().__init__(parent)
        for text, data in items:
            self.addItem(text, data)
        self.currentIndexChanged.connect(lambda i: self.chosen.emit(self.itemData(i)))

    def set_data_silent(self, data) -> None:
        index = self.findData(data)
        if index >= 0:
            self.blockSignals(True)
            self.setCurrentIndex(index)
            self.blockSignals(False)


class Segmented(QWidget):
    """Mutually exclusive buttons in a row; emits the selected data."""

    selected = Signal(object)

    def __init__(self, options: Sequence[Tuple[str, object]], parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._data: List[object] = []
        for index, (text, data) in enumerate(options):
            button = QToolButton()
            button.setText(text)
            button.setCheckable(True)
            button.setProperty("variant", "segment")
            if index == 0:
                button.setProperty("position", "first")
            elif index == len(options) - 1:
                button.setProperty("position", "last")
            self._group.addButton(button, index)
            self._data.append(data)
            layout.addWidget(button)
        self._group.idClicked.connect(lambda i: self.selected.emit(self._data[i]))

    def set_value(self, data) -> None:
        if data in self._data:
            self._group.button(self._data.index(data)).setChecked(True)


class ValueGrid(QWidget):
    """Read-only label/value pairs with mono values, cheap to refresh."""

    def __init__(self, labels: Sequence[str], columns: int = 1, parent: Optional[QWidget] = None):
        super().__init__(parent)
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(SPACE.m)
        grid.setVerticalSpacing(SPACE.xs + 2)
        self._values: Dict[str, QLabel] = {}
        for index, text in enumerate(labels):
            row, col = divmod(index, columns)
            label = caption(text)
            value = QLabel("–")
            value.setProperty("role", "value")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            grid.addWidget(label, row, col * 2)
            grid.addWidget(value, row, col * 2 + 1)
            self._values[text] = value

    def set(self, label: str, text: str, colour: Optional[str] = None) -> None:
        value = self._values[label]
        if value.text() != text:
            value.setText(text)
        style = f"color: {colour};" if colour else ""
        if value.styleSheet() != style:
            value.setStyleSheet(style)

    def clear(self) -> None:
        for value in self._values.values():
            value.setText("–")
            value.setStyleSheet("")
