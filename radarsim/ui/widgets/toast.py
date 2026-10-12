"""Transient notifications that slide in above the transport bar."""

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget
from shiboken6 import isValid

from ..motion import animate_property, fade_in
from ..theme import MOTION, SPACE, palette
from ..theme.icons import icon

_KIND_COLOUR = {"info": "text_muted", "ok": "ok", "warning": "accent", "error": "error"}
_KIND_ICON = {"info": "info", "ok": "info", "warning": "warning", "error": "warning"}


class Toast(QFrame):
    def __init__(self, parent: QWidget, text: str, kind: str = "info", timeout_ms: int = 3200):
        super().__init__(parent)
        p = palette()
        colour = getattr(p, _KIND_COLOUR.get(kind, "text_muted"))
        self.setStyleSheet(
            f"QFrame {{ background: {p.raised}; border: 1px solid {p.line_strong};"
            f" border-left: 2px solid {colour}; border-radius: 3px; }}"
            "QLabel { border: none; background: transparent; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE.m, SPACE.s, SPACE.lg, SPACE.s)
        layout.setSpacing(SPACE.s)
        glyph = QLabel()
        glyph.setPixmap(icon(_KIND_ICON.get(kind, "info"), 16, colour).pixmap(16, 16))
        layout.addWidget(glyph)
        message = QLabel(text)
        message.setWordWrap(True)
        message.setMaximumWidth(420)
        layout.addWidget(message)
        self.adjustSize()
        QTimer.singleShot(timeout_ms, self.close)


class ToastHost:
    """Stacks toasts in the bottom-right corner of ``parent``."""

    def __init__(self, parent: QWidget, bottom_margin: int = 64):
        self._parent = parent
        self._bottom = bottom_margin
        self._toasts = []

    def show(self, text: str, kind: str = "info") -> None:
        toast = Toast(self._parent, text, kind)
        toast.destroyed.connect(lambda *_: self._forget(toast))
        toast.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._toasts.append(toast)
        self._layout(animate_last=True)
        toast.show()
        toast.raise_()
        fade_in(toast, MOTION.normal)

    def _forget(self, toast) -> None:
        if toast in self._toasts:
            self._toasts.remove(toast)
            if isValid(self._parent):  # the window may already be gone at shutdown
                self._layout()

    def _layout(self, animate_last: bool = False) -> None:
        y = self._parent.height() - self._bottom
        for index, toast in enumerate(reversed(self._toasts)):
            y -= toast.height() + SPACE.s
            x = self._parent.width() - toast.width() - SPACE.lg
            target = QPoint(x, y)
            if animate_last and index == 0:
                animate_property(toast, b"pos", QPoint(x, y + 12), target, MOTION.normal)
            else:
                toast.move(target)

    def relayout(self) -> None:
        self._layout()
