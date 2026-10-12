"""Transport bar: play/pause, single step, restart, speed, record, replay scrubber."""

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QSlider, QToolButton

from ..i18n import tr
from ..theme import SPACE
from ..theme.icons import icon
from ..widgets import Segmented

SPEEDS = (0.5, 1.0, 2.0, 5.0, 10.0)


def _button(name: str, tip: str, checkable: bool = False) -> QToolButton:
    button = QToolButton()
    button.setIcon(icon(name, 18))
    button.setIconSize(QSize(18, 18))
    button.setToolTip(tip)
    button.setCheckable(checkable)
    button.setAutoRaise(True)
    return button


class TransportBar(QFrame):
    play_toggled = Signal()
    step_requested = Signal()
    restart_requested = Signal()
    speed_changed = Signal(float)
    record_toggled = Signal(bool)
    seek_requested = Signal(float)
    exit_replay = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setProperty("rule", "top")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(SPACE.m, SPACE.xs, SPACE.m, SPACE.xs)
        layout.setSpacing(SPACE.xs)

        self.play = _button("pause", tr("Play / pause (Space)"))
        self.play.clicked.connect(self.play_toggled)
        self.step = _button("step", tr("Single step (.)"))
        self.step.clicked.connect(self.step_requested)
        self.restart = _button("reset", tr("Restart scenario (R)"))
        self.restart.clicked.connect(self.restart_requested)
        for widget in (self.play, self.step, self.restart):
            layout.addWidget(widget)

        layout.addSpacing(SPACE.m)
        self.speed = Segmented([(f"{s:g}×", s) for s in SPEEDS])
        self.speed.set_value(1.0)
        self.speed.selected.connect(lambda value: self.speed_changed.emit(float(value)))
        layout.addWidget(self.speed)

        layout.addSpacing(SPACE.m)
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, 1000)
        self.scrub.sliderMoved.connect(lambda v: self.seek_requested.emit(v / 1000.0 * self._duration))
        self.scrub.setVisible(False)
        layout.addWidget(self.scrub, 1)
        self._spacer_index = layout.count()
        layout.addStretch(1)

        self.replay_label = QLabel("")
        self.replay_label.setProperty("role", "value")
        self.replay_label.setVisible(False)
        layout.addWidget(self.replay_label)
        self.leave_replay = _button("close", tr("Leave replay"))
        self.leave_replay.clicked.connect(self.exit_replay)
        self.leave_replay.setVisible(False)
        layout.addWidget(self.leave_replay)

        self.record = _button("record", tr("Record to HDF5"), checkable=True)
        self.record.toggled.connect(self.record_toggled)
        layout.addWidget(self.record)
        self._duration = 1.0

    def set_running(self, running: bool) -> None:
        self.play.setIcon(icon("pause" if running else "play", 18))
        self.step.setEnabled(not running)

    def set_recording(self, recording: bool) -> None:
        self.record.blockSignals(True)
        self.record.setChecked(recording)
        self.record.blockSignals(False)
        self.record.setIcon(icon("record", 18, "#e2685e" if recording else None))

    def set_replay(self, active: bool) -> None:
        self.scrub.setVisible(active)
        self.replay_label.setVisible(active)
        self.leave_replay.setVisible(active)
        self.record.setVisible(not active)
        self.restart.setVisible(not active)
        self.step.setVisible(not active)

    def set_replay_position(self, time_s: float, duration_s: float) -> None:
        self._duration = max(duration_s, 1e-6)
        if not self.scrub.isSliderDown():
            self.scrub.setValue(int(time_s / self._duration * 1000))
        self.replay_label.setText(f"{time_s:6.1f} / {duration_s:.1f} s")
