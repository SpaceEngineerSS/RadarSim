"""
Short, purposeful motion.

Every animation is a 120-260 ms ease-out (see ``theme.MOTION``). Motion can be
switched off globally ("Reduce motion" in the View menu); animations then jump
straight to their end state.
"""

from PySide6.QtCore import (
    QAbstractAnimation,
    QEasingCurve,
    QObject,
    QPropertyAnimation,
    QVariantAnimation,
)
from PySide6.QtWidgets import QGraphicsOpacityEffect, QWidget

from .theme import MOTION

_reduced = False


def set_reduced_motion(value: bool) -> None:
    global _reduced
    _reduced = bool(value)


def reduced_motion() -> bool:
    return _reduced


def ease_out() -> QEasingCurve:
    return QEasingCurve(QEasingCurve.Type.OutCubic)


def animate_property(
    target: QObject, prop: bytes, start, end, duration: int = MOTION.normal
) -> QPropertyAnimation:
    """Start (and return) a property animation; jumps to ``end`` when reduced."""
    animation = QPropertyAnimation(target, prop, target)
    animation.setDuration(0 if _reduced else duration)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.setEasingCurve(ease_out())
    animation.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
    return animation


def fade_in(widget: QWidget, duration: int = MOTION.normal) -> None:
    """Fade a widget in from transparent; the effect is removed afterwards."""
    if _reduced:
        return
    effect = QGraphicsOpacityEffect(widget)
    widget.setGraphicsEffect(effect)
    animation = animate_property(effect, b"opacity", 0.0, 1.0, duration)
    animation.finished.connect(lambda: widget.setGraphicsEffect(None))


def tween(owner: QObject, start: float, end: float, on_value, duration: int = MOTION.normal):
    """Animate a float, calling ``on_value(v)`` each frame."""
    animation = QVariantAnimation(owner)
    animation.setDuration(0 if _reduced else duration)
    animation.setStartValue(float(start))
    animation.setEndValue(float(end))
    animation.setEasingCurve(ease_out())
    animation.valueChanged.connect(on_value)
    animation.start(QAbstractAnimation.DeletionPolicy.DeleteWhenStopped)
    return animation
