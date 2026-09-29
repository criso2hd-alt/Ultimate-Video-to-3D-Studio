"""First-run welcome and the spotlight tutorial.

Same look and behaviour as the sibling DLSS converter's tour: the window is dimmed, one real control
at a time is lit with a breathing outline, and a small card beside it explains what it is for. The
overlay is presentation only - it points at the widgets the person will actually use, scrolls the
side rail to reach them, and never changes a setting.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import (
    Property,
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .widgets import FONT_DISPLAY, FONT_MONO, apply_font

#: Increment only when people who have already seen the tutorial should be offered a new one.
ONBOARDING_VERSION = 1


def _css_cubic_bezier(x1: float, y1: float, x2: float, y2: float) -> QEasingCurve:
    """The same timing curves the browser prototype (and the DLSS app) use."""
    curve = QEasingCurve(QEasingCurve.Type.BezierSpline)
    curve.addCubicBezierSegment(QPointF(x1, y1), QPointF(x2, y2), QPointF(1.0, 1.0))
    return curve


_FALLBACK_PALETTE = {
    "panel_hi": "#101b2c", "base": "#0d1625", "line": "#29405d", "ink": "#f5f9ff",
    "ink_dim": "#9bacc2", "ink_faint": "#61748d", "signal": "#42dcf5", "signal_light": "#69eaff",
    "signal_deep": "#159bc3", "on_accent": "#041017",
}


@dataclass(frozen=True)
class TourStep:
    """One piece of the real window to explain."""

    title: str
    body: str
    target: QWidget
    #: Run just before the step is shown - scroll the rail to the target, switch tab, and so on.
    prepare: Callable[[], None] | None = field(default=None, compare=False)


class WelcomeDialog(QDialog):
    """Bridge into the tour: what the app does, in a few lines, and a choice to take the tour or skip it."""

    FEATURES = (
        ("AI depth, any GPU", "Depth Anything V2 on DirectML / CoreML - no CUDA, no special card."),
        ("Every 3D format", "Side-by-side, top-and-bottom, anaglyph, colour + depth."),
        ("See it before you convert", "A live 3D preview with views that prove the depth, no glasses needed."),
        ("Grade it too", "Professional colour correction and .cube LUTs, with live scopes."),
        ("All the codecs", "H.264, H.265, AV1, VP9, ProRes, DNxHR, FFV1 - 10-bit HDR stays HDR."),
    )

    def __init__(self, icon_path: Path | None, parent: QWidget | None = None,
                 palette: dict[str, str] | None = None) -> None:
        super().__init__(parent)
        c = {**_FALLBACK_PALETTE, **(palette or {})}
        self.setWindowTitle("Welcome")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 24, 26, 24)
        layout.setSpacing(13)

        head = QHBoxLayout()
        head.setSpacing(16)
        if icon_path is not None and icon_path.is_file():
            glyph = QLabel()
            glyph.setPixmap(QPixmap(str(icon_path)).scaled(
                64, 64, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            glyph.setFixedSize(64, 64)
            head.addWidget(glyph)
        titles = QVBoxLayout()
        titles.setSpacing(4)
        eyebrow = QLabel("WELCOME  ·  FREE, ALWAYS")
        eyebrow.setObjectName("onboardingEyebrow")
        apply_font(eyebrow, family=FONT_MONO, size=8.5, spacing=2.0, caps=True)
        title = QLabel("Turn flat video into 3D.")
        title.setObjectName("onboardingTitle")
        apply_font(title, family=FONT_DISPLAY, size=19)
        titles.addWidget(eyebrow)
        titles.addWidget(title)
        head.addLayout(titles, 1)
        layout.addLayout(head)

        body = QLabel(
            "Choose a video, pick a 3D view under the picture, tune it, and convert. "
            "A short tour will point at each part of the window and say what it is for - "
            "it takes about a minute, and you can replay it any time from Settings.")
        body.setObjectName("onboardingBody")
        body.setWordWrap(True)
        layout.addWidget(body)

        for name, text in self.FEATURES:
            label = QLabel(f'<span style="color:{c["signal"]};">●</span>&nbsp;&nbsp;<b>{name}</b> - {text}')
            label.setObjectName("onboardingBody")
            label.setWordWrap(True)
            layout.addWidget(label)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Show me around")
        self.start_button.setObjectName("onboardingPrimary")
        self.skip_button = QPushButton("Skip tutorial")
        self.skip_button.setObjectName("secondary")
        self.start_button.clicked.connect(self.accept)
        self.skip_button.clicked.connect(self.reject)
        buttons.addWidget(self.start_button, 1)
        buttons.addWidget(self.skip_button)
        layout.addSpacing(6)
        layout.addLayout(buttons)

        self.setStyleSheet(f"""
            QDialog {{ background: {c['panel_hi']}; color: {c['ink']}; }}
            QLabel#onboardingEyebrow {{ color: {c['signal']}; }}
            QLabel#onboardingTitle {{ color: {c['ink']}; font-weight: 700; }}
            QLabel#onboardingBody {{ color: {c['ink_dim']}; font-size: 12px; }}
            QPushButton {{ min-height: 36px; border-radius: 8px; padding: 0 16px; }}
            QPushButton#onboardingPrimary {{
                color: {c['on_accent']}; font-weight: 700;
                border: 1px solid {c['signal_light']};
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {c['signal_light']}, stop:1 {c['signal_deep']});
            }}
            QPushButton#secondary {{
                color: {c['ink_dim']}; border: 1px solid {c['line']}; background: {c['base']};
            }}
        """)


class SpotlightOverlay(QWidget):
    """Dim the window and teach against one real control at a time."""

    finished = Signal()
    skipped = Signal()
    step_changed = Signal(int)

    def __init__(self, parent: QWidget, steps: list[TourStep], palette: dict[str, str] | None = None) -> None:
        super().__init__(parent)
        c = {**_FALLBACK_PALETTE, **(palette or {})}
        self._accent = QColor(c["signal"])
        self._steps = steps
        self._index = 0
        self._target = QRect()
        self._veil_opacity = 0.0
        self._pulse_phase = 0.0
        self._done = False
        self.setObjectName("spotlightOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.card = QFrame(self)
        self.card.setObjectName("tourCard")
        self.card.setFixedWidth(350)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(18, 16, 18, 15)
        card_layout.setSpacing(9)

        self.eyebrow = QLabel()
        self.eyebrow.setObjectName("tourEyebrow")
        apply_font(self.eyebrow, family=FONT_MONO, size=8, spacing=1.8, caps=True)
        self.title = QLabel()
        self.title.setObjectName("tourTitle")
        apply_font(self.title, family=FONT_DISPLAY, size=13)
        self.body = QLabel()
        self.body.setObjectName("tourBody")
        self.body.setWordWrap(True)
        card_layout.addWidget(self.eyebrow)
        card_layout.addWidget(self.title)
        card_layout.addWidget(self.body)

        footer = QHBoxLayout()
        self.skip_button = QPushButton("Skip tour")
        self.skip_button.setObjectName("tourSecondary")
        self.back_button = QPushButton("Back")
        self.back_button.setObjectName("tourSecondary")
        self.next_button = QPushButton("Next")
        self.next_button.setObjectName("tourPrimary")
        self.skip_button.clicked.connect(self._skip)
        self.back_button.clicked.connect(self._back)
        self.next_button.clicked.connect(self._next)
        footer.addStretch(1)
        footer.addWidget(self.skip_button)
        footer.addWidget(self.back_button)
        footer.addWidget(self.next_button)
        card_layout.addLayout(footer)

        self._card_opacity = QGraphicsOpacityEffect(self.card)
        self.card.setGraphicsEffect(self._card_opacity)
        self._card_opacity.setOpacity(0.0)

        # 320 ms for the spotlight and 300 ms for the card, both travelling to the next control
        # rather than cutting - the "guided camera" feel of the prototype.
        self._spot_animation = QPropertyAnimation(self, b"spotlightRect", self)
        self._spot_animation.setDuration(320)
        self._spot_animation.setEasingCurve(_css_cubic_bezier(0.25, 0.1, 0.25, 1.0))
        self._card_animation = QPropertyAnimation(self.card, b"geometry", self)
        self._card_animation.setDuration(300)
        self._card_animation.setEasingCurve(_css_cubic_bezier(0.25, 0.1, 0.25, 1.0))
        self._fade_animation = QPropertyAnimation(self, b"veilOpacity", self)
        self._fade_animation.setDuration(350)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._card_fade = QPropertyAnimation(self._card_opacity, b"opacity", self)
        self._card_fade.setDuration(300)
        self._card_fade.setEasingCurve(QEasingCurve.Type.OutCubic)

        self._pulse = QVariantAnimation(self)
        self._pulse.setStartValue(0.0)
        self._pulse.setEndValue(math.tau)
        self._pulse.setDuration(1800)
        self._pulse.setLoopCount(-1)
        self._pulse.valueChanged.connect(self._set_pulse)

        self.setStyleSheet(f"""
            QFrame#tourCard {{
                background: {c['panel_hi']}; border: 1px solid {c['line']}; border-radius: 12px;
            }}
            QLabel#tourEyebrow {{ color: {c['signal']}; }}
            QLabel#tourTitle {{ color: {c['ink']}; font-weight: 700; }}
            QLabel#tourBody {{ color: {c['ink_dim']}; font-size: 11px; }}
            QLabel#tourCounter {{ color: {c['ink_faint']}; font-family: "IBM Plex Mono"; }}
            QPushButton {{ min-height: 30px; border-radius: 7px; padding: 0 14px; }}
            QPushButton#tourSecondary {{
                color: {c['ink_dim']}; background: {c['base']}; border: 1px solid {c['line']};
            }}
            QPushButton#tourPrimary {{
                color: {c['on_accent']}; font-weight: 700; border: 1px solid {c['signal_light']};
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 {c['signal_light']}, stop:1 {c['signal_deep']});
            }}
        """)
        self._show_step(animate=False)

    @property
    def step_index(self) -> int:
        return self._index

    def start(self) -> None:
        self.setGeometry(self.parentWidget().rect())
        self.raise_()
        self.show()
        self.setFocus()
        self._show_step(animate=False)
        self._fade_animation.setStartValue(0.0)
        self._fade_animation.setEndValue(1.0)
        self._fade_animation.start()
        self._card_fade.setStartValue(0.0)
        self._card_fade.setEndValue(1.0)
        self._card_fade.start()
        self._pulse.start()

    def _show_step(self, *, animate: bool = True) -> None:
        if not self._steps:
            self._finish()
            return
        step = self._steps[self._index]
        if step.prepare is not None:
            step.prepare()
        last = self._index == len(self._steps) - 1
        self.eyebrow.setText(f"TUTORIAL  ·  {self._index + 1} / {len(self._steps)}")
        self.title.setText(f"{self._index + 1} · {step.title}")
        self.body.setText(step.body)
        self.next_button.setText("Done" if last else "Next")
        self.back_button.setVisible(self._index > 0)
        self.skip_button.setVisible(not last)
        self._fit_card()
        self.step_changed.emit(self._index)
        self._sync_geometry(animate=animate)
        # A tab switch or a scroll settles a moment later: measure the target again once it has.
        QTimer.singleShot(0, self.refresh_target)
        QTimer.singleShot(90, self.refresh_target)
        self.update()

    def _fit_card(self) -> None:
        """Size the card to its wrapped text. A word-wrapped label only reports its true height once it
        knows its width, so give it that first - otherwise the last lines are cut off."""
        inner = self.card.width() - 36
        self.body.setFixedWidth(inner)
        self.body.setFixedHeight(self.body.heightForWidth(inner))
        layout = self.card.layout()
        layout.invalidate()
        layout.activate()
        self.card.setFixedHeight(layout.sizeHint().height())

    def refresh_target(self) -> None:
        """Recalculate after a scroll or a layout change moved the highlighted control."""
        if not self._done:                       # a timer can fire after the tour has ended
            self._sync_geometry(animate=True)
            self.update()

    def _sync_geometry(self, *, animate: bool = False) -> None:
        if not self._steps:
            return
        target = self._steps[self._index].target
        top_left = self.mapFromGlobal(target.mapToGlobal(QPoint(0, 0)))
        raw = QRect(top_left, target.size()).adjusted(-7, -7, 7, 7)
        target_rect = raw.intersected(self.rect().adjusted(5, 5, -5, -5))

        margin = 16
        width = self.card.width()
        height = self.card.height()
        right_x = target_rect.right() + margin
        left_x = target_rect.left() - width - margin
        if right_x + width <= self.width() - margin:
            x = right_x
        elif left_x >= margin:
            x = left_x
        else:
            x = max(margin, min(self.width() - width - margin, target_rect.center().x() - width // 2))

        below_y = target_rect.bottom() + margin
        above_y = target_rect.top() - height - margin
        if below_y + height <= self.height() - margin:
            y = below_y
        elif above_y >= margin:
            y = above_y
        else:
            y = max(margin, min(self.height() - height - margin, target_rect.center().y() - height // 2))
        card_rect = QRect(x, y, width, height)
        if animate and not self._target.isEmpty() and card_rect != self.card.geometry():
            self._spot_animation.stop()
            self._spot_animation.setStartValue(self._target)
            self._spot_animation.setEndValue(target_rect)
            self._spot_animation.start()
            self._card_animation.stop()
            self._card_animation.setStartValue(self.card.geometry())
            self._card_animation.setEndValue(card_rect)
            self._card_animation.start()
        else:
            self._target = target_rect
            self.card.setGeometry(card_rect)

    def _get_spotlight_rect(self) -> QRect:
        return self._target

    def _set_spotlight_rect(self, rect: QRect) -> None:
        self._target = rect
        self.update()

    spotlightRect = Property(QRect, _get_spotlight_rect, _set_spotlight_rect)

    def _get_veil_opacity(self) -> float:
        return self._veil_opacity

    def _set_veil_opacity(self, value: float) -> None:
        self._veil_opacity = float(value)
        self.update()

    veilOpacity = Property(float, _get_veil_opacity, _set_veil_opacity)

    def _set_pulse(self, value: object) -> None:
        self._pulse_phase = float(value)
        self.update()

    def _next(self) -> None:
        if self._index >= len(self._steps) - 1:
            self._finish()
            return
        self._index += 1
        self._show_step()

    def _back(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._show_step()

    def _skip(self) -> None:
        self._done = True
        self._stop_animations()
        self.hide()
        self.skipped.emit()
        self.deleteLater()

    def _finish(self) -> None:
        self._done = True
        self._stop_animations()
        self.hide()
        self.finished.emit()
        self.deleteLater()

    def _stop_animations(self) -> None:
        """Never leave a native animation callback queued after the overlay dies."""
        for animation in (self._spot_animation, self._card_animation, self._fade_animation,
                          self._card_fade, self._pulse):
            animation.stop()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt name
        key = event.key()
        if key == Qt.Key.Key_Escape:
            self._skip()
        elif key in (Qt.Key.Key_Right, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self._next()
        elif key == Qt.Key.Key_Left:
            self._back()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._sync_geometry()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        shade = QPainterPath()
        shade.setFillRule(Qt.FillRule.OddEvenFill)
        shade.addRect(self.rect())
        if not self._target.isEmpty():
            shade.addRoundedRect(self._target, 10, 10)
        painter.fillPath(shade, QColor(2, 7, 14, int(205 * self._veil_opacity)))
        if not self._target.isEmpty():
            # A breathing glow around the lit control, so it feels live while it is being read.
            pulse = (math.sin(self._pulse_phase) + 1.0) * 0.5
            glow = QColor(self._accent)
            glow.setAlpha(int((26 + 30 * pulse) * self._veil_opacity))
            painter.setPen(QPen(glow, 9 + 7 * pulse))
            painter.drawRoundedRect(self._target.adjusted(-3, -3, 3, 3), 13, 13)
            painter.setPen(QPen(self._accent, 2))
            painter.drawRoundedRect(self._target, 10, 10)
