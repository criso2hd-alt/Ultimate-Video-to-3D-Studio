"""Reusable pieces of the interface.

The chrome (module cards, view bar, timeline, download dialog, segmented
control) is carried over from the DLSS app so the two look and behave alike; the
colour-correction controls, system gauge and 3D preview view are new.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPointF,
    QRect,
    QRectF,
    QSize,
    Qt,
    QTimer,
    QVariantAnimation,
    Signal,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QCursor,
    QFont,
    QImage,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

#: The bundled type families (registered in app._load_bundled_fonts). Named here
#: so the whole UI reaches for the same three faces the mockup uses.
FONT_DISPLAY = "Archivo SemiBold"          # wordmark, card titles, Convert
FONT_SANS = "IBM Plex Sans"                # body: labels, chips, buttons
FONT_MONO = "IBM Plex Mono"                # small-caps tags, values, readouts


def apply_font(
    widget,
    *,
    family: str | None = None,
    size: float | None = None,
    weight=None,
    spacing: float | None = None,
    caps: bool = False,
) -> None:
    """Set face, size, weight, letter-spacing and caps on a widget in one call.

    Letter-spacing is the reason this exists: Qt Style Sheets silently ignore the
    ``letter-spacing`` property, so the mockup's spaced-out caps — the wordmark,
    the card titles, the RUNTIME pill — can only be had by setting it on the
    QFont here, per widget.
    """
    from PySide6.QtGui import QFont

    font = widget.font()
    if family:
        font.setFamily(family)
    if size is not None:
        font.setPointSizeF(size)
    if weight is not None:
        font.setWeight(weight)
    if spacing is not None:
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
    if caps:
        font.setCapitalization(QFont.Capitalization.AllUppercase)
    widget.setFont(font)


def stage_placeholder(icon: str, title: str, subtitle: str = "") -> QWidget:
    """A centred icon + title + hint, for the empty state of a preview stage.

    So the Video and Sequence tabs read as "your clip / frames appear here"
    rather than a black void — the same welcoming empty state the single-image
    drop zone gives, in the same visual language.
    """
    holder = QWidget()
    box = QVBoxLayout(holder)
    box.setAlignment(Qt.AlignmentFlag.AlignCenter)
    box.setSpacing(8)
    glyph = QLabel(icon)
    glyph.setAlignment(Qt.AlignmentFlag.AlignCenter)
    apply_font(glyph, size=34)
    head = QLabel(title)
    head.setObjectName("phTitle")
    head.setAlignment(Qt.AlignmentFlag.AlignCenter)
    box.addWidget(glyph)
    box.addWidget(head)
    if subtitle:
        sub = QLabel(subtitle)
        sub.setObjectName("hint")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(sub)
    return holder

class ModuleCard(QFrame):
    """A titled card: the title sits *inside* a header strip with a divider under
    it, then a padded body — the mockup's ``.mod`` block, not a QGroupBox.

    QGroupBox hangs its title on the border and cannot draw the divider or carry
    a right-aligned tag, which is most of why the sidebar read as a plain form
    rather than the mockup's instrument panel. The optional ``tag`` is the little
    mono caption on the right of the header (RENODX · DLAA, NEW).
    """

    #: Emitted when a checkable card's header toggle changes.
    toggled = Signal(bool)

    def __init__(
        self, title: str, tag: str = "", checkable: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("modCard")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QFrame()
        head.setObjectName("modHead")
        head_row = QHBoxLayout(head)
        head_row.setContentsMargins(16, 12, 16, 11)
        head_row.setSpacing(9)

        self._check: "QCheckBox | None" = None
        if checkable:
            # The title itself is the on/off control, like the old checkable
            # group box — a checkbox whose label is the card title.
            from PySide6.QtWidgets import QCheckBox

            self._check = QCheckBox(title)
            self._check.setObjectName("modTitle")
            apply_font(self._check, family=FONT_DISPLAY, size=9.5, spacing=2.4, caps=True)
            self._check.toggled.connect(self._on_toggle)
            head_row.addWidget(self._check)
            self.title_label = self._check
        else:
            self.title_label = QLabel(title)
            self.title_label.setObjectName("modTitle")
            apply_font(self.title_label, family=FONT_DISPLAY, size=9.5, spacing=2.4, caps=True)
            head_row.addWidget(self.title_label)
        head_row.addStretch(1)
        if tag:
            tag_label = QLabel(tag)
            tag_label.setObjectName("modTag")
            apply_font(tag_label, family=FONT_MONO, size=7.5, spacing=1.4, caps=True)
            head_row.addWidget(tag_label)
        outer.addWidget(head)

        self._body_widget = QWidget()
        self.body = QVBoxLayout(self._body_widget)
        self.body.setContentsMargins(16, 14, 16, 15)
        self.body.setSpacing(13)
        outer.addWidget(self._body_widget)

    def add(self, widget: QWidget) -> QWidget:
        self.body.addWidget(widget)
        return widget

    def add_layout(self, layout) -> "object":
        self.body.addLayout(layout)
        return layout

    # -- checkable proxy (so a card can stand in for a checkable QGroupBox) --

    def _on_toggle(self, on: bool) -> None:
        # Grey the body when off, exactly like a checkable group box.
        self._body_widget.setEnabled(on)
        self.toggled.emit(on)

    def setChecked(self, on: bool) -> None:  # noqa: N802 - Qt-style name
        if self._check is not None:
            self._check.setChecked(on)
            # Sync the body even when the state did not change (so an initial
            # unchecked card greys its body without relying on a toggle signal).
            self._body_widget.setEnabled(on)

    def isChecked(self) -> bool:  # noqa: N802 - Qt-style name
        return self._check.isChecked() if self._check is not None else True


def to_qimage_u8(image_rgb: np.ndarray) -> QImage:
    """8-bit RGB to a QImage that owns its buffer.

    The copy at the end is not redundant: QImage wraps the numpy buffer without
    taking a reference, so returning an uncopied view hands Qt a pointer that is
    freed as soon as the temporary array goes out of scope.
    """
    data = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    height, width = data.shape[:2]
    return QImage(data.data, width, height, width * 3, QImage.Format.Format_RGB888).copy()


def to_qimage(image_rgb: np.ndarray) -> QImage:
    """0..1 float RGB to an 8-bit QImage that owns its buffer."""
    return to_qimage_u8(np.clip(image_rgb, 0.0, 1.0) * 255.0)


#: Overlay plate colours, straight from the mockup (--tag background, --line).
_OVERLAY_PLATE = QColor(6, 10, 18, 190)
_OVERLAY_LINE = QColor(35, 48, 74)
_OVERLAY_INK = QColor(201, 213, 230)      # --ink-dim, the non-accent pill text
_OVERLAY_READ = QColor(147, 162, 188)     # --ink-faint-ish, the readout text


def _overlay_font(painter: QPainter, size: float, *, caps: bool = True) -> None:
    """Set the painter to the mockup's overlay face: mono, spaced, small caps.

    Explicitly mono rather than the widget's inherited face because these chips
    are readouts - a resolution, a state - and the prototype sets them in IBM
    Plex Mono so the digits line up and the caps read as instrument labels.
    """
    font = QFont(FONT_MONO)
    font.setPointSizeF(size)
    font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 118)
    font.setCapitalization(
        QFont.Capitalization.AllUppercase if caps else QFont.Capitalization.MixedCase
    )
    painter.setFont(font)


def _draw_chip(
    painter: QPainter, rect: QRectF, text: str, *, accent: bool, signal: QColor,
    ink: QColor = _OVERLAY_INK, radius: float = 7.0,
) -> None:
    """Fill one overlay chip: translucent plate, hairline border, centred text.

    The accent variant is the DLSS side - cyan text over a cyan-tinted border -
    so the eye reads "this half is the neural result" without a legend.
    """
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_OVERLAY_PLATE)
    painter.drawRoundedRect(rect, radius, radius)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if accent:
        border = QColor(signal)
        border.setAlpha(160)
        painter.setPen(QPen(border, 1))
    else:
        painter.setPen(QPen(_OVERLAY_LINE, 1))
    painter.drawRoundedRect(rect, radius, radius)
    painter.setPen(signal if accent else ink)
    painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)


def _corner_label(
    painter: QPainter, text: str, image: QRectF, widget: QWidget, align_left: bool,
    *, accent: bool = False, signal: QColor | None = None,
) -> None:
    """Draw a pill at a top corner of `image`, clamped inside `widget`.

    The anchor is the image's own corner, so at fit the pills sit on the picture
    - top-left and top-right - wherever it is letterboxed. Clamping to the widget
    is what makes them sticky: zoom in and the image corners leave the screen,
    but the pill holds at the visible edge rather than scrolling away with the
    pixels it is describing.

    The translucent plate keeps mono caps legible over a blown-out sky or a
    white wall, which a plain light label would disappear into.
    """
    if not text:
        return
    if signal is None:
        signal = widget.palette().highlight().color()
    _overlay_font(painter, 8.5)
    metrics = painter.fontMetrics()
    pad_x, pad_y = 10.0, 5.0
    chip_w = metrics.horizontalAdvance(text) + pad_x * 2
    chip_h = metrics.height() + pad_y * 2
    margin = 12.0

    top = min(max(image.top() + margin, margin), widget.height() - chip_h - margin)
    if align_left:
        x = min(max(image.left() + margin, margin), widget.width() - chip_w - margin)
    else:
        x = max(min(image.right() - margin - chip_w, widget.width() - chip_w - margin),
                margin)
    _draw_chip(painter, QRectF(x, top, chip_w, chip_h), text, accent=accent, signal=signal)


def _paint_divider(
    painter: QPainter, x: float, top: float, bottom: float, signal: QColor,
) -> None:
    """The prototype's glowing seam: a vertical gradient bar with a round grip.

    A flat line reads as a crop mark. The gradient fading at the ends, the halo
    and the grabbable disc read instead as a control - which is what it is, the
    thing the user drags to compare - so it invites the drag rather than just
    marking where the two halves meet.
    """
    grad = QLinearGradient(x, top, x, bottom)
    clear = QColor(signal)
    clear.setAlpha(0)
    grad.setColorAt(0.0, clear)
    grad.setColorAt(0.5, signal)
    grad.setColorAt(1.0, clear)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QBrush(grad))
    painter.drawRect(QRectF(x - 1.0, top, 2.0, bottom - top))

    centre = QPointF(x, (top + bottom) / 2.0)
    # Concentric translucent rings stand in for the mockup's CSS box-shadow glow,
    # the same trick the tutorial spotlight uses - the painter has no box-shadow.
    for radius, alpha in ((23.0, 45), (18.0, 80)):
        halo = QColor(signal)
        halo.setAlpha(alpha)
        painter.setPen(QPen(halo, 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(centre, radius, radius)
    painter.setBrush(QColor(8, 17, 28))
    painter.setPen(QPen(signal, 1.5))
    painter.drawEllipse(centre, 16.0, 16.0)

    # Two chevrons for the grip, drawn rather than set as a glyph so the arrow
    # never depends on a font that may not carry ⟺.
    grip = QPen(signal, 1.6)
    grip.setCapStyle(Qt.PenCapStyle.RoundCap)
    grip.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(grip)
    cy = centre.y()
    left = QPolygonF([QPointF(x - 3, cy - 4), QPointF(x - 7, cy), QPointF(x - 3, cy + 4)])
    right = QPolygonF([QPointF(x + 3, cy - 4), QPointF(x + 7, cy), QPointF(x + 3, cy + 4)])
    painter.drawPolyline(left)
    painter.drawPolyline(right)


def _paint_readout(
    painter: QPainter, widget: QWidget, text: str, *, band: float | None = None,
) -> None:
    """A resolution + zoom chip pinned bottom-right, like the mockup's readout.

    Bottom-right rather than centred: it is a passive instrument reading, so it
    belongs out of the way at the corner, not floating over the middle of the
    picture the user is judging.

    ``band`` confines the chip to a reserved strip of that height at the very
    bottom, for the side-by-side view: there the panes must stay pixel-identical
    for comparison, so the chip has to sit clear of the image rather than float
    over it as it does on the single and wipe views.
    """
    if not text:
        return
    _overlay_font(painter, 8.5, caps=False)
    metrics = painter.fontMetrics()
    chip_w = metrics.horizontalAdvance(text) + 22.0
    margin = 12.0
    if band is not None:
        chip_h = min(metrics.height() + 8.0, band - 2.0)
        top = widget.height() - band + (band - chip_h) / 2.0
    else:
        chip_h = metrics.height() + 12.0
        top = widget.height() - chip_h - margin
    rect = QRectF(widget.width() - chip_w - margin, top, chip_w, chip_h)
    _draw_chip(
        painter, rect, text, accent=False, signal=widget.palette().highlight().color(),
        ink=_OVERLAY_READ, radius=9.0,
    )

class CanvasView(QWidget):
    """Shared zoom and pan for the image views.

    Wheel zooms about the cursor, right-drag pans, double-click fits again.
    Right rather than left because the comparison view already uses left-drag
    for its divider, and losing that to panning would be a bad trade.

    Zooming matters more here than in most viewers: people are running 6K and 8K
    renders through this and judging changes — pore detail, fabric weave, hair
    silhouettes — that simply are not visible in a fit-to-window view.
    """

    MIN_ZOOM = 1.0
    MAX_ZOOM = 32.0

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning = False
        self._pan_from = QPointF(0.0, 0.0)
        # The overlay layer - pills and the resolution readout - is hidden until
        # the pointer is over the picture, then faded in by the StageHost. It
        # starts invisible so a converted image is unobstructed the instant it
        # lands, and the instruments arrive only when the person reaches for them.
        self._chrome_opacity = 0.0
        self._hover_listener = None
        self.setMinimumSize(480, 320)

    # -- hover-revealed overlay chrome ---------------------------------------

    def set_chrome_opacity(self, value: float) -> None:
        """Fade the pills/readout in or out, driven by the StageHost's hover."""
        value = float(value)
        if value != self._chrome_opacity:
            self._chrome_opacity = value
            self.update()

    def set_hover_listener(self, callback) -> None:
        """Let the StageHost hear when the pointer enters or leaves the picture."""
        self._hover_listener = callback

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._hover_listener is not None:
            self._hover_listener(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._hover_listener is not None:
            self._hover_listener(False)
        super().leaveEvent(event)

    # -- geometry ------------------------------------------------------------

    def _viewport(self) -> QRectF:
        """The area one image is laid out in.

        The whole widget, unless a subclass splits it into panes. Everything
        below works in this space rather than in widget coordinates, which is
        what lets a two-pane view drive both panes from a single zoom and pan
        - they are not synchronised, they are literally the same numbers.
        """
        return QRectF(0.0, 0.0, float(self.width()), float(self.height()))

    def _to_viewport(self, point: QPointF) -> QPointF:
        """Widget coordinates into the space `_viewport` describes."""
        return point

    def _fit_rect(self, size) -> QRectF:
        """The image at 100% fit, centred, ignoring zoom and pan."""
        view = self._viewport()
        if size.width() <= 0 or size.height() <= 0 or view.isEmpty():
            return QRectF()
        scale = min(view.width() / size.width(), view.height() / size.height())
        width, height = size.width() * scale, size.height() * scale
        return QRectF(
            view.left() + (view.width() - width) / 2,
            view.top() + (view.height() - height) / 2,
            width,
            height,
        )

    def _display_rect(self, size) -> QRectF:
        base = self._fit_rect(size)
        if base.isEmpty():
            return base
        width, height = base.width() * self._zoom, base.height() * self._zoom
        left = base.center().x() - width / 2 + self._pan.x()
        top = base.center().y() - height / 2 + self._pan.y()
        return QRectF(left, top, width, height)

    def _content_size(self):
        """Subclasses return the pixmap size they are drawing, or None."""
        return None

    def reset_view(self) -> None:
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    def _clamp_pan(self) -> None:
        """Keep some of the image on screen at all times."""
        size = self._content_size()
        if size is None:
            return
        rect = self._display_rect(size)
        view = self._viewport()
        margin_x = max(0.0, (rect.width() - view.width()) / 2)
        margin_y = max(0.0, (rect.height() - view.height()) / 2)
        self._pan.setX(float(np.clip(self._pan.x(), -margin_x, margin_x)))
        self._pan.setY(float(np.clip(self._pan.y(), -margin_y, margin_y)))

    # -- interaction ---------------------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt name
        size = self._content_size()
        if size is None:
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        previous = self._zoom
        self._zoom = float(np.clip(previous * (1.25**steps), self.MIN_ZOOM, self.MAX_ZOOM))
        if self._zoom == previous:
            return

        # Keep whatever is under the cursor under the cursor. Without this,
        # zooming always creeps towards the centre and you lose the detail you
        # were aiming at.
        cursor = self._to_viewport(event.position())
        before = self._display_rect(size)
        if before.width() > 0 and before.height() > 0:
            u = (cursor.x() - before.left()) / before.width()
            v = (cursor.y() - before.top()) / before.height()
            self._pan = QPointF(0.0, 0.0) + self._pan  # copy
            after = self._display_rect(size)
            self._pan.setX(self._pan.x() + cursor.x() - (after.left() + u * after.width()))
            self._pan.setY(self._pan.y() + cursor.y() - (after.top() + v * after.height()))
        if self._zoom <= self.MIN_ZOOM:
            self._pan = QPointF(0.0, 0.0)
        self._clamp_pan()
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self._pan_from = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if self._panning:
            delta = event.position() - self._pan_from
            self._pan_from = event.position()
            self._pan += delta
            self._clamp_pan()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = False
            self.unsetCursor()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        self.reset_view()

    def _zoom_caption(self) -> str:
        return "" if self._zoom <= 1.001 else f"{self._zoom:.1f}x  ·  right-drag to pan"

    def _readout_text(self, size) -> str:
        """The bottom-right instrument reading: native resolution, then zoom.

        The resolution is the picture's real pixel size, not the on-screen rect,
        because that is what the person cares about when judging an 8K render;
        the zoom is appended only when it is doing something.
        """
        if size is None or size.width() <= 0 or size.height() <= 0:
            return ""
        dims = f"{size.width()} × {size.height()}"
        return dims if self._zoom <= 1.001 else f"{dims}   ·   {self._zoom:.1f}×"


class HoverBar(QFrame):
    """A translucent control bar that reports pointer enter/leave to its host.

    It floats over the picture and carries the view controls. Because it tells
    the StageHost when the pointer is on it, moving from the image onto the bar
    keeps the whole overlay revealed rather than flickering it away.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("viewBar")
        self._hover_listener = None

    def set_hover_listener(self, callback) -> None:
        self._hover_listener = callback

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._hover_listener is not None:
            self._hover_listener(True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._hover_listener is not None:
            self._hover_listener(False)
        super().leaveEvent(event)


class StageHost(QWidget):
    """Holds the image stack and floats a hover-revealed control bar over it.

    The whole overlay layer - the SOURCE/DLSS pills, the resolution readout and
    the view bar - fades in when the pointer is over the picture and fades out
    when it leaves, so the image is unobstructed while it is being studied and
    the instruments are there the instant they are reached for. The fade is one
    animation driving both the painted chrome (on the views) and the bar, so the
    two can never disagree about whether they are shown.
    """

    #: How long the overlay takes to fade in or out. Short enough to feel
    #: immediate, long enough not to read as a hard cut.
    FADE_MS = 160

    def __init__(
        self, stack: QStackedWidget, bar: HoverBar | None = None, parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._stack = stack
        self._bar = bar
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(stack)

        self._effect = None
        if bar is not None:                      # the bar is optional: the pills and readout fade regardless
            bar.setParent(self)
            self._effect = QGraphicsOpacityEffect(bar)
            bar.setGraphicsEffect(self._effect)
            self._effect.setOpacity(0.0)
            bar.setVisible(False)

        self._opacity = 0.0
        self._target = 0.0
        self._pending_hide = False
        self._fade = QVariantAnimation(self)
        self._fade.setDuration(self.FADE_MS)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade.valueChanged.connect(self._apply)

        # The views exist before the host is built, so bind their hover once.
        self._views = stack.findChildren(CanvasView)
        for view in self._views:
            view.set_hover_listener(self._hover)
        if bar is not None:
            bar.set_hover_listener(self._hover)

    def _position_bar(self) -> None:
        """Bottom-left, matching the mockup's stagefoot; the readout is painted
        bottom-right by the view, so the two share the strip without colliding."""
        if self._bar is None:
            return
        margin = 14
        self._bar.adjustSize()
        self._bar.move(margin, max(margin, self.height() - self._bar.height() - margin))
        self._bar.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._position_bar()
        super().resizeEvent(event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._position_bar()
        super().showEvent(event)

    def _hover(self, entered: bool) -> None:
        if entered:
            self._pending_hide = False
            self._reveal(True)
        else:
            # Moving from a view onto the bar (or a bar button) fires a leave
            # before the next enter, so never hide on the spot: wait, then hide
            # only if the pointer really is off the whole stage by then.
            self._pending_hide = True
            QTimer.singleShot(60, self._maybe_hide)

    def _maybe_hide(self) -> None:
        if not self._pending_hide:
            return
        inside = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        if not inside:
            self._reveal(False)

    def _reveal(self, shown: bool) -> None:
        self._target = 1.0 if shown else 0.0
        if self._target == self._opacity:
            return
        if shown and self._bar is not None:
            self._bar.setVisible(True)
            self._position_bar()
        self._fade.stop()
        self._fade.setStartValue(self._opacity)
        self._fade.setEndValue(self._target)
        self._fade.start()

    def _apply(self, value: object) -> None:
        self._opacity = float(value)
        if self._bar is not None:
            self._effect.setOpacity(self._opacity)
            if self._opacity <= 0.001 and self._target == 0.0:
                self._bar.setVisible(False)
        for view in self._views:
            view.set_chrome_opacity(self._opacity)


def format_bytes(count: float) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit in ("B", "KB") else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return "less than a second"
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f} min {seconds % 60:.0f} s"
    return f"{minutes / 60:.1f} hours"


class DownloadDialog(QDialog):
    """First-run download, with the two numbers people actually want.

    Rate is averaged over a trailing window rather than since-the-start: these
    downloads resume, and a run that picks up at 80% would otherwise show a
    wildly optimistic rate for its whole life. The window also keeps the
    estimate from lurching every time a file finishes.
    """

    #: Seconds of history used for the rate estimate.
    WINDOW = 8.0

    #: Emitted when the user asks to abandon the step (only if enable_cancel was
    #: called). The owner is responsible for stopping the work and closing this.
    cancelled = Signal()

    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setMinimumWidth(460)
        # No close button: the work continues regardless of the dialog, and a
        # titlebar X that silently does nothing is worse than not having one.
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        self._heading = QLabel(title)
        layout.addWidget(self._heading)
        if subtitle:
            note = QLabel(subtitle)
            note.setObjectName("hint")
            note.setWordWrap(True)
            layout.addWidget(note)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setTextVisible(False)
        layout.addWidget(self._bar)

        self._detail = QLabel("Starting…")
        self._detail.setObjectName("hint")
        layout.addWidget(self._detail)

        self._history: list[tuple[float, int]] = []

    def enable_cancel(self, text: str = "Skip") -> None:
        """Add a button that lets the user abandon this step.

        Off by default: the first-run downloads carry on whether the dialog is
        up or not, so a Cancel there would be a lie. The runtime check is the
        opposite - it can wedge the GPU, and the user needs a way out that is not
        force-quitting the whole app. Clicking it only emits `cancelled`; the
        owner stops the work and closes the dialog, so the button can never look
        like it did something while the process is actually still running.
        """
        row = QHBoxLayout()
        row.addStretch(1)
        button = QPushButton(text)
        button.setObjectName("secondary")
        button.clicked.connect(self.cancelled.emit)
        row.addWidget(button)
        self.layout().addLayout(row)

    def set_heading(self, text: str) -> None:
        self._heading.setText(text)

    def set_status(self, text: str) -> None:
        self._detail.setText(text)

    def set_busy(self) -> None:
        """Show indeterminate progress for work with no meaningful percentage."""
        self._bar.setRange(0, 0)

    def mark_complete(self) -> None:
        """Fill the bar on success.

        The folder-watching reporter can deliver a final sample slightly under
        the total — files are renamed out of ``.incomplete`` as they land, so
        the measured size dips at the very end — and a bar that stops at 95%
        looks like a download that gave up.
        """
        self._bar.setRange(0, 1000)
        self._bar.setValue(1000)

    def update_bytes(self, done: int, total: int) -> None:
        now = time.monotonic()
        self._history.append((now, done))
        while len(self._history) > 2 and now - self._history[0][0] > self.WINDOW:
            self._history.pop(0)

        if total > 0:
            self._bar.setRange(0, 1000)
            self._bar.setValue(int(min(1000, done / total * 1000)))
        else:
            # Unknown total: a busy indicator beats a bar stuck at zero.
            self._bar.setRange(0, 0)

        parts = [f"{format_bytes(done)} of {format_bytes(total)}" if total else format_bytes(done)]
        span = now - self._history[0][0]
        moved = done - self._history[0][1]
        if span >= 1.0 and moved > 0:
            rate = moved / span
            parts.append(f"{format_bytes(rate)}/s")
            if total > done:
                parts.append(f"about {format_duration((total - done) / rate)} left")
        self._detail.setText("  —  ".join(parts))

class SegmentedControl(QWidget):
    """A row of equal-width, mutually exclusive buttons — a styled radio group.

    The mockup replaces the Style and Detail-mode dropdowns with these: the two
    or three choices are all visible and one tap wide, instead of hidden behind a
    combo. ``options`` are ``str`` or ``(label, data)``; ``changed`` carries the
    selected index and ``current_data`` returns the chosen payload.
    """

    changed = Signal(int)

    def __init__(self, options, current: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._data: list = []
        for index, option in enumerate(options):
            label, data = option if isinstance(option, tuple) else (option, option)
            button = QPushButton(label)
            button.setObjectName("chip")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            row.addWidget(button, 1)
            self._group.addButton(button, index)
            self._data.append(data)
        self._group.idClicked.connect(self.changed)
        if options:
            self._group.button(min(current, len(options) - 1)).setChecked(True)

    def current_index(self) -> int:
        return self._group.checkedId()

    def current_data(self):
        index = self._group.checkedId()
        return self._data[index] if 0 <= index < len(self._data) else None

    def set_index(self, index: int) -> None:
        button = self._group.button(index)
        if button is not None:
            button.setChecked(True)


class TimelineWidget(QWidget):
    """A scrub bar with Premiere-style In/Out brackets and scroll-to-zoom.

    Frames are the unit throughout - the playhead, the In and Out points, and
    the visible window are all frame indices - because the conversion works in
    frames and a range that does not land on exact frames would convert a
    slightly different span than the one shown. Time is only ever a label.

    Zoom is a visible window ``[_view_lo, _view_hi)`` of the whole ``0..total``
    range; the wheel narrows or widens it about the cursor, so you can place an
    In point on frame 4137 of a five-thousand-frame clip without fighting a bar
    where every frame is a third of a pixel.
    """

    seeked = Signal(int)       # playhead moved by the user
    scrub_finished = Signal()  # the user let go of the playhead
    in_changed = Signal(int)
    out_changed = Signal(int)

    _HANDLE = 8.0              # half-width of a draggable bracket, in pixels
    _MIN_SPAN = 2             # never zoom tighter than this many frames

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(62)
        self.setMouseTracking(True)
        self._total = 0
        self._fps = 24.0
        self._playhead = 0
        self._in = 0
        self._out = 0
        self._range_mode = False
        self._view_lo = 0
        self._view_hi = 0          # exclusive; == total when fully zoomed out
        self._drag: str | None = None   # "playhead" | "in" | "out" | "pan"
        self._pan_anchor = 0.0
        self._buffered: tuple[int, int] | None = None     # frames rendered ahead of the playhead
        self._buffering = False

    # -- state ---------------------------------------------------------------

    def set_buffered(self, lo: int, hi: int, buffering: bool = False) -> None:
        """Show how much has been rendered ahead: frames ``lo`` .. ``hi`` (like a video site's
        buffer bar). ``hi <= lo`` clears it."""
        new = (int(lo), int(hi)) if hi > lo else None
        if new != self._buffered or buffering != self._buffering:
            self._buffered, self._buffering = new, bool(buffering)
            self.update()

    def set_duration(self, total_frames: int, fps: float) -> None:
        self._total = max(0, int(total_frames))
        self._fps = fps or 24.0
        self._playhead = 0
        self._in = 0
        self._out = max(0, self._total - 1)
        self._view_lo = 0
        self._view_hi = self._total
        self.update()

    def set_playhead(self, frame: int) -> None:
        """Called from the player as it advances; does not emit seeked."""
        self._playhead = int(np.clip(frame, 0, max(0, self._total - 1)))
        self.update()

    def set_range_mode(self, on: bool) -> None:
        self._range_mode = bool(on)
        self.update()

    def set_in_out(self, in_frame: int, out_frame: int) -> None:
        last = max(0, self._total - 1)
        self._in = int(np.clip(in_frame, 0, last))
        self._out = int(np.clip(out_frame, self._in, last))
        self.update()

    def in_out(self) -> tuple[int, int]:
        return self._in, self._out

    def range_seconds(self) -> float:
        return (self._out - self._in + 1) / self._fps if self._total else 0.0

    # -- geometry ------------------------------------------------------------

    def _track(self) -> QRectF:
        return QRectF(self._HANDLE, 6.0, max(1.0, self.width() - 2 * self._HANDLE), 26.0)

    def _span(self) -> int:
        return max(1, self._view_hi - self._view_lo)

    def _frame_to_x(self, frame: float) -> float:
        track = self._track()
        return track.left() + (frame - self._view_lo) / self._span() * track.width()

    def _x_to_frame(self, x: float) -> int:
        track = self._track()
        if track.width() <= 0:
            return self._view_lo
        frac = (x - track.left()) / track.width()
        return int(round(self._view_lo + frac * self._span()))

    # -- interaction ---------------------------------------------------------

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt name
        if self._total <= 0:
            return
        steps = event.angleDelta().y() / 120.0
        if not steps:
            return
        pivot = self._x_to_frame(event.position().x())
        factor = 0.8 ** steps   # scroll up zooms in
        new_span = int(np.clip(round(self._span() * factor), self._MIN_SPAN, self._total))
        # Keep the frame under the cursor under the cursor.
        frac = (pivot - self._view_lo) / self._span()
        lo = int(round(pivot - frac * new_span))
        lo = int(np.clip(lo, 0, max(0, self._total - new_span)))
        self._view_lo = lo
        self._view_hi = min(self._total, lo + new_span)
        self.update()

    def _nearest_handle(self, x: float) -> str:
        candidates = [("playhead", self._playhead)]
        if self._range_mode:
            candidates += [("in", self._in), ("out", self._out)]
        best, best_dist = "playhead", 1e9
        for name, frame in candidates:
            dist = abs(self._frame_to_x(frame) - x)
            if dist < best_dist:
                best, best_dist = name, dist
        # Grab a bracket only when genuinely near it; otherwise treat a click as
        # a seek, which is what a click on empty track should do.
        return best if best_dist <= self._HANDLE * 2 else "seek"

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if self._total <= 0:
            return
        if event.button() == Qt.MouseButton.MiddleButton:
            self._drag = "pan"
            self._pan_anchor = event.position().x()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        target = self._nearest_handle(event.position().x())
        if target == "seek":
            self._drag = "playhead"
            self._apply_drag(event.position().x())
        else:
            self._drag = target
            self._apply_drag(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        if self._drag == "pan":
            track = self._track()
            dx = event.position().x() - self._pan_anchor
            self._pan_anchor = event.position().x()
            shift = int(round(-dx / track.width() * self._span()))
            lo = int(np.clip(self._view_lo + shift, 0, max(0, self._total - self._span())))
            self._view_lo, self._view_hi = lo, lo + self._span()
            self.update()
        elif self._drag:
            self._apply_drag(event.position().x())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt name
        was = self._drag
        self._drag = None
        if was == "playhead":
            self.scrub_finished.emit()

    def _apply_drag(self, x: float) -> None:
        frame = int(np.clip(self._x_to_frame(x), 0, max(0, self._total - 1)))
        if self._drag == "playhead":
            self._playhead = frame
            self.seeked.emit(frame)
        elif self._drag == "in":
            self._in = min(frame, self._out)
            self.in_changed.emit(self._in)
        elif self._drag == "out":
            self._out = max(frame, self._in)
            self.out_changed.emit(self._out)
        self.update()

    # -- painting ------------------------------------------------------------

    def _paint_empty(self, painter: QPainter) -> None:
        """What the timeline looks like before a video is open: an empty track with faint ticks, the
        buffer rail, and the playhead parked at the start - so the bar is there (and the tutorial has
        something to point at), but with no numbers and nothing to drag."""
        track = self._track()
        painter.fillRect(track, QColor(32, 36, 44))
        painter.setPen(QPen(QColor(58, 64, 78), 1))
        ticks = max(2, int(track.width() // 28))                      # about one every 28 px, whatever the width
        for n in range(1, ticks):
            x = track.left() + track.width() * n / ticks
            major = n % 5 == 0
            inset = 5.0 if major else 10.0
            painter.drawLine(QPointF(x, track.top() + inset), QPointF(x, track.bottom() - inset))
        painter.fillRect(QRectF(track.left(), track.bottom() + 3.0, track.width(), 3.0), QColor(40, 45, 56))
        px = track.left()
        dim = QColor(150, 158, 172)
        painter.setPen(QPen(dim, 1))
        painter.drawLine(QPointF(px, track.top() - 4), QPointF(px, track.bottom() + 4))
        painter.setBrush(dim)
        painter.drawPolygon(QPolygonF([QPointF(px - 5, track.top() - 4), QPointF(px + 5, track.top() - 4),
                                       QPointF(px, track.top() + 2)]))

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), self.palette().window())
        if self._total <= 0:
            self._paint_empty(painter)
            return
        track = self._track()
        painter.fillRect(track, QColor(38, 42, 51))

        highlight = self.palette().highlight().color()
        if self._range_mode:
            x_in = self._frame_to_x(self._in)
            x_out = self._frame_to_x(self._out)
            sel = QRectF(x_in, track.top(), max(1.0, x_out - x_in), track.height())
            fill = QColor(highlight)
            fill.setAlpha(70)
            painter.fillRect(sel, fill)
            painter.setPen(QPen(highlight, 2))
            for x in (x_in, x_out):
                painter.drawLine(QPointF(x, track.top() - 3), QPointF(x, track.bottom() + 3))
                # A little bracket foot so it reads as a handle.
                foot = 6.0 if x == x_in else -6.0
                painter.drawLine(QPointF(x, track.top() - 3), QPointF(x + foot, track.top() - 3))
                painter.drawLine(QPointF(x, track.bottom() + 3), QPointF(x + foot, track.bottom() + 3))

        # Buffer bar: a slim rail under the track with the pre-rendered range in red, the
        # way a video site shows how far ahead it has loaded.
        rail = QRectF(track.left(), track.bottom() + 3.0, track.width(), 3.0)
        painter.fillRect(rail, QColor(46, 52, 64))
        if self._buffered is not None:
            x0 = min(max(self._frame_to_x(self._buffered[0]), track.left()), track.right())
            x1 = min(max(self._frame_to_x(self._buffered[1]), track.left()), track.right())
            if x1 > x0:
                colour = QColor(255, 176, 60) if self._buffering else QColor(255, 72, 72)
                painter.fillRect(QRectF(x0, rail.top(), x1 - x0, rail.height()), colour)

        # Playhead.
        px = self._frame_to_x(self._playhead)
        painter.setPen(QPen(QColor(240, 240, 240), 1))
        painter.drawLine(QPointF(px, track.top() - 4), QPointF(px, track.bottom() + 4))
        painter.setBrush(QColor(240, 240, 240))
        painter.drawPolygon(
            QPolygonF([
                QPointF(px - 5, track.top() - 4),
                QPointF(px + 5, track.top() - 4),
                QPointF(px, track.top() + 2),
            ])
        )

        # Labels: current time, and the selection length when ranging.
        painter.setPen(QColor(147, 162, 188))
        label_font = QFont(FONT_MONO)
        label_font.setPointSizeF(8.5)
        painter.setFont(label_font)
        painter.drawText(
            QRectF(0, track.bottom() + 9, self.width(), 16),
            Qt.AlignmentFlag.AlignLeft,
            f"  {_fmt_tc(self._playhead, self._fps)}",
        )
        right = (
            f"In {_fmt_tc(self._in, self._fps)}  Out {_fmt_tc(self._out, self._fps)}  "
            f"({self.range_seconds():.1f}s)  "
            if self._range_mode
            else f"{_fmt_tc(max(0, self._total - 1), self._fps)}  "
        )
        painter.drawText(
            QRectF(0, track.bottom() + 9, self.width(), 16),
            Qt.AlignmentFlag.AlignRight,
            right,
        )


def _fmt_tc(frame: int, fps: float) -> str:
    total_seconds = frame / (fps or 24.0)
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    frames = int(round(frame % (fps or 24.0)))
    return f"{minutes:02d}:{seconds:02d}.{frames:02d}"



class ElidedLabel(QLabel):
    """A one-line label that shortens its middle with "…" instead of forcing the layout wider.

    A long file path has no place to wrap, so a plain word-wrapped QLabel takes
    the whole path as its minimum width and pushes its card past the edge of the
    rail. This one reports a tiny minimum and elides at paint time; the full text
    is in the tooltip.
    """

    def __init__(self, text: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""
        self.setMinimumWidth(40)
        self.setText(text)

    def setText(self, text: str) -> None:  # noqa: N802 - Qt name
        self._full = text
        self.setToolTip(text)
        self._elide()

    def _elide(self) -> None:
        metrics = self.fontMetrics()
        width = max(self.width() - 2, 40)
        super().setText(metrics.elidedText(self._full, Qt.TextElideMode.ElideMiddle, width))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._elide()
        super().resizeEvent(event)

    def minimumSizeHint(self):  # noqa: N802 - Qt name
        from PySide6.QtCore import QSize

        return QSize(40, super().minimumSizeHint().height())


class CardColumns(QWidget):
    """Lays cards out in as many columns as the width allows, and reflows as it changes.

    A plain vertical stack wastes a wide window and overflows a short one. Here the
    column count follows the available width (each column at least ``min_width`` wide,
    at most ``max_width``), and each card goes into whichever column is currently
    shortest, so the columns stay balanced and the reading order stays roughly
    top-to-bottom, left-to-right. Cards are only reparented when the column count
    actually changes, so dragging the window edge does not churn the layout.
    """

    def __init__(self, cards: list[QWidget], min_width: int = 380, max_width: int = 560,
                 spacing: int = 12, max_columns: int = 3, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._cards = list(cards)
        self._min, self._max, self._gap, self._max_cols = min_width, max_width, spacing, max_columns
        self._count = 0
        self._columns: list[QWidget] = []
        self._row = QHBoxLayout(self)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(spacing)
        self._relayout(1)

    def column_count(self) -> int:
        return self._count

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt name
        # Only ever ask for one column's width. Left to itself the layout's minimum is the width of
        # the *current* number of columns, so once a wide window had produced three, the page could
        # never shrink back to two: it stayed three columns wide and scrolled sideways.
        hint = super().minimumSizeHint()
        return QSize(min(hint.width(), self._min), hint.height())

    def _wanted(self, width: int) -> int:
        fit = (width + self._gap) // (self._min + self._gap)
        return max(1, min(len(self._cards), self._max_cols, int(fit)))

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt name
        wanted = self._wanted(event.size().width())
        if wanted != self._count:
            self._relayout(wanted)
        super().resizeEvent(event)

    def _relayout(self, count: int) -> None:
        old = self._columns
        heights = [0] * count
        columns = []
        layouts = []
        for _ in range(count):
            holder = QWidget()
            holder.setMaximumWidth(self._max)
            col = QVBoxLayout(holder)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(self._gap)
            columns.append(holder)
            layouts.append(col)
        for card in self._cards:
            shortest = heights.index(min(heights))
            layouts[shortest].addWidget(card)            # reparents the card
            heights[shortest] += card.sizeHint().height() + self._gap
        for col in layouts:
            col.addStretch(1)
        # Swap: empty the row (the cards already moved into the new columns), then refill it.
        while self._row.count():
            self._row.takeAt(0)
        for holder in old:
            holder.deleteLater()
        for holder in columns:
            self._row.addWidget(holder, 1)
            if self.isVisible():
                holder.show()                       # Qt would show it a loop turn later; do it now
        self._row.addStretch(0)
        self._row.activate()
        self._columns, self._count = columns, count
        self.updateGeometry()
