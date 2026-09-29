"""Controls new to this app: grading sliders, collapsible cards, curves, wheels, scopes, gauge, preview.

Chrome shared with the DLSS app (module cards, timeline, view bar, download
dialog) lives in ``widgets``; everything here was written for the 3D studio.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEasingCurve, QPointF, QRectF, QSize, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QConicalGradient,
    QFont,
    QImage,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .widgets import (
    FONT_DISPLAY,
    FONT_MONO,
    CanvasView,
    ElidedLabel,
    _corner_label,
    _overlay_font,
    _OVERLAY_READ,
    _paint_readout,
    apply_font,
    to_qimage_u8,
)


class ParamSlider(QWidget):
    """A labelled slider with a live readout; double-click to reset.

    Integer-backed (Qt sliders are), with a fixed number of decimals, so a
    -100..100 grading slider reads as whole numbers the way Lumetri's do while a
    stops-of-exposure slider reads to two places.
    """

    changed = Signal(float)

    def __init__(self, label: str, minimum: float, maximum: float, value: float,
                 default: float | None = None, decimals: int = 0, suffix: str = "",
                 tooltip: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._decimals = decimals
        self._scale = 10 ** decimals
        self._default = value if default is None else default
        self._suffix = suffix
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 3, 0, 3)
        layout.setSpacing(2)
        header = QHBoxLayout()
        self._name = QLabel(label)
        self._value = QLabel()
        self._value.setObjectName("hint")
        header.addWidget(self._name)
        header.addStretch(1)
        header.addWidget(self._value)
        layout.addLayout(header)
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(int(round(minimum * self._scale)), int(round(maximum * self._scale)))
        self._slider.setValue(int(round(value * self._scale)))
        self._slider.valueChanged.connect(self._moved)
        layout.addWidget(self._slider)
        tip = (tooltip + "\n\n" if tooltip else "") + "Double-click to reset."
        self.setToolTip(tip)
        self._update_text()
        self._name.mouseDoubleClickEvent = lambda _e: self.reset()      # type: ignore[method-assign]
        self._slider.mouseDoubleClickEvent = lambda _e: self.reset()    # type: ignore[method-assign]

    def value(self) -> float:
        return self._slider.value() / self._scale

    def _update_text(self) -> None:
        v = self.value()
        text = f"{v:+.{self._decimals}f}" if self._slider.minimum() < 0 else f"{v:.{self._decimals}f}"
        self._value.setText(text + self._suffix)

    def _moved(self, _raw: int) -> None:
        self._update_text()
        self.changed.emit(self.value())

    def set_value(self, value: float, emit: bool = False) -> None:
        self._slider.blockSignals(not emit)
        self._slider.setValue(int(round(value * self._scale)))
        self._slider.blockSignals(False)
        self._update_text()

    def reset(self) -> None:
        self._slider.setValue(int(round(self._default * self._scale)))


class CollapsibleCard(QFrame):
    """A slim sliver with an enable switch that opens into a full panel.

    Closed it is one line - the title, an on/off checkbox and a chevron - so a
    heavy panel (colour correction, LUTs) costs almost no room until it is
    wanted. Enabling and opening are independent: leave a grade running with the
    panel folded away, or open it to look without applying anything.
    """

    toggled = Signal(bool)       # the enable switch
    opened = Signal(bool)        # the panel opening or closing

    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("sliver")
        self.setProperty("open", False)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        head = QWidget()
        row = QHBoxLayout(head)
        row.setContentsMargins(14, 9, 10, 9)
        row.setSpacing(9)
        self._enable = QCheckBox(title)
        self._enable.setObjectName("modTitle")
        apply_font(self._enable, family=FONT_DISPLAY, size=8.5, spacing=1.0, caps=True)
        self._enable.toggled.connect(self.toggled.emit)
        row.addWidget(self._enable)
        self._sub = ElidedLabel(subtitle)
        self._sub.setObjectName("sliverSub")
        row.addWidget(self._sub, 1)
        self._chevron = QPushButton("▸")
        self._chevron.setObjectName("chevron")
        self._chevron.setCursor(Qt.CursorShape.PointingHandCursor)
        self._chevron.setToolTip("Open or close the panel")
        self._chevron.clicked.connect(lambda: self.set_open(not self._open))
        row.addWidget(self._chevron)
        outer.addWidget(head)
        head.mousePressEvent = self._head_pressed          # type: ignore[method-assign]

        self._body = QWidget()
        self.body = QVBoxLayout(self._body)
        self.body.setContentsMargins(14, 4, 14, 14)
        self.body.setSpacing(10)
        self._body.setVisible(False)
        outer.addWidget(self._body)
        self._open = False

    def _head_pressed(self, event) -> None:
        # Clicking the empty part of the header opens it, like a disclosure row.
        if event.button() == Qt.MouseButton.LeftButton and not self._enable.underMouse():
            self.set_open(not self._open)

    def add(self, widget: QWidget) -> QWidget:
        self.body.addWidget(widget)
        return widget

    def add_layout(self, layout) -> None:
        self.body.addLayout(layout)

    def set_open(self, on: bool) -> None:
        self._open = bool(on)
        self._body.setVisible(self._open)
        self._chevron.setText("▾" if self._open else "▸")
        self.setProperty("open", self._open)
        self.style().unpolish(self)
        self.style().polish(self)
        self.opened.emit(self._open)

    def is_open(self) -> bool:
        return self._open

    def set_enabled_state(self, on: bool, emit: bool = False) -> None:
        self._enable.blockSignals(not emit)
        self._enable.setChecked(on)
        self._enable.blockSignals(False)

    def is_enabled_state(self) -> bool:
        return self._enable.isChecked()

    def set_subtitle(self, text: str) -> None:
        self._sub.setText(text)


class SectionTabs(QWidget):
    """A row of underline tabs (Basic / Creative / Curves ...)."""

    changed = Signal(int)

    def __init__(self, labels: list[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self._group = QButtonGroup(self)
        for i, label in enumerate(labels):
            button = QPushButton(label)
            button.setObjectName("sectionTab")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self._group.addButton(button, i)
            row.addWidget(button, 1)
        self._group.idClicked.connect(self.changed)
        self._group.button(0).setChecked(True)


class SystemGauge(QFrame):
    """The command bar's whole-machine readout: GPU, encoder, VRAM, CPU and RAM.

    Vendor-neutral by construction - the numbers come from the operating system
    (see ``sysinfo``), so a Radeon, an Arc, an RTX and an Apple GPU all show the
    same cells. The ENC cell matters most here: the fixed-function video engine
    is what does the work during a hardware-encoded export, while the GPU's 3D
    engine is busy only with depth.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("gauge")
        row = QHBoxLayout(self)
        row.setContentsMargins(13, 6, 13, 6)
        row.setSpacing(14)

        name_col = QVBoxLayout()
        name_col.setSpacing(1)
        lab = QLabel("GPU")
        lab.setObjectName("gaugeLab")
        apply_font(lab, family=FONT_MONO, size=7.5, spacing=1.6, caps=True)
        self._name = QLabel("detecting…")
        self._name.setObjectName("gpuName")
        apply_font(self._name, family=FONT_DISPLAY, size=10.5)
        name_col.addWidget(lab)
        name_col.addWidget(self._name)
        row.addLayout(name_col)

        self._cells: dict[str, tuple[QProgressBar, QLabel]] = {}
        for key, title, width in (("gpu", "GPU", 56), ("enc", "ENC", 56), ("vram", "VRAM", 92),
                                  ("cpu", "CPU", 56), ("ram", "RAM", 92)):
            self._cells[key] = self._add_cell(row, title, width)

    def _add_cell(self, row: QHBoxLayout, title: str, width: int):
        col = QVBoxLayout()
        col.setSpacing(3)
        col.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.setSpacing(6)
        t = QLabel(title)
        t.setObjectName("gaugeDim")
        apply_font(t, family=FONT_MONO, size=7.0, spacing=1.2, caps=True)
        v = QLabel("—")
        v.setObjectName("gaugeVal")
        apply_font(v, family=FONT_MONO, size=8.0)
        head.addWidget(t)
        head.addStretch(1)
        head.addWidget(v)
        bar = QProgressBar()
        bar.setObjectName("miniBar")
        bar.setRange(0, 1000)
        bar.setTextVisible(False)
        bar.setFixedHeight(4)
        col.addLayout(head)
        col.addWidget(bar)
        holder = QWidget()
        holder.setLayout(col)
        holder.setFixedWidth(width)
        row.addWidget(holder)
        return bar, v

    #: Cells dropped when the window is too narrow for the whole gauge; the essentials stay.
    _OPTIONAL_CELLS = ("enc", "ram")

    compact = False

    def set_compact(self, on: bool) -> None:
        """Show only GPU, VRAM and CPU (hiding encoder load and RAM) on a narrow window."""
        if on == self.compact:
            return
        self.compact = on
        for key in self._OPTIONAL_CELLS:
            self._cells[key][0].parentWidget().setVisible(not on)

    def set_gpu_name(self, name: str) -> None:
        self._name.setText(name or "No GPU")

    def _set(self, key: str, fraction: float | None, text: str) -> None:
        bar, label = self._cells[key]
        label.setText(text)
        bar.setValue(0 if fraction is None else int(max(0.0, min(1.0, fraction)) * 1000))
        hot = fraction is not None and fraction > 0.9
        if bar.property("hot") != hot:
            bar.setProperty("hot", hot)
            bar.style().unpolish(bar)
            bar.style().polish(bar)

    def update_load(self, load) -> None:
        for key, value in (("gpu", load.gpu_percent), ("enc", load.encoder_percent), ("cpu", load.cpu_percent)):
            self._set(key, None if value is None else value / 100.0, "—" if value is None else f"{value:.0f}%")

        def mem(used, total):
            if used is None or not total:
                return None, "—"
            return used / total, f"{used / (1 << 30):.1f}/{total / (1 << 30):.0f}G"

        self._set("vram", *mem(load.vram_used, load.vram_total))
        self._set("ram", *mem(load.ram_used, load.ram_total))


class PreviewView(CanvasView):
    """The live 3D preview: one picture, or two alternating (wiggle), with zoom and pan.

    Holds RGB uint8 arrays painted scaled to fit. Chrome (the mode badge and
    resolution readout) fades in with the pointer, like the rest of the app.
    """

    #: A click anywhere on the picture (a left click that was not the end of a double-click).
    clicked = Signal()

    #: The play/pause button fades away after the pointer has rested this long (while playing).
    OVERLAY_IDLE_MS = 1600

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._images: list[QImage] = []
        self._index = 0
        self._badge = ""
        self._note = ""
        self._wiggle = QTimer(self)
        self._wiggle.setInterval(130)          # ~7.5 Hz: the classic "wigglegram" rate
        self._wiggle.timeout.connect(self._flip)
        # The big play/pause button in the middle: it shows while the pointer is moving over the
        # picture and fades out when it rests or leaves, like a streaming player's.
        self.setMouseTracking(True)
        self._playing = False
        self._hovering = False
        self._overlay = 0.0
        self._swallow_release = False
        self._fade = QVariantAnimation(self)
        self._fade.setDuration(180)
        self._fade.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade.valueChanged.connect(self._set_overlay)
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.setInterval(self.OVERLAY_IDLE_MS)
        self._idle.timeout.connect(self._rest)

    # -- play/pause overlay and click ----------------------------------------------

    def set_playing(self, playing: bool) -> None:
        self._playing = bool(playing)
        if self._hovering:
            self._show_overlay()                # the button changes shape: let it be seen
        self.update()

    def _set_overlay(self, value) -> None:
        self._overlay = float(value)
        self.update()

    def _fade_to(self, target: float) -> None:
        running = self._fade.state() == QVariantAnimation.State.Running
        if running and self._fade.endValue() == target:
            return                              # already heading there: do not restart the fade on every mouse move
        if not running and abs(self._overlay - target) < 1e-3:
            return
        self._fade.stop()
        self._fade.setStartValue(self._overlay)
        self._fade.setEndValue(target)
        self._fade.start()

    def _show_overlay(self) -> None:
        if self._images:
            self._fade_to(1.0)
            self._idle.start()

    def _rest(self) -> None:
        if self._playing:                       # paused, the button stays for as long as the pointer is here
            self._fade_to(0.0)

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._hovering = True
        self._show_overlay()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt name
        self._hovering = False
        self._idle.stop()
        self._fade_to(0.0)
        super().leaveEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().mouseMoveEvent(event)
        if self._hovering:
            self._show_overlay()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt name
        super().mouseReleaseEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return
        if self._swallow_release:               # the second half of a double-click
            self._swallow_release = False
            return
        if self._images and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt name
        zoomed = self._zoom > 1.001
        super().mouseDoubleClickEvent(event)    # double-click still fits the picture again
        self._swallow_release = True
        if zoomed:
            self.clicked.emit()                 # the first click already toggled playback; undo that

    def _content_size(self):
        return self._images[0].size() if self._images else None

    def _flip(self) -> None:
        if len(self._images) > 1:
            self._index = (self._index + 1) % len(self._images)
            self.update()

    def set_images(self, arrays: list[np.ndarray], badge: str = "", wiggle: bool = False) -> None:
        """Show one array, or several to alternate between. Arrays are HxWx3 uint8."""
        size = QSize(arrays[0].shape[1], arrays[0].shape[0])
        fresh = not self._images or self._images[0].size() != size
        self._images = [to_qimage_u8(a) for a in arrays]
        self._index = min(self._index, len(self._images) - 1)
        self._badge = badge
        if wiggle and len(self._images) > 1:
            if not self._wiggle.isActive():
                self._wiggle.start()
        else:
            self._wiggle.stop()
            self._index = 0
        if fresh:
            self.reset_view()
        self.update()

    def set_note(self, text: str) -> None:
        self._note = text
        self.update()

    def clear(self) -> None:
        self._images = []
        self._wiggle.stop()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt name
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if not self._images:
            return
        image = self._images[self._index]
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self._display_rect(image.size())
        painter.drawImage(rect, image)
        if self._note:
            painter.save()
            _overlay_font(painter, 8.5, caps=False)
            painter.setPen(_OVERLAY_READ)
            painter.drawText(QRectF(0, self.height() - 26, self.width(), 22),
                             Qt.AlignmentFlag.AlignCenter, self._note)
            painter.restore()
        if self._chrome_opacity > 0.01:
            painter.save()
            painter.setOpacity(self._chrome_opacity)
            if self._badge:
                _corner_label(painter, self._badge, rect, self, True, accent=True)
            _paint_readout(painter, self, self._readout_text(image.size()))
            painter.restore()
        if self._overlay > 0.01:
            self._paint_transport(painter)

    def _paint_transport(self, painter: QPainter) -> None:
        """The round play/pause button, in the middle of the picture: it shows what a click will do."""
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        o = self._overlay
        c = QPointF(self.width() / 2.0, self.height() / 2.0)
        r = min(38.0, min(self.width(), self.height()) * 0.16)
        painter.setPen(QPen(QColor(255, 255, 255, int(70 * o)), 1.5))
        painter.setBrush(QColor(8, 14, 22, int(165 * o)))
        painter.drawEllipse(c, r, r)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(255, 255, 255, int(240 * o)))
        if self._playing:                                    # click = pause
            bar_w, bar_h, gap = r * 0.20, r * 0.72, r * 0.16
            for dx in (-gap - bar_w, gap):
                painter.drawRoundedRect(QRectF(c.x() + dx, c.y() - bar_h / 2, bar_w, bar_h), 2.0, 2.0)
        else:                                                # click = play
            s = r * 0.42
            painter.drawPolygon(QPolygonF([QPointF(c.x() - s * 0.7, c.y() - s * 1.05),
                                           QPointF(c.x() - s * 0.7, c.y() + s * 1.05),
                                           QPointF(c.x() + s * 1.15, c.y())]))
        painter.restore()


class CurveEditor(QWidget):
    """A tone curve edited by dragging points: click to add, double-click to remove."""

    changed = Signal(list)

    _PAD = 8.0
    _HIT = 9.0

    def __init__(self, color: QColor | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self._color = color or QColor(230, 236, 246)
        self._points: list[list[float]] = [[0.0, 0.0], [1.0, 1.0]]
        self._drag: int | None = None

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, w: int) -> int:  # noqa: N802
        return w

    def set_color(self, color: QColor) -> None:
        self._color = color
        self.update()

    def set_points(self, points) -> None:
        self._points = [list(map(float, p)) for p in points]
        self.update()

    def points(self) -> list[list[float]]:
        return [list(p) for p in self._points]

    def _box(self) -> QRectF:
        side = min(self.width(), self.height()) - 2 * self._PAD
        return QRectF(self._PAD, self._PAD, side, side)

    def _to_px(self, p) -> QPointF:
        b = self._box()
        return QPointF(b.left() + p[0] * b.width(), b.bottom() - p[1] * b.height())

    def _from_px(self, pos: QPointF) -> list[float]:
        b = self._box()
        return [min(max((pos.x() - b.left()) / b.width(), 0.0), 1.0),
                min(max((b.bottom() - pos.y()) / b.height(), 0.0), 1.0)]

    def _hit(self, pos: QPointF) -> int | None:
        for i, p in enumerate(self._points):
            q = self._to_px(p)
            if (q.x() - pos.x()) ** 2 + (q.y() - pos.y()) ** 2 <= self._HIT ** 2:
                return i
        return None

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if e.button() != Qt.MouseButton.LeftButton:
            return
        hit = self._hit(e.position())
        if hit is None:
            self._points.append(self._from_px(e.position()))
            self._points.sort(key=lambda p: p[0])
            hit = self._hit(e.position())
        self._drag = hit
        self.changed.emit(self.points())
        self.update()

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if self._drag is None:
            return
        x, y = self._from_px(e.position())
        i = self._drag
        if i in (0, len(self._points) - 1):
            x = self._points[i][0]                       # end points slide vertically only
        else:
            x = min(max(x, self._points[i - 1][0] + 0.01), self._points[i + 1][0] - 0.01)
        self._points[i] = [x, y]
        self.changed.emit(self.points())
        self.update()

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self._drag = None

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        hit = self._hit(e.position())
        if hit is not None and 0 < hit < len(self._points) - 1:
            del self._points[hit]
            self.changed.emit(self.points())
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        from .grade import monotone_curve

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        b = self._box()
        painter.fillRect(b, QColor(8, 12, 19))
        painter.setPen(QPen(QColor(35, 48, 74), 1))
        for i in range(1, 4):
            painter.drawLine(QPointF(b.left() + i * b.width() / 4, b.top()), QPointF(b.left() + i * b.width() / 4, b.bottom()))
            painter.drawLine(QPointF(b.left(), b.top() + i * b.height() / 4), QPointF(b.right(), b.top() + i * b.height() / 4))
        painter.drawRect(b)
        painter.setPen(QPen(QColor(70, 84, 110), 1, Qt.PenStyle.DashLine))
        painter.drawLine(b.bottomLeft(), b.topRight())
        curve = monotone_curve(self._points, 128)
        path = QPainterPath(self._to_px([0.0, float(curve[0])]))
        for i in range(1, 128):
            path.lineTo(self._to_px([i / 127.0, float(curve[i])]))
        painter.setPen(QPen(self._color, 2))
        painter.drawPath(path)
        painter.setBrush(QColor(8, 12, 19))
        for p in self._points:
            painter.drawEllipse(self._to_px(p), 4.5, 4.5)


class ColourWheel(QWidget):
    """A colour-balance wheel: drag the puck toward a hue; distance from centre is strength."""

    changed = Signal(float, float)       # hue degrees, amount 0..100

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = title
        self._hue = 0.0
        self._amount = 0.0
        self.setMinimumSize(92, 104)
        self.setToolTip("Drag toward a colour. Double-click to reset.")

    def set_state(self, hue: float, amount: float) -> None:
        self._hue, self._amount = float(hue), float(amount)
        self.update()

    def _disc(self) -> QRectF:
        side = min(self.width() - 8, self.height() - 22)
        return QRectF((self.width() - side) / 2, 4, side, side)

    def _apply(self, pos: QPointF) -> None:
        d = self._disc()
        dx, dy = pos.x() - d.center().x(), pos.y() - d.center().y()
        self._amount = float(min(np.hypot(dx, dy) / (d.width() / 2), 1.0) * 100.0)
        self._hue = float(np.degrees(np.arctan2(-dy, dx)) % 360.0)
        self.changed.emit(self._hue, self._amount)
        self.update()

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self._apply(e.position())

    def mouseMoveEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        if e.buttons() & Qt.MouseButton.LeftButton:
            self._apply(e.position())

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self._hue = self._amount = 0.0
        self.changed.emit(0.0, 0.0)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        d = self._disc()
        cone = QConicalGradient(d.center(), 0.0)
        for i in range(0, 361, 30):
            cone.setColorAt(i / 360.0, QColor.fromHsv(int(i) % 360, 200, 210))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(cone))
        painter.drawEllipse(d)
        fade = QRadialGradient(d.center(), d.width() / 2)
        fade.setColorAt(0.0, QColor(20, 26, 38, 235))
        fade.setColorAt(1.0, QColor(20, 26, 38, 0))
        painter.setBrush(QBrush(fade))
        painter.drawEllipse(d)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(90, 104, 130), 1))
        painter.drawEllipse(d)
        r = d.width() / 2 * self._amount / 100.0
        a = np.radians(self._hue)
        puck = QPointF(d.center().x() + r * np.cos(a), d.center().y() - r * np.sin(a))
        painter.setPen(QPen(QColor(255, 255, 255), 1.6))
        painter.setBrush(QColor(12, 18, 28))
        painter.drawEllipse(puck, 4.5, 4.5)
        painter.setPen(QColor(147, 162, 188))
        painter.setFont(QFont(FONT_MONO, 7))
        painter.drawText(QRectF(0, self.height() - 16, self.width(), 14), Qt.AlignmentFlag.AlignCenter, self._title.upper())


class ScopeView(QFrame):
    """Histogram / waveform / vectorscope of the picture on screen, like Lumetri Scopes."""

    MODES = ("Histogram", "Waveform", "Vectorscope")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("scopeBox")
        self.setMinimumHeight(120)
        self.setMaximumHeight(150)
        self._mode = 0
        self._frame: np.ndarray | None = None
        self._image: QImage | None = None
        self.setToolTip("Click to switch: histogram, waveform, vectorscope.")

    def mousePressEvent(self, e: QMouseEvent) -> None:  # noqa: N802
        self._mode = (self._mode + 1) % len(self.MODES)
        self._recompute()

    def set_frame(self, rgb_u8: np.ndarray | None) -> None:
        # No further subsampling: the frame arrives already small (240 px), and 1/9 of that leaves
        # too few pixels for a histogram to mean anything.
        self._frame = None if rgb_u8 is None else np.ascontiguousarray(rgb_u8)
        self._recompute()

    def _recompute(self) -> None:
        if self._frame is None or not self.isVisible():
            self._image = None
        else:
            self._image = to_qimage_u8(_scope_image(
                self._frame, self._mode, max(self.width() - 2, 64), max(self.height() - 2, 48)))
        self.update()

    def resizeEvent(self, e) -> None:  # noqa: N802
        self._recompute()
        super().resizeEvent(e)

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        if self._image is not None:
            painter.drawImage(1, 1, self._image)
        painter.setPen(QColor(147, 162, 188))
        painter.setFont(QFont(FONT_MONO, 7))
        painter.drawText(8, 14, self.MODES[self._mode].upper())


def _scope_image(frame: np.ndarray, mode: int, w: int, h: int) -> np.ndarray:
    """Render a scope to an RGB uint8 image of size (h, w)."""
    out = np.zeros((h, w, 3), np.float32)
    if mode == 0:                                            # histogram, RGB overlaid
        # 256 real bins, smoothed and then stretched to the width: histogramming straight into
        # `w` bins leaves every few bins empty (a comb). The scale ignores the two end bins and
        # a few tall spikes - a letterboxed or clipped frame piles up to a third of its pixels
        # in bin 0, and scaling by that flattens the whole picture's histogram to a sliver.
        hists = np.stack([np.histogram(frame[..., c], bins=256, range=(0, 256))[0] for c in range(3)]
                         ).astype(np.float32)
        hists = np.stack([np.convolve(row, (0.25, 0.5, 0.25), mode="same") for row in hists])
        scale = max(float(np.percentile(hists[:, 2:254], 98.5)) * 1.15, 1.0)
        xs = np.linspace(0, 255, w)
        rows = np.arange(h)[:, None]
        top = h - 16                                         # the label sits above the plot
        for c, tint in enumerate(((255, 70, 70), (70, 255, 90), (80, 130, 255))):
            curve = np.clip(np.interp(xs, np.arange(256), hists[c]) / scale, 0.0, 1.0)
            heights = (curve ** 0.85 * top).astype(int)
            fill = rows >= (h - heights)[None, :]
            out += fill[..., None] * (np.array(tint, np.float32) * 0.45)
    elif mode == 1:                                          # luma waveform
        luma = frame.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        cols = np.linspace(0, w - 1, luma.shape[1]).astype(int)
        rows = np.clip((1.0 - luma / 255.0) * (h - 1), 0, h - 1).astype(int)
        for x in range(luma.shape[1]):
            np.add.at(out, (rows[:, x], cols[x]), (40, 220, 130))
        out *= 1.4
    else:                                                    # vectorscope
        r, g, b = (frame[..., i].astype(np.float32) / 255.0 for i in range(3))
        u = -0.147 * r - 0.289 * g + 0.436 * b
        v = 0.615 * r - 0.515 * g - 0.100 * b
        size = min(w, h) - 8
        cx, cy = w // 2, h // 2
        xs = np.clip((cx + u / 0.45 * size / 2).astype(int), 0, w - 1)
        ys = np.clip((cy - v / 0.62 * size / 2).astype(int), 0, h - 1)
        np.add.at(out, (ys.ravel(), xs.ravel()), (60, 210, 255))
        yy, xx = np.ogrid[:h, :w]
        out[np.abs(np.hypot(xx - cx, yy - cy) - size / 2) < 0.8] = (45, 60, 90)
    return np.clip(out, 0, 255).astype(np.uint8)
