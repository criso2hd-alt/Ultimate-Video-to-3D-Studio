"""The rail's control panels: colour correction, LUT, 3D and export.

Each panel edits a settings object *in place* and emits ``changed``; the window
listens once and refreshes the preview, so a panel never needs to know about
the player, the preview or the export.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import encoders, lut, paths, stereo
from .controls import (
    CollapsibleCard,
    ColourWheel,
    CurveEditor,
    ParamSlider,
    ScopeView,
    SectionTabs,
)
from .grade import IDENTITY_CURVE, ColourSettings, Wheel
from .settings import ExportSettings, ThreeDSettings
from .widgets import FONT_DISPLAY, ModuleCard, SegmentedControl, apply_font


def _row(label: str, widget: QWidget, width: int = 168) -> QHBoxLayout:
    row = QHBoxLayout()
    row.addWidget(QLabel(label))
    row.addStretch(1)
    widget.setMinimumWidth(width)
    row.addWidget(widget)
    return row


class ColourPanel(CollapsibleCard):
    """Lumetri-style colour correction: Basic, Creative, Curves, Wheels, Vignette + scopes."""

    changed = Signal()

    def __init__(self, s: ColourSettings, parent: QWidget | None = None) -> None:
        super().__init__("Enable colour correction", "", parent)
        self._s = s
        self._sliders: dict[str, ParamSlider] = {}
        self.set_enabled_state(s.enabled)
        self.toggled.connect(self._enable_toggled)

        self.scope = ScopeView()
        self.add(self.scope)

        tabs = SectionTabs(["Basic", "Creative", "Curves", "Wheels", "Vignette"])
        self.add(tabs)
        self._stack = QStackedWidget()
        tabs.changed.connect(self._stack.setCurrentIndex)
        self.add(self._stack)

        self._stack.addWidget(self._page([
            ("Temperature", "temperature", -100, 100, 0, "Cooler (blue) to warmer (orange)."),
            ("Tint", "tint", -100, 100, 0, "Green to magenta."),
            ("Exposure", "exposure", -4, 4, 2, "Stops of light, in linear light."),
            ("Contrast", "contrast", -100, 100, 0, "S-curve around mid-grey."),
            ("Highlights", "highlights", -100, 100, 0, "Recover or push the bright areas."),
            ("Shadows", "shadows", -100, 100, 0, "Lift or deepen the dark areas."),
            ("Whites", "whites", -100, 100, 0, "The white clipping point."),
            ("Blacks", "blacks", -100, 100, 0, "The black clipping point."),
            ("Saturation", "saturation", -100, 100, 0, "-100 is black and white."),
        ]))
        self._stack.addWidget(self._page([
            ("Faded film", "faded_film", 0, 100, 0, "Lifts the blacks and softens the whites."),
            ("Sharpen", "sharpen", 0, 100, 0, "Edge sharpening applied last."),
            ("Vibrance", "vibrance", -100, 100, 0, "Saturation that leaves skin tones alone."),
        ]))
        self._stack.addWidget(self._curves_page())
        self._stack.addWidget(self._wheels_page())
        self._stack.addWidget(self._page([
            ("Amount", "vignette", -100, 100, 0, "Darken (+) or lighten (-) the edges."),
            ("Midpoint", "vignette_midpoint", 0, 100, 0, "How far from the edge it starts."),
            ("Roundness", "vignette_roundness", 0, 100, 0, "Follow the frame, or a true circle."),
            ("Feather", "vignette_feather", 0, 100, 0, "How soft the transition is."),
        ]))

        reset = QPushButton("Reset all")
        reset.setObjectName("secondary")
        reset.clicked.connect(self.reset_all)
        self.add(reset)

    # -- building ---------------------------------------------------------

    def _page(self, rows) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 4, 0, 0)
        col.setSpacing(2)
        for label, attr, lo, hi, dec, tip in rows:
            default = ColourSettings().__dict__[attr]
            slider = ParamSlider(label, lo, hi, getattr(self._s, attr), default, dec, tooltip=tip)
            slider.changed.connect(lambda v, a=attr: self._set(a, v))
            self._sliders[attr] = slider
            col.addWidget(slider)
        col.addStretch(1)
        return page

    def _curves_page(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 4, 0, 0)
        self._curve_names = (("Master", "curve_master", QColor(232, 238, 249)),
                             ("Red", "curve_red", QColor(255, 96, 96)),
                             ("Green", "curve_green", QColor(96, 230, 120)),
                             ("Blue", "curve_blue", QColor(100, 150, 255)))
        self._curve_pick = SegmentedControl([n for n, _a, _c in self._curve_names])
        self._curve_pick.changed.connect(self._curve_channel)
        col.addWidget(self._curve_pick)
        self._curve = CurveEditor()
        self._curve.changed.connect(self._curve_edited)
        col.addWidget(self._curve)
        hint = QLabel("Click to add a point · drag to shape · double-click to remove")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        col.addWidget(hint)
        self._curve_channel(0)
        return page

    def _wheels_page(self) -> QWidget:
        page = QWidget()
        col = QVBoxLayout(page)
        col.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        self._wheels = []
        for title, attr in (("Shadows", "wheel_shadows"), ("Midtones", "wheel_midtones"),
                            ("Highlights", "wheel_highlights")):
            wheel = ColourWheel(title)
            w: Wheel = getattr(self._s, attr)
            wheel.set_state(w.hue, w.amount)
            wheel.changed.connect(lambda hue, amount, a=attr: self._wheel_moved(a, hue, amount))
            self._wheels.append((attr, wheel))
            row.addWidget(wheel)
        col.addLayout(row)
        for title, attr in (("Shadows brightness", "wheel_shadows"), ("Midtones brightness", "wheel_midtones"),
                            ("Highlights brightness", "wheel_highlights")):
            slider = ParamSlider(title, -100, 100, getattr(self._s, attr).luma, 0)
            slider.changed.connect(lambda v, a=attr: self._wheel_luma(a, v))
            self._sliders[attr + ".luma"] = slider
            col.addWidget(slider)
        col.addStretch(1)
        return page

    # -- editing ----------------------------------------------------------

    def _set(self, attr: str, value: float) -> None:
        setattr(self._s, attr, value)
        self.changed.emit()

    def _enable_toggled(self, on: bool) -> None:
        self._s.enabled = on
        if on and not self._body.isVisible():
            self.set_open(True)         # switching it on should show what it does
        self.changed.emit()

    def _curve_channel(self, index: int) -> None:
        _name, attr, color = self._curve_names[index]
        self._curve.set_color(color)
        self._curve.set_points(getattr(self._s, attr))

    def _curve_edited(self, points: list) -> None:
        attr = self._curve_names[max(0, self._curve_pick.current_index())][1]
        setattr(self._s, attr, points)
        self.changed.emit()

    def _wheel_moved(self, attr: str, hue: float, amount: float) -> None:
        w: Wheel = getattr(self._s, attr)
        w.hue, w.amount = hue, amount
        self.changed.emit()

    def _wheel_luma(self, attr: str, value: float) -> None:
        getattr(self._s, attr).luma = value
        self.changed.emit()

    def reset_all(self) -> None:
        fresh = ColourSettings()
        keep = (self._s.enabled, self._s.lut_enabled, self._s.lut_path, self._s.lut_intensity, self._s.lut_before)
        for key, value in fresh.__dict__.items():
            setattr(self._s, key, value)
        (self._s.enabled, self._s.lut_enabled, self._s.lut_path,
         self._s.lut_intensity, self._s.lut_before) = keep
        self.refresh()
        self.changed.emit()

    def refresh(self) -> None:
        for attr, slider in self._sliders.items():
            if attr.endswith(".luma"):
                slider.set_value(getattr(self._s, attr[:-5]).luma)
            else:
                slider.set_value(getattr(self._s, attr))
        for attr, wheel in self._wheels:
            w: Wheel = getattr(self._s, attr)
            wheel.set_state(w.hue, w.amount)
        self._curve_channel(max(0, self._curve_pick.current_index()))
        self.set_enabled_state(self._s.enabled)


class LutPanel(CollapsibleCard):
    """Load a ``.cube`` look and apply it, before or after the grade."""

    changed = Signal()

    def __init__(self, s: ColourSettings, parent: QWidget | None = None) -> None:
        super().__init__("Look · LUT", "load a .cube", parent)
        self._s = s
        self.set_enabled_state(s.lut_enabled)
        self.toggled.connect(self._toggled)

        self.combo = QComboBox()
        self.combo.currentIndexChanged.connect(self._picked)
        self.add(self.combo)

        row = QHBoxLayout()
        browse = QPushButton("Load .cube…")
        browse.setObjectName("secondary")
        browse.clicked.connect(self._browse)
        folder = QPushButton("Open LUT folder")
        folder.setObjectName("secondary")
        folder.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths.luts_dir()))))
        row.addWidget(browse)
        row.addWidget(folder)
        self.add_layout(row)

        self.intensity = ParamSlider("Intensity", 0, 100, s.lut_intensity, 100, 0, "%")
        self.intensity.changed.connect(self._intensity)
        self.add(self.intensity)

        self.before = QCheckBox("Apply before the colour correction (input LUT)")
        self.before.setChecked(s.lut_before)
        self.before.setToolTip("A camera-log conversion belongs first; a creative look belongs last.")
        self.before.toggled.connect(self._before)
        self.add(self.before)

        self.status = QLabel("")
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        self.add(self.status)
        self.refresh_list()

    def refresh_list(self) -> None:
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("— none —", "")
        found = lut.available_luts(paths.luts_dir())
        for path in found:
            self.combo.addItem(path.stem, str(path))
        if self._s.lut_path and not any(str(p) == self._s.lut_path for p in found):
            self.combo.addItem(Path(self._s.lut_path).stem, self._s.lut_path)
        index = self.combo.findData(self._s.lut_path)
        self.combo.setCurrentIndex(max(0, index))
        self.combo.blockSignals(False)
        self._validate()

    def _validate(self) -> None:
        if not self._s.lut_path:
            self.status.setText(f"Drop .cube files in the LUT folder, or load one from anywhere.")
            self.set_subtitle("load a .cube")
            return
        try:
            loaded = lut.load_cube(Path(self._s.lut_path))
            self.status.setText(f"{loaded.size}³ table{' · ' + loaded.title if loaded.title else ''}")
            self.set_subtitle(Path(self._s.lut_path).stem)
        except ValueError as error:
            self.status.setText(f"Cannot use this LUT: {error}")

    def _toggled(self, on: bool) -> None:
        self._s.lut_enabled = on
        if on and not self._s.lut_path and self.combo.count() > 1:
            self.combo.setCurrentIndex(1)       # first available look, so switching on does something
        elif on and not self._body.isVisible():
            self.set_open(True)
        self.changed.emit()

    def _picked(self) -> None:
        self._s.lut_path = self.combo.currentData() or ""
        self._validate()
        self.changed.emit()

    def _browse(self) -> None:
        chosen, _ = QFileDialog.getOpenFileName(self, "Load a LUT", "", "LUT (*.cube)")
        if not chosen:
            return
        self._s.lut_path = chosen
        self._s.lut_enabled = True
        self.set_enabled_state(True)
        self.refresh_list()
        self.changed.emit()

    def _intensity(self, value: float) -> None:
        self._s.lut_intensity = value
        self.changed.emit()

    def _before(self, on: bool) -> None:
        self._s.lut_before = on
        self.changed.emit()


class ThreeDPanel(ModuleCard):
    """The 2D -> 3D conversion controls."""

    changed = Signal()

    def __init__(self, s: ThreeDSettings, parent: QWidget | None = None) -> None:
        super().__init__("3D conversion", "AI depth", checkable=True, parent=parent)
        self._s = s
        self.setChecked(s.enabled)
        self.toggled.connect(self._enabled)

        self.layout_box = QComboBox()
        for layout in stereo.LAYOUTS:
            self.layout_box.addItem(layout.label, layout.key)
        self.layout_box.setCurrentIndex(max(0, self.layout_box.findData(s.layout)))
        self.layout_box.currentIndexChanged.connect(self._layout)
        self.add_layout(_row("Output", self.layout_box))
        self.layout_note = QLabel("")
        self.layout_note.setObjectName("hint")
        self.layout_note.setWordWrap(True)
        self.add(self.layout_note)
        self._layout()

        self.strength = ParamSlider("3D strength", 0, 100, s.strength, 50, 0,
                                    tooltip="How far the two eyes are pushed apart. Cinema is around 30-50; "
                                            "VR headsets tolerate more. Too much is uncomfortable.")
        self.strength.changed.connect(lambda v: self._set("strength", v))
        self.add(self.strength)
        self.convergence = ParamSlider("Depth position", 0, 100, s.convergence, 55, 0,
                                       tooltip="Where the screen sits inside the scene. Low pulls everything "
                                               "out of the screen; high pushes it behind, like a window.")
        self.convergence.changed.connect(lambda v: self._set("convergence", v))
        self.add(self.convergence)

        self.near = ParamSlider("Foreground depth", 0, 200, s.near_scale, 100, 0, "%",
                                "Scales only the part that comes out of the screen.")
        self.near.changed.connect(lambda v: self._set("near_scale", v))
        self.far = ParamSlider("Background depth", 0, 200, s.far_scale, 100, 0, "%",
                               "Scales only the part behind the screen.")
        self.far.changed.connect(lambda v: self._set("far_scale", v))
        self.temporal = ParamSlider("Depth stability", 0, 100, s.temporal, 60, 0,
                                    tooltip="Steadies depth over time so it does not pump or shimmer. "
                                            "Very high can lag behind fast motion.")
        self.temporal.changed.connect(lambda v: self._set("temporal", v))
        self.blur = ParamSlider("Depth softening", 0, 100, s.depth_blur, 0, 0,
                                tooltip="Blurs the depth map to hide stretching at edges, at the cost of sharper 3D.")
        self.blur.changed.connect(lambda v: self._set("depth_blur", v))
        self.edge = QCheckBox("Snap depth edges to the picture")
        self.edge.setChecked(s.edge_aware)
        self.edge.setToolTip("Aligns the depth map's edges with the image's edges. Removes halos around people.")
        self.edge.toggled.connect(lambda on: self._set("edge_aware", on))
        self.swap = QCheckBox("Swap left / right eyes")
        self.swap.setChecked(s.swap_eyes)
        self.swap.setToolTip("If the depth looks inside-out, or for cross-eyed viewing.")
        self.swap.toggled.connect(lambda on: self._set("swap_eyes", on))

        self.advanced = QPushButton("Advanced ▸")
        self.advanced.setObjectName("link")
        self.advanced.setCheckable(True)
        self.advanced.toggled.connect(self._advanced)
        self.add(self.advanced)
        self._adv_widgets = [self.near, self.far, self.temporal, self.blur, self.edge, self.swap]
        for widget in self._adv_widgets:
            self.add(widget)
            widget.setVisible(False)

    def _advanced(self, on: bool) -> None:
        self.advanced.setText("Advanced ▾" if on else "Advanced ▸")
        for widget in self._adv_widgets:
            widget.setVisible(on)

    def _enabled(self, on: bool) -> None:
        self._s.enabled = on
        self.changed.emit()

    def _layout(self) -> None:
        key = self.layout_box.currentData()
        self._s.layout = key
        self.layout_note.setText(stereo.LAYOUTS_BY_KEY[key].note)
        self.changed.emit()

    def _set(self, attr: str, value) -> None:
        setattr(self._s, attr, value)
        self.changed.emit()


class ExportPanel(ModuleCard):
    """Format, encoder, quality, size, audio and HDR."""

    changed = Signal()

    HEIGHTS = (("Same as source", 0), ("2160p (4K)", 2160), ("1440p", 1440), ("1080p", 1080), ("720p", 720))

    def __init__(self, s: ExportSettings, parent: QWidget | None = None) -> None:
        super().__init__("Export", "", parent=parent)
        self._s = s
        self._probe: encoders.ProbeReport | None = None

        self.family_box = QComboBox()
        for fam in encoders.FAMILIES:
            self.family_box.addItem(fam.label, fam.key)
        self.family_box.setCurrentIndex(max(0, self.family_box.findData(s.family)))
        self.family_box.currentIndexChanged.connect(self._family)
        self.add_layout(_row("Format", self.family_box))
        self.family_note = QLabel("")
        self.family_note.setObjectName("hint")
        self.family_note.setWordWrap(True)
        self.add(self.family_note)

        self.encoder_box = QComboBox()
        self.encoder_box.currentIndexChanged.connect(self._encoder)
        self.add_layout(_row("Encoder", self.encoder_box))

        self.profile_box = QComboBox()
        self.profile_box.currentIndexChanged.connect(self._profile)
        self._profile_row = _row("Profile", self.profile_box)
        self.add_layout(self._profile_row)

        self.quality = ParamSlider("Quality", 0, 100, s.quality, 60, 0,
                                   tooltip="Higher is larger and closer to the original. 60 is a good default.")
        self.quality.changed.connect(lambda v: self._set("quality", int(v)))
        self.add(self.quality)

        self.height_box = QComboBox()
        for label, value in self.HEIGHTS:
            self.height_box.addItem(label, value)
        self.height_box.setCurrentIndex(max(0, self.height_box.findData(s.max_height)))
        self.height_box.currentIndexChanged.connect(lambda: self._set("max_height", self.height_box.currentData()))
        self.height_box.setToolTip("Height of each eye. Half layouts squeeze two eyes into this frame.")
        self.add_layout(_row("Size (per eye)", self.height_box))

        self.audio_box = QComboBox()
        for label, value in (("Copy original", "copy"), ("Re-encode (AAC / Opus)", "aac"), ("No audio", "none")):
            self.audio_box.addItem(label, value)
        self.audio_box.setCurrentIndex(max(0, self.audio_box.findData(s.audio)))
        self.audio_box.currentIndexChanged.connect(lambda: self._set("audio", self.audio_box.currentData()))
        self.add_layout(_row("Audio", self.audio_box))

        self.hdr_box = QComboBox()
        self.hdr_box.addItem("Keep HDR (10-bit)", "keep")
        self.hdr_box.addItem("Convert to SDR", "tonemap")
        self.hdr_box.setCurrentIndex(max(0, self.hdr_box.findData(s.hdr_mode)))
        self.hdr_box.currentIndexChanged.connect(lambda: self._set("hdr_mode", self.hdr_box.currentData()))
        self.hdr_box.setToolTip("Only matters when the source is HDR. H.264 cannot carry HDR.")
        self.hdr_row = _row("HDR", self.hdr_box)
        self.add_layout(self.hdr_row)

        self.range_box = QComboBox()
        self.range_box.addItem("Whole clip", "whole")
        self.range_box.addItem("Select In/Out", "range")
        self.range_box.setCurrentIndex(max(0, self.range_box.findData(s.range_mode)))
        self.range_box.currentIndexChanged.connect(lambda: self._set("range_mode", self.range_box.currentData()))
        self.range_box.setToolTip("Select In/Out shows brackets on the timeline: drag them to the part you want, "
                                  "scroll to zoom for a precise edit.")
        self.add_layout(_row("Range", self.range_box))
        self._family()

    def set_probe(self, report: encoders.ProbeReport | None) -> None:
        self._probe = report
        self._fill_encoders()

    def _fill_encoders(self) -> None:
        fam = encoders.FAMILIES_BY_KEY[self.family_box.currentData()]
        self.encoder_box.blockSignals(True)
        self.encoder_box.clear()
        best = self._probe.defaults.get(fam.key) if self._probe else None
        self.encoder_box.addItem(
            f"Auto — {encoders.encoder_label(best)}" if best else "Auto", "auto")
        if self._probe:
            for name in encoders.ladder(fam, []):
                if self._probe.results.get(name):
                    self.encoder_box.addItem(encoders.encoder_label(name), name)
        index = self.encoder_box.findData(self._s.encoder)
        self.encoder_box.setCurrentIndex(max(0, index))
        self.encoder_box.blockSignals(False)

    def _set(self, attr: str, value) -> None:
        setattr(self._s, attr, value)
        self.changed.emit()

    def _family(self) -> None:
        key = self.family_box.currentData()
        fam = encoders.FAMILIES_BY_KEY[key]
        self._s.family = key
        self.family_note.setText(fam.note)
        self.quality.setVisible(fam.uses_quality)
        self.profile_box.blockSignals(True)
        self.profile_box.clear()
        for i, p in enumerate(fam.profiles):
            self.profile_box.addItem(p.label, i)
        default = min(self._s.profile, max(len(fam.profiles) - 1, 0))
        self.profile_box.setCurrentIndex(default if fam.profiles else -1)
        self.profile_box.blockSignals(False)
        for i in range(self._profile_row.count()):
            item = self._profile_row.itemAt(i).widget()
            if item is not None:
                item.setVisible(bool(fam.profiles))
        self._fill_encoders()
        self.changed.emit()

    def _encoder(self) -> None:
        self._s.encoder = self.encoder_box.currentData() or "auto"
        self.changed.emit()

    def _profile(self) -> None:
        self._s.profile = max(0, self.profile_box.currentIndex())
        self.changed.emit()

    def set_hdr_visible(self, hdr_source: bool) -> None:
        for i in range(self.hdr_row.count()):
            widget = self.hdr_row.itemAt(i).widget()
            if widget is not None:
                widget.setVisible(hdr_source)


def open_folder(path: Path) -> None:
    """Reveal a folder in the platform's file manager."""
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
