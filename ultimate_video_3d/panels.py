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
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import encoders, lut, paths, ratecontrol, stereo, video
from . import zones as zonesmod
from .controls import (
    CollapsibleCard,
    ColourWheel,
    CurveEditor,
    DepthGauge,
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
    fit_requested = Signal(float)           # look through the whole film and set the strength so its strongest moment sits at this % of the width
    #: The person picked a depth model ("fast" / "quality"). The window decides: it may need a download first, and calls
    #: ``set_depth_model`` with what is really in use (so a cancelled download puts the box back).
    model_requested = Signal(str)

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

        self.model_box = QComboBox()
        self.model_box.addItem("Fast", "fast")
        self.model_box.addItem("High quality", "quality")
        self.model_box.setToolTip(
            "The AI that works out depth.\n\nFast: Depth Anything V2 Small, built in. Quick enough for smooth previews on "
            "any graphics card.\n\nHigh quality: Depth Anything 3 Mono-Large. Keeps people and objects on clearer layers "
            "and holds distance better (landscapes, rooms). A one-time download of about 1.3 GB; slower on most cards "
            "(conversions only a little), fast with the NVIDIA add-on in Settings > Performance.")
        self.model_box.currentIndexChanged.connect(lambda: self.model_requested.emit(self.model_box.currentData()))
        self.add_layout(_row("Depth model", self.model_box))
        self.set_depth_model(s.depth_model)

        self.gauge = DepthGauge()
        self.add(self.gauge)
        self.strength = ParamSlider("3D strength", 0, 100, s.strength, 50, 0,
                                    tooltip="How far the two eyes are pushed apart. Cinema is around 30-50; "
                                            "VR headsets tolerate more. Too much is uncomfortable.")
        self.strength.changed.connect(lambda v: self._set("strength", v))
        self.strength.setToolTip(self.strength.toolTip() + "\n\nThe coloured strip under the slider shows how strong the "
                                 "effect is for this picture: green = comfortable, amber = strong, red = too strong.")
        self.add(self.strength)
        # One setting for the whole film: find its strongest moment and scale the strength so that moment is just comfortable.
        self.fit_comfortable = QPushButton("Fit to comfort")
        self.fit_strong = QPushButton("Fit to strong")
        for button, limit, tip in (
            (self.fit_comfortable, zonesmod.COMFORT, "Looks through the whole video (or the In to Out part) for the moment that comes out of "
             "or sinks behind the screen the most, and sets the 3D strength so that moment is just comfortable. Every other moment is "
             "scaled with it: one setting for the film, nothing animates."),
            (self.fit_strong, zonesmod.STRONG, "The same, but allows the strongest moment to reach the strong limit (twice as far)."),
        ):
            button.setObjectName("secondary")
            button.setToolTip(tip)
            button.clicked.connect(lambda _=False, v=limit: self.fit_requested.emit(float(v)))
        fit_row = QHBoxLayout()
        fit_row.addWidget(self.fit_comfortable)
        fit_row.addWidget(self.fit_strong)
        self.add_layout(fit_row)
        self.fit_note = QLabel("")
        self.fit_note.setObjectName("hint")
        self.fit_note.setWordWrap(True)
        self.fit_note.setVisible(False)
        self.add(self.fit_note)
        self.convergence = ParamSlider("Depth position", 0, 100, s.convergence, 55, 0,
                                       tooltip="Where the screen sits inside the scene. Low pulls everything "
                                               "out of the screen; high pushes it behind, like a window.")
        self.convergence.changed.connect(lambda v: self._set("convergence", v))
        self.convergence.setToolTip(self.convergence.toolTip() + "\n\nThe coloured strip under the slider shows what this picture does at each setting: warm = nearly all of it comes out of the screen, light = split across the screen (the white tick is where half of it is in front), cool = nearly all of it is behind the screen.")
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
        self.quality = QComboBox()
        self.quality.addItem("Standard", "standard")
        self.quality.addItem("Smoother objects", "smooth")
        self.quality.addItem("Smoother, more detail (slower export)", "smooth_high")
        self.quality.setCurrentIndex(max(0, self.quality.findData(s.depth_quality)))
        self.quality.setToolTip("How the depth map is cleaned up. Standard is the original. Smoother objects flattens the inside of "
                                "people and things, which can look cleaner on busy footage. The last one is the "
                                "same at a larger size when converting; the preview stays quick.")
        self.quality.currentIndexChanged.connect(lambda: self._set("depth_quality", self.quality.currentData()))
        self.quality_row = QWidget()                      # a layout cannot be shown/hidden with the Advanced toggle
        row = _row("Depth cleanup", self.quality)
        row.setContentsMargins(0, 0, 0, 0)
        self.quality_row.setLayout(row)
        self.swap = QCheckBox("Swap left / right eyes")
        self.swap.setChecked(s.swap_eyes)
        self.swap.setToolTip("If the depth looks inside-out, or for cross-eyed viewing.")
        self.swap.toggled.connect(lambda on: self._set("swap_eyes", on))

        self.advanced = QPushButton("Advanced ▸")
        self.advanced.setObjectName("link")
        self.advanced.setCheckable(True)
        self.advanced.toggled.connect(self._advanced)
        self.add(self.advanced)
        self._adv_widgets = [self.near, self.far, self.temporal, self.blur, self.edge, self.quality_row, self.swap]
        for widget in self._adv_widgets:
            self.add(widget)
            widget.setVisible(False)

    def set_depth_model(self, key: str) -> None:
        """Show ``key`` as the model in use, without asking for it again."""
        self.model_box.blockSignals(True)
        self.model_box.setCurrentIndex(max(0, self.model_box.findData(key)))
        self.model_box.blockSignals(False)

    def set_fit_state(self, running: bool, text: str = "") -> None:
        """Show a film-wide scan's progress or result under the buttons; while it runs either button stops it."""
        self.fit_comfortable.setText("Stop" if running else "Fit to comfort")
        self.fit_strong.setEnabled(not running)
        self.fit_note.setText(text)
        self.fit_note.setVisible(bool(text))

    def set_depth_stats(self, stats: zonesmod.DepthStats) -> None:
        """Show where the current picture sits, on the gauge and as the marks under Depth position."""
        self.gauge.set_stats(stats)
        marks = zonesmod.convergence_marks(stats)
        self.convergence.set_zones(*(marks if marks else (None, None, None)))
        limits = zonesmod.strength_marks(stats, self.strength.value())
        if limits is None:
            self.strength.set_strip(None)
        else:
            comfort, strong = limits
            self.strength.set_strip([(0.0, comfort, "good"), (comfort, strong, "strong"), (strong, 100.0, "too")])

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
    audio_track_chosen = Signal(int)              # position of the chosen audio track (so the preview can play that one)

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
                                   tooltip="Higher is larger and closer to the original. 60 is a good default. The bitrate and the "
                                           "file size below move with it (as estimates); type either of them to aim at it exactly.")
        self.quality.changed.connect(self._quality_moved)
        self.add(self.quality)

        # Quality, bitrate and file size are one choice seen three ways: change one and the others follow (see ratecontrol.py).
        self._rate_context = None                      # asks the window: ((width, height), fps, seconds, audio bits/s) or None
        self._syncing = False
        self.bitrate_box = QSpinBox()
        self.bitrate_box.setRange(100, 2_000_000)
        self.bitrate_box.setSingleStep(250)
        self.bitrate_box.setSuffix(" kb/s")
        self.bitrate_box.setKeyboardTracking(False)    # a typed number counts when it is finished, not at every digit
        self.bitrate_box.setMinimumWidth(130)
        self.bitrate_box.setToolTip("The video's bitrate. Type one to make the encode aim at it; the quality and the file size follow.")
        self.bitrate_box.valueChanged.connect(self._bitrate_typed)
        self.size_box = QSpinBox()
        self.size_box.setRange(1, 100_000_000)
        self.size_box.setSingleStep(50)
        self.size_box.setSuffix(" MB")
        self.size_box.setKeyboardTracking(False)
        self.size_box.setMinimumWidth(130)
        self.size_box.setToolTip("The size of the whole file (picture and sound) for the part you are converting. Type one to make the "
                                 "encode land on it; the bitrate and the quality follow.")
        self.size_box.valueChanged.connect(self._size_typed)
        self.rate_note = QLabel("")
        self.rate_note.setObjectName("hint")
        self.rate_note.setWordWrap(True)
        self.rate_block = QWidget()
        rate_layout = QVBoxLayout(self.rate_block)
        rate_layout.setContentsMargins(0, 0, 0, 0)
        rate_layout.setSpacing(6)
        rate_layout.addLayout(_row("Video bitrate", self.bitrate_box))
        rate_layout.addLayout(_row("File size", self.size_box))
        rate_layout.addWidget(self.rate_note)
        self.add(self.rate_block)

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

        # The tracks of the open file, once there is one: which sound to keep and which subtitles (see video.list_tracks).
        self._audio_tracks: list = []
        self._subtitle_tracks: list = []
        self._auto_mkv = False                    # "Save as" went to MKV by itself, because a subtitle was chosen (and goes back by itself)
        self._switching = False
        self.audio_track_box = QComboBox()
        self.audio_track_box.currentIndexChanged.connect(self._audio_track)
        self.audio_track_row = _row("Audio track", self.audio_track_box)
        self.add_layout(self.audio_track_row)
        self.audio_note = QLabel("")
        self.audio_note.setObjectName("hint")
        self.audio_note.setWordWrap(True)
        self.add(self.audio_note)
        self.subtitle_box = QComboBox()
        self.subtitle_box.currentIndexChanged.connect(self._subtitle_track)
        self.subtitle_row = _row("Subtitles", self.subtitle_box)
        self.add_layout(self.subtitle_row)
        self.container_box = QComboBox()
        self.container_box.addItem("MP4", "auto")                  # the formats above are the MP4 ones: H.264 / H.265 / AV1 / VP9 in MP4
        self.container_box.addItem("Matroska (MKV)", "mkv")
        self.container_box.setCurrentIndex(max(0, self.container_box.findData(s.container)))
        self.container_box.currentIndexChanged.connect(self._container)
        self.container_box.setToolTip("MP4 plays everywhere, but turns TrueHD / DTS sound into AAC and cannot hold subtitles. MKV keeps "
                                      "subtitles and lossless sound (TrueHD, DTS-HD) exactly as they are.")
        self.container_row = _row("Save as", self.container_box)
        self.add_layout(self.container_row)
        self._show_track_rows()

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

    # -- the open file's audio and subtitle tracks ----------------------------------------------------------

    def set_tracks(self, audio: list, subtitles: list) -> None:
        """Show the tracks of the file that was just opened (video.Track lists) and go back to the defaults for it."""
        self._audio_tracks, self._subtitle_tracks = list(audio), list(subtitles)
        if self._auto_mkv:
            self._switch_container("auto", False)              # a new file: the MKV a subtitle asked for is not carried over to it
        self._s.audio_track = self._s.subtitle_track = -1
        for box in (self.audio_track_box, self.subtitle_box):
            box.blockSignals(True)
            box.clear()
        self.audio_track_box.addItem("Best available (automatic)", -1)
        for track in self._audio_tracks:
            self.audio_track_box.addItem(track.label(), track.index)
        self.subtitle_box.addItem("None", -1)
        for track in self._subtitle_tracks:
            self.subtitle_box.addItem(track.label(), track.index)
        for box in (self.audio_track_box, self.subtitle_box):
            box.setCurrentIndex(0)
            box.blockSignals(False)
        self._show_track_rows()
        self.changed.emit()

    def choose_tracks(self, audio_index: int, subtitle_index: int) -> None:
        """Select tracks by stream index (the folder dialog's choice for the file it opened)."""
        for box, value in ((self.audio_track_box, audio_index), (self.subtitle_box, subtitle_index)):
            at = box.findData(value)
            if at >= 0:
                box.setCurrentIndex(at)

    def set_container(self, container: str) -> None:
        at = self.container_box.findData(container)
        if at >= 0:
            self.container_box.setCurrentIndex(at)

    def _wants_mkv(self) -> bool:
        family = encoders.FAMILIES_BY_KEY[self.family_box.currentData()]
        return family.suffix == ".mkv" or (family.suffix == ".mp4" and self._s.container == "mkv")

    def _show_track_rows(self) -> None:
        self._sync_rates()                                        # the sound and the container decide how much of the file is picture
        many_audio = len(self._audio_tracks) > 1
        for row, visible in ((self.audio_track_row, many_audio), (self.subtitle_row, bool(self._subtitle_tracks))):
            for i in range(row.count()):
                widget = row.itemAt(i).widget()
                if widget is not None:
                    widget.setVisible(visible)
        family = encoders.FAMILIES_BY_KEY[self.family_box.currentData()]
        for i in range(self.container_row.count()):
            widget = self.container_row.itemAt(i).widget()
            if widget is not None:
                widget.setVisible(family.suffix == ".mp4")
        self.subtitle_box.setEnabled(self._wants_mkv())
        self.subtitle_box.setToolTip("" if self._wants_mkv() else
                                     "Subtitles are kept in MKV files: choose one and 'Save as' switches to MKV.")
        self.audio_note.setVisible(False)
        track = next((t for t in self._audio_tracks if t.index == self._s.audio_track), None)
        if track is not None and self._s.audio == "copy" and not self._wants_mkv() and \
                track.codec not in video.MP4_AUDIO and not track.codec.startswith("pcm"):
            self.audio_note.setText(f"{track.format_name} cannot go in an MP4 as it is, so it becomes AAC. "
                                    "Choose 'Save as: Matroska (MKV)' to keep it untouched.")
            self.audio_note.setVisible(True)

    def _audio_track(self) -> None:
        self._s.audio_track = int(self.audio_track_box.currentData() if self.audio_track_box.currentData() is not None else -1)
        self._show_track_rows()
        self.changed.emit()
        self.audio_track_chosen.emit(self._ordinal_of(self._s.audio_track))

    def _ordinal_of(self, index: int) -> int:
        """Position among the audio tracks (what the preview player counts by); the best one when none is chosen."""
        if index < 0:
            index = video.best_audio_index(self._audio_tracks)
        return next((i for i, t in enumerate(self._audio_tracks) if t.index == index), -1)

    def _subtitle_track(self) -> None:
        self._s.subtitle_track = int(self.subtitle_box.currentData() if self.subtitle_box.currentData() is not None else -1)
        if self._s.subtitle_track >= 0 and self._s.container != "mkv" and \
                encoders.FAMILIES_BY_KEY[self.family_box.currentData()].suffix == ".mp4":
            self._switch_container("mkv", True)                   # a subtitle can only be kept in an MKV: switch for them
        elif self._s.subtitle_track < 0 and self._auto_mkv:
            self._switch_container("auto", False)                 # ... and back to MP4 when they no longer want one
        self._show_track_rows()
        self.changed.emit()

    def _switch_container(self, container: str, automatic: bool) -> None:
        """Change "Save as" ourselves (``automatic``: because of a subtitle choice, so it can be undone the same way)."""
        self._switching = True
        self.set_container(container)
        self._switching = False
        self._auto_mkv = automatic

    def _container(self) -> None:
        if not self._switching:
            self._auto_mkv = False                                # the person chose it: it stays until they change it
        self._s.container = self.container_box.currentData() or "auto"
        if self._s.container != "mkv" and self._s.subtitle_track >= 0 and \
                encoders.FAMILIES_BY_KEY[self.family_box.currentData()].suffix == ".mp4":
            self.subtitle_box.setCurrentIndex(0)                  # no MKV, no subtitles
        self._show_track_rows()
        self.changed.emit()

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
        self._sync_rates()
        self.changed.emit()

    # -- quality, bitrate and file size: three views of one choice ---------------------------------------------

    def set_rate_context(self, provider) -> None:
        """``provider()`` returns ((width, height), fps, seconds, audio bits/s) for what would be converted now, or None."""
        self._rate_context = provider
        self._sync_rates()

    def refresh_rates(self) -> None:
        """What is being converted changed (file, 3D layout, size, range, sound): recompute the other two values."""
        self._sync_rates()

    def _quality_moved(self, value: float) -> None:
        self._s.quality = int(value)
        self._s.rate_mode = "quality"
        self._sync_rates()
        self.changed.emit()

    def _bitrate_typed(self, kbps: int) -> None:
        if self._syncing:
            return
        self._s.rate_mode, self._s.video_kbps = "bitrate", int(kbps)
        self._sync_rates()
        self.changed.emit()

    def _size_typed(self, megabytes: int) -> None:
        if self._syncing:
            return
        self._s.rate_mode, self._s.target_mb = "size", float(megabytes)
        self._sync_rates()
        self.changed.emit()

    def _sync_rates(self) -> None:
        if not hasattr(self, "rate_block"):
            return
        fam = encoders.FAMILIES_BY_KEY[self.family_box.currentData()]
        self.rate_block.setVisible(fam.uses_quality)
        context = self._rate_context() if self._rate_context else None
        for box in (self.bitrate_box, self.size_box):
            box.setEnabled(context is not None)
        if context is None:
            self.rate_note.setText("Open a video to see the bitrate and file size this will make.")
            return
        size, fps, seconds, audio = context
        mode = self._s.rate_mode
        if mode == "bitrate":
            bps = max(ratecontrol.MIN_VIDEO_BPS, self._s.video_kbps * 1000)
        elif mode == "size" and self._s.target_mb > 0:
            bps = ratecontrol.video_bps_for_size(self._s.target_mb * 1e6, audio, seconds)
        else:
            mode = "quality"
            bps = ratecontrol.bitrate_for_quality(size, fps, self._s.quality, fam.key)
        quality = self._s.quality if mode == "quality" else ratecontrol.quality_for_bitrate(bps, size, fps, fam.key)
        megabytes = ratecontrol.size_bytes(bps, audio, seconds) / 1e6
        self._syncing = True
        try:
            if mode != "quality":
                self.quality.set_value(quality)                # the slider shows what the typed number comes to
            self.bitrate_box.setValue(max(self.bitrate_box.minimum(), round(bps / 1000)))
            self.size_box.setValue(max(1, round(self._s.target_mb if mode == "size" else megabytes)))
        finally:
            self._syncing = False
        mbps, how = bps / 1e6, ratecontrol.describe_size(megabytes)
        if mode == "quality":
            text = (f"Constant quality: about {mbps:.1f} Mb/s and {how} for this part (an estimate - it depends on the picture). "
                    "Type a bitrate or a file size to aim at it exactly.")
        elif mode == "bitrate":
            text = f"Aiming at {mbps:.1f} Mb/s: the file will be about {how}. Quality {quality} is the nearest slider position."
        else:
            text = (f"Aiming at {ratecontrol.describe_size(self._s.target_mb)}: {mbps:.1f} Mb/s of video"
                    + (f" and {audio / 1000:.0f} kb/s of sound" if audio else "") + f". Quality {quality} is the nearest slider position.")
        if mode != "quality" and fam.key == "vp9":
            text += " VP9 aims loosely: the real size can run well over. H.265 or H.264 land much closer."
        elif mode != "quality" and self._s.encoder not in ("auto",) and encoders.is_hardware(self._s.encoder):
            text += " Hardware encoders can land up to about 10% off."
        if mode != "quality" and quality == 0:
            text += " That is below the slider's lowest quality: expect a rough picture."
        elif mode != "quality" and quality == 100:
            text += " That is above the slider's top quality: more than the picture needs."
        self.rate_note.setText(text)

    def _family(self) -> None:
        key = self.family_box.currentData()
        fam = encoders.FAMILIES_BY_KEY[key]
        self._s.family = key
        self.family_note.setText(fam.note)
        self.quality.setVisible(fam.uses_quality)
        self._sync_rates()
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
        if hasattr(self, "container_row"):
            self._show_track_rows()
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
