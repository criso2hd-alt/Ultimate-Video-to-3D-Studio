"""The dark "instrument cockpit" theme, as interchangeable palettes.

Carried over unchanged from the DLSS app so the two feel like one family. Each
palette is a full colour world with a resting accent (``signal``) and a warm
action accent (``heat``, the Convert hover), so colour reads as activity rather
than decoration. The stylesheet is one template; a palette fills its tokens.
Glows come from a drop-shadow in code (QSS has no box-shadow) and the dark
canvas from a Fusion dark QPalette.
"""

from __future__ import annotations

from string import Template as _Template

from PySide6.QtGui import QColor, QPalette

#: Curated, not open-ended: five tastes rather than a colour picker, so every
#: choice is one someone designed. Order is the order shown in the menu.
PALETTES: dict[str, dict[str, str]] = {
    "Neural Cyan": dict(
        ground="#0A0E15", panel_hi="#141d2c", panel_lo="#101825", base="#0b1220",
        line="#23304a", line_soft="#1a2436", ink="#E8EEF9", ink_dim="#93A2BC",
        ink_faint="#5C6B85", signal="#37E1FF", signal_deep="#0d8fb8",
        signal_light="#5eeaff", heat="#FF7A3C", heat_deep="#d9531f",
        heat_light="#ffd0a8", on_accent="#04121a",
    ),
    "Ember": dict(
        ground="#100C09", panel_hi="#20180F", panel_lo="#17110B", base="#140F0A",
        line="#3B2C1E", line_soft="#241A12", ink="#F6ECE2", ink_dim="#B79E88",
        ink_faint="#7C6857", signal="#FF9838", signal_deep="#C25E13",
        signal_light="#FFB566", heat="#FF4E67", heat_deep="#C21F38",
        heat_light="#FF9DAB", on_accent="#1A0D02",
    ),
    "Violet Flux": dict(
        ground="#0C0A17", panel_hi="#191529", panel_lo="#130F22", base="#100C1E",
        line="#2E2650", line_soft="#1C1636", ink="#ECE8F9", ink_dim="#A79BCC",
        ink_faint="#675C85", signal="#A96BFF", signal_deep="#6A2FD8",
        signal_light="#C295FF", heat="#37E1FF", heat_deep="#0d8fb8",
        heat_light="#8EF0FF", on_accent="#0D0420",
    ),
    "Emerald": dict(
        ground="#08120E", panel_hi="#10201A", panel_lo="#0C1913", base="#0A1611",
        line="#1E4436", line_soft="#142B22", ink="#E4F5EC", ink_dim="#8FBBA6",
        ink_faint="#567A68", signal="#35E0A1", signal_deep="#12A56A",
        signal_light="#6FF0C1", heat="#FFC24B", heat_deep="#D99320",
        heat_light="#FFE0A0", on_accent="#04140C",
    ),
    "Slate Mono": dict(
        ground="#0D1017", panel_hi="#171C26", panel_lo="#12161F", base="#0F131B",
        line="#2A3242", line_soft="#1D2330", ink="#E6EAF1", ink_dim="#97A0B2",
        ink_faint="#5E6675", signal="#8FB4E0", signal_deep="#567AA8",
        signal_light="#B3D0F0", heat="#E0A96B", heat_deep="#A8763C",
        heat_light="#F0D0A8", on_accent="#0A1420",
    ),
}
DEFAULT_THEME = "Neural Cyan"
STYLE_TEMPLATE = _Template("""
* { font-family: "IBM Plex Sans", "Segoe UI", system-ui, sans-serif; }
QMainWindow, QDialog { background: $ground; }
QWidget { background: transparent; color: $ink; font-size: 13px; }

QGroupBox {
    border: 1px solid $line_soft; border-radius: 14px; margin-top: 16px;
    padding: 15px 14px 13px 14px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $panel_hi, stop:1 $panel_lo);
}
QGroupBox::title {
    subcontrol-origin: margin; left: 14px; top: 2px; padding: 0 6px;
    color: $ink_dim; font-weight: 700;
}

QLabel { background: transparent; }
QLabel#hint { color: $ink_faint; }
QLabel#linkSep { color: $line; }
QLabel#footerVersion { color: $ink_faint; padding: 0 8px 0 4px; }
QLabel#onboardingEyebrow { color: $signal; }
QLabel#onboardingTitle { color: $ink; font-weight: 700; }
QListWidget#onboardingSources {
    background: $base; border: 1px solid $line_soft; border-radius: 9px;
    padding: 6px; color: $ink_dim;
}
QListWidget#onboardingSources::item { padding: 5px; }
QListWidget#onboardingSources::item:selected { background: $panel_hi; color: $ink; }

QFrame#dropZone {
    border: 2px dashed $line; border-radius: 16px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $panel_lo, stop:1 $ground);
}
QFrame#dropZone[hovering="true"] { border-color: $signal; background: $panel_hi; }

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal, stop:1 $signal_deep);
    color: $on_accent; border: none; border-radius: 9px; padding: 9px 16px; font-weight: 700;
}
QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal_light, stop:1 $signal_deep);
}
QPushButton:disabled { background: $panel_lo; color: $ink_faint; }
QPushButton#secondary {
    background: $panel_lo; color: $ink; border: 1px solid $line; font-weight: 600;
}
QPushButton#secondary:hover { border-color: $signal_deep; color: $ink; background: $panel_hi; }
QPushButton#secondary:checked {
    background: $base; color: $signal; border: 1px solid $signal_deep;
}
QPushButton#secondary:disabled { color: $ink_faint; border-color: $line_soft; background: $panel_lo; }
QPushButton#convert {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal_light, stop:0.5 $signal, stop:1 $signal_deep);
    color: $on_accent; border-radius: 12px; padding: 15px; font-weight: 800; font-size: 15px;
}
QPushButton#convert:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $heat_light, stop:0.5 $heat, stop:1 $heat_deep);
}
QPushButton#convert:disabled { background: $panel_lo; color: $ink_faint; }
QPushButton#link {
    background: transparent; color: $ink_faint; border: none; padding: 4px 2px; font-weight: 500;
}
QPushButton#link:hover { color: $signal; }
QPushButton#chip {
    background: $base; color: $ink_dim; border: 1px solid $line; border-radius: 9px;
    padding: 7px 13px; font-weight: 500;
}
QPushButton#chip:hover { color: $ink; border-color: $signal_deep; }
QPushButton#chip:checked {
    color: $on_accent; font-weight: 600; border-color: transparent;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal, stop:1 $signal_deep);
}

QSlider:horizontal { min-height: 22px; }
QSlider::groove:horizontal { height: 6px; border-radius: 3px; background: $base; border: 1px solid $line_soft; margin: 0 9px; }
QSlider::sub-page:horizontal {
    height: 6px; border-radius: 3px; margin: 0 9px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $signal_deep, stop:1 $signal);
}
QSlider::add-page:horizontal { height: 6px; border-radius: 3px; background: $base; margin: 0 9px; }
QSlider::handle:horizontal {
    background: $signal_light; border: 2px solid $signal; width: 15px; height: 15px;
    margin: -6px -9px; border-radius: 9px;
}
QSlider::handle:horizontal:hover { border-color: $signal_light; }

QComboBox { background: $base; color: $ink; border: 1px solid $line; border-radius: 9px; padding: 7px 12px; min-height: 20px; }
QComboBox:hover { border-color: $signal_deep; }
QComboBox:disabled { color: $ink_faint; border-color: $line_soft; }
QComboBox::drop-down { border: none; width: 22px; }
/* A visible chevron, drawn from borders so it needs no bundled image. Without
   it the dropdown read as a plain box and people typed values in by hand. */
QComboBox::down-arrow {
    width: 0; height: 0; margin-right: 8px;
    border-left: 5px solid transparent; border-right: 5px solid transparent;
    border-top: 6px solid $ink_dim;
}
QComboBox::down-arrow:hover { border-top-color: $signal; }
QComboBox::down-arrow:disabled { border-top-color: $line; }
QComboBox QAbstractItemView {
    background: $panel_lo; color: $ink; border: 1px solid $line;
    selection-background-color: $line; outline: none; padding: 4px;
}
QSpinBox { background: $base; color: $ink; border: 1px solid $line; border-radius: 9px; padding: 6px 8px; min-height: 20px; }
QSpinBox:hover { border-color: $signal_deep; }
/* Qt's default spin arrows are near-invisible on the dark theme. Draw our own
   from borders (like the combo chevron) so up/down are legible. */
QSpinBox::up-button, QSpinBox::down-button {
    subcontrol-origin: border; width: 18px; background: $panel_hi; border-left: 1px solid $line;
}
QSpinBox::up-button { subcontrol-position: top right; border-top-right-radius: 8px; }
QSpinBox::down-button { subcontrol-position: bottom right; border-bottom-right-radius: 8px; }
QSpinBox::up-button:hover, QSpinBox::down-button:hover { background: $panel_hi; border-left-color: $signal_deep; }
QSpinBox::up-arrow {
    width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 5px solid $ink_dim;
}
QSpinBox::down-arrow {
    width: 0; height: 0; border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid $ink_dim;
}
QSpinBox::up-arrow:hover { border-bottom-color: $signal; }
QSpinBox::down-arrow:hover { border-top-color: $signal; }
QSpinBox::up-arrow:disabled { border-bottom-color: $line; }
QSpinBox::down-arrow:disabled { border-top-color: $line; }

QCheckBox { color: $ink; spacing: 8px; background: transparent; }
QCheckBox::indicator { width: 17px; height: 17px; border-radius: 5px; border: 1px solid $line; background: $base; }
QCheckBox::indicator:hover { border-color: $signal; }
QCheckBox::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal, stop:1 $signal_deep); border-color: $signal;
}

QProgressBar {
    background: $base; border: 1px solid $line_soft; border-radius: 6px; height: 12px;
    text-align: center; color: $ink_dim; font-size: 11px;
}
QProgressBar::chunk {
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $signal_deep, stop:1 $signal);
}

QTabWidget::pane { border: none; }
/* Full-width tab band: its own strip under the command bar, with a baseline
   that runs the whole width of the window. */
QTabWidget#mainTabs { background: #0a0f18; }
QTabWidget#mainTabs::pane { border: none; border-top: 1px solid $line_soft; top: -1px; }
QTabBar { background: #0a0f18; qproperty-drawBase: 0; padding: 3px 10px 0 10px; }
QTabBar::tab {
    background: transparent; color: $ink_dim; padding: 11px 16px; margin-right: 2px;
    border: none; border-bottom: 2px solid transparent; font-weight: 500;
}
QTabBar::tab:hover { color: $ink; }
QTabBar::tab:selected { color: $signal; border-bottom: 2px solid $signal; }
QPushButton#tabAction {
    background: transparent; color: $ink_faint; border: none; padding: 8px 18px; font-weight: 500;
}
QPushButton#tabAction:hover { color: $signal; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: $line; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: $signal_deep; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 2px; }
QScrollBar::handle:horizontal { background: $line; border-radius: 5px; min-width: 30px; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; width: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QStatusBar { background: $ground; color: $ink_faint; border-top: 1px solid $line_soft; }
QToolTip { background: $panel_lo; color: $ink; border: 1px solid $line; padding: 6px 8px; }

/* command bar */
QFrame#commandBar {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #0d1420, stop:1 $ground);
    border: none; border-bottom: 1px solid $line_soft;
}
QLabel#brandGlyph {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 $signal, stop:1 $heat);
    color: $on_accent; border-radius: 8px; font-weight: 800; font-size: 15px;
    min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px;
}
QLabel#wordmark { color: $ink; font-weight: 800; font-size: 15px; letter-spacing: 2px; }
QLabel#wordmark #accent { color: $signal; }
QLabel#wordmarkSub { color: $ink_faint; font-size: 9px; letter-spacing: 4px; }
QFrame#gauge { background: $base; border: 1px solid $line; border-radius: 11px; }
QLabel#gaugeLab { color: $ink_faint; font-size: 9px; letter-spacing: 2px; }
QLabel#gpuName { color: $ink; font-weight: 700; font-size: 12px; }
QLabel#vramRead { color: $ink_dim; font-size: 10px; }
QProgressBar#vramBar { background: $base; border: 1px solid $line_soft; border-radius: 3px; }
QProgressBar#vramBar::chunk {
    border-radius: 2px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $signal_deep, stop:1 $signal);
}
QLabel#runtimePill {
    font-size: 10px; letter-spacing: 1px; padding: 7px 13px; border-radius: 9px;
    border: 1px solid #204a3a; background: #0c1a15; color: #54E39B;
}
QLabel#runtimePill[state="setup"] {
    border: 1px solid #4a3a20; background: #1a140c; color: $heat;
}

/* footer */
QFrame#footer { background: #0a0f18; border: none; border-top: 1px solid $line_soft; }
QLabel#footFile { color: $ink_dim; font-size: 11px; }

/* module cards (title inside a divider header, then a padded body) */
QFrame#modCard {
    border: 1px solid $line_soft; border-radius: 14px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $panel_hi, stop:1 $panel_lo);
}
QFrame#modHead { background: transparent; border: none; border-bottom: 1px solid $line_soft; }
QLabel#modTitle { color: $ink_dim; background: transparent; }
QCheckBox#modTitle { color: $ink_dim; background: transparent; spacing: 9px; }
QLabel#modTag { color: $ink_faint; background: transparent; }

/* preview stage (Video / Sequence): the bordered panel content appears in */
QFrame#stage {
    border: 1px solid $line_soft; border-radius: 14px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $panel_lo, stop:1 $ground);
}
QLabel#phTitle { color: $ink_dim; background: transparent; }

/* floating view bar (hover-revealed over the image, mockup's stagefoot) */
QFrame#viewBar {
    background: rgba(10, 16, 24, 0.82); border: 1px solid $line; border-radius: 12px;
}
QFrame#viewBarSep { background: $line; border: none; margin: 5px 0; }
QFrame#viewBar QPushButton#viewChip {
    background: transparent; color: $ink_dim; border: none; border-radius: 8px;
    padding: 6px 12px; font-weight: 500; font-size: 12px;
}
QFrame#viewBar QPushButton#viewChip:hover { color: $ink; background: $panel_hi; }
QFrame#viewBar QPushButton#viewChip:checked {
    color: $on_accent; font-weight: 600;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal, stop:1 $signal_deep);
}
QFrame#viewBar QPushButton#viewChip:disabled { color: $ink_faint; background: transparent; }
/* the permanent view bar under the picture: the same chips, stretched across the stage */
QFrame#modeBar {
    background: $panel_lo; border: 1px solid $line_soft; border-radius: 12px;
}
QFrame#modeBar QPushButton#viewChip {
    background: transparent; color: $ink_dim; border: none; border-radius: 8px;
    padding: 7px 6px; font-weight: 500; font-size: 12px;
}
QFrame#modeBar QPushButton#viewChip:hover { color: $ink; background: $panel_hi; }
QFrame#modeBar QPushButton#viewChip:checked {
    color: $on_accent; font-weight: 600;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal, stop:1 $signal_deep);
}
QFrame#viewBar QPushButton#fullResChip {
    background: $signal_deep; color: $on_accent; border: 1px solid $signal;
    border-radius: 8px; padding: 6px 14px; font-weight: 700; font-size: 12px;
}
QFrame#viewBar QPushButton#fullResChip:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $signal, stop:1 $signal_deep);
    color: $on_accent;
}

/* support banner at the very bottom: warm brown, with the yellow button and an X (closes it for the session) */
QFrame#coffeeBanner {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a140c, stop:0.5 #211810, stop:1 #1a140c);
    border: none; border-top: 1px solid #3b2c1e;
}
QLabel#coffeeText { color: #d9b98a; font-size: 12px; background: transparent; }
QPushButton#coffeeClose {
    background: transparent; color: #b89a70; border: none; border-radius: 13px;
    padding: 0; font-size: 12px; font-weight: 700;
}
QPushButton#coffeeClose:hover { background: rgba(255, 220, 160, 0.14); color: #f3dcb4; }
QPushButton#coffeeButton {
    background: #FFDD00; color: #1a1408; border: none; border-radius: 8px;
    padding: 6px 16px; font-weight: 800; font-size: 12px;
}
QPushButton#coffeeButton:hover { background: #ffe94d; }

/* collapsible card: a slim sliver that opens into a full panel */
QFrame#sliver {
    border: 1px solid $line_soft; border-radius: 12px;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 $panel_hi, stop:1 $panel_lo);
}
QFrame#sliver[open="true"] { border-color: $line; }
QPushButton#chevron {
    background: transparent; border: none; color: $ink_dim; padding: 2px 6px; font-size: 13px;
}
QPushButton#chevron:hover { color: $signal; }
QLabel#sliverSub { color: $ink_faint; font-size: 11px; }

/* colour panel section tabs */
QPushButton#sectionTab {
    background: transparent; color: $ink_faint; border: none; border-bottom: 2px solid transparent;
    border-radius: 0; padding: 6px 3px; font-weight: 600; font-size: 11px;
}
QPushButton#sectionTab:hover { color: $ink; }
QPushButton#sectionTab:checked { color: $signal; border-bottom: 2px solid $signal; }

/* system gauge cells */
QLabel#gaugeVal { color: $ink; font-size: 11px; font-weight: 600; }
QLabel#gaugeDim { color: $ink_faint; font-size: 9px; letter-spacing: 1px; }
QProgressBar#miniBar { background: $base; border: 1px solid $line_soft; border-radius: 2px; }
QProgressBar#miniBar::chunk { border-radius: 1px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 $signal_deep, stop:1 $signal); }
QProgressBar#miniBar[hot="true"]::chunk { background: $heat; }
QLabel#encoderPill {
    font-size: 10px; letter-spacing: 1px; padding: 7px 13px; border-radius: 9px;
    border: 1px solid #204a3a; background: #0c1a15; color: #54E39B;
}
QPushButton#updatePill {
    font-size: 10px; letter-spacing: 1px; padding: 7px 13px; border-radius: 9px; font-weight: 700;
    border: 1px solid $heat; background: #1a140c; color: $heat;
}
QPushButton#updatePill:hover { background: $heat; color: $on_accent; }
QLabel#encoderPill[state="software"] { border: 1px solid #4a3a20; background: #1a140c; color: $heat; }

QFrame#scopeBox { background: #06090f; border: 1px solid $line_soft; border-radius: 8px; }

/* Placed last on purpose: the earlier "QWidget { background: transparent }" rule
   otherwise wins over the window background for dialogs and leaves them unpainted. */
QMainWindow, QDialog { background: $ground; }
QFrame#tabCorner { background: #0a0f18; border: none; }
""")


def style_for(name: str) -> str:
    """The stylesheet for a palette name, falling back to the default."""
    return STYLE_TEMPLATE.substitute(PALETTES.get(name, PALETTES[DEFAULT_THEME]))


def qpalette_for(name: str) -> QPalette:
    """The Fusion QPalette for a palette, for the canvas and native controls."""
    p = PALETTES.get(name, PALETTES[DEFAULT_THEME])
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(p["ground"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(p["ink"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(p["base"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(p["panel_lo"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(p["ink"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(p["panel_lo"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(p["ink"]))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(p["panel_lo"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(p["ink"]))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(p["ink_faint"]))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(p["signal"]))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor(p["on_accent"]))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText, QPalette.ColorRole.WindowText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(p["ink_faint"]))
    return pal


#: The active stylesheet, reassigned when the theme changes. Dialogs read this
#: at creation, so a switch reaches every window opened afterwards.
STYLE = style_for(DEFAULT_THEME)
