"""Regenerates ultimate_video_3d/support.py (the masked banner link + message + digest). After changing URL/TEXT
here, copy the printed pin into _SUPPORT_PIN in app.py. Only the project owner should ever do this."""
import base64
import hashlib
from pathlib import Path

URL = "https://buymeacoffee.com/criso2hdj"
TEXT = ("Ultimate Video to 3D Studio is free, and will always stay free. If it saves you time "
        "or money, a coffee will keep it going and help me improve the app in the future.")
KEY = hashlib.sha256(b"uv3d/support/mask").digest()[:16]


def mask(text: str) -> str:
    raw = text.encode("utf-8")
    return base64.b64encode(bytes(b ^ KEY[i % len(KEY)] for i, b in enumerate(raw))).decode()


digest = hashlib.sha256((URL + "\n" + TEXT).encode("utf-8")).hexdigest()
print("pin", digest[:16])

source = f'''"""The support banner: what it says, where it points, and the checks that keep it that way.

This project is free to use and asks for one thing in return: that the "Buy me a coffee" banner
stays and keeps pointing at its author (see LICENSE). The banner can be closed for the current
session with its X, and comes back the next time the app starts - it is never remembered as closed.

Nobody can be *stopped* from editing an application whose source they hold; what this module does
is make an edit visible and pointless rather than silent:

* the official link and message are not plain string literals (searching the source for the URL, or
  changing one, finds nothing to change) but a masked blob, pinned by a SHA-256 digest that is also
  held in ``app.py``;
* whatever is actually on screen is checked against the blob - the text, the button, that the
  banner is visible, at the bottom, unstyled away - at start-up, every few seconds, and before a
  conversion; a banner that does not match is put back;
* the button opens the *decoded official* link, whatever the widget holds;
* a copy that keeps failing the check switches its Convert buttons off and says where the
  official version is.

That is a deterrent plus the licence, not DRM: a person determined to patch the program can. The
licence forbids redistributing it (modified or not), and official builds come only from the
project's GitHub releases.
"""

from __future__ import annotations

import base64
import hashlib
import hmac

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton, QWidget

from .widgets import ElidedLabel

_KEY = hashlib.sha256(b"uv3d/support/mask").digest()[:16]
_URL = "{mask(URL)}"
_TEXT = "{mask(TEXT)}"
#: SHA-256 of ``url + "\\n" + text``. ``app.py`` holds the first 16 characters as a second, separate pin.
DIGEST = "{digest}"
BUTTON_TEXT = "\\u2615  Buy me a coffee"


def _unmask(blob: str) -> str:
    raw = base64.b64decode(blob)
    return bytes(b ^ _KEY[i % len(_KEY)] for i, b in enumerate(raw)).decode("utf-8")


def official_url() -> str:
    return _unmask(_URL)


def official_text() -> str:
    return _unmask(_TEXT)


def intact() -> bool:
    """The masked link and message are the ones the digest was made from (i.e. nobody edited the blob)."""
    try:
        made = hashlib.sha256((official_url() + "\\n" + official_text()).encode("utf-8")).hexdigest()
    except Exception:  # noqa: BLE001 - a damaged blob is a failed check, not a crash
        return False
    return hmac.compare_digest(made, DIGEST)


def require_intact() -> None:
    """Called at the start of a conversion."""
    if not intact():
        raise RuntimeError("This copy of Ultimate Video to 3D Studio has been modified. "
                           "Please use the official version from its GitHub page.")


def open_official_page() -> None:
    QDesktopServices.openUrl(QUrl(official_url()))


class SupportBanner(QFrame):
    """Warm-brown strip at the very bottom of the window: message, the yellow button, and an X."""

    HEIGHT = 42

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        #: Only ever set by the X, and only for this run - never saved anywhere.
        self.closed_for_session = False
        self.setObjectName("coffeeBanner")
        self.setFixedHeight(self.HEIGHT)
        row = QHBoxLayout(self)
        row.setContentsMargins(20, 6, 12, 6)
        row.setSpacing(12)
        # No stretch spacers: the message takes all the room left of the button (centred),
        # so it is only ever shortened on a genuinely narrow window.
        self._text = ElidedLabel(official_text())
        self._text.setObjectName("coffeeText")
        self._text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        row.addWidget(self._text, 1)
        self._button = QPushButton(BUTTON_TEXT)
        self._button.setObjectName("coffeeButton")
        self._button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._button.clicked.connect(open_official_page)          # never the widget's own idea of the link
        row.addWidget(self._button)
        self._close = QPushButton("\\u2715")
        self._close.setObjectName("coffeeClose")
        self._close.setCursor(Qt.CursorShape.PointingHandCursor)
        self._close.setFixedSize(26, 26)
        self._close.setToolTip("Hide this for now. It comes back the next time you open the app.")
        self._close.clicked.connect(self._dismiss)
        row.addWidget(self._close)

    def _dismiss(self) -> None:
        self.closed_for_session = True
        self.hide()

    # -- integrity ---------------------------------------------------------------

    def verify(self) -> bool:
        """Is this banner what it should be? (If it was closed with the X, that counts as fine.)"""
        if not intact():
            return False
        if self._text._full != official_text() or self._button.text() != BUTTON_TEXT:
            return False
        if self.styleSheet() or self._text.styleSheet() or self._button.styleSheet():
            return False                                            # a style that could hide or recolour it away
        if self.graphicsEffect() is not None or self._button.graphicsEffect() is not None:
            return False
        if self.closed_for_session:
            return True                                             # closed by the user: nothing more to check
        window = self.window()
        if not window.isVisible():
            return True                                             # minimised or not shown yet
        if not (self.isVisible() and self._button.isVisible() and self._button.isEnabled()):
            return False
        if self.height() < 30 or self._button.height() < 20 or self.width() < window.width() * 0.8:
            return False
        parent = self.parentWidget()
        layout = parent.layout() if parent is not None else None
        if layout is None or layout.itemAt(layout.count() - 1).widget() is not self:
            return False                                            # no longer the last thing in the window
        return True
'''
Path(__file__).resolve().parent.parent.joinpath("ultimate_video_3d", "support.py").write_text(source, encoding="utf-8")
print("written")
