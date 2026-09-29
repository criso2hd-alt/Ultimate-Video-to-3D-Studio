"""Tell the user when a newer version exists. That is all it does.

At launch the app asks GitHub for the newest published release and compares its
version with its own. If GitHub's is higher, the window shows a notice with a link
to the download page. Nothing is downloaded, nothing is replaced, nothing restarts:
the user decides, and installs the new version themselves.

Deliberately small. It sends one anonymous GET request to api.github.com (no
identifiers, no telemetry), gives up after a few seconds, and treats *every* failure
- offline, rate-limited, no releases yet, a private repository - as "nothing to
report". An update check must never be the reason the app misbehaves.

The release list is only readable without a token when the repository is public. Until
then the check simply finds nothing.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from . import __version__

REPO = "criso2hd-alt/Ultimate-Video-to-3D-Studio"
#: The newest published (non-draft, non-prerelease) release.
LATEST_API = f"https://api.github.com/repos/{REPO}/releases/latest"
#: Where the user is sent to download it.
RELEASES_PAGE = f"https://github.com/{REPO}/releases/latest"

_VERSION = re.compile(r"^\s*v?(\d+(?:\.\d+)*)\s*(?:[-+._]?\s*(.*?))?\s*$", re.IGNORECASE)


def parse_version(text: str) -> tuple[tuple[int, ...], int]:
    """``"v0.10.1"`` -> ``((0, 10, 1), 1)``. The second item is 1 for a final release, 0 for
    a pre-release (``0.2.0-beta1``), so a pre-release sorts *below* its own final version.
    Raises ``ValueError`` for something that is not a version."""
    match = _VERSION.match(text or "")
    if not match:
        raise ValueError(f"not a version: {text!r}")
    numbers = tuple(int(part) for part in match.group(1).split("."))
    return numbers, 0 if match.group(2) else 1


def is_newer(candidate: str, current: str = __version__) -> bool:
    """Whether ``candidate`` is a higher version than ``current`` (trailing zeros do not matter)."""
    try:
        (a, a_final), (b, b_final) = parse_version(candidate), parse_version(current)
    except ValueError:
        return False
    width = max(len(a), len(b))
    a, b = a + (0,) * (width - len(a)), b + (0,) * (width - len(b))
    return (a, a_final) > (b, b_final)


@dataclass(frozen=True)
class UpdateInfo:
    version: str            # e.g. "0.2.0" (no leading "v")
    url: str                # the page to send the user to
    notes: str = ""         # the release notes, if any
    name: str = ""


def _fetch(url: str, timeout: float) -> dict | None:
    request = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": f"UltimateVideo3DStudio/{__version__}",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    with urllib.request.urlopen(request, timeout=timeout) as response:      # noqa: S310 - fixed https host
        data = json.load(response)
    return data if isinstance(data, dict) else None


def check_for_update(current: str = __version__, url: str = LATEST_API, timeout: float = 6.0) -> UpdateInfo | None:
    """The newer release if there is one, else ``None``. Never raises."""
    try:
        release = _fetch(url, timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return None                                # offline, 404 (private / no releases), rate limit ...
    except Exception:  # noqa: BLE001 - nothing here may take the app down
        return None
    if not release or release.get("draft") or release.get("prerelease"):
        return None
    tag = str(release.get("tag_name") or "").strip()
    if not tag or not is_newer(tag, current):
        return None
    return UpdateInfo(
        version=tag.lstrip("vV"),
        url=str(release.get("html_url") or RELEASES_PAGE),
        notes=str(release.get("body") or "").strip(),
        name=str(release.get("name") or "").strip(),
    )
