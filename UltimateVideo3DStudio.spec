# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Windows (portable folder) and macOS (.app).

    Windows:  scripts\\build_release.ps1     -> release\\UltimateVideo3DStudio\\
    macOS:    scripts/build_mac.sh           -> release/Ultimate Video to 3D Studio.app

PyAV is deliberately excluded: the ~35 MB FFmpeg component is fetched on first
use (see ultimate_video_3d/bootstrap.py) rather than shipped in every copy.
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH)
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform == "win32"

ASSETS = ROOT / "ultimate_video_3d" / "assets"
datas = [
    (str(ASSETS / "fonts"), "ultimate_video_3d/assets/fonts"),
    (str(ASSETS / "icon.png"), "ultimate_video_3d/assets"),
]
# Only the half-precision depth model ships (50 MB); the fp32 file, if present in a
# dev checkout, would double the download for a 25% slower model.
datas += [(str(f), "ultimate_video_3d/assets/onnx") for f in (ASSETS / "onnx").glob("*.fp16.onnx")]
datas += collect_data_files("onnxruntime")

hiddenimports = collect_submodules("numba") + ["psutil", "platformdirs"]

# Nothing here needs these; leaving them out keeps the bundle small.
excludes = [
    "av",                         # downloaded on first use
    "tkinter", "matplotlib", "scipy", "pandas", "IPython", "notebook",
    "torch", "tensorflow",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick3D",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtBluetooth", "PySide6.QtSensors",
]

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="UltimateVideo3DStudio",
    console=False,                # windowed; crash.log is the record (see crashlog.py)
    icon=str(ROOT / "icon" / ("app.icns" if IS_MAC else "app.ico")),
    upx=False,
)

coll = COLLECT(exe, a.binaries, a.datas, name="UltimateVideo3DStudio", upx=False)

if IS_MAC:
    app = BUNDLE(
        coll,
        name="Ultimate Video to 3D Studio.app",
        icon=str(ROOT / "icon" / "app.icns"),
        bundle_identifier="app.ultimatevideo3d.studio",
        info_plist={
            "CFBundleDisplayName": "Ultimate Video to 3D Studio",
            "CFBundleShortVersionString": "0.1.0",
            "NSHighResolutionCapable": True,
            "NSHumanReadableCopyright": "Free to use",
        },
    )
