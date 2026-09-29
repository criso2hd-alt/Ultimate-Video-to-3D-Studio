"""Vendor-neutral hardware detection and live load readings.

Nothing here assumes NVIDIA. The whole point of the app is that it runs on any
GPU, so the readings come from the operating system rather than a vendor tool:

- **Windows** - DXGI lists the adapters (name, vendor, dedicated VRAM, LUID) and
  the PDH performance counters that Task Manager itself reads give live GPU
  engine load, encoder load and VRAM use for NVIDIA, AMD and Intel alike.
- **macOS** - ``system_profiler`` lists the GPU and ``ioreg`` reports the
  accelerator's utilisation; Apple silicon has unified memory, so "VRAM" is the
  system RAM the GPU shares.
- CPU and RAM come from psutil on every platform.

Every query degrades to ``None``. A missing counter must show a dash in the
gauge, never stop the app from starting.
"""

from __future__ import annotations

import ctypes
import re
import subprocess
import sys
from dataclasses import dataclass, field

VENDOR_NVIDIA = "nvidia"
VENDOR_AMD = "amd"
VENDOR_INTEL = "intel"
VENDOR_APPLE = "apple"
VENDOR_OTHER = "other"

_PCI_VENDORS = {
    0x10DE: VENDOR_NVIDIA,
    0x1002: VENDOR_AMD,
    0x1022: VENDOR_AMD,
    0x8086: VENDOR_INTEL,
    0x106B: VENDOR_APPLE,
}

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class GpuInfo:
    name: str
    vendor: str
    vram_bytes: int = 0          # dedicated VRAM; 0 when the GPU shares system memory
    luid: str = ""               # Windows adapter id, for matching perf counters
    luids: list[str] = field(default_factory=list)  # same GPU can enumerate twice
    unified_memory: bool = False

    @property
    def short_name(self) -> str:
        name = self.name
        for junk in ("NVIDIA GeForce ", "NVIDIA ", "AMD Radeon ", "AMD ", "Intel(R) ", "(R)", "(TM)"):
            name = name.replace(junk, "")
        return " ".join(name.split()) or self.name

    def signature(self) -> str:
        """Stable id for "is this the same GPU as last launch" (encoder-probe cache key)."""
        return f"{self.vendor}:{self.name}:{self.vram_bytes}"


@dataclass
class Load:
    """One live reading. Any field is None when the platform cannot supply it."""

    cpu_percent: float | None = None
    ram_used: int | None = None
    ram_total: int | None = None
    gpu_percent: float | None = None
    encoder_percent: float | None = None
    vram_used: int | None = None
    vram_total: int | None = None


# --- GPU discovery --------------------------------------------------------


def detect_gpus() -> list[GpuInfo]:
    """Every real GPU, best first (dedicated VRAM, then discrete vendors)."""
    try:
        if sys.platform == "win32":
            gpus = _detect_windows()
        elif sys.platform == "darwin":
            gpus = _detect_macos()
        else:
            gpus = _detect_linux()
    except Exception:  # noqa: BLE001 - detection is advisory
        gpus = []
    # A single card can enumerate twice (a second LUID from a virtual display
    # driver, for instance). Fold duplicates together but remember every LUID,
    # since the performance counters are keyed by whichever one carries the work.
    merged: dict[tuple, GpuInfo] = {}
    for gpu in gpus:
        key = (gpu.name, gpu.vendor, gpu.vram_bytes)
        if key in merged:
            merged[key].luids.append(gpu.luid)
        else:
            gpu.luids = [gpu.luid] if gpu.luid else []
            merged[key] = gpu
    gpus = list(merged.values())
    gpus.sort(key=lambda g: (g.vram_bytes, g.vendor != VENDOR_INTEL), reverse=True)
    return gpus


def primary_gpu(gpus: list[GpuInfo] | None = None) -> GpuInfo | None:
    gpus = detect_gpus() if gpus is None else gpus
    return gpus[0] if gpus else None


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8),
    ]


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128), ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint), ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint), ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
        ("LuidLow", ctypes.c_ulong), ("LuidHigh", ctypes.c_long), ("Flags", ctypes.c_uint),
    ]


def _com_call(obj: ctypes.c_void_p, index: int, restype, *argtypes):
    """Call slot ``index`` of a COM object's vtable."""
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
    prototype = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)
    return prototype(vtable[index])


def _detect_windows() -> list[GpuInfo]:
    dxgi = ctypes.WinDLL("dxgi")
    # IID_IDXGIFactory1 = 770aae78-f26f-4dba-a829-253c83d1b387
    iid = _GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0 or not factory:
        return []
    found: list[GpuInfo] = []
    try:
        enum = _com_call(factory, 12, ctypes.c_long, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p))
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum(factory, index, ctypes.byref(adapter)) != 0 or not adapter:
                break
            index += 1
            try:
                desc = _AdapterDesc1()
                get_desc = _com_call(adapter, 10, ctypes.c_long, ctypes.POINTER(_AdapterDesc1))
                if get_desc(adapter, ctypes.byref(desc)) != 0:
                    continue
                # Flags bit 1 is DXGI_ADAPTER_FLAG_SOFTWARE; 0x1414 is Microsoft's
                # "Basic Render Driver". Neither is a GPU worth listing.
                if desc.Flags & 2 or desc.VendorId == 0x1414:
                    continue
                vendor = _PCI_VENDORS.get(desc.VendorId, VENDOR_OTHER)
                found.append(GpuInfo(
                    name=desc.Description.strip(), vendor=vendor,
                    vram_bytes=int(desc.DedicatedVideoMemory),
                    luid=f"luid_0x{desc.LuidHigh & 0xFFFFFFFF:08X}_0x{desc.LuidLow & 0xFFFFFFFF:08X}",
                ))
            finally:
                _com_call(adapter, 2, ctypes.c_ulong)(adapter)
    finally:
        _com_call(factory, 2, ctypes.c_ulong)(factory)
    return found


def _detect_macos() -> list[GpuInfo]:
    import json

    out = subprocess.run(
        ["system_profiler", "SPDisplaysDataType", "-json"],
        capture_output=True, text=True, timeout=10, check=False,
    ).stdout
    gpus = []
    for entry in json.loads(out).get("SPDisplaysDataType", []):
        name = entry.get("sppci_model") or entry.get("_name") or "GPU"
        vendor_text = str(entry.get("spdisplays_vendor", "")).lower()
        apple = "apple" in name.lower() or "apple" in vendor_text
        vendor = VENDOR_APPLE if apple else (
            VENDOR_AMD if "amd" in vendor_text or "radeon" in name.lower()
            else VENDOR_INTEL if "intel" in vendor_text else VENDOR_OTHER
        )
        vram = 0
        match = re.search(r"(\d+)\s*(GB|MB)", str(entry.get("spdisplays_vram", "")), re.I)
        if match and not apple:
            vram = int(match.group(1)) * (1 << 30 if match.group(2).upper() == "GB" else 1 << 20)
        gpus.append(GpuInfo(name=name, vendor=vendor, vram_bytes=vram, unified_memory=apple))
    return gpus


def _detect_linux() -> list[GpuInfo]:
    try:
        out = subprocess.run(["lspci", "-mm"], capture_output=True, text=True, timeout=5, check=False).stdout
    except OSError:
        return []
    gpus = []
    for line in out.splitlines():
        if "VGA" in line or "3D controller" in line:
            low = line.lower()
            vendor = (VENDOR_NVIDIA if "nvidia" in low else VENDOR_AMD if "amd" in low or "ati" in low
                      else VENDOR_INTEL if "intel" in low else VENDOR_OTHER)
            gpus.append(GpuInfo(name=line.split('"')[5] if line.count('"') >= 6 else line, vendor=vendor))
    return gpus


# --- live readings --------------------------------------------------------


class _Pdh:
    """Windows performance counters, wildcarded, read as name->value dicts.

    These are the same counters Task Manager's GPU tab draws. They need two
    samples to produce a rate, which is why ``sample`` is called on a timer and
    the first reading after start-up is empty.
    """

    PDH_FMT_DOUBLE = 0x200
    PDH_FMT_NOCAP100 = 0x8000
    PDH_MORE_DATA = 0x800007D2

    class _Item(ctypes.Structure):
        _fields_ = [("name", ctypes.c_wchar_p), ("status", ctypes.c_ulong),
                    ("value", ctypes.c_double)]

    def __init__(self) -> None:
        self._pdh = ctypes.WinDLL("pdh")
        self._query = ctypes.c_void_p()
        if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
            raise OSError("PdhOpenQuery failed")
        self._counters: dict[str, ctypes.c_void_p] = {}
        for key, path in (
            ("engine", r"\GPU Engine(*)\Utilization Percentage"),
            ("dedicated", r"\GPU Adapter Memory(*)\Dedicated Usage"),
        ):
            handle = ctypes.c_void_p()
            if self._pdh.PdhAddEnglishCounterW(self._query, path, 0, ctypes.byref(handle)) == 0:
                self._counters[key] = handle
        self._pdh.PdhCollectQueryData(self._query)

    def sample(self) -> dict[str, dict[str, float]]:
        if self._pdh.PdhCollectQueryData(self._query) != 0:
            return {}
        return {key: self._read(handle) for key, handle in self._counters.items()}

    def _read(self, handle) -> dict[str, float]:
        size = ctypes.c_ulong(0)
        count = ctypes.c_ulong(0)
        fmt = self.PDH_FMT_DOUBLE | self.PDH_FMT_NOCAP100
        status = self._pdh.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(size), ctypes.byref(count), None)
        if (status & 0xFFFFFFFF) != self.PDH_MORE_DATA or not size.value:
            return {}
        buffer = ctypes.create_string_buffer(size.value)
        if self._pdh.PdhGetFormattedCounterArrayW(handle, fmt, ctypes.byref(size), ctypes.byref(count), buffer) != 0:
            return {}
        items = ctypes.cast(buffer, ctypes.POINTER(self._Item))
        return {items[i].name: items[i].value for i in range(count.value) if items[i].status in (0, 0x00000000)}

    def close(self) -> None:
        try:
            self._pdh.PdhCloseQuery(self._query)
        except Exception:  # noqa: BLE001
            pass


class LoadMonitor:
    """Polls CPU, RAM and the primary GPU. Cheap enough for a 1-2 s timer."""

    def __init__(self, gpu: GpuInfo | None) -> None:
        import psutil

        self._psutil = psutil
        self._gpu = gpu
        self._pdh: _Pdh | None = None
        psutil.cpu_percent(None)
        if sys.platform == "win32" and gpu is not None:
            try:
                self._pdh = _Pdh()
            except Exception:  # noqa: BLE001 - counters are optional
                self._pdh = None

    def read(self) -> Load:
        ps = self._psutil
        mem = ps.virtual_memory()
        load = Load(cpu_percent=ps.cpu_percent(None), ram_used=int(mem.used), ram_total=int(mem.total))
        gpu = self._gpu
        if gpu is None:
            return load
        load.vram_total = gpu.vram_bytes or (int(mem.total) if gpu.unified_memory else None)
        if self._pdh is not None and gpu.luids:
            self._fill_windows(load, gpu)
        elif sys.platform == "darwin":
            self._fill_macos(load, gpu, mem)
        return load

    def _fill_windows(self, load: Load, gpu: GpuInfo) -> None:
        data = self._pdh.sample() if self._pdh else {}
        engines = data.get("engine", {})
        if engines:
            # Sum every process's use of one engine type, then take the busiest
            # engine - Task Manager's own rule for the single headline percentage.
            by_type: dict[str, float] = {}
            for name, value in engines.items():
                if not any(luid.lower() in name.lower() for luid in gpu.luids):
                    continue
                kind = name.rsplit("engtype_", 1)[-1]
                by_type[kind] = by_type.get(kind, 0.0) + value
            if by_type:
                encoders = [v for k, v in by_type.items() if "encode" in k.lower()]
                others = [v for k, v in by_type.items() if "encode" not in k.lower() and "decode" not in k.lower()]
                load.gpu_percent = min(100.0, max(others, default=0.0))
                load.encoder_percent = min(100.0, max(encoders, default=0.0))
        used = sum(
            v for n, v in data.get("dedicated", {}).items()
            if any(luid.lower() in n.lower() for luid in gpu.luids)
        )
        if used:
            load.vram_used = int(used)

    def _fill_macos(self, load: Load, gpu: GpuInfo, mem) -> None:
        try:
            out = subprocess.run(
                ["ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"],
                capture_output=True, text=True, timeout=3, check=False,
            ).stdout
        except OSError:
            return
        match = re.search(r'"Device Utilization %"\s*=\s*(\d+)', out)
        if match:
            load.gpu_percent = float(match.group(1))
        if gpu.unified_memory:
            load.vram_used = int(mem.used)
        else:
            match = re.search(r'"vramUsedBytes"\s*=\s*(\d+)', out)
            if match:
                load.vram_used = int(match.group(1))

    def close(self) -> None:
        if self._pdh is not None:
            self._pdh.close()
            self._pdh = None


def format_bytes(count: float) -> str:
    gib = count / (1 << 30)
    return f"{gib:.1f} GB" if gib >= 1 else f"{count / (1 << 20):.0f} MB"
