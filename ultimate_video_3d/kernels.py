"""One lock for every Numba parallel kernel launch.

The kernels release the GIL (``nogil=True``) so decode, depth and the UI keep
running while a warp or grade is on the CPU cores. But Numba's default threading
layer (``workqueue``) does not support two threads launching parallel kernels at
the same instant - it aborts the process - and this app has several threads that
launch them (preview render thread, depth thread, export stages).

So launches are serialised through this lock. A kernel is 5-15 ms and uses every
core anyway, so nothing is lost by taking turns; everything *else* still overlaps.
"""

from __future__ import annotations

import threading

LOCK = threading.RLock()


def configure_threads() -> None:
    """Cap the CPU thread pools so no one library starves the others.

    Left alone, Numba, OpenCV, the AV1 decoder and ONNX Runtime each start a worker
    per logical core - on a 24-thread machine that is ~100 threads fighting for 24
    cores. The symptom is not slowness but *stalls*: the decode thread simply does
    not get scheduled for a while and the picture stutters. Measured on a 4K clip,
    a Numba pool of 6-12 with OpenCV at 4 gave a steady 30 fps in every run, where
    the defaults swung between 10 and 30. ``UV3D_THREADS`` overrides the Numba count.
    """
    import os
    import sys

    # Several Python threads run at once (decode, depth, render, display, the UI). At
    # the default 5 ms switch interval a thread that becomes ready - the display thread
    # waking for the next frame - can wait 20 ms+ for the interpreter, and misses its
    # frame slot. 1 ms keeps wake-ups on time for a negligible switching cost.
    sys.setswitchinterval(0.001)

    cores = os.cpu_count() or 4
    try:
        import numba

        want = int(os.environ.get("UV3D_THREADS", "0")) or max(2, min(8, cores // 3))
        numba.set_num_threads(max(1, min(want, numba.config.NUMBA_NUM_THREADS)))
    except Exception:  # noqa: BLE001 - an unset cap only costs some smoothness
        pass
    try:
        import cv2

        cv2.setNumThreads(max(2, min(4, cores // 6)))
    except Exception:  # noqa: BLE001
        pass


configure_threads()
