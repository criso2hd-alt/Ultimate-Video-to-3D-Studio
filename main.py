import os
import sys

# A PyInstaller --windowed build has no console, so sys.stdout and sys.stderr are
# None, and anything that writes to them dies with "'NoneType' object has no
# attribute 'write'". Guard first, before importing anything that might write
# while being imported.
for _name in ("stdout", "stderr"):
    if getattr(sys, _name, None) is None:
        setattr(sys, _name, open(os.devnull, "w", encoding="utf-8"))

# Right after the streams are safe and before anything heavy: turn a silent
# "crashes to desktop" into a written crash.log.
from ultimate_video_3d import crashlog  # noqa: E402

crashlog.install()

# A frozen build fetches PyAV on first use; a copy from an earlier run must be on
# the import path before anything tries to import it.
from ultimate_video_3d import bootstrap  # noqa: E402

bootstrap.activate_av()

if __name__ == "__main__":
    if "--selftest" in sys.argv:
        from ultimate_video_3d.selftest import run_selftest

        sys.exit(run_selftest())
    from ultimate_video_3d.app import main

    main()
