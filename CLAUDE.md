# Ultimate Video to 3D Studio

A free desktop app that converts **flat video into stereoscopic 3D** using AI
depth: side-by-side, top-and-bottom, anaglyph, colour+depth. A direct competitor
to Owl3D, free, with a Buy Me a Coffee banner that returns on every launch (closable per session with its X).

It shares its look and its video backbone with the sibling project
`DLSS5 IMAGE Converter` (same theme, layout, timeline, encoder-probe idea), with
everything DLSS removed.

## The pipeline

decode (PyAV) -> grade -> depth (Depth Anything V2 on ONNX Runtime) -> refine ->
stereo warp (Numba) -> pack -> encode (hardware ladder) -> mux audio.

Three worker threads (reader / depth / warp+pack, plus a writer thread) so
decode, the GPU depth model, the CPU warp and the encoder overlap. See
`pipeline.py`.

## Non-negotiable constraints

- **Every GPU on PC, and a clean path to macOS.** No CUDA dependency, ever.
  Depth = ONNX Runtime (DirectML on Windows, CoreML on Mac, CPU fallback).
  Warp/grade/tonemap = Numba kernels (CPU, all cores). Encoders = the fixed
  function ones in every driver (NVENC / AMF / QSV / VideoToolbox), software as
  the floor. Anything Windows-only (`ctypes.windll`, DXGI, PDH) lives behind a
  `sys.platform` check in `sysinfo.py` and degrades to `None`.
- **"Present" is not "usable."** An encoder is only trusted after `can_open()`
  has opened it and encoded a frame at the real size. The first-launch probe is
  saved in settings, keyed by GPU signature, and re-run if the GPU changes.
- **Live preview is a proxy (<= 720p)**, but runs the *same* grade/depth/warp
  code as the export, only smaller. Never write a second implementation for the
  preview.
- **Colour is stated, never assumed.** Untagged HD video is BT.709. 10-bit HDR
  stays 10-bit with PQ/HLG + BT.2020 tags; tone mapping happens only for depth
  input, the on-screen preview, and an SDR export.
- Depth is estimated from the **ungraded** frame.

## Preview playback and performance (hard-won - read before touching)

The UI thread must never touch a video frame. Qt's `QVideoSink` path forced every 4K
frame through the UI thread (convert + shrink) and the app became unresponsive at 5 fps.
Now `player.FrameFeeder` decodes on its own thread, scales to a proxy *inside FFmpeg*
(`frame.reformat(width, height, interpolation="AREA")`), and hands it straight to the
engine. Qt only plays audio (no video sink) and its position is the master clock.

Two ways to get a picture on screen, chosen by `PreviewPlayer`:
- **live** (`preview.PreviewEngine`, depth thread + render thread, newest-job-wins): paused
  stills, scrubbing, slider tweaks, plain-2D playback.
- **buffered** (`buffered.BufferedPlayback`): while a *3D* preview plays, decode -> depth ->
  render x2 -> JPEG runs 5-10 s ahead into `RenderCache`; a display thread shows the frame
  the clock is on. Underruns pause with a "Buffering" note. Rendered on a 24 fps grid so it
  fills faster than it plays. Depth reuse on alternate frames while the buffer is <60% full.

Things that each cost a debugging session:
- **Numba kernels must be `nogil=True`**, otherwise every 10 ms warp freezes decode, depth
  and the UI. But Numba's default threading layer aborts on concurrent parallel launches, so
  every launch goes through `kernels.LOCK`. A *tiny* kernel must not take that lock (it would
  queue behind long warps) - make it serial instead (`depth._to_nchw`).
- **Cap thread pools** (`kernels.configure_threads`): Numba ~8, OpenCV 4. With defaults (24
  each) the decode thread is starved and fps swings 10<->30. Also `sys.setswitchinterval(0.001)`
  so the display thread wakes on time.
- **One ONNX session, one lock** (`depth.shared_engine`, `_ORT_LOCK`): two DirectML sessions
  created/run concurrently crash the driver (access violation). ORT spin-waiting is off for
  GPU providers.
- The depth model is a fixed 518 px: preview resolution does **not** change its cost. Lowering
  the proxy (540p default) helps the render/decode side only.
- Preview views use the fused `stereo.pack_frame` kernel, not numpy (numpy anaglyph/uint8
  conversions cost ~25 ms at 720p).
- Never call Qt objects (`QMediaPlayer`) from a worker thread - use a signal (`_audio_command`).
- Wheel events over sliders/combos scroll the rail instead (`app.WheelGuard`); they only
  change a control that has focus.
- **Updates are notice-only** (`updates.py`): compare GitHub's latest release tag with
  `__version__`, show an UPDATE pill linking to the release page. Never download or replace
  anything (a self-replacing updater was considered and deliberately rejected as too complex).
  It reads `api.github.com/.../releases/latest`, which needs a *public* repo; a private one
  quietly yields "no update". Every failure is silent.
- **First-run welcome + spotlight tutorial** (`onboarding.py`, copied in look and behaviour from the
  DLSS app): a welcome dialog, then a dimmed window with one real control lit at a time (12 steps; the
  side-rail steps scroll the rail via `TourStep.prepare`, the Settings step switches tab). Offered once
  (`settings.onboarding_version` vs `ONBOARDING_VERSION`), only after the first-launch encoder probe
  has finished, and replayable from Settings > Help. It never changes a setting. Tests and harnesses
  set `UV3D_SKIP_TOUR=1` (see `tests/conftest.py`) - a fresh settings file would otherwise pop a modal
  dialog over them. Bump `ONBOARDING_VERSION` only when existing users should see a new tour.
- **The support banner** (`support.py`, `SupportBanner`): warm brown strip at the very bottom, yellow
  button, and an X that hides it for the *session only* (a plain in-memory flag - never in settings, so
  it is back at every launch). The link and message are a masked blob in `support.py`, pinned by a
  SHA-256 digest that `app.py` also pins (`_SUPPORT_PIN`); the button opens the *decoded* official link,
  not whatever the widget holds. `MainWindow._guard_support` re-checks the live banner (text, button,
  visible, last in the layout, no stylesheet/effect) at start, every 7 s and before a conversion; a
  wrong banner is rebuilt, a second failure disables Convert and points to the official GitHub page;
  `pipeline.convert_video` also calls `support.require_intact()`. It is a deterrent and a licence term
  (LICENSE clause 11), not DRM - anyone who patches out all of it can; do not pretend otherwise, and do
  not add a plain `buymeacoffee` string anywhere (a test greps for it). If the link or wording ever
  changes, re-run the generator logic: new masked blobs + new `DIGEST` + the pin in `app.py`.
- The app opens **paused, on the plain video**; the 3D effect is opt-in by picking a 3D view on
  the permanent `modeBar` under the picture (2D is the first chip; there is no separate 3D switch).
- **Paused = still working.** With a 3D view chosen and the video paused, `BufferedPlayback.prefill`
  keeps the look-ahead renderer running (`PAUSED_FILL_SECONDS` ahead) into `RenderCache`, which
  stores the JPEGs as files in `scratch/preview-cache/<pid>/`. Play then starts at once from the cache.
  A scrub restarts the fill from the new spot (debounced).
- **Caches are kept, not rebuilt.** One `RenderCache` per *look* (`BufferedPlayback._select`: the hash of
  `PreviewParams` + proxy height), kept until the video changes; nothing is evicted behind the playhead.
  Output -> Anaglyph -> Output finds Output still complete. Limits: 1 GB total (`_housekeeping` drops
  other looks oldest-first, then the frames farthest from the playhead), at most `MAX_LOOKS` looks, and
  no growth once under 2 GB of disk is free. Stale folders from crashed runs are swept at start-up.
- **A keyframe-only scrub picture is not a position.** `FrameFeeder.approximate` is set while one is
  delivered; on a short clip the keyframe is frame 0, and it used to yank the playhead (and the clock)
  to the start before the exact frame arrived. The window also waits ~70 ms before the first keyframe
  preview, so a plain click never shows it at all.
- **Never block the UI thread on the cache.** `RenderCache.clear()` swaps the dict and bumps
  `generation` (renderers write under the generation they started with, so a run that is winding
  down cannot refill the emptied cache); the files are deleted on a background thread. Use
  `BufferedRenderer.request_stop()` from the UI thread, not `stop()` (which joins). Deleting ~1000
  files and joining the render threads on the UI thread froze the window for a second.
- Do not measure the machine's speed from a throttled run: the tuning measurement must come from an
  unthrottled fill (depth reuse while the buffer is under 60% - dropping it made the paused fill
  half as fast and picked 15 fps on a machine that does 40).
- **The depth model is loaded and run once at launch** (`warmup._warm_model`, background thread) and
  stays resident: `depth.shared_engine` returns the same session forever. Play never loads or
  re-warms anything (measured: ~1 s load + ~0.2 s first inference, then ~18 ms per frame).
- Wheel over a slider/combo never changes it, even if it has focus (a clicked control stays focused,
  which is how the wheel used to slip through). `CardColumns.minimumSizeHint` is one column wide,
  otherwise a page that once had 3 columns can never shrink back.
- The histogram scales by the 98.5th percentile of the interior bins, not the tallest: a
  letterboxed frame puts a third of its pixels in bin 0 and flattened everything else to a sliver.

## Conventions

- Python 3.12, PySide6, `from __future__ import annotations` everywhere.
- Heavy imports (`av`, `onnxruntime`) happen lazily inside functions, so start-up
  is instant and a missing video component can be offered for download.
- Comments say **why**. If a line looks odd, the comment says what breaks
  without it (see the Numba `prange` unsigned-index note in `stereo._pack`).
- `except Exception:  # noqa: BLE001` around anything optional: a broken driver
  or a missing counter must degrade, never block startup.
- Numba kernels: `prange` indices are unsigned - cast with `np.int64(y)` before
  mixing with signed ints or the index becomes a float and typing fails.
- Model weights and downloads go in the per-user data dir via `paths.py`.

## Layout

- `ultimate_video_3d/` - the package
  - `app.py` window, video page, workers, first-run · `panels.py` rail panels
  - `player.py` preview playback (decode thread + Qt audio clock) · `buffered.py`
    look-ahead render cache · `preview.py` live preview engine · `kernels.py` thread caps + kernel lock
  - `controls.py` new widgets (sliders, curves, wheels, scopes, gauge, preview view)
  - `widgets.py` chrome shared with the DLSS app · `theme.py` palettes + QSS
  - `pipeline.py` conversion
  - `stereo.py` view synthesis + layouts · `depth.py` model + temporal stabiliser
  - `grade.py` Lumetri-style colour · `lut.py` .cube · `hdr.py` PQ/HLG tone map
  - `video.py` decode/encode/mux · `encoders.py` ladder + probe · `sysinfo.py` hardware
  - `bootstrap.py` PyAV download · `cloud.py` 3D orbit view · `selftest.py`
- `tests/` - `pytest`; `tests/shot.py` is a visual harness that launches the real window.
- `scripts/` - setup / run / build for Windows (`.ps1`) and macOS (`.sh`).

## Testing

`python -m pytest` (~45 tests, ~4 s, no GPU/model needed - a depth stand-in is
used). `python main.py --selftest` prints GPUs, encoder results and depth/warp
timings. For UI work, `tests/shot.py <video> <outdir> [mode...]` saves screenshots;
`tests/perf_preview.py` and `tests/perf_buffer.py` measure live-preview fps / UI stalls and
buffered-playback behaviour on a real clip (numbers to beat: 2D 30 fps; 3D anaglyph ~25-30 fps;
buffered: no underruns after the start-up wait).

## Licence

Source-available, **not** open source (see `LICENSE`): free to use, but no resale,
no redistribution, no reuse of the code elsewhere. Depth Anything V2 **Small** is
Apache-2.0 and is what ships; Base/Large are CC-BY-NC - do not bundle them.

## Planned for 0.2: macOS, and copying your own discs to a file first

macOS is written but untested (0.2 will test it on an Apple Silicon MacBook Pro). The Source card already
shows a disabled "Choose external disc (coming soon)" button (`VideoPage.pick_disc`): the disc is first
copied/re-encoded to a file on disk as fast as the drive allows, and the normal pipeline then runs on that
file. Work list: drive detection, title picker, DVD deinterlace / anamorphic SAR / pulldown, audio-track and
subtitle choice (MKV output), pause and resume for movie-length jobs, disk-space check before copying.
