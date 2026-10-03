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

- **Depth refinement** (`depth.refine`, `ThreeDSettings.depth_quality`): the default `standard` is the original single
  picture-steered guided filter at 640 px - it gave excellent *3D results* in real use, so leave it the default.
  `smooth` / `smooth_high` (opt-in, Advanced > Depth cleanup) first median-filter and flatten each object with a
  depth-steered filter, then snap only the outline: much cleaner *depth maps* (the standard filter drags picture
  texture into them), but the final stereo differs little, and it costs ~7 ms/frame (640 px) or ~25 ms (960 px,
  exports only; the preview always uses 640). Lesson: judge depth changes by the rendered 3D, not the depth map.

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

## 0.2 work in progress (branch `0.2-dev`)

**Disc import** - the Source card's "Choose external disc..." button opens `discdialog.DiscDialog`:
pick a drive (or a folder that holds VIDEO_TS / BDMV), pick a title (longest first), and `disc.copy_title`
stream-copies it to `data/discs/<label> - Title N.mkv` (no re-encode: fast, lossless), which is then opened like
any file. `disc.py` is pure Python: DVD titles come from the `VTS_nn_0.IFO` program-chain tables (cells = sector
runs; decoy and short chains dropped), Blu-ray titles from `BDMV/PLAYLIST/*.mpls`. Copies run on a worker
thread, can be cancelled, write `*.part` and rename, and check free disk space first.
- Disc clocks restart at every cell / clip; `_Retimer` keeps timestamps rising. Each stream tracks its own
  place among the restarts because picture runs ahead of sound in the file (a single global shift broke this).
- `disc.is_scrambled` looks at the first 4 MB (TS sync bytes / DVD scramble flag); a scrambled title is
  reported at once instead of letting FFmpeg chew noise for a minute. The message is `disc.PROTECTED`.
- `video.display_width`: DVD pixels are not square (32:27). `probe()` reports the display width and
  `frames()` scales to it; the preview proxy derives from `info`, so everything downstream sees the real shape.
- Two modes in the dialog. *Copy to disk first* (default) writes the MKV above. *Convert straight from the disc*
  keeps no copy: `disc.DiscStream` stands in for the file path in `pipeline.convert_video` (it has `.info`, `.stem`,
  `.frames()`, `.audio_path`, `.discard()`); the disc is read once, front to back, the picture is decoded as it
  arrives and the soundtrack is written to a small temp MKV in the same pass, muxed on at the end. It cannot seek,
  so it always converts the whole title (the range box is disabled) and the preview is a ~90 s sample cut from a
  quarter of the way in (`copy_title(start_fraction=, max_seconds=)`), held in `scratch/`. `page.disc_stream` is set
  while a disc title is open; opening any file clears it.
- Timestamps are rebased so the first keyframe sits at 0; sound before the first picture is dropped.
- Tests use synthetic discs only (`tests/test_disc.py`); a real drive in the machine is masked in the dialog test.
**Pause and resume** (`resume.py`, `video.SegmentedWriter` / `join_segments`, `pipeline.convert_video(resume=)`):
the video is encoded as segments (`SEGMENT_FRAMES`, ~1 min each) in `scratch/resume/<job key>/`, with a `state.json`
listing the finished ones. Pause = close the open segment at once (so nothing encoded is lost), keep the folder,
yield `stopped` with message `paused`. Convert again on the same file/range/settings finds it (`resume.find`,
the key hashes path+size+mtime, range, and every export/3D/colour setting) and the window asks Continue / Start
over / Cancel (`MainWindow._ask_resume`). Resuming re-reads `PREROLL` frames before the resume point so the depth
smoothing has settled, then writes only new frames; at the end the segments are joined by copying packets
(`join_segments` keeps decode times strictly rising - a two-frame segment has no B-frame delay). A different
encoder than the one that made the segments discards them. Errors also keep progress. Disc-direct conversions
never resume (forward-only). Stale jobs (>14 days) are swept at launch. The Stop button reads "Pause" for files.
**Library tab** (`library.py` data, `librarypage.py` UI): a grid of the person's 3D videos with a player beside it.
New conversions are added automatically (`MainWindow._record_conversion`, also from the batch dialog), and the
person can add their own ("Add videos..."); a layout is guessed for those from the file name and shape
(`guess_layout`) and can be changed from the right-click menu. Nothing is copied: `library.json` in the data dir
just remembers paths, layout and a few facts; thumbnails (one eye) are cut lazily on a one-thread pool.
On a normal screen the player offers only **Original** (as stored) and **Anaglyph** (`library.view_of`, eyes
split by `split_eyes`, half layouts stretched back); it reuses `player.FrameFeeder` + `Clock` with Qt audio.
Remove = forget; Delete = Recycle Bin (`send_to_trash`; never a permanent delete; nothing happens if the bin
fails). Windows will not move an open file, so a delete first closes the player *and waits for its decode
thread*, holds the thumbnail lock, and retries for ~2 s **with the event loop running** (Qt's audio backend
releases a file only when its events are delivered - a plain sleep keeps the lock). Do not look tabs up by
position (`tabs.setCurrentWidget`): Library sits between Video and Settings. The VR player (planned) will play
from this same list.
- Still to do: deinterlace and 3:2 pulldown for DVD (needs a real DVD to check), audio-track / subtitle choice,
  pause and resume for movie-length copies and conversions, Blu-ray playlists that repeat one clip (decoys).

## Depth zones (0.2): showing where things will sit relative to the screen

`zones.py` turns `stereo.disparity_map` (signed pixels, positive = in front of the screen) into percent of the
picture width, and is used three ways so the picture and the controls always agree: the **Zones** view on the mode bar
(`zone_image`: warm = out of the screen, cool = behind it, a white line where the disparity crosses zero = on the
screen, plus a legend bottom right); the **gauge** at the top of the 3D panel (`controls.DepthGauge`: histogram of the
picture's depth for the *current* settings, comfort bands at `COMFORT`=1% and `STRONG`=2% of the width, and the
extremes); and **coloured strips under two sliders** (`controls.MarkedSlider`): *3D strength* green/amber/red
(`strength_marks`, exact because disparity is linear in strength) and *Depth position* warm/light/cool with a tick at
the median (`convergence_marks`, from the 10th/90th depth percentiles - not the 2nd/98th, because the depth model
rescales every scene to span 0..1, which would always read full range). Stats ride on `PreviewResult.stats["zones"]`
and reach `ThreeDPanel.set_depth_stats` from the live path only (buffered playback does not carry them). The percent
thresholds are rules of thumb for comfort, not a standard: tune them if testers disagree.

## VR (planned for 0.2, Windows first)

Native OpenXR player, integrated (not a file you open elsewhere): *Preview in VR* renders a ~5 s clip at export
settings and loops it muted until the headset is worn (session state FOCUSED); a button or Space starts it with
sound. The Library plays in VR too. A simple room (reflective floor, walls tinted by the video's colours) comes
first, more scenes later. No OpenXR on a Mac: it gets a WebXR page later (HTTPS needed), same branch, platform-gated.
**Spike result** (`scripts/vr_probe.py`, `scripts/vr_spike.py`, 2026-10-02): Quest 3 over Link, active runtime
Meta/Oculus 1.208 (SteamVR and Virtual Desktop runtimes also installed). It offers OpenGL, D3D11 and Vulkan; an
OpenGL session through `pyopenxr` (Apache-2.0) + `glfw` + `PyOpenGL` opened, reached FOCUSED and drew eye images
(READY took ~12 s from launch). Recommended 2064 x 2272 per eye. Use OpenGL (D3D11 would need COM from Python).
Eye order confirmed on the headset: the first view from `view_loop` is the LEFT eye (red), the second the RIGHT
(blue). Note the Quest's proximity sensor puts it to sleep if it is not worn or covered: hold it when testing.
Stereo confirmed in the headset (2026-10-02): the test card's OUT squares come toward the viewer and BEHIND ones sink
away, 72 fps steady for a minute. The owner is modelling a custom baked room in Blender: its file contract is in
`docs/vr-room-spec.md` (GLB + lit / dim / screen-only bakes, `Screen`, `Seat_*`, `BiasGlow`, `FloorMirror`); the
procedural room in `vr/renderer.py` is the stand-in until it arrives.
TODO (VR): make the player screen-size aware. The owner's room has a ~5.19 m wide screen (235 in), so 1% of the picture
width is 5.2 cm of eye separation and 'behind' disparity must stay under ~1.2% of the width (IPD 6.3 cm) or the eyes
diverge; a depth-limit/convergence shift keyed on the `Screen` mesh width, with a visible note when it acts. The
comfort bands in `zones.py` assume a smaller screen.
**Baked rooms (built 2026-10-02).** `vr/gltf.py` reads a .glb (nodes, meshes, embedded images), `vr/room.py` turns a folder
(`room.glb` + `room.json` + per-group `_dim`/`_screen` JPEGs) into a `Room` (groups of world-space triangles with
lit/dim/screen pictures, `Seat_*` eye positions, the fixed `ScreenQuad`, the `BiasGlow` quad). `renderer.BAKED` draws it:
`mix(dim, lit, room_light) + screen_bake * 2^-ev / screen_emission * picture_colour` in linear light, the room's own screen as
a black panel with the picture letterboxed inside (`Screen.from_quad`, `picture_aspect`), and the `BiasGlow` halo tinted from
the picture's four edge colours (additive, fades out before its quad edge). `session.set_room/set_seat`: the person is moved
onto a seat with `geometry.rig_to_seat` (head lands on the seat's eye position, yaw matches, only about the vertical);
`state.LightFader` dims the lights while the film plays and brings them to `light_level_paused` on pause; the right stick
steps seats. `vr/controller.py` is what the window talks to (waits for the headset, previews loop muted and the first
press unmutes); `app.py` has the Library's Play in VR, Space routing, and Settings > Virtual reality (room folder, or the
first folder with a room.glb under `data/rooms/`). Preview any room without a headset:
`python scripts/vr_room_preview.py <room_folder> <out_dir> --seat Seat_R2_Centre`; in the headset:
`python scripts/vr_demo.py <video> sbs_half 120 --room <room_folder>`.
Baking lessons (all hit for real): `smart_project(correct_aspect=True)` reads the material's texture image aspect, which is
0x0 in a background run, and squashed a 9.5x5.2 m wall to a 14:1 sliver (18% atlas use; 89% with `correct_aspect=False`);
`average_islands_scale` also wrecked packing; apply every object's transform before unwrapping (the unwrap ignores object
scale); the first unwrap in a Blender session packs badly, so retry on low coverage; the owner's file was saved with an object
in Edit Mode, which breaks `transform_apply` (go to Object Mode first); a background command over 10 minutes is killed, so
bake per group/state and persist each group's UVs (`uv_<group>.npz`) so a later export matches every bake.
Not yet checked: SteamVR and Virtual Desktop runtimes (`XR_RUNTIME_JSON=<json>` selects one for a single process
without touching the system setting).
The runtime prints a lot of its own log to stdout: never parse stdout for results.

## Planned for 0.2: macOS

macOS is written but untested (0.2 will test it on an Apple Silicon MacBook Pro).
