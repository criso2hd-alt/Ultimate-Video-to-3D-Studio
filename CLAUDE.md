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
  DLSS app): a welcome dialog, then a dimmed window with one real control lit at a time (15 steps since 0.2.0; the
  side-rail steps scroll the rail via `TourStep.prepare`, the Library / Watch in VR / Settings steps switch tab).
  `ONBOARDING_VERSION` is 2: everyone who saw the 0.1 tour is offered the new one once. Offered once
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

## Folders of rips, audio tracks and subtitles (0.2)

Source card > "Open a folder of videos..." (`folderdialog.FolderDialog`): lists every video under a folder (3 levels down,
`MIN_BYTES` 2 MB to skip menu loops) longest first, each with its audio and subtitle tracks; per file the person picks the
audio and subtitle track, then *Open the selected* (back to the normal window) or *Convert the selected / all* (pre-fills the
batch converter, each job `(file, audio_idx, sub_idx)`; `BatchWorker` deep-copies the settings per job). Unprotected rips only
- nothing here touches AACS/CSS.
- `video.list_tracks` -> `Track` (disc-style names: Dolby TrueHD / Digital Plus, DTS-HD MA, PGS "Blu-ray picture", ...);
  `best_audio_index` skips commentaries and prefers the marked default. FFmpeg reads ASS subtitles back as codec `ssa`.
- Picture subtitles (PGS / VobSub) can only be copied into an MKV, never converted. Subtitles are kept only in MKV; choosing one
  switches `ExportSettings.container` to `mkv` (MP4 formats only; ProRes etc. are unchanged).
- Audio MP4/MOV cannot hold (TrueHD, DTS, DTS-HD: not in `video.MP4_AUDIO`) is re-encoded to AAC there and copied untouched in
  an MKV; the panel says so. In/Out ranges re-time the subtitle packets. `src.seek(0)` before the second demux pass.
- The preview plays the chosen audio track (`PreviewPlayer.set_audio_track`, by position in file order).
- Tests build real MKVs with PyAV (`tests/test_tracks.py::make_rip`: ac3 5.1 eng, aac fre, commentary, two ASS tracks).

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
**The menu in the headset** (`vr/menu.py`, `Menu`): a 1280x800 panel painted with QPainter into a QImage (no GL in the
class, so it is tested without a headset), put on a textured quad by `renderer.draw_eye(..., panel=)` and hung 1.5 m in front
of the person, 20 cm below the eyes, tilted up (`geometry.panel_matrix`). Pages: `help` (shows by itself the first time the
headset is worn and closes after 16 s, or `videos` instead if nothing is playing and the Library has entries), `main` (Play,
Videos, Help, Lights Auto/On/Off, Seat, Volume, Reset view, Leave VR), `videos` (3x2 thumbnails a page). Open/close:
left menu button or B/Y. **Right thumbstick press = reset the view** (`Hand.reset_pressed`, the `reset` action in `_Inputs`).
The pointer ray goes to whichever is nearer, panel or screen; the trigger on the panel presses the hovered button, so a menu
click never also toggles play. `Menu.press` returns action tuples; `session._apply` does the headset's own (lights mode on
`LightFader.mode`, seat, recentre, play) and emits `menu_action` for the window's (`video`, `volume`, `exit`), which
`VRController._on_menu_action` carries out. The Library reaches the headset as `Tile`s via `MainWindow._push_vr_videos`
(on Play in VR, after a conversion, and on every tab change). The Play in VR button is always shown and fades out, with the
reason as its tooltip, when `vr.unavailable_reason()` is non-empty; a runtime with no headset behind it (XR
FORM_FACTOR_UNAVAILABLE) becomes "Please connect your VR headset" in a box (`session.friendly_error`, `MainWindow._vr_failed`).
**Seats and screenshots (headset feedback, 2026-10-03).** The right stick goes to the seat it points at (`state.pick_seat`: within 60
degrees of the push, in the seat's own frame - forward is toward the screen - nearest wins; one move per push, the stick must
return to the middle). Changing seat keeps the person's turn (`geometry.rig_to_seat(..., keep_turn=rig)`); only the first sit and
Reset view (right stick press, or the menu button) face them the way the seat faces - before, every seat change spun the whole room
to wherever they were looking. Meta's own Meta-button+trigger screenshot does not capture a Link session, so the menu has a
Screenshot button: it closes the menu, waits `SHOT_DELAY` (3 s), reads the LEFT eye image straight out of the swapchain
framebuffer (`session._save_eye_view`, menu and rays hidden for that frame) and saves a PNG to Pictures/Ultimate Video to 3D Studio
(`paths.screenshots_dir`), then reopens the menu with a note. The seat < > buttons in the menu still step through the list.
**The projector (planned beam effect).** The owner's room gets a ceiling-hung projector. In Blender: a `Projector` box (stand-in until they
model it) on a thin `Projector_Mount` pole through the centre slot between the acoustic panels, and an empty `Projector_Lens`
(arrow pointing at the screen) at the lens, in collection `06_PROJECTOR`; a wire `BeamGuide_TEMP` shows the beam's outline in
the viewport (delete it, it is hide_render). `scripts/blender/place_projector.py` found the spot by testing 231 rays from the
lens to the whole screen against all geometry (lens (0, -7.0, 4.155), throw 6.96 m = ratio 1.34, body hanging below the panels,
pole through the slot at y = -7.1). The new file is `docs/room-template/room_template_projector.blend` (ignored by git like the
other room_template*.blend). Runtime beam, not built yet: a four-sided additive pyramid from the lens to the picture's corners
(so it narrows for letterboxed films), brightness/tint from the picture, only while playing with the lights down, soft edges, a
slow noise shimmer; the app needs nothing from the model but the `Projector_Lens` position.
**The second bake (2026-10-03, `room-build/home_theatre_v2`, package `room-build/room_home_theatre_v2`).** What went wrong the first time and
how `scripts/blender/bake_room.py` handles it now:
- *Black wood (chair console, bin, wall-base framing)*: those pieces were inside-out (faces pointing into the solid; the imported chair is
  stored mirrored). A bake lights a face from the side its normal points to, so they got no light; an ordinary render hides it by shading
  both sides. `fix_inside_out` (called through `prepare()` in the bake AND the export phase, so the UV layouts still line up) flips, in memory,
  each connected piece whose back looks out into open room while its front looks into an enclosed space (24 fixed hemisphere rays each way
  from up to 48 faces; fixed seed; the .blend is never saved). The first versions of the test were wrong in instructive ways: "hits a back face"
  misses a wholly inside-out solid, and "more free distance behind" flips hollow leather skins. Proven with a swap to plain emission (bakes)
  vs plain Principled (stays black) vs flipping the faces (lit).
- *Logo*: a Cycles bake does not composite a see-through sticker. `make_logo_opaque` forces its alpha to 1 for the chair-group bake only; then
  `--phase logoinfo` writes `logo_info.json`, `composite_logo.py` lays the sticker onto the WALL atlas (lit, dim, screen-only, each with its
  own exposure undone and redone) by the PNG's real alpha, and `EXPORT_SKIP` leaves the logo object out of room.glb. The wall keeps the glow the
  logo casts (baked with the real see-through logo). Do not mix the `Lightbox` plate in: it is not visible in Blender's own render.
- *Hallway*: group `hall` = Floor.001 + Wall_Back.002 at 6144 px (277 px/m; was 84 at 2048), which also frees the plaster-wall atlas.
- Run order: `--only <group>` per group (lit, dim, screen), then `--phase logoinfo`, `python scripts/blender/composite_logo.py <dir>`, `--phase export`.
  The output dir is made absolute (a relative one was resolved from the .blend's folder and the export could not find the images).
- A bake stops at once if a visible mesh is in no group or a listed name is missing (`--allow-unbaked` overrides); `ProjectorLensGlow` is hidden in
  every bake (the app will draw the beam and lens glow live).
**Getting into VR without the mouse, and sound (2026-10-03).** `VRController.watch(True)` (switched on by `MainWindow._update_vr_watch`
while the Library tab is open and `settings.vr_watch`) polls `vr.headset_present()` (an OpenXR instance + one system query, off
the UI thread, every 5 s) and starts the session by itself when a headset appears; a session that finds none while watching is silent
(not the "Please connect your headset" box). Leaving VR from the menu (`stop()`) pauses watching until VR is asked for again. The
Library's Play in VR button is enabled with nothing selected (`vr_requested(None)`: the app starts the session and the headset's first
wear opens the Videos page). Sound: `apply_audio()` routes the film to the output device whose name matches `AUDIO_HINTS` ("Oculus
Virtual Audio Device" for Meta Link), at session creation, on every wear and when a film is shown, so the person never has to visit
Windows' sound settings; the menu's Sound row (`("audio", +-1)`) cycles Automatic and each device, and the choice is saved in
`settings.vr_audio_device`. The Meta runtime's own answer (XR_OCULUS_audio_device_guid) is not used; name matching was enough.
**Glow (2026-10-03).** Every `BiasGlow*` mesh is exported and drawn (`room.glows`): the shader measures the 3D distance from each
fragment to the screen rectangle (past the edges, plus how far in front of the screen's plane) and tints with the nearest picture
edge, so one formula serves the wall behind the screen, the floor, the top and the sides (`GLOW_REACH`, `GLOW_GAIN` per surface kind,
`GLOW_STRENGTH` in renderer.py). Ceiling clutter is not followed: a plane only lights what it covers.
**Third bake (home_theatre_v3).** Island gaps are 28 px (`ISLAND_GAP` x `MARGIN_PX`) with a 20 px bake margin so no black is left to bleed
into the edges of the ceiling panels (the first two bakes packed islands 6 px apart and left the rest black; mipmapping blended it in);
the flat ceiling is its own 2048 group; the nine chairs are split per row (`split_chairs`, three 4096 atlases at ~620 px/m with 10 px
gaps, was one at 288); `Wood2` (the gold lattice, driven by a Metallic map) is baked as diffuse because shine cannot be baked.
GPU memory of the room is now ~1.6 GB (lit+dim+screen, mipmapped); the old one was ~0.7 GB.
**Spill glow, projector and the beam option (2026-10-03, after headset feedback).** The glow around the screen is no longer an edge colour
(copying the nearest edge pixel smeared each picture row straight out sideways, and averaging the four edges turned a half-orange,
half-blue picture grey above and below): `_GLOW_FS` mirror-tiles the picture outward over the glow planes (flipped over every edge and
corner), blurs it into mush (`lod` 7+, 3x3 taps, wider with distance) and *multiplies* the wall's own colour (`GL_DST_COLOR, GL_ONE`:
wall * (1 + glow)) so the plaster texture shows through instead of a flat wash lying on it. `GLOW_STRENGTH` 4.2 is tuned for that blend.
The BiasGlow planes are invisible canvases (they only mark where spill may land); nothing draws them.
The room's ceiling projector: the `Projector_Lens` empty is exported in room.glb (`room.projector_lens`). `renderer._draw_projector`
draws, only with the lights down: the *beam* (`_BEAM_FS`: a pyramid lens -> picture corners, far faces only; each pixel intersects its eye
ray with five planes and sums 8 haze samples, tinted by the part of the picture each sample aims at, soft sides, dust drift and motes)
and the *lens* (`_LENS_FS`: a camera-facing sprite, as bright as the picture). The beam alone has an on/off switch and a brightness
(`BEAM_LEVELS` 25%..200%): the headset menu's Beam row, Settings > Virtual reality, `settings.vr_beam` / `vr_beam_level`,
`VRController.set_beam` / `beam_changed`. The lens always glows. Do not name a renderer method `_draw_beam`: that is the pointer ray.
**Quality / bitrate / file size (`ratecontrol.py`, ExportPanel).** Three linked controls for one choice: the Quality slider, a video bitrate
(kb/s) and a file size (MB). `ExportSettings.rate_mode` ("quality" | "bitrate" | "size") is whichever the person changed last and is what the
encode follows; the other two are shown as what it works out to (`ExportPanel._sync_rates`, with `set_rate_context` fed by
`MainWindow._rate_context`: output frame size, fps, length of the In/Out range, sound bits/s; it is refreshed on file, layout, range, sound and format
changes). Quality -> bitrate is an *estimate* (bits-per-pixel model x codec efficiency, `ratecontrol.bitrate_for_quality`); a typed bitrate or
size is aimed at for real: `encoders.bitrate_options` per encoder, `VideoWriter(bitrate_bps=)`, `pipeline.convert_video` turns a size into a bitrate
(`video_bps_for_size`: size / length minus the sound, which counts what will really be stored - copied DTS-HD in an MKV, AAC in an MP4). Measured
here: x264/x265/SVT-AV1 within 1-9%, NVENC ~10% over (so `bitrate_bias` 0.95 and a ceiling equal to the target), libvpx-vp9 1-pass 1.4-1.9x over whatever
the options (the panel says so). Resumed jobs hash the new fields, so a different target starts fresh. "Save as" now says MP4 (it used to say
"Same as the format"), and the MKV a subtitle asks for undoes itself when the subtitle is dropped or another file opens (`_auto_mkv`).
**Black bars (`bars.py`).** Many films are 2.35:1 with the bars burned into a 16:9 frame. `bars.detect` samples 24 frames (0.9 s on a 25 GB rip) and
reports, per eye picture, how much of each side is black in all of them (a row is picture if 0.5% of it is bright in >= 2 frames, so dark scenes
and subtitles can only make bars look smaller; one row of bar is kept). `VRMedia` scans in a background thread when a film opens, `VRMedia.crop` /
`.aspect` give the real picture, the renderer shows `crop_rect(rect, crop)` (so the glow and beam follow) and the room's screen fits that shape.
Settings > VR > "Cut the black bars off wide films" (`vr_trim_bars`). Not applied to conversions. Desktop preview unchanged.
**Starting VR safely, and choosing the room (2026-10-03, after "GraphicsRequirementsCallMissingError" and "LimitReachedError" in the headset).**
The Meta runtime keeps state per *process*: the 5-second headset probe (`vr.headset_present`: create an instance, ask for the system, destroy
it) could run in the middle of a session's start-up and make it forget the graphics requirements had been asked for; and a start-up that
stopped half way left its instance behind, so every later try was refused. Now `vr.XR_LOCK` is held by the probe (non-blocking: it simply skips)
and by `session.GuardedContext` (wraps pyopenxr's ContextObject: starts under the lock, frees a half-made start-up, lets go once the session
exists). A real failure also stops the automatic retrying (`VRController._on_failed` pauses watching until VR is asked for again). A session
that starts by itself (headset appeared) used to open in the plain built-in room because only Play in VR loaded the chosen room:
`VRController.room_provider` (the app's `_vr_room_dir`) now loads it in `start()`. Rooms page: `vr/rooms.py` lists the folders under data/rooms with a
room.glb (name from room.json, picture from `thumbnail.jpg|png` / `preview.jpg|png` in the room folder), the menu shows a Rooms button
(only with more than one room) and a 3x2 Rooms page, a pick calls `use_room` live (seat and film stay) and `room_chosen` saves it as `vr_room_dir`.
The plain built-in room (mirror floor, no baked room) is always the last card (`rooms.PLAIN`, `plain_thumbnail()` paints its picture); choosing it stores `vr_room_dir = "plain"`
(Settings > "Use the plain room" does the same; an empty setting means "the first baked room"). The home theatre ships as the cinema-screen version only.
**Home theatre screen is now cinematic (room v4).** `scripts/blender/set_screen_size.py` makes a copy of the .blend with only the Screen quad resized
(6.6 x 2.76 m, 2.39:1, same bottom edge) and `bake_room.py --phase export` re-exports into a copy of the bake folder: nothing is re-baked and every
other piece (groups, glow planes, seats, lens) is identical to v3 (checked). The owner's own .blend files are untouched; their working file still has
the old screen unless they apply the same numbers.
**Next rooms (2026-10-03): blockouts made, owner models them.** `scripts/blender/make_blockout.py imax|drivein <out.blend>` writes a rough
starting file in the home theatre's layout (never overwrites; `--force` to). *Grand Cinema* (`room_imax_blockout.blend`): a huge IMAX
house, screen 28 x 14.7 m (1.9:1), 20 stadium rows (1.2 m tread, 0.3 m riser), only the centre seat of each row (`Seat_R<k>_Centre`,
`REF_SeatSlot_R<k>`), ribbed side walls, sloping ceiling, the five BiasGlow planes; row 10 is the sweet spot (~53 degrees).
*Drive-in* (`room_drivein_blockout.blend`): open-air, 15 x 8.44 m screen on a tower, two parking ramps, a Chevy pickup and a Cadillac with open
cabins and two seats each (`Seat_Truck_*`, `Seat_Cadillac_*`, labelled), window speakers (`AudioSource_Speaker_*` empties for a future tinny-
speaker audio filter), filler cars, `Sky_Dome` for the HDRI, a projector booth 60 m back (`Projector_Lens`). To do in the app for them:
(1) the depth limit for big screens (behind-screen separation must stay under ~6.3 cm: only ~0.2% of a 28 m screen), (2) scale the glow
reach and the beam falloff by screen size / throw (both are tuned for the 5 m screen and a 7 m throw), (3) `bake_room.py` GROUPS/default
seat are home-theatre specific and need a per-room config.
Idea agreed but not built: many films are 2.35:1 with the bars burned into a 16:9 frame; detect those bars and fit the real picture
area, and make the room's Screen object 2.35:1 (Blender change + screen-light re-bake only).
**Pointer rays are drawn last, on top, without depth test** (`renderer.draw_eye`, 5-element tuples (origin, direction, length,
colour, landed) with a dot at the tip when `landed`). Depth-tested they were hidden piecemeal by chair backs and the console and
looked "cut in half and shifted" in the headset; the menu panel also showed them ghosting through it. Reproduce without a
headset by drawing a lap-height ray at the menu offscreen (the repro lives in the session notes; the test is
`test_pointer_rays_are_never_hidden_by_the_room`).
`scripts/blender/find_bake_problems.py` lists (and selects in Edit Mode) faces that cannot bake: pointing into solid (INSIDE) or sunk
under 3 cm into another surface (BURIED). The owner's first bake had a black console bin and wall-base framing for exactly that
reason; the logo (`icon_transparent`, PNG alpha over a `Lightbox` face) baked opaque black because a Cycles bake does not composite
a transparent surface over what is behind it - plan: bake only the lightbox face and ship the logo as an alpha decal.
Not yet checked: SteamVR and Virtual Desktop runtimes (`XR_RUNTIME_JSON=<json>` selects one for a single process
without touching the system setting).
The runtime prints a lot of its own log to stdout: never parse stdout for results.

## Edge artifacts, conversion speed, and starting VR (2026-10-04)

**Holes at the edges of near objects (`stereo._warp`).** The owner saw black dithery dots in one eye and stretched stripes in the other,
worse with more 3D strength. Cause: the hole filler mirrored the background texture across a hole, but when the hole was longer than the
run of picture beside it the mirror ran on into the *next* hole, which is still black at that point (right eye: holes are filled left to
right, so the right-hand source was always unfilled); and a one- or two-pixel sliver of blended edge squeezed between two holes was taken
for background and repeated as a stripe. Now: slivers (`SLIVER` = 2) become part of the hole, the fill reflects back and forth *inside* the
run of real picture (`REACH` = 48), and the invented pixels get a light 1-2-1 smoothing. Reproduce on the old kernel with random-depth rows
(`test_ragged_depth_leaves_no_black_dots_in_either_eye`); 3 real frames at strength 100 went from ~2,500 stray black pixels to 0.
**Is the GPU used when converting? Yes - and that is not where the time goes.** Measured on an RTX 4080 with a 1080p H.264 rip, SBS half, h265:
the encoder is `hevc_nvenc` (`nvidia-smi dmon -s u`: enc 6-11%, dec 0%, SM ~30% = the depth model on DirectML); Task Manager's default GPU graphs
show 3D/Compute, not the Video Encode engine, so it looks idle. The CPU pays for everything else: decode + YUV->RGB, float conversion, depth
refinement (cv2), the Numba warp and pack, RGB->YUV for the encoder. What was wasted (all fixed, 11 -> ~20 fps here): a fresh 25 MB float
picture per frame costs ~8 ms of page faults each (`video.frames(ring=)` re-uses 16 pictures, `stereo.WarpBuffers` keeps the two eyes: the warp
fell from 35 to ~20 ms), the float conversion was two passes and two temporaries, swscale's RGB->YUV is single-threaded and took 20 ms
(`stereo.rgb_to_yuv420`, a Numba kernel, 2.3 ms; 8-bit SDR 4:2:0 only, others still use swscale; `video._NO_DIRECT_YUV` switches it off for
tests), and the progress thumbnail converted the whole packed frame to float before shrinking it. A ring picture is only valid until 16 more
frames were yielded (the queues hold at most 10 in flight: `pipeline.PICTURE_RING`); a pair made with `WarpBuffers` is overwritten by the next
call. Not done: hardware *decode* (PyAV 18 has `HWAccel` for d3d11va/cuda; tried - on this 1080p rip it saved 15 CPU-ms of decoding but the
NV12 download + conversion cost more than that, so it only pays for 4K HEVC; worth an opt-in later), and the preview window itself is CPU:
decode and render on the CPU, depth on the GPU, painted with QPainter (a QOpenGLWidget would move the last step to the GPU).
**Starting VR is a button now, not a guess.** The Library has a Start VR / Stop VR button with a status line beside it
(`LibraryPage.set_vr_state`, fed by `VRController.state_changed`: off, starting, waiting, worn, stopping). Automatic headset watching is
OFF by default (`vr_watch`, Settings: "Also look for my headset by itself..."). `VRController.start_vr()` clears an earlier failure and
starts; the runtime has `START_TIMEOUT` (40 s) to accept the session (`VRSession.connected`, emitted when the instance/session/GL context exist) -
after that the person is told - while waiting for the *headset* never times out. `stop()` no longer blocks the window: state "stopping", polled
every 150 ms, and a session that has not ended after `STOP_PATIENCE` (12 s) is let go of (`_abandon_session`) so a fresh one can be made;
`stop(blocking=True)` is for quitting. Everything is written to `vr.log` in the data folder (`vr/log.py`: requests, runtime connect, session
states, errors with tracebacks, watchdog) - read it first when VR "sticks". LimitReached / GraphicsRequirementsCallMissing / RuntimeFailure
now say "close other VR apps, check Quest Link, press Start VR again".

**VR start-up, really fixed (2026-10-04, from `vr.log`).** The first session in a process worked; every later one failed with
GraphicsRequirementsCallMissingError and then LimitReachedError until the runtime reclaimed the leaked instance. Two bugs in how we used
pyopenxr, both visible in the log: (1) `ContextObject.__init__(session_create_info=xr.SessionCreateInfo())` is ONE shared default object; the
first session fills in `.next` with its graphics binding, so every later session skips creating its graphics, never asks for the graphics
requirements, and fails - `VRSession._run` now passes a new `SessionCreateInfo()` per try; (2) pyopenxr's `__exit__` begins with
`self.default_action_set` / `self.space`, which a half-finished start-up never made, so it died on its first line and left the instance
alive (LimitReached) - `session.release_context` frees each piece on its own. `GuardedContext(factory, attempts=4)` also retries a refusal
of those two kinds after 2.5 s with a fresh context. `vr.log` (beside the exe, `vr/log.py`) is what found all this: read it first.
The old "look for my headset" setting is now `vr_watch` (the key changed so a saved True from the unreliable builds is dropped; default off).
**In the headset (2026-10-04).** The main menu has a timeline (click to jump) with -1 min / -10 s / +10 s / +1 min buttons
(`Menu._timeline`, actions `("skip", ms)` / `("seek", fraction)` carried out by `VRController`); the left stick still steps 10 s. The
picture clock no longer copies Qt's audio position (a headset sound device that stalled or jumped dragged the picture with it - judder no pause
cured): `VRMedia._audio_position` only slews small differences, jumps for big ones, and `_check_audio` restarts a sound that stopped moving.
PyOpenGL's per-call error check is off (`vr/__init__.py`): both eyes of the home theatre took 7.7 ms and now 3.9 ms (GPU is not the limit: an
RTX 4080 draws the room with the lights down and the beam on in 3.9 ms; the rest is Python overhead and GIL pressure). The diary logs fps and
the feeder's dropped/late frames every 30 s while a film plays. `app._exit_watchdog` ends the process 8 s after the window closes if Python is
still waiting on a thread - a lingering process kept the VR runtime busy and blocked the next start.

**Gauge while playing, and "Fit to comfort" (2026-10-04).** The depth gauge, the strength strip and the depth-position marks used to move only when the
picture was paused (stats rode on the live path's `PreviewResult` only). `buffered.CachedFrame.zones` now carries each frame's `zones.DepthStats` dict through
the cache (it survives the spill to disk) and `BufferedPlayback.image` emits it as a 6th element; `MainWindow._on_buffered_image` updates the gauge at most every
`GAUGE_EVERY` (0.15 s), and still accepts the old 5-part payload. The 3D panel has **Fit to comfort / Fit to strong** (`ThreeDPanel.fit_requested`): `comfort.scan`
samples 240 places in the film (or the In/Out range; ~0.14 s each, ~35 s on a feature, run by `app.FitScan` on its own thread, the button turns into Stop), measures
each with the person's current settings through the same depth model/refine/disparity code as the preview (`measure_picture`), takes the worst
frame (`max(out_pct, back_pct)`, the gauge's 99th percentile) and sets ONE strength for the whole film: `strength * limit / worst`
(disparity is linear in strength), clamped to 0..100 (a gentle film is raised to fit). Limits are `zones.COMFORT` 1% / `zones.STRONG` 2% of the width. Nothing animates and no
scene is treated differently. Because the depth model rescales every frame to span 0..1, most frames of a film read nearly alike (Cloudy: median = worst = 1.4% at
strength 50), so the answer is stable; it can miss a one-second flash between samples (the note says how many places were checked).

## Planned for 0.2: macOS

macOS is written but untested (0.2 will test it on an Apple Silicon MacBook Pro).
