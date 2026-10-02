# VR room: what the app needs from the Blender file

One room = one folder, `room_<name>/`, loaded by the app (`ultimate_video_3d/vr/room.py`, to be written). Everything
is baked, so the headset runs it easily; the app adds the live parts (the video, the screen's colour spill, the
reflection, the LED glow, the light dimming).

## Units and axes
Start from the template: `docs/room-template/room_template.blend` (made by `scripts/blender/make_room_template.py`).
- Metres. Floor at **z = 0** in Blender (Z up). Export glTF Binary (`.glb`) with Blender's default **+Y up** conversion.
- **Blender axes: the screen wall is at y = 0 and the room extends toward -Y. The viewers sit at negative y and look
  toward +Y.** (The exporter turns Blender +Y into glTF -Z, which is the way the app looks.) x = 0 is the middle of the
  room and +x is the viewer's right. An Empty with **zero rotation already looks at the screen**: its green Y arrow
  is its forward. (An earlier version of this note said -Y; that was wrong.)

## Files in the folder
| file | what |
| --- | --- |
| `room.glb` | all geometry, with the **lit** bake as its base colour texture |
| `room_dim.png` | the **dim** bake (movie lighting), same UVs and size as the lit texture |
| `room_screen.png` | the **screen-only** bake (below), same UVs and size |
| `room.json` | optional: name, default seat, notes (see the end) |

Use ONE material and ONE texture atlas for all the baked room (4096 x 4096 is plenty; 8192 is fine on a desktop GPU).
Textures: PNG or JPEG, sRGB (colour), not linear.

## The three bakes (same UV layout, same resolution)
1. **Lit** (`room.glb`'s texture): normal room lights ON (spotlights, lamps). Bake type *Combined*.
2. **Dim** (`room_dim.png`): the lights as they are during a film, e.g. only small floor/step lights or nothing.
3. **Screen only** (`room_screen.png`): every light OFF, and the screen plane set as an **emitter** (white, strength
   about 5-10). Bake *Combined*. The app multiplies this by the picture's live average colour, so the walls, ceiling
   and seats glow red in a red scene, blue in a blue one, with the room's real materials shading it.

The app mixes: `result = mix(dim, lit, light_level) + screen_bake * picture_colour * strength`. So "lights up when the film
pauses" is just `light_level` fading from 0 to 1.

## The glow without a mirror (carpet, matte rooms)
The **screen-only bake is the global illumination**: Cycles bounces the screen's light off the carpet, walls, ceiling and
furniture, so the room is lit *by the film* with real bounce, not a flat tint. Bake it in Cycles (not EEVEE), Combined,
Diffuse bounces 4 or more, denoise on, into the same atlas. The app multiplies it by the picture's live colour.
`BiasGlow` is the separate LED halo behind the screen, tinted from the picture's four edges by the same colour feed.
**Optional zoned bakes** (better): bake the screen's left half and right half separately (or four quadrants) as grayscale
irradiance and pack them into the channels of one RGBA image (`room_screen_zones.png`, R=left G=right, or R=TL G=TR B=BL
A=BR). Then each side of the room takes the colour of the matching side of the picture. The app accepts either.

## Screen size and depth (read this if the screen is bigger than ~120 in)
Picture depth is a percentage of the screen's width, so a bigger virtual screen means more real separation between the
eyes. Behind-the-screen separation must stay under the eye distance (about 6.3 cm) or the eyes diverge. On a screen
W metres wide that is about `0.063 / W * 100` percent: 2.3% at 120 in (2.66 m), but only about 1.2% at 235 in (5.19 m).
The VR player will limit this automatically for the room's `Screen` size.

## Seats: eye positions
Eyes are about 0.7 m above the seat cushion: seated on the floor about 1.15 m, on a 0.43 m riser about 1.58 m, on a 0.86 m
riser about 2.0 m. Comfortable picture width is 35-45 degrees: distance = screen width / (2 * tan(half the angle)).
Keep the top of the picture under about 25 degrees above the eye line for the main seats.

## Named objects (the names matter; the app finds them by name)
- **`Screen`**: a single quad, 16:9, **100 in = 2.214 x 1.245 m** or **120 in = 2.657 x 1.494 m**, facing the seats,
  normal pointing into the room. UV: map the whole quad 0..1 (the picture fills it; a wider film gets black bars).
  Give it an unlit/black material; the app draws the video over it. Keep the real frame (bezel) a separate object.
- **`Seat_<name>`**: an **Empty** per sitting position, e.g. `Seat_Centre`, `Seat_Front`, `Seat_Couch_Left`. Put it
  where the **eyes** are (about 1.1-1.2 m high when seated), rotated so its **-Z** axis points at the screen. The app moves
  you so your head is at that empty and you look where it looks. Optional custom property `label` for the menu name.
- **`BiasGlow`**: a flat quad on the wall **behind the screen**, a bit larger than it (the soft halo area of a TV
  backlight). The app tints it from the picture's four edges (top, bottom, left, right) like an Ambilight strip.
- **`FloorMirror`** (optional, **polished floors only**): a flat quad exactly on the floor area that should reflect the
  screen. The app draws the fading reflection on it. **Omit it for carpet or any matte floor**: the screen-only bake below
  already puts the screen's light on the floor, which is what makes carpet look lit by the film.
- **`Spawn`** (optional): an Empty used if there are no seats. 

Everything else (walls, seats, furniture, props, logos) is just baked geometry in the one atlas; the logos on the side
walls are simply part of the texture.

## Do not
- No lights, cameras or animations needed in the GLB. No lightmaps inside the GLB (they come as the PNGs).
- Don't apply the screen's video texture; the app does that.
- Keep it under about 300 thousand triangles.

## `room.json` (optional)
```json
{ "name": "Home theatre", "default_seat": "Seat_Centre", "light_level_paused": 0.6 }
```

## Why 100-120 inch matters for 3D
Depth is shown as a **percent of the screen's width**, so a bigger screen means more real depth in centimetres. At
120 in (2.66 m wide), 1% is 2.7 cm between the eyes: comfortable. That is why the app's comfort limits are percentages.
