# VR room: what the app needs from the Blender file

One room = one folder, `room_<name>/`, loaded by the app (`ultimate_video_3d/vr/room.py`, to be written). Everything
is baked, so the headset runs it easily; the app adds the live parts (the video, the screen's colour spill, the
reflection, the LED glow, the light dimming).

## Units and axes
- Metres. Floor at **y = 0**. Export glTF Binary (`.glb`) with Blender's default **+Y up** conversion.
- The front of the room (the screen wall) is **-Z**, the viewer looks toward -Z. (Blender: face -Y when you model it;
  the exporter converts.) If it comes out the other way I will flip it with one switch, so don't worry about it.

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

## Named objects (the names matter; the app finds them by name)
- **`Screen`**: a single quad, 16:9, **100 in = 2.214 x 1.245 m** or **120 in = 2.657 x 1.494 m**, facing the seats,
  normal pointing into the room. UV: map the whole quad 0..1 (the picture fills it; a wider film gets black bars).
  Give it an unlit/black material; the app draws the video over it. Keep the real frame (bezel) a separate object.
- **`Seat_<name>`**: an **Empty** per sitting position, e.g. `Seat_Centre`, `Seat_Front`, `Seat_Couch_Left`. Put it
  where the **eyes** are (about 1.1-1.2 m high when seated), rotated so its **-Z** axis points at the screen. The app moves
  you so your head is at that empty and you look where it looks. Optional custom property `label` for the menu name.
- **`BiasGlow`**: a flat quad on the wall **behind the screen**, a bit larger than it (the soft halo area of a TV
  backlight). The app tints it from the picture's four edges (top, bottom, left, right) like an Ambilight strip.
- **`FloorMirror`** (optional): a flat quad exactly on the floor area that should reflect the screen. The app draws the
  fading reflection on it. Omit it for a matte floor.
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
