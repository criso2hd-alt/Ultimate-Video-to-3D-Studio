"""Draws the VR scene with OpenGL: the virtual screen, the room around it, and the controller pointers.

It knows nothing about OpenXR. The caller makes an OpenGL context current, binds the framebuffer for one eye, and
calls :meth:`Renderer.draw_eye` with that eye's view and projection. That is also how the tests drive it, drawing
into an ordinary off-screen framebuffer, so most of the VR picture can be checked without a headset.

The room is the only "set" for now: a floor that mirrors the picture, and walls and ceiling that pick up its
average colour as light, so a red scene washes the walls red. The reflection is worked out in the floor's own
shader (a mirrored ray is traced to the screen) rather than by drawing the scene twice.
"""

from __future__ import annotations

import time

import numpy as np
from OpenGL import GL

from . import geometry as geo

ROOM, VOID, BAKED = "room", "void", "baked"

# Room size in metres, in the same world as the screen (floor at y = 0, centred on x = 0).
ROOM_HALF_WIDTH, ROOM_FRONT, ROOM_BACK, ROOM_HEIGHT = 5.0, 3.0, 9.0, 4.2

# Colour: the video is display-encoded (sRGB-ish) and is shown *untouched* on the screen. The room is lit in
# linear light (decode, light, encode) and everything is written display-encoded into an sRGB swapchain, which the
# headset's compositor decodes again. A plain 8-bit UNORM swapchain is read as linear by the Meta runtime, which
# brightens everything a second time and bands the darks; hence the sRGB format chosen in session.py.
_COMMON = """
vec3 to_linear(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
vec3 to_display(vec3 c) { return pow(max(c, vec3(0.0)), vec3(1.0 / 2.2)); }
uniform float u_time;
float hash(vec2 p) {
    p = fract(p * vec2(443.897, 441.423));
    p += dot(p, p.yx + 19.19);
    return fract((p.x + p.y) * p.x);
}
// +-1 level of triangular noise, so smooth gradients do not show as steps
vec3 dither(vec3 c) {
    vec2 f = gl_FragCoord.xy + fract(u_time) * 61.0;
    float n = hash(f) + hash(f.yx + 7.31) - 1.0;
    return c + vec3(n) / 255.0;
}
"""

_SCREEN_VS = """#version 330 core
layout(location = 0) in vec2 p;
uniform mat4 u_mvp;
out vec2 v_uv;
void main() {
    gl_Position = u_mvp * vec4(p, 0.0, 1.0);
    v_uv = vec2(p.x + 0.5, 0.5 - p.y);          // (0, 0) is the picture's top-left corner
}
"""

_SCREEN_FS = "#version 330 core\n" + _COMMON + """
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec4 u_rect;                            // the part of the frame this eye sees: u0 v0 u1 v1
uniform float u_gain;
out vec4 colour;
void main() {
    vec2 uv = mix(u_rect.xy, u_rect.zw, v_uv);
    colour = vec4(dither(texture(u_tex, uv).rgb * u_gain), 1.0);
}
"""

_ROOM_VS = """#version 330 core
layout(location = 0) in vec3 p;
layout(location = 1) in vec3 n;
uniform mat4 u_vp;
out vec3 v_pos;
out vec3 v_nrm;
void main() {
    v_pos = p;
    v_nrm = n;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

_ROOM_FS = "#version 330 core\n" + _COMMON + """
in vec3 v_pos;
in vec3 v_nrm;
uniform sampler2D u_tex;
uniform vec4 u_rect;
uniform vec3 u_glow;                            // the picture's average colour (display-encoded)
uniform vec3 u_eye;
uniform vec3 u_screen_centre;
uniform mat4 u_screen_inv;                      // world -> the screen's unit quad
uniform float u_gain;
uniform float u_light;                          // 0 when there is no picture yet
out vec4 colour;

// What a polished floor shows here: trace the mirrored ray back to the screen. The picture is brightest where
// the floor meets the screen's lower edge and fades with the distance the light travelled (and blurs), the way
// a real reflection on a glossy floor dissolves into it instead of being a flipped copy.
vec3 reflection_of_screen(vec3 pos) {
    vec3 v = normalize(pos - u_eye);
    vec3 r = vec3(v.x, -v.y, v.z);
    vec3 o = (u_screen_inv * vec4(pos, 1.0)).xyz;
    vec3 d = (u_screen_inv * vec4(r, 0.0)).xyz;
    if (abs(d.z) < 1e-5) return vec3(0.0);
    float t = -o.z / d.z;                       // metres travelled to the screen
    if (t <= 0.0) return vec3(0.0);
    vec2 xy = o.xy + t * d.xy;
    vec2 e = smoothstep(vec2(0.5), vec2(0.40), abs(xy));
    float edge = e.x * e.y;
    if (edge <= 0.0) return vec3(0.0);
    vec2 uv = mix(u_rect.xy, u_rect.zw, vec2(xy.x + 0.5, 0.5 - xy.y));
    // How high up the screen is the point being reflected? The bottom of the picture reflects strongly, the top
    // hardly at all: that gradient is what makes a floor reflection melt into the floor instead of being a flipped copy.
    float hy = max(pos.y + r.y * t, 0.0);
    float blur = clamp(log2(1.0 + t * 1.2 + hy * 2.5), 0.0, 6.0);
    float fade = exp(-hy * 0.85) * exp(-t * 0.10);
    return to_linear(textureLod(u_tex, uv, blur).rgb * u_gain) * edge * fade;
}

void main() {
    vec3 n = normalize(v_nrm);
    vec3 to_screen = u_screen_centre - v_pos;
    float dist = length(to_screen);
    float facing = max(dot(n, to_screen / max(dist, 1e-3)), 0.0);
    float falloff = 1.0 / (1.0 + 0.20 * dist * dist);
    vec3 glow = to_linear(u_glow);
    float albedo = n.y > 0.5 ? 0.030 : (n.y < -0.5 ? 0.020 : 0.075);
    vec3 lit = vec3(albedo) * (0.06 + u_light * 9.0 * facing * falloff * glow);
    if (n.y > 0.5) {
        float grazing = 1.0 - clamp(abs(normalize(v_pos - u_eye).y), 0.0, 1.0);
        float fresnel = 0.28 + 0.60 * pow(grazing, 2.5);
        lit += reflection_of_screen(v_pos) * fresnel * u_light;
    }
    colour = vec4(dither(to_display(lit)), 1.0);
}
"""

_BAKED_VS = """#version 330 core
layout(location = 0) in vec3 p;
layout(location = 1) in vec2 t;
uniform mat4 u_vp;
out vec2 v_uv;
void main() {
    v_uv = t;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

# The owner's baked room. Each group has three pictures on one UV layout: the lights on, the lights as during a film, and
# only the screen's light. Mixed here in linear light: lights up or down, plus the screen's glow in the picture's colour.
_BAKED_FS = "#version 330 core\n" + _COMMON + """
in vec2 v_uv;
uniform sampler2D u_lit;
uniform sampler2D u_dim;
uniform sampler2D u_scr;
uniform float u_light;                          // 0 = film lighting, 1 = lights up
uniform float u_scr_scale;                      // 2^-ev: the screen-only bake is stored brightened by ev stops
uniform float u_scr_gain;                       // how strongly the screen lights the room (0 with no picture)
uniform vec3 u_glow;                            // the picture's average colour (display-encoded)
out vec4 colour;
void main() {
    vec3 lit = to_linear(texture(u_lit, v_uv).rgb);
    vec3 dim = to_linear(texture(u_dim, v_uv).rgb);
    vec3 scr = to_linear(texture(u_scr, v_uv).rgb) * u_scr_scale * to_linear(u_glow) * u_scr_gain;
    colour = vec4(dither(to_display(mix(dim, lit, u_light) + scr)), 1.0);
}
"""

# The halo behind the screen, like a TV's LED backlight: the colour of the picture's nearest edge, fading with distance.
_GLOW_VS = """#version 330 core
layout(location = 0) in vec3 p;
uniform mat4 u_vp;
out vec3 v_world;
void main() {
    v_world = p;
    gl_Position = u_vp * vec4(p, 1.0);
}
"""

_GLOW_FS = "#version 330 core\n" + _COMMON + """
in vec3 v_world;
uniform vec2 u_centre;                          // the screen's centre and half size, in the wall's x / y
uniform vec2 u_half;
uniform vec3 u_top;                             // the picture's edge colours (display-encoded)
uniform vec3 u_bottom;
uniform vec3 u_left;
uniform vec3 u_right;
uniform float u_strength;
out vec4 colour;
void main() {
    vec2 d = v_world.xy - u_centre;
    float wt = max(d.y - u_half.y, 0.0), wb = max(-d.y - u_half.y, 0.0);
    float wl = max(-d.x - u_half.x, 0.0), wr = max(d.x - u_half.x, 0.0);
    float outside = length(vec2(max(abs(d.x) - u_half.x, 0.0), max(abs(d.y) - u_half.y, 0.0)));
    vec3 near = (to_linear(u_top) * wt + to_linear(u_bottom) * wb + to_linear(u_left) * wl + to_linear(u_right) * wr)
                / max(wt + wb + wl + wr, 1e-4);
    // a tight halo that is gone before the quad's own edge (the margins are 0.5-0.7 m), so no rectangle shows
    float glow = exp(-outside * 4.0) * smoothstep(0.0, 0.04, outside) * (1.0 - smoothstep(0.22, 0.48, outside));
    colour = vec4(to_display(near * glow * u_strength), 1.0);               // added to what is already there
}
"""

_FLAT_VS = """#version 330 core
layout(location = 0) in vec3 p;
uniform mat4 u_mvp;
void main() { gl_Position = u_mvp * vec4(p, 1.0); }
"""

_FLAT_FS = """#version 330 core
uniform vec4 u_colour;
out vec4 colour;
void main() { colour = u_colour; }
"""


_PANEL_FS = """#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform float u_alpha;
out vec4 colour;
void main() {
    vec4 c = texture(u_tex, v_uv);              // already display-encoded, straight alpha
    colour = vec4(c.rgb, c.a * u_alpha);
}
"""


def _compile(vertex: str, fragment: str) -> int:
    from OpenGL.GL import shaders

    return shaders.compileProgram(
        shaders.compileShader(vertex, GL.GL_VERTEX_SHADER),
        shaders.compileShader(fragment, GL.GL_FRAGMENT_SHADER),
    )


def _room_triangles() -> np.ndarray:
    """The room as triangles of (x, y, z, nx, ny, nz): the inside of a box, normals pointing in."""
    w, f, b, h = ROOM_HALF_WIDTH, ROOM_FRONT, ROOM_BACK, ROOM_HEIGHT
    faces = [
        # (four corners counter-clockwise as seen from inside), normal
        (((-w, 0, f), (w, 0, f), (w, 0, -b), (-w, 0, -b)), (0, 1, 0)),             # floor
        (((-w, h, -b), (w, h, -b), (w, h, f), (-w, h, f)), (0, -1, 0)),            # ceiling
        (((-w, 0, -b), (w, 0, -b), (w, h, -b), (-w, h, -b)), (0, 0, 1)),           # back wall (behind the screen)
        (((w, 0, f), (-w, 0, f), (-w, h, f), (w, h, f)), (0, 0, -1)),              # wall behind the viewer
        (((-w, 0, f), (-w, 0, -b), (-w, h, -b), (-w, h, f)), (1, 0, 0)),           # left wall
        (((w, 0, -b), (w, 0, f), (w, h, f), (w, h, -b)), (-1, 0, 0)),              # right wall
    ]
    out = []
    for corners, normal in faces:
        for i in (0, 1, 2, 0, 2, 3):
            out.append((*corners[i], *normal))
    return np.array(out, np.float32)


def _cube_triangles() -> np.ndarray:
    """A unit cube centred on the origin, for the thin pointer beams and cursor dots."""
    c = np.array([(x, y, z) for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], np.float32)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = []
    for a, b, d, e in quads:
        tris += [c[a], c[b], c[d], c[a], c[d], c[e]]
    return np.array(tris, np.float32)


class Renderer:
    def __init__(self) -> None:
        self._ready = False
        self.video_size = (0, 0)
        self.has_picture = False
        self.gain = 1.0
        self.room = None                       # a vr.room.Room, once set_room() has been called
        self._room_gl = None                   # its OpenGL objects, made on first use
        self.room_light = 1.0                  # 0 = film lighting, 1 = lights up
        self.edge_colours = np.zeros((4, 3), np.float32)       # top, bottom, left, right of the picture, 0..1

    # -- set up -----------------------------------------------------------------------------------

    def init_gl(self) -> None:
        """Compile shaders and create buffers. Call once, with the OpenGL context current."""
        self._screen = _compile(_SCREEN_VS, _SCREEN_FS)
        self._room = _compile(_ROOM_VS, _ROOM_FS)
        self._flat = _compile(_FLAT_VS, _FLAT_FS)
        self._baked = _compile(_BAKED_VS, _BAKED_FS)
        self._glow_program = _compile(_GLOW_VS, _GLOW_FS)
        self._panel = _compile(_SCREEN_VS, _PANEL_FS)
        self._panel_texture = 0
        self._panel_size = (0, 0)
        self._vao_quad = self._vao(np.array([(-.5, -.5), (.5, -.5), (.5, .5), (-.5, -.5), (.5, .5), (-.5, .5)], np.float32),
                                   [(0, 2, 0)], 8)
        room = _room_triangles()
        self._room_count = len(room)
        self._vao_room = self._vao(room, [(0, 3, 0), (1, 3, 12)], 24)
        cube = _cube_triangles()
        self._cube_count = len(cube)
        self._vao_cube = self._vao(cube, [(0, 3, 0)], 12)
        self._texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        try:                                                    # sharper at a slant, where a screen is mostly seen
            GL.glTexParameterf(GL.GL_TEXTURE_2D, 0x84FE, 8.0)   # GL_TEXTURE_MAX_ANISOTROPY_EXT
        except Exception:  # noqa: BLE001 - not every driver has it
            pass
        black = np.zeros((2, 2, 3), np.uint8)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, 2, 2, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, black)
        self.glow = np.zeros(3, np.float32)
        self._ready = True

    @staticmethod
    def _vao(data: np.ndarray, attributes: list[tuple[int, int, int]], stride: int) -> int:
        vao = int(GL.glGenVertexArrays(1))
        vbo = int(GL.glGenBuffers(1))
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        for index, count, offset in attributes:
            GL.glEnableVertexAttribArray(index)
            GL.glVertexAttribPointer(index, count, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(offset))
        GL.glBindVertexArray(0)
        return vao

    # -- the picture ------------------------------------------------------------------------------

    def upload(self, frame: np.ndarray, layout: str = "") -> None:
        """Make this decoded frame (uint8, height x width x 3, rows from the top) the one on the screen.

        ``layout`` says where one eye's picture is within the frame, so the edge colours that light the wall behind the
        screen come from the picture itself and not from the seam between the two eyes."""
        height, width = frame.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        data = np.ascontiguousarray(frame)
        if (width, height) != self.video_size:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, width, height, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, data)
            self.video_size = (width, height)
        else:
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, data)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        # the room is lit by the picture's average colour: a tiny sample is enough and nearly free
        step_y, step_x = max(1, height // 24), max(1, width // 48)
        self.glow = (frame[::step_y, ::step_x].reshape(-1, 3).mean(axis=0) / 255.0).astype(np.float32)
        u0, v0, u1, v1 = geo.eye_uv_rect(layout, 0)
        eye = frame[int(v0 * height):max(int(v1 * height), 1), int(u0 * width):max(int(u1 * width), 1)]
        eh, ew = eye.shape[:2]
        band_y, band_x = max(1, eh // 10), max(1, ew // 10)
        sample = lambda part: (part[::max(1, part.shape[0] // 16), ::max(1, part.shape[1] // 16)]  # noqa: E731
                               .reshape(-1, 3).mean(axis=0) / 255.0)
        self.edge_colours = np.array([sample(eye[:band_y]), sample(eye[-band_y:]),
                                      sample(eye[:, :band_x]), sample(eye[:, -band_x:])], np.float32)
        self.has_picture = True

    def set_room(self, room) -> None:
        """Use this baked room (a ``vr.room.Room``) for the BAKED scene; the GPU copies are made when first drawn."""
        self.room = room
        self._room_gl = None

    @staticmethod
    def _picture_texture(picture: np.ndarray) -> int:
        texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        height, width = picture.shape[:2]
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB8, width, height, 0, GL.GL_RGB, GL.GL_UNSIGNED_BYTE,
                        np.ascontiguousarray(picture))
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        try:
            GL.glTexParameterf(GL.GL_TEXTURE_2D, 0x84FE, 8.0)       # anisotropic filtering where the driver has it
        except Exception:  # noqa: BLE001
            pass
        return texture

    @staticmethod
    def _indexed_vao(vertices: np.ndarray, indices: np.ndarray, attributes: list[tuple[int, int, int]], stride: int) -> int:
        vao = int(GL.glGenVertexArrays(1))
        GL.glBindVertexArray(vao)
        vbo = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        for index, count, offset in attributes:
            GL.glEnableVertexAttribArray(index)
            GL.glVertexAttribPointer(index, count, GL.GL_FLOAT, GL.GL_FALSE, stride, GL.ctypes.c_void_p(offset))
        ebo = int(GL.glGenBuffers(1))
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
        GL.glBindVertexArray(0)
        return vao

    def _upload_room(self) -> None:
        groups = []
        black = self._picture_texture(np.zeros((2, 2, 3), np.uint8))
        for group in self.room.groups:
            vertices = np.ascontiguousarray(np.c_[group.positions, group.uv].astype(np.float32))
            lit = self._picture_texture(group.lit)
            groups.append({
                "vao": self._indexed_vao(vertices, group.indices.astype(np.uint32), [(0, 3, 0), (1, 2, 12)], 20),
                "count": len(group.indices), "lit": lit,
                "dim": self._picture_texture(group.dim) if group.dim is not None else lit,      # no dim bake: lights stay up
                "scr": self._picture_texture(group.screen) if group.screen is not None else black,
                # undo the brightening the bake was stored with (ev stops), then the emission it was baked at: a picture's
                # white is emission strength 1, and the bake used more so that there was something to measure
                "scale": 2.0 ** -group.screen_ev / max(self.room.screen_emission, 1e-6),
                "has_screen": group.screen is not None,
            })
        glow = None
        if self.room.glow is not None:
            quad = self.room.glow
            glow = {"vao": self._indexed_vao(np.ascontiguousarray(quad.positions, np.float32),
                                              quad.indices.astype(np.uint32), [(0, 3, 0)], 12),
                    "count": len(quad.indices)}
        self._room_gl = {"groups": groups, "glow": glow}

    def _draw_baked_room(self, vp: np.ndarray, now: float) -> None:
        if self._room_gl is None:
            self._upload_room()
        program = self._baked
        GL.glUseProgram(program)
        self._m4(program, "u_vp", vp)
        for name, unit in (("u_lit", 0), ("u_dim", 1), ("u_scr", 2)):
            GL.glUniform1i(GL.glGetUniformLocation(program, name), unit)
        GL.glUniform1f(GL.glGetUniformLocation(program, "u_light"), float(self.room_light))
        GL.glUniform3f(GL.glGetUniformLocation(program, "u_glow"), *self.glow)
        GL.glUniform1f(GL.glGetUniformLocation(program, "u_time"), now)
        # the picture lights the room most when the room lights are down
        gain = (1.0 - 0.75 * float(self.room_light)) * self.gain if self.has_picture else 0.0
        GL.glUniform1f(GL.glGetUniformLocation(program, "u_scr_gain"), gain)
        for group in self._room_gl["groups"]:
            for unit, key in ((0, "lit"), (1, "dim"), (2, "scr")):
                GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
                GL.glBindTexture(GL.GL_TEXTURE_2D, group[key])
            GL.glUniform1f(GL.glGetUniformLocation(program, "u_scr_scale"), group["scale"])
            GL.glBindVertexArray(group["vao"])
            GL.glDrawElements(GL.GL_TRIANGLES, group["count"], GL.GL_UNSIGNED_INT, None)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        glow = self._room_gl["glow"]
        if glow is not None and self.has_picture and self.room.screen is not None:
            program = self._glow_program
            GL.glUseProgram(program)
            self._m4(program, "u_vp", vp)
            screen = self.room.screen
            GL.glUniform2f(GL.glGetUniformLocation(program, "u_centre"), float(screen.centre[0]), float(screen.centre[1]))
            GL.glUniform2f(GL.glGetUniformLocation(program, "u_half"), screen.width / 2, screen.height / 2)
            for name, colour in zip(("u_top", "u_bottom", "u_left", "u_right"), self.edge_colours):
                GL.glUniform3f(GL.glGetUniformLocation(program, name), *map(float, colour))
            GL.glUniform1f(GL.glGetUniformLocation(program, "u_strength"), 0.8 * (1.0 - 0.8 * float(self.room_light)))
            GL.glUniform1f(GL.glGetUniformLocation(program, "u_time"), now)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE)
            GL.glDepthMask(GL.GL_FALSE)
            GL.glBindVertexArray(glow["vao"])
            GL.glDrawElements(GL.GL_TRIANGLES, glow["count"], GL.GL_UNSIGNED_INT, None)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)

    def set_panel_image(self, rgba: np.ndarray) -> None:
        """The menu panel's picture: height x width x 4 uint8, straight alpha, row 0 at the top."""
        h, w = rgba.shape[:2]
        if not self._panel_texture:
            self._panel_texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._panel_texture)
        if self._panel_size != (w, h):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
            try:
                GL.glTexParameterf(GL.GL_TEXTURE_2D, 0x84FE, 8.0)       # anisotropic: small text stays readable at a slant
            except Exception:  # noqa: BLE001 - not every driver has it
                pass
            self._panel_size = (w, h)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, np.ascontiguousarray(rgba))
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)

    def clear_picture(self) -> None:
        self.has_picture = False
        self.glow = np.zeros(3, np.float32)

    # -- one eye ----------------------------------------------------------------------------------

    def draw_eye(self, view: np.ndarray, projection: np.ndarray, eye: int, eye_position, screen: geo.Screen,
                 layout: str, scene: str = ROOM, pointers: tuple = (), panel: np.ndarray | None = None) -> None:
        """Draw everything for one eye into the currently bound framebuffer (its viewport already set).

        ``pointers`` is a sequence of (origin, direction, length, colour) beams to draw (the controllers' rays).
        ``panel`` places the menu (unit quad -> world) when it is open; nothing is drawn if it is None.
        """
        if not self._ready:
            self.init_gl()
        GL.glDisable(GL.GL_FRAMEBUFFER_SRGB)               # the shaders write display-encoded values themselves
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClearColor(0.01, 0.01, 0.015, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        vp = projection @ view
        rect = geo.eye_uv_rect(layout, eye)
        now = float(time.monotonic() % 1000.0)
        model = screen.model_matrix()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        if scene == ROOM:
            program = self._room
            GL.glUseProgram(program)
            self._m4(program, "u_vp", vp)
            self._m4(program, "u_screen_inv", np.linalg.inv(model))
            GL.glUniform1i(GL.glGetUniformLocation(program, "u_tex"), 0)
            GL.glUniform4f(GL.glGetUniformLocation(program, "u_rect"), *rect)
            GL.glUniform3f(GL.glGetUniformLocation(program, "u_glow"), *self.glow)
            GL.glUniform3f(GL.glGetUniformLocation(program, "u_eye"), *map(float, eye_position))
            GL.glUniform3f(GL.glGetUniformLocation(program, "u_screen_centre"), *map(float, screen.centre()))
            GL.glUniform1f(GL.glGetUniformLocation(program, "u_gain"), self.gain)
            GL.glUniform1f(GL.glGetUniformLocation(program, "u_light"), 1.0 if self.has_picture else 0.0)
            GL.glUniform1f(GL.glGetUniformLocation(program, "u_time"), now)
            GL.glBindVertexArray(self._vao_room)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._room_count)

        if scene == BAKED and self.room is not None:
            self._draw_baked_room(vp, now)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)

        if screen.fixed is not None:                          # the room's screen: a black panel the picture sits on
            GL.glUseProgram(self._flat)
            self._m4(self._flat, "u_mvp", vp @ screen.physical_matrix())
            GL.glUniform4f(GL.glGetUniformLocation(self._flat, "u_colour"), 0.004, 0.004, 0.006, 1.0)
            GL.glBindVertexArray(self._vao_quad)
            GL.glDepthMask(GL.GL_FALSE)                       # no depth written, or the picture (same plane) would fail the test
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            GL.glDepthMask(GL.GL_TRUE)

        if self.has_picture:
            program = self._screen
            GL.glUseProgram(program)
            self._m4(program, "u_mvp", vp @ model)
            GL.glUniform1i(GL.glGetUniformLocation(program, "u_tex"), 0)
            GL.glUniform4f(GL.glGetUniformLocation(program, "u_rect"), *rect)
            GL.glUniform1f(GL.glGetUniformLocation(program, "u_gain"), self.gain)
            GL.glUniform1f(GL.glGetUniformLocation(program, "u_time"), now)
            GL.glBindVertexArray(self._vao_quad)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        if panel is not None and self._panel_texture:
            GL.glUseProgram(self._panel)
            self._m4(self._panel, "u_mvp", vp @ panel)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._panel_texture)
            GL.glUniform1i(GL.glGetUniformLocation(self._panel, "u_tex"), 0)
            GL.glUniform1f(GL.glGetUniformLocation(self._panel, "u_alpha"), 1.0)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            GL.glDepthMask(GL.GL_FALSE)
            GL.glDisable(GL.GL_DEPTH_TEST)               # a menu is never hidden by the room: the chair in front of you must not cover it
            GL.glBindVertexArray(self._vao_quad)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthMask(GL.GL_TRUE)
            GL.glDisable(GL.GL_BLEND)
        if pointers:
            # The controllers' rays go last and on top of everything: a ray is a tool, not part of the room. Depth-tested, the
            # chair backs and the console hid pieces of it (it looked cut in half and shifted), and the menu showed it ghosting
            # through. A small dot marks where it lands on something it can press.
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glUseProgram(self._flat)
            GL.glBindVertexArray(self._vao_cube)
            for pointer in pointers:
                origin, direction, length, colour = pointer[:4]
                self._draw_beam(vp, origin, direction, length, colour)
                if len(pointer) > 4 and pointer[4]:
                    d = np.asarray(direction, np.float64)
                    self._draw_dot(vp, np.asarray(origin, np.float64) + d / (np.linalg.norm(d) or 1.0) * length, colour)
            GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def _draw_dot(self, vp: np.ndarray, position, colour, size: float = 0.014) -> None:
        """A small cube where a pointer ray lands."""
        m = np.diag([size, size, size, 1.0])
        m[:3, 3] = position
        self._m4(self._flat, "u_mvp", vp @ m)
        GL.glUniform4f(GL.glGetUniformLocation(self._flat, "u_colour"), *colour)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._cube_count)

    def _draw_beam(self, vp: np.ndarray, origin, direction, length: float, colour, thickness: float = 0.004) -> None:
        """A thin box from ``origin`` along ``direction``: the controller's pointing ray."""
        d = np.asarray(direction, np.float64)
        d = d / (np.linalg.norm(d) or 1.0)
        up = np.array([0.0, 1.0, 0.0]) if abs(d[1]) < 0.99 else np.array([1.0, 0.0, 0.0])
        x = np.cross(up, -d)
        x /= np.linalg.norm(x)
        y = np.cross(-d, x)
        m = np.eye(4)
        m[:3, 0], m[:3, 1], m[:3, 2] = x * thickness, y * thickness, -d * length          # box along -z of its own frame
        m[:3, 3] = np.asarray(origin, np.float64) + d * (length / 2)
        program = self._flat
        self._m4(program, "u_mvp", vp @ m)
        GL.glUniform4f(GL.glGetUniformLocation(program, "u_colour"), *colour)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._cube_count)

    @staticmethod
    def _m4(program: int, name: str, matrix: np.ndarray) -> None:
        GL.glUniformMatrix4fv(GL.glGetUniformLocation(program, name), 1, GL.GL_TRUE, np.asarray(matrix, np.float32))
