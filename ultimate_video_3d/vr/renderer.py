"""Draws the VR scene with OpenGL: the virtual screen, the room around it, and the controller pointers.

It knows nothing about OpenXR. The caller makes an OpenGL context current, binds the framebuffer for one eye, and
calls :meth:`Renderer.draw_eye` with that eye's view and projection. That is also how the tests drive it, drawing
into an ordinary off-screen framebuffer, so most of the VR picture can be checked without a headset.

The room is the only "set" for now: a floor that mirrors the picture, and walls and ceiling that pick up its
average colour as light, so a red scene washes the walls red. The reflection is worked out in the floor's own
shader (a mirrored ray is traced to the screen) rather than by drawing the scene twice.
"""

from __future__ import annotations

import numpy as np
from OpenGL import GL

from . import geometry as geo

ROOM, VOID = "room", "void"

# Room size in metres, in the same world as the screen (floor at y = 0, centred on x = 0).
ROOM_HALF_WIDTH, ROOM_FRONT, ROOM_BACK, ROOM_HEIGHT = 5.0, 3.0, 9.0, 4.2

_SCREEN_VS = """#version 330 core
layout(location = 0) in vec2 p;
uniform mat4 u_mvp;
out vec2 v_uv;
void main() {
    gl_Position = u_mvp * vec4(p, 0.0, 1.0);
    v_uv = vec2(p.x + 0.5, 0.5 - p.y);          // (0, 0) is the picture's top-left corner
}
"""

_SCREEN_FS = """#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec4 u_rect;                            // the part of the frame this eye sees: u0 v0 u1 v1
uniform float u_gain;
out vec4 colour;
void main() {
    vec2 uv = mix(u_rect.xy, u_rect.zw, v_uv);
    colour = vec4(texture(u_tex, uv).rgb * u_gain, 1.0);
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

_ROOM_FS = """#version 330 core
in vec3 v_pos;
in vec3 v_nrm;
uniform sampler2D u_tex;
uniform vec4 u_rect;
uniform vec3 u_glow;                            // the picture's average colour, as light
uniform vec3 u_eye;
uniform vec3 u_screen_centre;
uniform mat4 u_screen_inv;                      // world -> the screen's unit quad
uniform float u_gain;
uniform float u_light;                          // 0 when there is no picture yet
out vec4 colour;

vec3 reflection_of_screen(vec3 pos) {
    // Where would a mirror at this point of the floor show the screen? Trace the mirrored ray back to it.
    vec3 v = normalize(pos - u_eye);
    vec3 r = vec3(v.x, -v.y, v.z);
    vec3 o = (u_screen_inv * vec4(pos, 1.0)).xyz;
    vec3 d = (u_screen_inv * vec4(r, 0.0)).xyz;
    if (abs(d.z) < 1e-5) return vec3(0.0);
    float t = -o.z / d.z;
    if (t <= 0.0) return vec3(0.0);
    vec2 xy = o.xy + t * d.xy;
    // the picture, softened the farther the light travelled (a slightly glossy floor), fading out at its edges
    vec2 e = smoothstep(vec2(0.5), vec2(0.42), abs(xy));
    float edge = e.x * e.y;
    if (edge <= 0.0) return vec3(0.0);
    vec2 uv = vec2(xy.x + 0.5, 0.5 - xy.y);
    uv = mix(u_rect.xy, u_rect.zw, uv);
    float blur = clamp(log2(1.0 + t * 1.6), 0.0, 5.0);
    return textureLod(u_tex, uv, blur).rgb * u_gain * edge;
}

void main() {
    vec3 n = normalize(v_nrm);
    vec3 to_screen = u_screen_centre - v_pos;
    float dist = length(to_screen);
    float facing = max(dot(n, to_screen / max(dist, 1e-3)), 0.0);
    float falloff = 1.0 / (1.0 + 0.045 * dist * dist);
    vec3 albedo = n.y > 0.5 ? vec3(0.07, 0.07, 0.09) : (n.y < -0.5 ? vec3(0.05, 0.05, 0.06) : vec3(0.16, 0.15, 0.15));
    vec3 lit = albedo * (0.10 + u_light * 2.6 * facing * falloff * u_glow * 1.8);
    if (n.y > 0.5) {
        float fresnel = 0.22 + 0.5 * pow(1.0 - clamp(abs(normalize(v_pos - u_eye).y), 0.0, 1.0), 3.0);
        lit += reflection_of_screen(v_pos) * fresnel * u_light;
        // pooled colour on the floor around and in front of the screen
        lit += u_glow * u_light * 0.10 * falloff * 2.0;
    }
    colour = vec4(lit, 1.0);
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

    # -- set up -----------------------------------------------------------------------------------

    def init_gl(self) -> None:
        """Compile shaders and create buffers. Call once, with the OpenGL context current."""
        self._screen = _compile(_SCREEN_VS, _SCREEN_FS)
        self._room = _compile(_ROOM_VS, _ROOM_FS)
        self._flat = _compile(_FLAT_VS, _FLAT_FS)
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

    def upload(self, frame: np.ndarray) -> None:
        """Make this decoded frame (uint8, height x width x 3, rows from the top) the one on the screen."""
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
        self.has_picture = True

    def clear_picture(self) -> None:
        self.has_picture = False
        self.glow = np.zeros(3, np.float32)

    # -- one eye ----------------------------------------------------------------------------------

    def draw_eye(self, view: np.ndarray, projection: np.ndarray, eye: int, eye_position, screen: geo.Screen,
                 layout: str, scene: str = ROOM, pointers: tuple = ()) -> None:
        """Draw everything for one eye into the currently bound framebuffer (its viewport already set).

        ``pointers`` is a sequence of (origin, direction, length, colour) beams to draw (the controllers' rays).
        """
        if not self._ready:
            self.init_gl()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glClearColor(0.01, 0.01, 0.015, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        vp = projection @ view
        rect = geo.eye_uv_rect(layout, eye)
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
            GL.glBindVertexArray(self._vao_room)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._room_count)

        if self.has_picture:
            program = self._screen
            GL.glUseProgram(program)
            self._m4(program, "u_mvp", vp @ model)
            GL.glUniform1i(GL.glGetUniformLocation(program, "u_tex"), 0)
            GL.glUniform4f(GL.glGetUniformLocation(program, "u_rect"), *rect)
            GL.glUniform1f(GL.glGetUniformLocation(program, "u_gain"), self.gain)
            GL.glBindVertexArray(self._vao_quad)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)

        if pointers:
            GL.glUseProgram(self._flat)
            GL.glBindVertexArray(self._vao_cube)
            for origin, direction, length, colour in pointers:
                self._draw_beam(vp, origin, direction, length, colour)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

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
