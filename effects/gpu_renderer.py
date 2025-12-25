# effects/gpu_renderer.py

import glfw
import OpenGL.GL as gl
import numpy as np
import cv2
from pathlib import Path
from interaction.control_state import ControlState

VERT_SRC = """
#version 330 core
out vec2 vTexCoord;
void main() {
    vec2 pos = vec2((gl_VertexID & 1) * 2 - 1, (gl_VertexID & 2) - 1);
    gl_Position = vec4(pos, 0.0, 1.0);
    vTexCoord = (pos + 1.0) * 0.5;
}
"""

class GPURenderer:
    """
    GPU distortion renderer.
    Chooses fragment shader based on ControlState.shader_mode.
    """

    def __init__(self, control_state: ControlState, width=640, height=360):
        self.control_state = control_state
        self.width = width
        self.height = height

        if not glfw.init():
            raise RuntimeError("Failed to init GLFW")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(width, height, "GPU Renderer", None, None)
        glfw.make_context_current(self.window)

        gl.glDisable(gl.GL_DEPTH_TEST)

        self.programs = {}
        self.current_mode = None
        self.fbo = gl.glGenFramebuffers(1)
        self.color_tex = gl.glGenTextures(1)
        self._init_fbo()

    def _init_fbo(self):
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.color_tex)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB,
            self.width, self.height, 0,
            gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
            gl.GL_TEXTURE_2D, self.color_tex, 0
        )

        status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        if status != gl.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Framebuffer incomplete")

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _load_fragment_shader(self, mode: str) -> int:
        shader_folder = Path("effects/shaders")
        mode_map = {
            "swirl": "distortion_swirl.frag",
            "ripple": "distortion_ripple.frag",
            "chromatic": "chromatic_abbreviation.frag",
             "neural_pulse": "neural_pulse.frag",
        }
        filename = mode_map.get(mode, "swirl.frag")
        frag_path = shader_folder / filename

        code = frag_path.read_text()

        vs = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vs, VERT_SRC)
        gl.glCompileShader(vs)
        if not gl.glGetShaderiv(vs, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(vs).decode())

        fs = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(fs, code)
        gl.glCompileShader(fs)
        if not gl.glGetShaderiv(fs, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(fs).decode())

        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glLinkProgram(prog)
        if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
            raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())

        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        return prog

    def _get_program_for_mode(self, mode: str) -> int:
        if mode != self.current_mode or mode not in self.programs:
            if mode in self.programs:
                gl.glDeleteProgram(self.programs[mode])
            prog = self._load_fragment_shader(mode)
            self.programs[mode] = prog
            self.current_mode = mode
        return self.programs[mode]

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        state = self.control_state.get_state()

        h, w = frame_bgr.shape[:2]
        if (w, h) != (self.width, self.height):
            frame_bgr = cv2.resize(frame_bgr, (self.width, self.height))

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = np.flipud(frame_rgb)

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self.fbo)
        gl.glViewport(0, 0, self.width, self.height)

        tex_in = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_in)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGB,
            self.width, self.height, 0,
            gl.GL_RGB, gl.GL_UNSIGNED_BYTE, frame_rgb
        )
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        prog = self._get_program_for_mode(state.shader_mode)
        gl.glUseProgram(prog)

        # Set uniforms
        time_val = float(glfw.get_time())
        u_time = gl.glGetUniformLocation(prog, "u_time")
        if u_time != -1:
            gl.glUniform1f(u_time, time_val)

        u_intensity = gl.glGetUniformLocation(prog, "u_intensity")
        if u_intensity != -1:
            gl.glUniform1f(u_intensity, state.distortion_intensity)

        u_audio = gl.glGetUniformLocation(prog, "u_audio")
        if u_audio != -1:
            gl.glUniform1f(u_audio, state.audio_energy)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, tex_in)
        u_tex = gl.glGetUniformLocation(prog, "u_tex")
        if u_tex != -1:
            gl.glUniform1i(u_tex, 0)

        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

        out_rgb = gl.glReadPixels(
            0, 0, self.width, self.height,
            gl.GL_RGB, gl.GL_UNSIGNED_BYTE
        )
        out_rgb = np.frombuffer(out_rgb, dtype=np.uint8).reshape(self.height, self.width, 3)
        out_rgb = np.flipud(out_rgb)

        gl.glDeleteTextures([tex_in])
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        return out_bgr

    def __del__(self):
        try:
            for prog in self.programs.values():
                gl.glDeleteProgram(prog)
            gl.glDeleteTextures([self.color_tex])
            gl.glDeleteFramebuffers(1, [self.fbo])
            glfw.terminate()
        except Exception:
            pass
