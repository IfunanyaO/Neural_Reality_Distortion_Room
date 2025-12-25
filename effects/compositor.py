# effects/compositor.py

import time
import cv2
import numpy as np
from typing import Optional

from interaction.control_state import ControlState
from effects.sd_image_filter import StableDiffusionImageFilter


class Compositor:
    def __init__(self, control_state: ControlState, use_sd: bool = True):
        self.control_state = control_state
        self.use_sd = use_sd
        self.sd_filter: Optional[StableDiffusionImageFilter] = None
        if use_sd:
            self.sd_filter = StableDiffusionImageFilter()
        self.last_sd_time = 0.0

    def apply(self, base_frame_bgr: np.ndarray) -> np.ndarray:
        """
        base_frame_bgr: image from Gaussian rendering (BGR)
        Returns final stylized/distorted frame (BGR).
        """
        state = self.control_state.get_state()
        out = base_frame_bgr.copy()

        # 1. Shader-type distortions (you’ll run those via GPU/OpenGL;
        # here we just leave a placeholder)
        # Example: apply simple CV2 warp as fallback
        if state.distortion_intensity > 0.05:
            out = self._simple_wave_warp(out, intensity=state.distortion_intensity)

        # 2. Occasional SD stylization (expensive)
        if self.use_sd and self.sd_filter is not None:
            now = time.time()
            # sd_frequency = expected fraction of seconds you want SD to run
            if now - self.last_sd_time > (1.0 / max(state.sd_frequency * 10.0, 0.1)):
                small = cv2.resize(out, (512, 512))
                try:
                    stylized = self.sd_filter.stylize(
                        small,
                        prompt=state.style_prompt,
                        strength=0.5 + 0.4 * state.distortion_intensity,
                        guidance_scale=7.5,
                        num_inference_steps=20,
                    )
                    out = cv2.resize(stylized, (out.shape[1], out.shape[0]))
                    self.last_sd_time = now
                except Exception as e:
                    print("[Compositor] SD error:", e)

        return out

    def _simple_wave_warp(self, img: np.ndarray, intensity: float) -> np.ndarray:
        """Fallback CPU warp so it still ‘moves’ without full GLSL pipeline."""
        h, w = img.shape[:2]
        dst = np.zeros_like(img)
        for y in range(h):
            shift = int(np.sin(y / 20.0 + time.time()) * 10 * intensity)
            dst[y] = np.roll(img[y], shift, axis=0)
        return dst
