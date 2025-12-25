import numpy as np
import torch
from PIL import Image
from typing import Optional
import time

from diffusers import StableDiffusionImg2ImgPipeline

class StableDiffusionStylizer:
    """
    Ultra-stable, low-memory SD stylizer using SSD-1B.
    Lazy-loads the model to avoid startup crashes.
    """

    def __init__(self, model_name="segmind/SSD-1B", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe: Optional[StableDiffusionImg2ImgPipeline] = None
        self.last_load_time = 0

        # Low RAM settings
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()

    def _ensure_loaded(self):
        """Lazy load SD the first time it's needed."""
        if self.pipe is not None:
            return

        print("[SD] Loading SSD-1B… (this happens only once)")

        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            safety_checker=None,
        )

        # Move to device
        self.pipe = self.pipe.to(self.device)

        # Memory optimizations
        try:
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
            self.pipe.enable_sequential_cpu_offload()
        except:
            pass

        torch.cuda.empty_cache()
        print("[SD] SSD-1B loaded successfully.")

    def _bgr_to_pil(self, frame_bgr: np.ndarray) -> Image.Image:
        rgb = frame_bgr[:, :, ::-1]
        return Image.fromarray(rgb)

    def _pil_to_bgr(self, img: Image.Image) -> np.ndarray:
        arr = np.array(img)
        return arr[:, :, ::-1].astype(np.uint8)

    @torch.inference_mode()
    def stylize(self, frame_bgr: np.ndarray, prompt: str, strength: float = 0.55) -> np.ndarray:
        self._ensure_loaded()

        init_image = self._bgr_to_pil(frame_bgr)

        result = self.pipe(
            prompt=prompt,
            image=init_image,
            strength=float(strength),
            guidance_scale=3.0,
            num_inference_steps=12,
        )

        out = result.images[0]
        return self._pil_to_bgr(out)
