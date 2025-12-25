# effects/sd_image_filter.py

import torch
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


class StableDiffusionImageFilter:
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to(device)
        self.device = device

    def _np_to_pil(self, img: np.ndarray) -> Image.Image:
        # assume BGR from OpenCV, convert to RGB
        if img.shape[2] == 3:
            img_rgb = img[:, :, ::-1]
        else:
            img_rgb = img
        return Image.fromarray(img_rgb)

    def _pil_to_np(self, img: Image.Image) -> np.ndarray:
        arr = np.array(img)
        # return BGR for OpenCV compatibility
        return arr[:, :, ::-1]

    def stylize(
        self,
        frame_bgr: np.ndarray,
        prompt: str = "melting chrome surreal XR world",
        strength: float = 0.5,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30
    ) -> np.ndarray:
        """
        Apply SD img2img style. Use lower resolution or strength for speed.
        """
        pil_img = self._np_to_pil(frame_bgr)

        result = self.pipe(
            prompt=prompt,
            image=pil_img,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).images[0]

        return self._pil_to_np(result)


if __name__ == "__main__":
    # quick sanity test
    import cv2
    sd_filter = StableDiffusionImageFilter()

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, (512, 512))
        stylized = sd_filter.stylize(small, prompt="liquid glass cyberpunk dreamscape", strength=0.6)
        cv2.imshow("original", small)
        cv2.imshow("stylized", stylized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
