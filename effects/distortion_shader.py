import numpy as np
import cv2

def _swirl_map(h, w, intensity: float):
    ys, xs = np.indices((h, w), dtype=np.float32)
    cx, cy = w / 2.0, h / 2.0
    x = xs - cx
    y = ys - cy
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    max_r = np.max(r) + 1e-6
    k = intensity * 3.0
    theta2 = theta + k * (1.0 - (r / max_r))

    xs2 = cx + r * np.cos(theta2)
    ys2 = cy + r * np.sin(theta2)
    return xs2.astype(np.float32), ys2.astype(np.float32)

def _ripple_map(h, w, intensity: float):
    ys, xs = np.indices((h, w), dtype=np.float32)
    cx, cy = w / 2.0, h / 2.0
    x = xs - cx
    y = ys - cy
    r = np.sqrt(x * x + y * y)
    k = 12.0 * intensity
    ripple = np.sin(r * 0.05 * k) * 5.0 * intensity

    xs2 = xs + (x / (r + 1e-6)) * ripple
    ys2 = ys + (y / (r + 1e-6)) * ripple
    return xs2.astype(np.float32), ys2.astype(np.float32)

def apply_distortion(
    frame_bgr,
    intensity: float = 0.3,
    mode: str = "swirl",
    audio_energy: float = 0.0,
):
    """
    CPU-based fallback distortion using cv2.remap.

    Args:
        frame_bgr: HxWx3 uint8 image
        intensity: 0–1
        mode: "swirl" | "ripple"
        audio_energy: optional modifier
    """
    if intensity <= 0.0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    intensity = float(max(0.0, min(1.0, intensity + audio_energy * 0.5)))

    if mode == "ripple":
        map_x, map_y = _ripple_map(h, w, intensity)
    else:
        map_x, map_y = _swirl_map(h, w, intensity)

    distorted = cv2.remap(
        frame_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return distorted
