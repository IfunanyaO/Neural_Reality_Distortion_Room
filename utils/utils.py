import os
import time
import uuid
import cv2
import psutil
import numpy as np
from datetime import datetime

# ============================================================
#   SAFE PATH GENERATION
# ============================================================

def ensure_dir(path: str):
    """Create directory if it does not exist."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"[utils] Failed to create directory: {path} | Error: {e}")
    return path


def session_root():
    """Return the root folder for sessions."""
    root = os.path.join(os.getcwd(), "sessions")
    return ensure_dir(root)


def new_session_folder():
    """Create a unique session folder with timestamp + UUID."""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join(session_root(), f"session_{ts}_{uuid.uuid4().hex[:6]}")
    return ensure_dir(folder)


# ============================================================
#   FILENAME HELPERS
# ============================================================

def timestamp_name(prefix="file", ext="png"):
    """Generate a unique filename with timestamp."""
    ts = datetime.now().strftime("%H-%M-%S")
    return f"{prefix}_{ts}_{uuid.uuid4().hex[:4]}.{ext}"


def numbered_file(prefix, number, ext="png"):
    """Generate sequential filenames."""
    return f"{prefix}_{number:05d}.{ext}"


# ============================================================
#   TIMER / PERFORMANCE MONITOR
# ============================================================

class FPSTimer:
    """Tracks FPS and returns smoothed values."""

    def __init__(self):
        self.last = time.time()
        self.fps = 0.0

    def tick(self):
        now = time.time()
        dt = now - self.last
        self.last = now
        if dt > 0:
            self.fps = (self.fps * 0.8) + (0.2 * (1.0 / dt))
        return self.fps


class SystemStats:
    """Tracks CPU & RAM for HUD overlay."""

    def __init__(self):
        self.cpu = 0.0
        self.ram = 0.0
        self.last_update = 0

    def update(self):
        now = time.time()
        if now - self.last_update > 0.5:
            self.last_update = now
            self.cpu = psutil.cpu_percent()
            self.ram = psutil.virtual_memory().percent
        return self.cpu, self.ram


# ============================================================
#   EFFECT INTENSITY SLIDER (*)
# ============================================================

class EffectIntensity:
    """Stores global effect intensity."""

    def __init__(self):
        self.value = 1.0   # 1.0 = 100%

    def increase(self):
        self.value = min(3.0, self.value + 0.1)
        print(f"[Intensity] = {self.value:.2f}")

    def decrease(self):
        self.value = max(0.2, self.value - 0.1)
        print(f"[Intensity] = {self.value:.2f}")


# ============================================================
#   ON-SCREEN MENU STATE ( ()
# ============================================================

class OnScreenMenu:
    """Toggles the big menu overlay."""
    def __init__(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled
        print("[MenuOverlay] Enabled =", self.enabled)
        return self.enabled


# ============================================================
#   AUTO FACE ZOOM ( ))
# ============================================================

class AutoFaceZoom:
    """Automatically zooms into face using smooth interpolation."""

    def __init__(self):
        self.enabled = False
        self.zoom_factor = 1.0
        self.target_zoom = 1.0

    def toggle(self):
        self.enabled = not self.enabled
        print("[FaceZoom] Enabled =", self.enabled)
        return self.enabled

    def update(self, face_box):
        if not self.enabled or face_box is None:
            self.target_zoom = 1.0
        else:
            (x, y, w, h) = face_box
            if w > 60:
                self.target_zoom = min(2.5, 1.3 + (h / 300))
            else:
                self.target_zoom = 1.1

        # interpolate
        self.zoom_factor = (self.zoom_factor * 0.85) + (self.target_zoom * 0.15)
        return self.zoom_factor


# ============================================================
#   LIVE EFFECT THUMBNAIL GRID (+)
# ============================================================

class ThumbnailGrid:
    """
    Stores thumbnails of modes (to show grid preview).
    """

    def __init__(self, mode_list):
        self.enabled = False
        self.modes = mode_list
        self.thumb_size = (120, 70)
        self.thumbs = {}

    def toggle(self):
        self.enabled = not self.enabled
        print("[ThumbGrid] Enabled =", self.enabled)
        return self.enabled

    def update(self, mode_key, frame):
        """Store frame thumbnail for active mode."""
        if not self.enabled:
            return

        small = cv2.resize(frame, self.thumb_size)
        self.thumbs[mode_key] = small

    def render(self, canvas):
        """Draw grid onto output canvas."""
        if not self.enabled:
            return canvas

        x = 10
        y = 10

        for mk in self.modes:
            if mk in self.thumbs:
                t = self.thumbs[mk]
                h, w = t.shape[:2]
                canvas[y:y+h, x:x+w] = t
                x += w + 8
                if x + w + 10 > canvas.shape[1]:
                    x = 10
                    y += h + 8

        return canvas


# ============================================================
#   CAROUSEL ENGINE (%)
# ============================================================

class EffectCarousel:
    """
    Handles effect preview cycling.
    """

    def __init__(self, mode_list):
        self.modes = mode_list
        self.index = 0
        self.active = False

    def next(self):
        self.index = (self.index + 1) % len(self.modes)
        return self.modes[self.index]

    def current(self):
        return self.modes[self.index]

    def toggle(self):
        self.active = not self.active
        return self.active


# ============================================================
#   AUTO-CYCLE ENGINE (^)
# ============================================================

class AutoCycle:
    """Automatically cycles effects every N seconds."""

    def __init__(self, mode_list, interval=4.0):
        self.modes = mode_list
        self.interval = interval
        self.last_switch = time.time()
        self.enabled = False
        self.index = 0

    def toggle(self):
        self.enabled = not self.enabled
        self.last_switch = time.time()
        return self.enabled

    def update(self):
        if not self.enabled:
            return None

        now = time.time()
        if now - self.last_switch >= self.interval:
            self.last_switch = now
            self.index = (self.index + 1) % len(self.modes)
            return self.modes[self.index]

        return None


# ============================================================
#   PRESET MANAGER ( { & } )
# ============================================================

class PresetState:
    """Holds mode + intensity for saving/loading."""
    
    def __init__(self):
        self.mode = "1"
        self.intensity = 1.0

    def update(self, mode, intensity):
        self.mode = mode
        self.intensity = intensity

    def package(self):
        return {"mode": self.mode, "intensity": self.intensity}

    def apply(self, preset_dict):
        self.mode = preset_dict.get("mode", "1")
        self.intensity = preset_dict.get("intensity", 1.0)


# ============================================================
#   VOICE COMMAND FLAG (|)
# ============================================================

class VoiceFlag:
    """Enables or disables voice recognition."""

    def __init__(self):
        self.enabled = False

    def toggle(self):
        self.enabled = not self.enabled
        print("[Voice] Enabled =", self.enabled)
        return self.enabled
