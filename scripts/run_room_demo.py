import random
import cv2
import numpy as np
import time
import os
import imageio
import json
import psutil
import math
import threading
import random
from collections import deque
from datetime import datetime

# ---- External utilities ----
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)
from utils.capture import SessionManager
from utils.utils import EffectCarousel, AutoCycle, FPSTimer
from utils.utils import numbered_file


gif_buffer = deque(maxlen=120)   # 4 seconds @ 30fps
echo_buffer = deque(maxlen=12)  # number of ghost clones


# global toggle
system_stats_enabled = False

# --- FPS + system overlay (_ key) ---
fps_overlay_enabled = False

# --- Thumbnail Grid (+ key) ---
#thumbnail_grid_enabled = False
thumbnail_scale = 0.18
show_thumbnail_grid = False

# --- Presets { } Keys ---
preset_save_pending = False
preset_load_pending = False
preset_file = os.path.join(os.getcwd(), "effect_preset.json")

# global slider
show_intensity_slider = False
effect_intensity = 0.5     # range: 0.0 - 1.0

# --- On-screen menu overlay (( key) ---
show_menu_overlay = False

# --- Auto Face Zoom () key) ---
auto_face_zoom = False
face_zoom_level = 1.0       # current smooth zoom
target_zoom = 1.0           # what we want to zoom to
audio_energy = 0.0
audio_delta = 0.0

# ===============================
# CLAP / IMPACT DETECTION (MODE 0)
# ===============================

prev_gray_clap = None
clap_energy = 0.0
CLAP_THRESHOLD = 22.0     # sensitivity (15–30)
CLAP_DECAY = 0.85
clap_burst_frames = 0


CLICK_GRID = False
GRID_X, GRID_Y = 10, 10  # corner where the grid is drawn
GRID_PER_ROW = 8
THUMB_W, THUMB_H = 160, 90
LABEL_H = 20
CELL_H = THUMB_H + LABEL_H

# --- Voice Commands ( | key ) ---
voice_commands_enabled = False
voice_last = ""  # store last voice result

try:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    HAS_VOICE = True
except ImportError:
    HAS_VOICE = False
    recognizer = None
    mic = None
    print("[Voice] speech_recognition not installed; voice commands disabled.")
# -------- Optional: audio input for reactive modes --------
try:
    import sounddevice as sd
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False
    sd = None
    print("[Audio] sounddevice not installed; audio-reactive modes will use a fake animation.\n"
          "        To enable real microphone reactivity: pip install sounddevice")

# -------- Optional: hand tracking + pose with MediaPipe --------
HAS_HANDS = False
HAS_POSE = False
mp = None
mp_hands = None
mp_pose = None
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    HAS_HANDS = True
    HAS_POSE = True
except ImportError:
    print("[MediaPipe] mediapipe not installed; E/L/G/I/V/X/Q will use fallbacks.\n"
          "           To enable hand + pose tracking: pip install mediapipe")

HAS_FACE_MESH = False
mp_face_mesh = None
try:
    mp_face_mesh = mp.solutions.face_mesh
    HAS_FACE_MESH = True
except:
    print("[MediaPipe] Face Mesh not available.")


# -------- Optional: CUDA GPU via OpenCV --------
HAS_CUDA = False
try:
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    if cuda_count > 0:
        HAS_CUDA = True
        print(f"[CUDA] OpenCV CUDA detected, {cuda_count} device(s) available.")
    else:
        print("[CUDA] No CUDA-capable devices reported by OpenCV.")
except Exception as e:
    print("[CUDA] CUDA not available in this OpenCV build:", e)


def try_open_camera():
    """Open default camera using common Windows backends."""
    backends = [
        (cv2.CAP_DSHOW, "CAP_DSHOW"),
        (cv2.CAP_MSMF, "CAP_MSMF"),
        (cv2.CAP_VFW, "CAP_VFW"),
        (0, "AUTO"),
    ]

    for backend, name in backends:
        print(f"[Camera] Trying {name}")
        cap = cv2.VideoCapture(0, backend)
        if cap.isOpened():
            print(f"[Camera] SUCCESS using {name}")
            return cap

    raise RuntimeError("Could not open camera on any backend.")

# FACE DETECTOR INITIALIZATION
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def draw_system_stats(frame, fps, audio_level, mode_name):
    """Overlay CPU, RAM, FPS, resolution, audio level, etc."""
    try:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
    except:
        cpu, ram = -1, -1  # psutil unavailable fallback

    h, w, _ = frame.shape

    stats = [
        f"FPS: {fps:.1f}",
        f"CPU: {cpu if cpu>=0 else 'N/A'}%",
        f"RAM: {ram if ram>=0 else 'N/A'}%",
        f"Resolution: {w}x{h}",
        f"Audio: {audio_level:.2f}",
        f"Mode: {mode_name}",
    ]

    y = 30
    for text in stats:
        cv2.putText(
            frame, text, (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0), 2,
            cv2.LINE_AA,
        )
        y += 28

    return frame

def draw_menu_overlay(frame, mode, mode_names, effect_intensity,
                      audio_level, fps_value,
                      recording, session_path,
                      show_hud, auto_on, carousel_on):
    """
    Draws a premium vertical translucent menu overlay on left side.
    """
    h, w, _ = frame.shape

    # Panel dimensions
    panel_w = int(w * 0.32)
    panel_h = int(h * 0.92)
    x0 = 10
    y0 = 10

    # Background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + panel_w, y0 + panel_h),
        (15, 15, 15),
        -1
    )
    frame[:] = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    # Title
    cv2.putText(frame, "NEURAL REALITY MENU", (x0 + 10, y0 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    y = y0 + 80

    # Current Mode
    cv2.putText(frame, f"Mode: {mode} - {mode_names.get(mode,'')}", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 35

    # Intensity
    cv2.putText(frame, f"Intensity: {effect_intensity:.2f}", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 1)
    y += 35

    # Audio Level
    cv2.putText(frame, f"Audio Level: {audio_level:.2f}", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    y += 35

    # FPS
    cv2.putText(frame, f"FPS: {fps_value:.1f}", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 255), 1)
    y += 35

    # Recording
    if recording:
        cv2.putText(frame, "● Recording...", (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Recording: OFF", (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 35

    # HUD
    cv2.putText(frame, f"HUD: {'ON' if show_hud else 'OFF'}", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 255), 1)
    y += 35

    # Auto Cycle
    cv2.putText(frame, f"Auto Cycle: {'ON' if auto_on else 'OFF'}", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 200), 1)
    y += 35

    # Carousel
    cv2.putText(frame, f"Carousel: {'ON' if carousel_on else 'OFF'}", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)
    y += 35

    # Session Path
    cv2.putText(frame, "Session Path:", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1)
    y += 30

    cv2.putText(frame, session_path[-30:], (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 35

    # Controls Help
    cv2.putText(frame, "Controls:", (x0 + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 1)
    y += 35

    help_lines = [
        "1-9,0,A-Z,/  Effects",
        "@ Photo    # Video    $ Export",
        "* Slider   ( Menu    , HUD",
        "% Carousel   ^ Auto Cycle",
        "] Intensity +    [ Intensity -",
        ". Quit     ESC Quit",
    ]

    for line in help_lines:
        cv2.putText(frame, line, (x0 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        y += 28

def handle_voice_commands(session, frame):
    global voice_last
    if not HAS_VOICE:
        return

    try:
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.1)
            audio = recognizer.listen(source, timeout=0.1, phrase_time_limit=1)
        text = recognizer.recognize_google(audio).lower()
        voice_last = text

        if "photo" in text:
            session.capture_photo(frame)
        elif "record" in text:
            session.toggle_video(frame.shape[1], frame.shape[0], fps=30)
        elif "next" in text:
            return "next"
        elif "previous" in text or "back" in text:
            return "prev"
    except Exception:
        pass

    return None

def apply_audio_wave(frame, t, audio_level, coords):
    strength = 8 + 40 * float(audio_level)
    wave = apply_wave(frame, t * 1.5, strength, coords)

    factor = 0.6 + 1.2 * float(audio_level)
    return np.clip(wave.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def apply_auto_face_zoom(frame, face_box, zoom_level):
    """
    Smooth cinematic zoom centered on face. 
    zoom_level: 1.0 = no zoom, higher = zoom in.
    """
    if zoom_level <= 1.01:
        return frame

    h, w, _ = frame.shape
    zoom = zoom_level

    # Determine zoom center
    if face_box is not None:
        x, y, fw, fh = face_box
        cx = x + fw // 2
        cy = y + fh // 2
    else:
        # fallback center of frame
        cx, cy = w // 2, h // 2

    # Compute crop box
    crop_w = int(w / zoom)
    crop_h = int(h / zoom)

    x1 = int(cx - crop_w // 2)
    y1 = int(cy - crop_h // 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Clamp boundaries
    x1 = max(0, min(x1, w - crop_w))
    y1 = max(0, min(y1, h - crop_h))
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Crop + upscale
    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return zoomed

def precompute_coords(width, height):
    """Precompute coordinate grids for warp-based effects."""
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    cx = width / 2.0
    cy = height / 2.0
    x = xx - cx
    y = yy - cy
    r = np.sqrt(x * x + y * y).astype(np.float32)
    theta = np.arctan2(y, x).astype(np.float32)
    r_max = np.max(r) + 1e-6
    return {
        "xx": xx,
        "yy": yy,
        "cx": cx,
        "cy": cy,
        "x": x,
        "y": y,
        "r": r,
        "theta": theta,
        "r_max": r_max,
    }


# ===================== CORE EFFECTS =====================

def apply_wave(frame, t, strength=10.0, coords=None):
    """Mode 1 / 6: wavy distortion."""
    h, w, _ = frame.shape
    if coords is None:
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        yy = yy.astype(np.float32)
        xx = xx.astype(np.float32)
    else:
        yy = coords["yy"]
        xx = coords["xx"]

    wave_x = (strength * np.sin(2 * np.pi * yy / 80.0 + t)).astype(np.float32)
    wave_y = (strength * np.cos(2 * np.pi * xx / 80.0 + t * 0.8)).astype(np.float32)

    map_x = (xx + wave_x).astype(np.float32)
    map_y = (yy + wave_y).astype(np.float32)

    distorted = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return distorted


def apply_swirl(frame, t, coords):
    """Mode 2: swirl around center."""
    xx = coords["xx"]
    yy = coords["yy"]
    x = coords["x"]
    y = coords["y"]
    r = coords["r"]
    theta = coords["theta"]
    r_max = coords["r_max"]

    strength = 2.0 + 1.0 * np.sin(t * 0.5)
    swirl = strength * np.exp(-r / (0.7 * r_max))
    theta2 = theta + swirl

    x2 = r * np.cos(theta2) + coords["cx"]
    y2 = r * np.sin(theta2) + coords["cy"]

    map_x = x2.astype(np.float32)
    map_y = y2.astype(np.float32)

    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_neon_edges(frame):
    """Mode 3: neon cyberpunk edges."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edges = cv2.dilate(edges, None, iterations=1)
    edges_col = cv2.applyColorMap(edges, cv2.COLORMAP_HOT)
    combined = cv2.addWeighted(frame, 0.4, edges_col, 0.6, 0)
    return combined


def apply_glitch(frame, t):
    """Mode 4: glitch effect with channel shifts and block jumps."""
    h, w, _ = frame.shape
    glitched = frame.copy()

    shift = int(5 + 10 * (0.5 + 0.5 * np.sin(t * 3)))
    M = np.float32([[1, 0, shift], [0, 1, 0]])
    b, g, r = cv2.split(glitched)
    r_shifted = cv2.warpAffine(r, M, (w, h), borderMode=cv2.BORDER_WRAP)
    g_shifted = cv2.warpAffine(g, -M, (w, h), borderMode=cv2.BORDER_WRAP)
    glitched = cv2.merge([b, g_shifted, r_shifted])

    strip_h = h // 10
    y0 = int((h - strip_h) * (0.5 + 0.5 * np.sin(t * 4))) % (h - strip_h)
    strip = glitched[y0:y0 + strip_h].copy()
    offset = int(20 * np.sin(t * 6))
    glitched[y0:y0 + strip_h] = np.roll(strip, offset, axis=1)

    return glitched

def detect_clap_motion(frame):
    """
    CPU-safe clap / impact detector using frame differencing.
    Returns True when a strong motion spike is detected.
    """
    global prev_gray_clap, clap_energy

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_gray_clap is None:
        prev_gray_clap = gray
        return False

    diff = cv2.absdiff(gray, prev_gray_clap)
    motion = np.mean(diff)

    # exponential smoothing
    clap_energy = clap_energy * CLAP_DECAY + motion * (1 - CLAP_DECAY)
    prev_gray_clap = gray

    return clap_energy > CLAP_THRESHOLD


# def generate_thumbnail_grid(frame, t, coords, audio_level):
    """Generate a 2x3 live preview grid of different effects."""
   #thumb_w = w // 3
    #thumb_h = h // 2

    # Pick 6 effects to preview
    #preview_modes = [
        #apply_wave,
        #apply_swirl,
        #apply_neon_edges,
        #apply_glitch,
        #apply_kaleidoscope,
        #apply_rgb_split
    #]

    #thumbnails = []

    for i, fx in enumerate(preview_modes):
        # Apply the effect safely
        try:
            if fx == apply_swirl:
                thumb = fx(frame, t, coords)
            elif fx == apply_wave:
                thumb = fx(frame, t, strength=10.0, coords=coords)
            elif fx == apply_rgb_split:
                thumb = fx(frame, t)
            else:
                thumb = fx(frame)
        except:
            thumb = frame.copy()  # fallback if effect errors

        thumb = cv2.resize(thumb, (thumb_w, thumb_h))
        thumbnails.append(thumb)

    # Build final grid
    top = np.hstack(thumbnails[:3])
    bottom = np.hstack(thumbnails[3:])
    grid = np.vstack([top, bottom])

    return grid

def apply_kaleidoscope(frame):
    """Mode 5 / part of O: kaleidoscope via mirroring."""
    h, w, _ = frame.shape
    half_w = w // 2
    left = frame[:, :half_w]
    right = cv2.flip(left, 1)
    k1 = np.hstack([left, right])

    half_h = h // 2
    top = k1[:half_h, :]
    bottom = cv2.flip(top, 0)
    k2 = np.vstack([top, bottom])
    k2 = cv2.resize(k2, (w, h))
    return k2


def apply_face_spotlight(frame, face_box, coords):
    """Mode 8 / W base: face spotlight + gentle warp."""
    h, w, _ = frame.shape
    yy, xx = coords["yy"], coords["xx"]

    if face_box is not None:
        x, y, fw, fh = face_box
        cx = x + fw / 2
        cy = y + fh / 2
    else:
        cx, cy = coords["cx"], coords["cy"]

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = max(w, h) * 0.4
    mask = np.clip(1.0 - (dist / radius), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=15, sigmaY=15)

    warp_strength = 0.25
    dx = (xx - cx) * (mask * warp_strength)
    dy = (yy - cy) * (mask * warp_strength)

    map_x = (xx - dx).astype(np.float32)
    map_y = (yy - dy).astype(np.float32)

    warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    mask_3 = cv2.merge([mask, mask, mask])
    background = (warped * 0.2).astype(np.uint8)
    spotlight = (warped * (0.7 + 0.3 * mask_3)).astype(np.uint8)
    out = (background * (1 - mask_3) + spotlight * mask_3).astype(np.uint8)

    return out


def apply_ghost_trails(frame, prev_trail, alpha=0.75):
    """Mode A: ghosting trails via exponential moving average."""
    if prev_trail is None:
        return frame.copy(), frame.copy()
    trail = cv2.addWeighted(frame, 1 - alpha, prev_trail, alpha, 0)
    return trail, trail


def apply_rgb_split(frame, t):
    """Mode B: RGB split warp."""
    h, w, _ = frame.shape
    b, g, r = cv2.split(frame)
    shift = int(5 + 10 * (0.5 + 0.5 * np.sin(t * 2)))

    M_right = np.float32([[1, 0, shift], [0, 1, 0]])
    M_left = np.float32([[1, 0, -shift], [0, 1, 0]])

    r_shifted = cv2.warpAffine(r, M_right, (w, h), borderMode=cv2.BORDER_REFLECT)
    b_shifted = cv2.warpAffine(b, M_left, (w, h), borderMode=cv2.BORDER_REFLECT)
    merged = cv2.merge([b_shifted, g, r_shifted])
    return merged

def draw_intensity_slider(frame, value):
    """
    Draws a premium UI horizontal slider on screen.
    value: 0.0 - 1.0
    """
    h, w, _ = frame.shape
    bar_w = int(w * 0.6)
    bar_h = 12
    x = int(w * 0.2)
    y = int(h * 0.88)

    # Background bar
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (60, 60, 60), -1)

    # Filled part
    fill = int(bar_w * value)
    cv2.rectangle(frame, (x, y), (x + fill, y + bar_h), (0, 255, 100), -1)

    # Border
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (255, 255, 255), 2)

    cv2.putText(frame,
                f"Intensity: {value:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2)


def apply_chromatic_aberration(frame, coords, amount=7.0):
    """Mode C: radial chromatic aberration (more intense)."""
    b, g, r = cv2.split(frame)

    xx = coords["xx"]
    yy = coords["yy"]
    cx = coords["cx"]
    cy = coords["cy"]
    x = xx - cx
    y = yy - cy
    rdist = np.sqrt(x * x + y * y)
    norm = rdist / (coords["r_max"] + 1e-6)

    shift_x = (x / (coords["r_max"] + 1e-6)) * norm * amount
    shift_y = (y / (coords["r_max"] + 1e-6)) * norm * amount

    def shift_channel(ch, sx_scale, sy_scale):
        map_x = (xx + shift_x * sx_scale).astype(np.float32)
        map_y = (yy + shift_y * sy_scale).astype(np.float32)
        return cv2.remap(ch, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    r2 = shift_channel(r, 1.8, 1.8)
    b2 = shift_channel(b, -1.8, -1.8)
    g2 = g

    merged = cv2.merge([b2, g2, r2])
    return merged


def apply_neural_edges(frame):
    """Mode D: 'neural' edge outlines."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    edges_col = cv2.applyColorMap(edges, cv2.COLORMAP_TURBO)
    out = cv2.addWeighted(frame, 0.6, edges_col, 0.8, 0)
    return out


def apply_music_visualizer(frame, t, audio_level, coords):
    """Mode 9: music-reactive visualizer."""
    base = apply_wave(frame, t * 1.5, strength=8 + 40 * audio_level, coords=coords)
    base = apply_kaleidoscope(base)

    hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue_shift = (audio_level * 90) * np.sin(t * 2)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (0.6 + audio_level * 0.8), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.7 + audio_level * 0.9), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out

def detect_clap_motion(prev_gray, gray, threshold=35_000):
    """
    Detect sudden motion spikes (clap / fast hand impact).
    CPU-safe, no ML.
    """
    if prev_gray is None:
        return False

    diff = cv2.absdiff(prev_gray, gray)
    motion_energy = np.sum(diff)

    return motion_energy > threshold

def apply_glitch_burst(frame, t):
    """Used for clap-triggered burst glitch (mode 0 events)."""
    noisy = apply_glitch(frame, t * 3.0)
    noise = np.random.randint(0, 60, frame.shape, dtype=np.uint8)
    out = cv2.addWeighted(noisy, 0.8, noise, 0.7, 0)
    return out


# -------- E = Hand-tracking based distortion --------
def apply_hand_distortion(frame, hand_landmarks, coords, t):
    """Mode E: hand-tracking based distortion."""
    h, w, _ = frame.shape
    yy, xx = coords["yy"], coords["xx"]

    if hand_landmarks is not None:
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        cx = float(sum(xs) / len(xs))
        cy = float(sum(ys) / len(ys))
    else:
        cx, cy = coords["cx"], coords["cy"]

    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    radius = max(w, h) * 0.45
    mask = np.clip(1.0 - (dist / radius), 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=10, sigmaY=10)

    swirl_strength = 3.0 + 2.0 * np.sin(t * 2.0)

    x0 = xx - cx
    y0 = yy - cy
    rr = np.sqrt(x0 * x0 + y0 * y0) + 1e-6
    theta = np.arctan2(y0, x0)

    swirl = swirl_strength * mask * np.exp(-rr / (radius * 1.2))
    theta2 = theta + swirl

    x2 = rr * np.cos(theta2) + cx
    y2 = rr * np.sin(theta2) + cy

    map_x = x2.astype(np.float32)
    map_y = y2.astype(np.float32)

    warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    mask3 = cv2.merge([mask, mask, mask])
    highlight = (warped * (0.8 + 0.4 * mask3)).astype(np.uint8)
    out = np.clip(highlight, 0, 255).astype(np.uint8)
    return out


# -------- F = GPU shader acceleration with CUDA --------
def apply_gpu_shader(frame, t, audio_level, coords):
    """Mode F: GPU shader acceleration with CUDA (if available)."""
    if not HAS_CUDA:
        cpu = apply_wave(frame, t, strength=10 + 15 * audio_level, coords=coords)
        cpu = apply_neon_edges(cpu)
        return cpu

    try:
        gpu_src = cv2.cuda_GpuMat()
        gpu_src.upload(frame)

        ksize = (0, 0)
        sigma = 3.0 + 5.0 * audio_level
        blur_filter = cv2.cuda.createGaussianFilter(
            gpu_src.type(), gpu_src.type(), ksize, sigma, sigma
        )
        gpu_blur = blur_filter.apply(gpu_src)

        src_blur = gpu_blur.download()
        gray = cv2.cvtColor(src_blur, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 120)
        edges_col = cv2.applyColorMap(edges, cv2.COLORMAP_TURBO)

        gpu_edges = cv2.cuda_GpuMat()
        gpu_edges.upload(edges_col)
        gpu_mix = cv2.cuda.addWeighted(gpu_src, 0.7, gpu_edges, 0.5, 0)
        mixed = gpu_mix.download()

        hsv = cv2.cvtColor(mixed, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_shift = 40 * np.sin(t * 1.7)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (0.7 + audio_level * 0.8), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.8 + audio_level * 0.8), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return out
    except Exception as e:
        print("[CUDA] Error in GPU shader mode, falling back to CPU:", e)
        cpu = apply_wave(frame, t, strength=10 + 15 * audio_level, coords=coords)
        cpu = apply_neon_edges(cpu)
        return cpu


# ===================== NEW FANCY MODES =====================

def apply_face_melt(frame, coords, t):
    """Mode H: VERY visible face melt (strong vertical smear from center)."""
    h, w, _ = frame.shape
    xx, yy = coords["xx"], coords["yy"]
    cx, cy = coords["cx"], coords["cy"]

    dy = (yy - cy).astype(np.float32)
    amount = 0.6 + 0.4 * np.sin(t * 1.5)
    offset = dy * amount

    map_x = xx.astype(np.float32)
    map_y = yy + offset

    melted = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)

    hsv = cv2.cvtColor(melted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.5 + 0.5 * np.sin(t * 2)), 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def get_face_and_mouth_box(face_box, coords):
    """Approximate mouth region from face box."""
    if face_box is None:
        h_cx, h_cy = coords["cx"], coords["cy"]
        w_box = coords["r_max"] * 0.6
        h_box = coords["r_max"] * 0.4
        x = int(h_cx - w_box / 2)
        y = int(h_cy)
        return (x, y, int(w_box), int(h_box))
    x, y, w, h = face_box
    mx = x
    my = y + int(h * 0.6)
    mw = w
    mh = int(h * 0.4)
    return (mx, my, mw, mh)


def apply_fire_water_mode(frame, mouth_center, hand_landmarks, fire=True):
    """
    G/I: Fire or Water from mouth + hands.
    Fire: orange/yellow; Water: blue/cyan.
    """
    h, w, _ = frame.shape
    overlay = frame.copy()

    if fire:
        color_core = (0, 180, 255)
        color_outer = (0, 80, 255)
    else:
        color_core = (255, 200, 0)
        color_outer = (255, 100, 0)

    # Mouth stream
    if mouth_center is not None:
        mx, my = mouth_center
        length = int(h * 0.6)
        for i in range(1, 10):
            alpha = i / 10.0
            end_point = (mx + int((alpha - 0.5) * 40), my + int(alpha * length))
            cv2.line(overlay, (mx, my), end_point,
                     tuple(int(c * (1 - alpha)) for c in color_core), 4)

        cv2.circle(overlay, (mx, my), 16, color_outer, 2)
        cv2.circle(overlay, (mx, my), 8, color_core, -1)

    # Hand streams
    hand_points = []
    if hand_landmarks is not None:
        for lm in hand_landmarks.landmark:
            hand_points.append((int(lm.x * w), int(lm.y * h)))

    for px, py in hand_points[::3]:
        length = int(h * 0.5)
        for i in range(1, 8):
            alpha = i / 8.0
            end_point = (px + int((alpha - 0.5) * 60), py + int(alpha * length))
            cv2.line(overlay, (px, py), end_point,
                     tuple(int(c * (1 - alpha)) for c in color_core), 3)
        cv2.circle(overlay, (px, py), 10, color_outer, 2)
        cv2.circle(overlay, (px, py), 5, color_core, -1)

    return cv2.addWeighted(frame, 0.6, overlay, 0.9, 0)


def apply_volumetric_face_warp(frame, face_box, coords, t):
    """Mode X: Volumetric face warp — very visible '3D bulge' on face."""
    h, w, _ = frame.shape
    xx, yy = coords["xx"], coords["yy"]

    if face_box is not None:
        x, y, fw, fh = face_box
        cx = x + fw / 2
        cy = y + fh / 2
        radius = max(fw, fh) * 0.7
    else:
        cx, cy = coords["cx"], coords["cy"]
        radius = min(w, h) * 0.4

    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx * dx + dy * dy)

    r = radius
    inside = dist < r
    dist_norm = np.zeros_like(dist)
    dist_norm[inside] = dist[inside] / r

    bulge_strength = 1.0 + 0.3 * np.sin(t * 2.0)
    factor = 1.0 - bulge_strength * (dist_norm ** 2)

    dx2 = dx * factor
    dy2 = dy * factor

    map_x = (cx + dx2).astype(np.float32)
    map_y = (cy + dy2).astype(np.float32)

    warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)

    # Highlight region
    mask = np.zeros((h, w), np.float32)
    mask[inside] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), 15)
    mask3 = cv2.merge([mask, mask, mask])
    glow = (warped * (0.9 + 0.6 * mask3)).astype(np.uint8)
    return glow


def apply_ai_hallucination(frame, t, audio_level):
    """Mode J: AI hallucination — neon + glitch + hue cycling."""
    out = apply_neon_edges(frame)
    out = apply_glitch(out, t * 1.5)
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + (50 * np.sin(t * 1.3))) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.1 + 0.5 * audio_level), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (0.8 + 0.6 * audio_level), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_depth_warp(frame, coords, t):
    """Mode K: Depth Estimation Warp (brightness -> strong radial warp)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    depth = gray
    xx, yy = coords["xx"], coords["yy"]
    cx, cy = coords["cx"], coords["cy"]

    max_offset = 45.0
    offset = (depth - 0.5) * 2.0 * max_offset

    dir_x = (xx - cx)
    dir_y = (yy - cy)
    mag = np.sqrt(dir_x * dir_x + dir_y * dir_y) + 1e-6
    dir_x /= mag
    dir_y /= mag

    map_x = xx + dir_x * offset
    map_y = yy + dir_y * offset

    warped = cv2.remap(frame, map_x.astype(np.float32), map_y.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return warped


def apply_particle_emitter(frame, particles):
    """Mode L: Particle emitter from hands (particles list maintained externally)."""
    out = frame.copy()
    for p in particles:
        x, y, vx, vy, life, color = p
        radius = max(1, int(4 + 3 * life))
        cv2.circle(out, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    return out


def update_particles(particles, hand_points, w, h):
    """Update + spawn particles from hand points."""
    new_particles = []

    # Update existing
    for (x, y, vx, vy, life, color) in particles:
        x += vx
        y += vy
        life *= 0.92
        if 0 <= x < w and 0 <= y < h and life > 0.05:
            new_particles.append((x, y, vx, vy, life, color))

    # Spawn from hand points
    for (hx, hy) in hand_points:
        for _ in range(12):
            angle = np.random.uniform(0, 2 * np.pi)
            speed = np.random.uniform(2, 6)
            vx = np.cos(angle) * speed
            vy = np.sin(angle) * speed
            color = (np.random.randint(150, 255),
                     np.random.randint(150, 255),
                     np.random.randint(150, 255))
            new_particles.append((hx, hy, vx, vy, 1.0, color))

    return new_particles


def apply_heat_vision(frame):
    """Mode M: heat vision using thermal colormap."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    out = cv2.addWeighted(frame, 0.3, heat, 0.7, 0)
    return out


def apply_neural_glow(frame, t):
    """Mode N: neural glow aura around edges."""
    edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 40, 120)
    glow = cv2.GaussianBlur(edges, (0, 0), 7)
    glow_color = cv2.applyColorMap(glow, cv2.COLORMAP_PLASMA)
    factor = 0.6 + 0.4 * (0.5 + 0.5 * np.sin(t * 2))
    out = cv2.addWeighted(frame, 0.6, glow_color, factor, 0)
    return out


def apply_depth_kaleidoscope(frame, coords):
    """Mode O: depth-kaleidoscope (fake depth via brightness)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    depth_col = cv2.applyColorMap((gray * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    combined = cv2.addWeighted(frame, 0.5, depth_col, 0.7, 0)
    kale = apply_kaleidoscope(combined)
    return kale


def apply_liquid_mirror(frame, coords, t):
    """Mode P: liquid mirror (horizontal wave mirror)."""
    h, w, _ = frame.shape
    xx, yy = coords["xx"], coords["yy"]
    cy = coords["cy"]

    wave = 10.0 * np.sin(2 * np.pi * (xx / 80.0) + t * 2.5)
    map_x = xx.astype(np.float32)
    map_y = (2 * cy - yy + wave).astype(np.float32)

    mirrored = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    out = cv2.addWeighted(frame, 0.4, mirrored, 0.7, 0)
    return out


def apply_hologram_base(frame, t):
    """Base hologram effect used inside Q mode."""
    h, w, _ = frame.shape
    base = frame.copy()
    ghost = np.roll(base, int(5 * np.sin(t * 2)), axis=1)
    holo = cv2.addWeighted(base, 0.6, ghost, 0.6, 0)

    overlay = holo.copy()
    for y in range(0, h, 4):
        cv2.line(overlay, (0, y), (w, y), (0, 0, 0), 1)
    holo = cv2.addWeighted(holo, 0.8, overlay, 0.4, 0)

    b, g, r = cv2.split(holo)
    b = np.clip(b * 1.4, 0, 255).astype(np.uint8)
    g = np.clip(g * 1.1, 0, 255).astype(np.uint8)
    r = np.clip(r * 0.5, 0, 255).astype(np.uint8)
    holo = cv2.merge([b, g, r])
    return holo


def apply_skeleton_hologram(frame, pose_landmarks, t):
    """Mode Q: 3D Body Skeleton Overlay Hologram."""
    holo = apply_hologram_base(frame, t)
    h, w, _ = holo.shape
    overlay = holo.copy()

    if pose_landmarks is not None:
        pts = {}
        for idx, lm in enumerate(pose_landmarks.landmark):
            pts[idx] = (int(lm.x * w), int(lm.y * h))

        def draw_bone(a, b):
            if a in pts and b in pts:
                cv2.line(overlay, pts[a], pts[b], (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(overlay, pts[a], 3, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, pts[b], 3, (0, 255, 255), -1, cv2.LINE_AA)

        # Common skeleton connections (MediaPipe pose indices)
        pairs = [
            (11, 13), (13, 15),  # left arm
            (12, 14), (14, 16),  # right arm
            (11, 12),            # shoulders
            (11, 23), (12, 24),  # torso
            (23, 25), (25, 27),  # left leg
            (24, 26), (26, 28),  # right leg
        ]
        for a, b in pairs:
            draw_bone(a, b)

    out = cv2.addWeighted(holo, 0.7, overlay, 0.9, 0)
    return out


def apply_cartoon(frame):
    """Mode R: cartoon shader."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  9, 2)
    color = cv2.bilateralFilter(frame, 9, 75, 75)
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(color, edges_col)
    return cartoon


def apply_slowmo_trails(frame, prev_trail):
    """Mode S: super slow-mo motion trails."""
    if prev_trail is None:
        return frame.copy(), frame.copy()
    trail = cv2.addWeighted(frame, 0.3, prev_trail, 0.7, 0)
    return trail, trail


def apply_vhs_crt(frame, t):
    """Mode T: VHS / CRT effect."""
    h, w, _ = frame.shape
    out = frame.copy()

    for y in range(0, h, 2):
        out[y:y+1, :] = (out[y:y+1, :] * 0.4).astype(np.uint8)

    b, g, r = cv2.split(out)
    shift = int(2 * np.sin(t * 1.2))
    r = np.roll(r, shift, axis=1)
    g = np.roll(g, -shift, axis=0)
    out = cv2.merge([b, g, r])

    noise = np.random.randint(0, 30, frame.shape, dtype=np.uint8)
    out = cv2.addWeighted(out, 0.9, noise, 0.5, 0)
    return out


def apply_shockwave(frame, coords, t, audio_level):
    """Mode U: center shockwave pulse."""
    xx, yy = coords["xx"], coords["yy"]
    cx, cy = coords["cx"], coords["cy"]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    r_max = coords["r_max"]

    wave_center = ((t * 150) % (r_max * 1.2))
    thickness = 40 + 60 * audio_level

    ring = np.exp(-((r - wave_center) ** 2) / (2 * (thickness ** 2)))
    ring = np.clip(ring * 3.0, 0.0, 1.0)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + ring * 120, 0, 255)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + ring * 80, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def apply_lightning(frame, face_box, hand_points, coords, t):
    """Mode V: Lightning between hand(s) and face center."""
    h, w, _ = frame.shape
    out = frame.copy()
    overlay = out.copy()

    if face_box is not None:
        x, y, fw, fh = face_box
        fcx = int(x + fw / 2)
        fcy = int(y + fh / 2)
    else:
        fcx, fcy = int(coords["cx"]), int(coords["cy"])

    points = [(fcx, fcy)] + hand_points

    for i in range(1, len(points)):
        x1, y1 = points[0]
        x2, y2 = points[i]
        segments = 12
        pts = []
        for s in range(segments + 1):
            t_s = s / segments
            px = x1 + (x2 - x1) * t_s
            py = y1 + (y2 - y1) * t_s
            jitter = 10 * (1.0 - abs(0.5 - t_s) * 2.0)
            px += np.random.uniform(-jitter, jitter)
            py += np.random.uniform(-jitter, jitter)
            pts.append((int(px), int(py)))
        for j in range(len(pts) - 1):
            cv2.line(overlay, pts[j], pts[j+1], (255, 255, 255), 2, cv2.LINE_AA)

    out = cv2.addWeighted(out, 0.7, overlay, 0.9, 0)
    return out


def apply_warp_portal(frame, face_box, coords, t):
    """Mode W: warp portal around face center."""
    xx, yy = coords["xx"], coords["yy"]

    if face_box is not None:
        x, y, fw, fh = face_box
        cx = x + fw / 2
        cy = y + fh / 2
        radius = max(fw, fh) * 0.9
    else:
        cx, cy = coords["cx"], coords["cy"]
        radius = min(frame.shape[0], frame.shape[1]) * 0.5

    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx * dx + dy * dy)
    inside = dist < radius

    swirl_strength = 4.0 + 2.0 * np.sin(t * 1.7)
    theta = np.arctan2(dy, dx)
    r = dist + 1e-6

    swirl = np.zeros_like(r)
    swirl[inside] = swirl_strength * (1.0 - (r[inside] / radius))

    theta2 = theta + swirl
    x2 = r * np.cos(theta2) + cx
    y2 = r * np.sin(theta2) + cy

    map_x = x2.astype(np.float32)
    map_y = y2.astype(np.float32)

    warped = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)

    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 25 * np.sin(t * 1.3)) % 180
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def apply_xray(frame):
    """Mode Y: X-RAY look."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    inv = cv2.bitwise_not(gray)
    inv_col = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
    edges_col = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(inv_col, 0.8, edges_col, 0.8, 0)
    return out


def init_matrix_rain(w, h):
    """Init state for Matrix Rain (Z)."""
    num_cols = max(10, w // 10)
    drops = np.random.randint(-h, 0, size=(num_cols,), dtype=np.int32)
    chars = [chr(c) for c in range(0x30A0, 0x30FF)]
    return {
        "num_cols": num_cols,
        "drops": drops,
        "chars": chars,
        "char_step": 18,
    }


def apply_matrix_rain(frame, matrix_state):
    """Z: Matrix rain overlay."""
    h, w, _ = frame.shape
    overlay = np.zeros_like(frame)
    num_cols = matrix_state["num_cols"]
    char_step = matrix_state["char_step"]
    drops = matrix_state["drops"]
    chars = matrix_state["chars"]

    col_width = max(8, w // num_cols)
    for i in range(num_cols):
        x = i * col_width + col_width // 4
        y = drops[i]
        for k in range(0, h, char_step):
            yy = (y + k) % h
            ch = random.choice(chars)
            cv2.putText(
                overlay, ch, (x, yy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        drops[i] = (drops[i] + random.randint(4, 10)) % h

    matrix_state["drops"] = drops
    return cv2.addWeighted(frame, 0.45, overlay, 0.9, 0)
 #   GIF EXPORT (& key)



def apply_optical_flow(prev_gray, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )

    h, w = gray.shape
    vis = frame.copy()

    step = 16
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            cv2.line(
                vis,
                (x, y),
                (int(x + fx * 3), int(y + fy * 3)),
                (0, 255, 255),
                1
            )

    return vis, gray


def draw_lightning(img, p1, p2, depth=4):
    if depth == 0:
        cv2.line(img, p1, p2, (255,255,255), 2)
        return
    mid = (
        (p1[0]+p2[0])//2 + np.random.randint(-20,20),
        (p1[1]+p2[1])//2 + np.random.randint(-20,20)
    )
    draw_lightning(img, p1, mid, depth-1)
    draw_lightning(img, mid, p2, depth-1)

def apply_teleport_flicker(frame, t):
    h, w, _ = frame.shape
    out = frame.copy()

    mask = np.random.rand(h, w) > (0.5 + 0.4 * math.sin(t * 8))
    out[mask] = 0

    blur = cv2.GaussianBlur(out, (0, 0), 15)
    return cv2.addWeighted(out, 0.6, blur, 0.8, 0)

def export_gif(buffer, session):
    """
    Unified GIF exporter compatible with SessionManager.
    """
    if not buffer:
        print("[GIF] Buffer empty.")
        return None

    # SessionManager already knows how to export GIF USING ITS OWN buffer
    try:
        return session.export_gif(frames=list(buffer))
    except TypeError:
        # If session.export_gif() expects NO arguments:
        session.gif_frames = list(buffer)
        return session.export_gif()

def build_thumbnail_grid(frame, mode_list, effect_functions, current_mode=None, cache={}):
    """
    Premium Thumbnail Grid:
    - Labels
    - Highlight current mode
    - Fade-in animation
    - Clickable regions
    - Safe padding for consistent vstack
    """

    THUMB_W, THUMB_H = 160, 90
    LABEL_H = 20
    CELL_H = THUMB_H + LABEL_H
    PER_ROW = 8

    # Prepare black placeholder
    black_thumb = np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
    black_label = np.zeros((LABEL_H, THUMB_W, 3), dtype=np.uint8)

    all_cells = []

    for idx, m in enumerate(mode_list):

        # Thumbnail caching → HUGE speed boost
        if m not in cache or cache[m]["time"] < time.time() - 0.5:
            try:
                thumb = effect_functions[m](frame.copy())
            except:
                thumb = frame.copy()

            thumb = cv2.resize(thumb, (THUMB_W, THUMB_H))

            # store in cache
            cache[m] = {
                "img": thumb.copy(),
                "time": time.time()
            }
        else:
            thumb = cache[m]["img"]

        # Label strip
        label = black_label.copy()
        cv2.putText(label, m, (5, LABEL_H - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255,255,255), 1)

        # Combine
        cell = np.vstack([thumb, label])

        # Highlight if active
        if m == current_mode:
            cv2.rectangle(cell, (0,0), (THUMB_W-1, CELL_H-1), (0,255,255), 2)

        all_cells.append(cell)

    # Pad rows
    rows = []
    empty_cell = np.vstack([black_thumb, black_label])

    for i in range(0, len(all_cells), PER_ROW):
        row = all_cells[i:i+PER_ROW]
        while len(row) < PER_ROW:
            row.append(empty_cell.copy())
        rows.append(np.hstack(row))

    # Stack
    grid = np.vstack(rows)
    return grid

def grid_mouse_event(event, x, y, flags, param):
    global CLICK_GRID, mode

    if not CLICK_GRID:
        return

    # If click happened inside grid
    gx, gy = GRID_X, GRID_Y
    if gx <= x < gx + GRID_PER_ROW * THUMB_W and gy <= y < gy + 4 * CELL_H:
        col = (x - gx) // THUMB_W
        row = (y - gy) // CELL_H
        idx = row * GRID_PER_ROW + col

        if idx < len(param["mode_list"]):
            selected = param["mode_list"][idx]
            print(f"[Grid] Click selected → {selected}")
            mode = selected


def apply_ai_glitch_fracture(frame, t):
    """Mode /: AI Glitch Fracture."""
    h, w, _ = frame.shape
    out = apply_glitch(frame, t * 2.0)

    num_blocks = 10
    for _ in range(num_blocks):
        bw = np.random.randint(w // 8, w // 3)
        bh = np.random.randint(h // 8, h // 3)
        x = np.random.randint(0, w - bw)
        y = np.random.randint(0, h - bh)
        block = out[y:y+bh, x:x+bw].copy()
        dx = np.random.randint(-40, 40)
        dy = np.random.randint(-40, 40)
        x2 = np.clip(x + dx, 0, w - bw)
        y2 = np.clip(y + dy, 0, h - bh)
        out[y2:y2+bh, x2:x2+bw] = block

    return out
def handle_mode_key(key, mode_names, particles, state):
    """
    Handles A–Z, 0–9, / mode switching.
    Resets trails, particles, glitch bursts.
    """
    ch = chr(key).upper()

    # valid modes
    if ch in mode_names:
        state["mode"] = ch
        print(f"[Mode] Switched to {ch}: {mode_names[ch]}")
        state["trail"] = None
        state["slowmo"] = None
        state["glitch_burst"] = 0
        particles.clear()
        return True

    return False

def handle_capture_key(key, session, out, w, h):
    """
    Handles @ # $  (photo, video, export)
    """
    if key == ord("@"):
        print("[Photo] Capture photo.")
        session.capture_photo(out)
        return True

    elif key == ord("#"):
        print("[Video] Toggle record.")
        session.toggle_video(w, h, fps=30)
        return True

    elif key == ord("$"):
        print("[Export] Export latest.")
        session.export_latest()
        return True

    return False

def handle_system_key(key, show_hud_state):
    """
    Handles HUD toggle, quit, ESC.
    show_hud_state is a dict: { "hud": True/False }
    """
    if key in [ord(","), ord("<")]:
        show_hud_state["hud"] = not show_hud_state["hud"]
        print("[HUD] Toggled:", show_hud_state["hud"])
        return "hud"

    elif key in [ord("."), ord(">"), 27]:  # . or ESC
        print("[System] Quit.")
        return "quit"

    return None

def handle_automation_key(key, auto_cycle, carousel):
    """
    Handles TAB, SPACE, ^, %, and preset keys.
    """
    # Auto-cycle toggle
    if key == ord("^"):  
        enabled = auto_cycle.toggle()
        print("[AutoCycle] Enabled:", enabled)
        return True

    # Carousel toggle
    if key == ord("%"):
        active = carousel.toggle()
        print("[Carousel] Active:", active)
        return True

    return False
def draw_fps(out, fps):
    cv2.putText(out, f"FPS: {fps:.1f}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

def draw_mode_name(out, mode, mode_names):
    name = mode_names.get(mode, "Unknown")
    cv2.putText(out, f"Mode {mode} — {name}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 255), 2)


def draw_audio_level(out, audio_level):
    bar_w = int(200 * float(audio_level))
    cv2.rectangle(out, (20, 90), (20 + bar_w, 110), (0, 255, 255), -1)
    cv2.rectangle(out, (20, 90), (220, 110), (255, 255, 255), 2)
    cv2.putText(out, f"Audio {audio_level:.2f}",
                (230, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 0), 1)

def draw_face_box(out, face_box):
    if face_box is None:
        return
    (x, y, w, h) = face_box
    cv2.rectangle(out, (x, y), (x + w, y + h),
                  (0, 255, 0), 2)

def draw_video_recording(out, recording):
    if recording:
        cv2.circle(out, (out.shape[1] - 40, 40),
                   12, (0, 0, 255), -1)
        cv2.putText(out, "REC",
                    (out.shape[1] - 100, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)


def draw_gif_recording(out, gif_active):
    if gif_active:
        cv2.putText(out, "GIF",
                    (out.shape[1] - 100, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 200, 255), 2)
def draw_on_screen_menu(out):
    """
    Minimal translucent black menu overlay.
    """
    overlay = out.copy()
    alpha = 0.45

    cv2.rectangle(overlay,
                  (0, 0),
                  (out.shape[1], out.shape[0]),
                  (0, 0, 0), -1)

    out[:] = cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0)
    cv2.putText(out, "ON-SCREEN MENU",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 3)

    cv2.putText(out, "Use ({ }), +, _, (, ), & | to access advanced tools",
                (40, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)

def draw_effect_intensity_slider(out, intensity):
    """
    intensity ∈ [0, 1].
    """
    x0, y0 = 20, out.shape[0] - 40
    bar_w = 240
    cursor = int(x0 + intensity * bar_w)

    cv2.rectangle(out, (x0, y0), (x0 + bar_w, y0 + 16),
                  (255, 255, 255), 1)
    cv2.circle(out, (cursor, y0 + 8), 7, (0, 255, 200), -1)
    cv2.putText(out, f"Intensity {intensity:.2f}",
                (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)
    
def apply_full_hud(out, fps, mode, mode_names,
                   audio_level, face_box,
                   recording, gif_active,
                   intensity, show_hud, show_menu, stats):
    """
    Draws all HUD elements in a consistent style.
    """

    if show_menu:
        draw_on_screen_menu(out)
        return out

    if not show_hud:
        return out

    draw_fps(out, fps)
    draw_mode_name(out, mode, mode_names)
    draw_audio_level(out, audio_level)
    draw_face_box(out, face_box)
    draw_video_recording(out, recording)
    draw_gif_recording(out, gif_active)
    draw_effect_intensity_slider(out, intensity)

    if stats:
        draw_system_stats(out)

    return out
def apply_for_preview(mode, frame, t, audio_level, coords, face_mesh_results, face_box):
    """
    Run any effect in a SAFE WAY so it never crashes when generating
    a small thumbnail. This prevents undefined variables from breaking preview.
    """
    try:
        m = mode.upper()

        if m == "1":
            return apply_wave(frame, t, strength=8.0, coords=coords)
        if m == "2":
            return apply_swirl(frame, t, coords)
        if m == "3":
            return apply_neon_edges(frame)
        if m == "4":
            return apply_glitch(frame, t)
        if m == "5":
            return apply_kaleidoscope(frame)
        if m == "6":
            strength = 6 + 35 * float(audio_level)
            return apply_wave(frame, t * 1.1, strength=strength, coords=coords)
        if m == "7":
            return frame.copy()
        if m == "8":
            return apply_face_spotlight(frame, face_box, coords)
        if m == "9":
            return apply_music_visualizer(frame, t, float(audio_level), coords)
        if m == "0":
            return apply_glitch(frame, t * 2)

        if m == "A":
            out, _ = apply_ghost_trails(frame, None, 0.8)
            return out
        if m == "B":
            return apply_rgb_split(frame, t)
        if m == "C":
            return apply_chromatic_aberration(frame, coords)
        if m == "D":
            return apply_neural_edges(frame)
        if m == "E":
            return frame  # preview-safe placeholder
        if m == "F":
            return apply_gpu_shader(frame, t, float(audio_level), coords)
        if m == "G":
            return apply_fire_water_mode(frame, None, None, fire=True)
        if m == "H":
            return apply_face_melt(frame, coords, t)
        if m == "I":
            return apply_fire_water_mode(frame, None, None, fire=False)
        if m == "J":
            return apply_ai_hallucination(frame, t, float(audio_level))
        if m == "K":
            return apply_depth_warp(frame, coords, t)
        if m == "L":
            return frame  # safe preview
        if m == "M":
            return apply_heat_vision(frame)
        if m == "N":
            return apply_neural_glow(frame, t)
        if m == "O":
            return apply_depth_kaleidoscope(frame, coords)
        if m == "P":
            return apply_liquid_mirror(frame, coords, t)
        if m == "Q":
            return frame  # No skeleton tracking in preview
        if m == "R":
            return apply_cartoon(frame)
        if m == "S":
            out, _ = apply_slowmo_trails(frame, None)
            return out
        if m == "T":
            return apply_vhs_crt(frame, t)
        if m == "U":
            return apply_shockwave(frame, coords, t, audio_level)
        if m == "V":
            return apply_lightning(frame, None, [], coords, t)
        if m == "W":
            return apply_warp_portal(frame, face_box, coords, t)
        if m == "X":
            return frame  # safe
        if m == "Y":
            return apply_xray(frame)
        if m == "Z":
            return apply_matrix_rain(frame, {"cols":None})
        if m == "/":
            return apply_ai_glitch_fracture(frame, t) 
        if m == ";":  # Optical Flow (preview-safe)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return apply_neon_edges(frame)
        if m == "=":  # Teleport Flicker
            return apply_teleport_flicker(frame, t)

        return frame.copy()

    except Exception as e:
        print("[Preview Error]", e)
        return frame.copy()
# ===================== MAIN =====================
def preset_path():
    return os.path.join(os.getcwd(), "preset.json")


def save_preset(mode, intensity):
    data = {
        "mode": mode,
        "intensity": float(intensity)
    }
    with open(preset_path(), "w") as f:
        json.dump(data, f, indent=4)
    print(f"[Preset] Saved → preset.json")


def load_preset():
    try:
        with open(preset_path(), "r") as f:
            data = json.load(f)
        print("[Preset] Loaded OK")
        return data["mode"], float(data["intensity"])
    except:
        print("[Preset] No preset found")
        return None
def handle_voice_spike(audio_level, current_mode, mode_list):
    """
    Simple V1 voice control:
    - Loud audio spike switches to next mode
    """
    if audio_level > 0.82:
        idx = mode_list.index(current_mode)
        idx = (idx + 1) % len(mode_list)
        print("[Voice] Loud spike → NEXT MODE")
        return mode_list[idx]
    return current_mode

def save_gif_from_frames(frames, path):
    import imageio

    if not frames:
        print("[GIF] No frames to save.")
        return

    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                  for f in frames]

    try:
        imageio.mimsave(path, rgb_frames, fps=24)
        print(f"[GIF] Saved → {path}")
    except Exception as e:
        print("[GIF] FAILED →", e)
# ============================================================
# FIX MISSING GLOBALS REQUIRED BY MAIN()
# ============================================================

effect_strength = 1.0
glitch_burst_frames = 0

show_hud = True
show_thumbnail_grid = False

fps_overlay_enabled = fps_overlay_enabled if 'fps_overlay_enabled' in globals() else False
system_stats_enabled = system_stats_enabled if 'system_stats_enabled' in globals() else False

show_menu_overlay = show_menu_overlay if 'show_menu_overlay' in globals() else False

auto_face_zoom = auto_face_zoom if 'auto_face_zoom' in globals() else False
face_zoom_level = 1.0
target_zoom = 1.0

gif_buffer = list(gif_buffer)

voice_commands_enabled = voice_commands_enabled if 'voice_commands_enabled' in globals() else False

preset_save_pending = False
preset_load_pending = False

def main(): 
    # decay clap impulse over time
    cap = try_open_camera()
    if cap is None or not cap.isOpened():
        raise RuntimeError("Camera failed to open")
    target_w, target_h = 640, 360
    coords = precompute_coords(target_w, target_h)

    # use globals for zoom toggling
    global effect_intensity
    global preset_save_pending
    global preset_load_pending
    global auto_face_zoom
    global face_zoom_level
    global fps_overlay_enabled
    global target_zoom
    #global thumbnail_grid_enabled
    global voice_last
    global voice_commands_enabled
    global system_stats_enabled
    global gif_buffer
    global show_intensity_slider
    global show_menu_overlay

    # Face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Hand tracker
    hands = None
    if HAS_HANDS and mp_hands is not None:
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[HandTracking] MediaPipe Hands initialized.")

    face_mesh = None
    if HAS_FACE_MESH:
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    # Pose tracker for 3D skeleton hologram
    pose = None
    if HAS_POSE and mp_pose is not None:
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        print("[Pose] MediaPipe Pose initialized.")
    # ---------------------------------------------------------
    # SESSION, CAPTURE + EXPORT
    # ---------------------------------------------------------
    session = SessionManager()       # <-- from utils/capture.py
    fps_timer = FPSTimer()           # <-- from utils/utils.py

    # ---------------------------------------------------------
    # AUTO-CYCLE + CAROUSEL
    # ---------------------------------------------------------
    mode_list = [
        "1","2","3","4","5","6","7","8","9","0",
        "A","B","C","D","E","F","G","H","I","J",
        "K","L","M","N","O","P","Q","R","S","T",
        "U","V","W","X","Y","Z","/", "[", "]", 
        ";", "\\", "-", "=", "<", ">", "?"
    ]

    carousel = EffectCarousel(mode_list)
    auto_cycle = AutoCycle(mode_list)

    # ---------------------------------------------------------
    # AUDIO STREAM
    # ---------------------------------------------------------
    # Audio
    audio_level = 0.0
    prev_audio_level = 0.0
    audio_gain = 20.0
    audio_stream = None
    clap_impulse = 0.0

    if HAS_AUDIO:
        def audio_callback(indata, frames, time_info, status):
            nonlocal audio_level, clap_impulse
            if status:
                return

            # raw microphone energy (no smoothing)
            raw = float(np.sqrt(np.mean(indata ** 2)))
            raw = min(raw * audio_gain, 1.0)

            # impulse = sudden increase (clap detection)
            impulse = max(0.0, raw - audio_level)

            # smooth baseline audio
            audio_level = 0.85 * audio_level + 0.15 * raw

            # strong clap detected
            if impulse > 0.25:
                clap_impulse = 1.0

        try:
            audio_stream = sd.InputStream(
                channels=1,
                samplerate=16000,
                callback=audio_callback,
            )
            audio_stream.start()
            print("[Audio] Microphone stream started for reactive modes.")
        except Exception as e:
            print(f"[Audio] Failed to start sounddevice InputStream: {e}")
            audio_stream = None
    else:
        print("[Audio] Running in fake audio mode (time-based animation).")

    window_name = "Neural Reality Distortion Room"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, grid_mouse_event, param={"mode_list": mode_list})


    mode = "1"
    show_hud = True
    trail_frame = None
    slowmo_trail = None
    glitch_burst_frames = 0
    clone_buffer = []

    particles = []
    matrix_state = init_matrix_rain(target_w, target_h)

    t0 = time.time()

    # slider strength for '*' key (avoid NameError)
    effect_strength = 1.0

    mode_names = {
        "1": "Wave",
        "2": "Swirl",
        "3": "Neon Edges",
        "4": "Glitch",
        "5": "Kaleidoscope",
        "6": "Audio Reactive (speak)",
        "7": "Raw Camera",
        "8": "Face Spotlight + Warp",
        "9": "Music Visualizer",
        "0": "Clap Glitch Bursts",
        "A": "Ghost Trails",
        "B": "RGB Split Warp",
        "C": "Chromatic Aberration",
        "D": "Neural Edge Outlines",
        "E": "Hand Distortion",
        "F": "GPU Shader (CUDA/CPU)",
        "G": "Fire From Mouth & Hands",
        "H": "Face Melt",
        "I": "Water From Mouth & Hands",
        "J": "AI Hallucination",
        "K": "Depth Estimation Warp",
        "L": "Hand Particle Emitter",
        "M": "Heat Vision",
        "N": "Neural Glow Aura",
        "O": "Depth-Kaleidoscope",
        "P": "Liquid Mirror",
        "Q": "3D Skeleton Hologram",
        "R": "Cartoon Shader",
        "S": "Super Slow-Mo Trails",
        "T": "VHS / CRT",
        "U": "Shockwave Pulse",
        "V": "Lightning (hands→face)",
        "W": "Warp Portal (face center)",
        "X": "Volumetric Face Warp",
        "Y": "X-RAY",
        "Z": "Matrix Rain",
        "/": "AI Glitch Fracture",
        ";": "apply_neon_edges",
        "=": "Teleportation Flicker",
    }
    # Map each mode key → corresponding function for thumbnails
    effect_functions = {
        "1": lambda f: apply_wave(f.copy(), t, 12.0, coords),
        "2": lambda f: apply_swirl(f.copy(), t, coords),
        "3": lambda f: apply_neon_edges(f.copy()),
        "4": lambda f: apply_glitch(f.copy(), t),
        "5": lambda f: apply_kaleidoscope(f.copy()),
        "6": lambda f: apply_wave(f.copy(), t*1.3, 20.0, coords),
        "7": lambda f: f.copy(),
        "8": lambda f: apply_face_spotlight(f.copy(), face_box, coords),
        "9": lambda f: apply_music_visualizer(f.copy(), t, float(audio_level), coords),
        "0": lambda f: apply_wave(f.copy(), t, 8.0, coords),
        "A": lambda f: apply_ghost_trails(f.copy(), None)[0],
        "B": lambda f: apply_rgb_split(f.copy(), t),
        "C": lambda f: apply_chromatic_aberration(f.copy(), coords),
        "D": lambda f: apply_neural_edges(f.copy()),
        "E": lambda f: apply_hand_distortion(f.copy(), hand_landmarks, coords, t),
        "F": lambda f: apply_gpu_shader(f.copy(), t, float(audio_level), coords),
        "G": lambda f: apply_fire_water_mode(f.copy(), mouth_center, hand_landmarks, fire=True),
        "H": lambda f: apply_face_melt(f.copy(), coords, t),
        "I": lambda f: apply_fire_water_mode(f.copy(), mouth_center, hand_landmarks, fire=False),
        "J": lambda f: apply_ai_hallucination(f.copy(), t, float(audio_level)),
        "K": lambda f: apply_depth_warp(f.copy(), coords, t),
        "L": lambda f: apply_particle_emitter(f.copy(), []),
        "M": lambda f: apply_heat_vision(f.copy()),
        "N": lambda f: apply_neural_glow(f.copy(), t),
        "O": lambda f: apply_depth_kaleidoscope(f.copy(), coords),
        "P": lambda f: apply_liquid_mirror(f.copy(), coords, t),
        "Q": lambda f: apply_skeleton_hologram(f.copy(), pose_landmarks, t),
        "R": lambda f: apply_cartoon(f.copy()),
        "S": lambda f: apply_slowmo_trails(f.copy(), None)[0],
        "T": lambda f: apply_vhs_crt(f.copy(), t),
        "U": lambda f: apply_shockwave(f.copy(), coords, t, float(audio_level)),
        "V": lambda f: apply_lightning(f.copy(), face_box, hand_points, coords, t),
        "W": lambda f: apply_warp_portal(f.copy(), face_box, coords, t),
        "X": lambda f: apply_volumetric_face_warp(f.copy(), face_box, coords, t),
        "Y": lambda f: apply_xray(f.copy()),
        "Z": lambda f: apply_matrix_rain(f.copy(), matrix_state),
        "/": lambda f: apply_ai_glitch_fracture(f.copy(), t),
        ";": lambda f: apply_neon_edges(f.copy()),  # safe preview
        "=": lambda f: apply_teleport_flicker(f.copy(), t),
    }

    print("[run_room_demo] Started.")
    print("  1–5: Wave / Swirl / Neon / Glitch / Kaleidoscope")
    print("  6: Audio Reactive  |  7: Raw  |  8: Face Spotlight  |  9: Music Visualizer  |  0: Clap Glitch")
    print("  A: Ghost Trails  |  B: RGB Split  |  C: Chromatic Ab.  |  D: Neural Edges  |  E: Hand Distortion  |  F: GPU Shader")
    print("  G: Fire  |  H: Face Melt  |  I: Water  |  J: AI Hallucination  |  K: Depth Warp  |  L: Hand Particles")
    print("  M: Heat Vision  |  N: Glow Aura  |  O: Depth-Kaleidoscope  |  P: Liquid Mirror  |  Q: 3D Skeleton Hologram")
    print("  R: Cartoon  |  S: Slow-Mo  |  T: VHS/CRT  |  U: Shockwave  |  V: Lightning  |  W: Warp Portal")
    print("  X: Volumetric Face Warp  |  Y: X-Ray  |  Z: Matrix Rain  |  /: Glitch Fracture")
    print("  ,: Toggle HUD  |  .: Quit  |  ESC: Quit")
    print(" @ = Photo   |   # = Video Start/Stop   |   $ = Social Export")
    print(" TAB = Carousel Preview   |   SPACE = Auto-cycle")
    print(" CTRL+S = Save full session backup")
    print("------------------------------------------------------------")

    while True:
        clap_impulse *= 0.85
        ret, frame = cap.read()
        if not ret:
            print("[run_room_demo] Camera read failed, retrying...")
            time.sleep(0.05)
            continue

        frame = cv2.resize(frame, (target_w, target_h))
        t = time.time() - t0
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clap_detected = False
        

        # IMPORTANT: update AFTER detection
        prev_gray = gray

        # always tick FPS onapply_magic_glyphsce per frame so `fps` is defined everywhere
        fps = fps_timer.tick()

        if not HAS_AUDIO or audio_stream is None:
            audio_level = 0.5 + 0.5 * np.sin(t * 2.0)

        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        face_box = None
        if len(faces) > 0:
            face_box = max(faces, key=lambda f: f[2] * f[3])
            
        # --- Proper Auto Face Zoom update ---
        # (we already declared globals at top of main)
        if auto_face_zoom:
            if face_box is not None:
                target_zoom = 1.35
            else:
                target_zoom = 1.0
        else:
            target_zoom = 1.0

        # Smooth transition (ease)
        face_zoom_level = 0.85 * face_zoom_level + 0.15 * target_zoom

        # Mouth center for fire/water
        mouth_center = None
        if face_box is not None:
            mx, my, mw, mh = get_face_and_mouth_box(face_box, coords)
            mouth_center = (int(mx + mw / 2), int(my + mh * 0.2))

        face_mesh_results = None
        if face_mesh is not None:
            rgb_face = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_mesh_results = face_mesh.process(rgb_face)


        # Hand landmarks (for E, L, G, I, V, X)
        hand_landmarks = None
        hand_points = []
        if hands is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_h = hands.process(rgb)
            if results_h.multi_hand_landmarks:
                hand_landmarks = results_h.multi_hand_landmarks[0]
                for lm in hand_landmarks.landmark:
                    hx = int(lm.x * target_w)
                    hy = int(lm.y * target_h)
                    hand_points.append((hx, hy))

        # Pose landmarks for Q
        pose_landmarks = None
        if pose is not None:
            rgb_pose = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_p = pose.process(rgb_pose)
            if results_p.pose_landmarks:
                pose_landmarks = results_p.pose_landmarks

         # -----------------------------------
        # AUTO CYCLE UPDATE
        # -----------------------------------
        new_mode = auto_cycle.update()
        if new_mode:
            mode = new_mode
            print("[AutoCycle] Now:", mode)

        # -----------------------------------
        # CAROUSEL PREVIEW
        # -----------------------------------
        if carousel.active:
            mode = carousel.current()

        # -----------------------------------
        # CLAP → GLITCH BURST
        # -----------------------------------
        if audio_level > 0.6 and prev_audio_level <= 0.6:
            glitch_frames = 10
        prev_audio_level = audio_level

        key_mode = mode.upper()
        out = frame.copy()

        # ----- Mode routing -----
        if key_mode == "1":
            strength = 5.0 + 35.0 * effect_intensity + 45.0 * clap_impulse
            out = apply_wave(frame, t, strength=strength, coords=coords)

        elif key_mode == "2":
            out = apply_swirl(frame, t, coords)
        elif key_mode == "3":
            out = apply_neon_edges(frame)
        elif key_mode == "4":
            out = apply_glitch(frame, t)
        elif key_mode == "5":
            out = apply_kaleidoscope(frame)
        elif key_mode == "6":
            strength = 5.0 + 45.0 * float(audio_level)
            out = apply_wave(frame, t * 1.3, strength=strength, coords=coords)
            factor = 0.7 + 0.8 * float(audio_level)
            out = np.clip(out.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        elif key_mode == "7":
            out = frame
        elif key_mode == "8":
            out = apply_face_spotlight(frame, face_box, coords)
        elif key_mode == "9":
            out = apply_music_visualizer(frame, t, float(audio_level), coords)
        elif key_mode == "0":
            if clap_detected:
                clap_burst_frames = 10   # stronger burst

            if clap_burst_frames > 0:
                burst = apply_glitch_burst(frame, t)

                # ENERGY BLOOM
                bloom = cv2.GaussianBlur(burst, (0, 0), 22)
                out = cv2.addWeighted(burst, 0.55, bloom, 1.1, 0)

                # RADIAL IMPACT ZOOM
                h, w = frame.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, 0, 1.03)
                out = cv2.warpAffine(out, M, (w, h))

                clap_burst_frames -= 1
            else:
                out = frame.copy()

        elif key_mode == "A":
            out, trail_frame = apply_ghost_trails(frame, trail_frame, alpha=0.8)
        elif key_mode == "B":
            out = apply_rgb_split(frame, t)
        elif key_mode == "C":
            out = apply_chromatic_aberration(frame, coords, amount=7.0)
        elif key_mode == "D":
            out = apply_neural_edges(frame)
        elif key_mode == "E":
            out = apply_hand_distortion(frame, hand_landmarks, coords, t)
        elif key_mode == "F":
            # GPU mode safely falls back to CPU when HAS_CUDA == False
            out = apply_gpu_shader(frame, t, float(audio_level), coords)
        elif key_mode == "G":
            out = apply_fire_water_mode(frame, mouth_center, hand_landmarks, fire=True)
        elif key_mode == "H":
            out = apply_face_melt(frame, coords, t)
        elif key_mode == "I":
            out = apply_fire_water_mode(frame, mouth_center, hand_landmarks, fire=False)
        elif key_mode == "J":
            out = apply_ai_hallucination(frame, t, float(audio_level))
        elif key_mode == "K":
            out = apply_depth_warp(frame, coords, t)
        elif key_mode == "L":
            particles[:] = update_particles(particles, hand_points, target_w, target_h)
            out = apply_particle_emitter(frame, particles)
        elif key_mode == "M":
            out = apply_heat_vision(frame)
        elif key_mode == "N":
            out = apply_neural_glow(frame, t)
        elif key_mode == "O":
            out = apply_depth_kaleidoscope(frame, coords)
        elif key_mode == "P":
            out = apply_liquid_mirror(frame, coords, t)
        elif key_mode == "Q":
            out = apply_skeleton_hologram(frame, pose_landmarks, t)
        elif key_mode == "R":
            out = apply_cartoon(frame)
        elif key_mode == "S":
            out, slowmo_trail = apply_slowmo_trails(frame, slowmo_trail)
        elif key_mode == "T":
            out = apply_vhs_crt(frame, t)
        elif key_mode == "U":
            intensity = max(audio_level, clap_impulse)
            out = apply_shockwave(frame, coords, t, intensity)
        elif key_mode == "V":
            out = apply_lightning(frame, face_box, hand_points, coords, t)
        elif key_mode == "W":
            out = apply_warp_portal(frame, face_box, coords, t)
        elif key_mode == "X":
            out = apply_volumetric_face_warp(frame, face_box, coords, t)
        elif key_mode == "Y":
            out = apply_xray(frame)
        elif key_mode == "Z":
            out = apply_matrix_rain(frame, matrix_state)
        elif key_mode == "/":
            out = apply_ai_glitch_fracture(frame, t)

        elif key_mode == ";":
            if prev_gray is None:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                out = frame
            else:
                out, prev_gray = apply_optical_flow(prev_gray, frame)

        elif key_mode == "=":
            out = apply_teleport_flicker(frame, t)
     
        else:
            out = frame

        # Clap glitch bursts overlay (mode 0)
        if key_mode == "0" and glitch_burst_frames > 0:
            burst_intensity = glitch_burst_frames / 12.0
            glitch_frame = apply_glitch_burst(frame, t)
            out = cv2.addWeighted(out, 1.0 - burst_intensity, glitch_frame, burst_intensity, 0)
            glitch_burst_frames -= 1

        # -----------------------------------
        # RECORD VIDEO
        # -----------------------------------
        session.write_video_frame(out)

        # ============================================================
        # NEW FEATURE HOOKS (Hybrid C)
        # ============================================================

        # 1) Add each frame to GIF buffer
        gif_buffer.append(out.copy())

        # 2) Auto Face Zoom
        # (we now apply it once later using face_zoom_level; keep this
        # placeholder but don't double-apply with wrong signature)
        # if auto_face_zoom and face_box is not None:
        #     out = apply_auto_face_zoom(out, face_box)

        # 3) FPS Overlay
        if fps_overlay_enabled:
            cv2.putText(out, f"{fps:.1f} FPS", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        # 4) Thumbnail Grid
        if show_thumbnail_grid:
            grid = build_thumbnail_grid(frame, list(effect_functions.keys()), effect_functions)

            # -------------------------------
            # Resize grid to fit overlay area
            # -------------------------------
            H, W, _ = out.shape
            max_w, max_h = W // 2, H // 2  # grid must fit inside 50% of frame

            gh, gw, _ = grid.shape
            scale = min(max_w / gw, max_h / gh)

            new_w = int(gw * scale)
            new_h = int(gh * scale)

            grid = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Place into top-left
            GRID_X = 10
            GRID_Y = 10
            h, w, _ = grid.shape

            out[GRID_Y:GRID_Y+h, GRID_X:GRID_X+w] = grid

        # 5) On-Screen Menu Overlay
        # (actual menu drawing is done below with full arguments)
        # kept here as structural placeholder

        # 6) Voice commands
        if voice_commands_enabled and HAS_VOICE:
            handle_voice_commands(session, out)

        #if thumbnail_grid_enabled:
            #thumb = generate_thumbnail_grid(frame, t, coords, audio_level)

            # scale thumbnail overlay
            #th, tw, _ = thumb.shape
            #out[10:10+th, 10:10+tw] = thumb


        # HUD
        if show_hud:
            name = mode_names.get(key_mode, "Unknown")
            hud_text = f"Mode {key_mode} - {name} | audio_level={audio_level:.2f}"
            cv2.putText(out, hud_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(
                out,
                "1-9,0,A-Z,/ modes | , HUD | . / ESC: quit",
                (20, target_h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
         # visual explosion flash on clap
        if clap_impulse > 0.1:
            flash = np.full_like(out, 255)
            out = cv2.addWeighted(out, 1.0, flash, 0.18 * clap_impulse, 0)

        if show_thumbnail_grid:
            grid = build_thumbnail_grid(frame, list(effect_functions.keys()), effect_functions)
            gh, gw, _ = grid.shape
            out[10:10+gh, 10:10+gw] = grid

        if show_intensity_slider:
            draw_intensity_slider(out, effect_intensity)

        if show_menu_overlay:
            draw_menu_overlay(out, 
                            mode, 
                            mode_names, 
                            effect_intensity,
                            audio_level,
                            fps,
                            session.recording,
                            session.session_path,
                            show_hud,
                            auto_cycle.enabled,
                            carousel.active)

        # Apply auto-zoom AFTER effects, using smooth zoom level
        if auto_face_zoom:
            out = apply_auto_face_zoom(out, face_box, face_zoom_level)

        if system_stats_enabled:
            out = draw_system_stats(
                out,
                fps=fps,
                audio_level=float(audio_level),
                mode_name=mode_names.get(key_mode, "Unknown"),
            )
  
        cv2.imshow(window_name, out)

        key = cv2.waitKey(1) & 0xFF
        if key != 255:
            # Digits
            if key in [ord(str(d)) for d in range(0, 10)]:
                mode = chr(key)
                print(f"[Mode] Switched to {mode}: {mode_names.get(mode.upper(), '')}")
                trail_frame = None
                slowmo_trail = None
                glitch_burst_frames = 0
                particles.clear()
                clap_burst_frames = 0
                global freeze_frame
                freeze_frame = None
            # Letters + slash
            elif chr(key).upper() in mode_names:
                mode = chr(key).upper()
                print(f"[Mode] Switched to {mode}: {mode_names.get(mode, '')}")
                trail_frame = None
                clap_burst_frames = 0
                slowmo_trail = None
                glitch_burst_frames = 0
                particles.clear()

            
            elif key == ord(","):
                show_hud = not show_hud
                print(f"[HUD] show_hud = {show_hud}")
            elif key == ord("."):
                print("[run_room_demo] Quit key pressed (ESC).")
                break
            elif key in [27]:  # ESC
                print("[run_room_demo] Quit key pressed (ESC).")
                break
            elif key == ord("!"):
                session.save_full_session_zip()
            elif key == ord("^"):
                enabled = auto_cycle.toggle()
                print("[AutoCycle] Enabled:", enabled)
            elif key == ord("%"):
                active = carousel.toggle()
                print("[Carousel] Active:", active)
            elif key == ord("@"):
                print("[Photo] Capture triggered.")
                session.capture_photo(out)
            elif key == ord("#"):
                print("[Video] Toggle record.")
                session.toggle_video(target_w, target_h, fps=30)
            elif key == ord("$"):
                print("[Export] Export latest.")
                session.export_latest()
            # --- GIF Export (& key) ---
            elif key == ord("&"):
                print("[GIF] Exporting last 4 seconds...")
                export_gif(gif_buffer, session)

            # --- Effect Intensity Slider (* key) ---
            elif key == ord("*"):
                #global effect_intensity
                effect_intensity += 0.1
                if effect_intensity > 1.0:
                    effect_intensity = 0.0
                print(f"[Intensity] Now = {effect_intensity:.2f}")


            # --- On-Screen Menu Overlay (( key) ---
            elif key == ord("("):
                show_menu_overlay = not show_menu_overlay
                print("[Menu] Overlay =", show_menu_overlay)

            # --- Auto Face Zoom () key) ---
            elif key == ord(")"):
                auto_face_zoom = not auto_face_zoom
                if auto_face_zoom:
                    print("[AutoFaceZoom] ENABLED")
                else:
                    print("[AutoFaceZoom] DISABLED")

            # --- FPS Overlay (_ key) ---
            elif key == ord("_"):
                fps_overlay_enabled = not fps_overlay_enabled
                print("[FPS] Overlay =", fps_overlay_enabled)
                system_stats_enabled = not system_stats_enabled
                print("[SystemStats] ENABLED" if system_stats_enabled else "[SystemStats] DISABLED")

            # --- Live Effect Thumbnail Grid (+ key) ---
            elif key == 43:   # ASCII for "+"
                #thumbnail_grid_enabled = not thumbnail_grid_enabled
                #print("[ThumbnailGrid] ENABLED" if thumbnail_grid_enabled else "[ThumbnailGrid] DISABLED")
                show_intensity_slider = not show_intensity_slider
                print("[Slider] Visible:", show_intensity_slider)

            # --- Save Preset ({ key) ---
            elif key == 123:  # ASCII for "{"
                preset_save_pending = True
                save_preset(mode, effect_intensity)
                print("[Preset] Save requested")

            # --- Load Preset (} key) ---
            elif key == 125:  # ASCII for "}"
                preset_load_pending = True
                loaded = load_preset()
                if loaded:
                    mode, effect_intensity = loaded
                print("[Preset] Load requested")

            # --- Voice Commands (| key) ---
            elif key == 124:  # ASCII for "|"
                if HAS_VOICE:
                    voice_commands_enabled = not voice_commands_enabled
                    print("[Voice] Enabled =", voice_commands_enabled)
                else:
                    print("[Voice] Module not installed.")

    cap.release()
    cv2.destroyAllWindows()

    if audio_stream is not None:
        audio_stream.stop()
        audio_stream.close()

    if hands is not None:
        hands.close()

    if pose is not None:
        pose.close()
    print("FRAME SHAPE =", out.shape)

    print("[run_room_demo] Finished.")


if __name__ == "__main__":
    main()
