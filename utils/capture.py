import os
import cv2
import time
import shutil
import zipfile
import imageio
import numpy as np
from datetime import datetime
from .utils import ensure_dir, session_root, new_session_folder, timestamp_name, numbered_file


# ============================================================
#   SESSION MANAGER
# ============================================================

class SessionManager:
    """
    Handles photos, videos, GIF export, presets, and full session zips.
    """

    def __init__(self):
        # Create unique session folder
        self.session_path = new_session_folder()
        print(f"[Session] Started new session: {self.session_path}")

        # Video
        self.recording = False
        self.video_writer = None
        self.video_counter = 0
        self.photo_counter = 0

        # GIF buffer (&)
        self.gif_seconds = 4
        self.gif_buffer = []
        self.gif_max = 120     # 4 seconds at 30 FPS

        # Voice
        self.voice_enabled = False

        # Presets
        self.preset_path = os.path.join(self.session_path, "presets.json")

    # ============================================================
    #   PHOTO CAPTURE (@)
    # ============================================================

    def capture_photo(self, frame):
        """Save single-frame photo."""
        self.photo_counter += 1
        filename = numbered_file("photo", self.photo_counter, ext="png")
        path = os.path.join(self.session_path, filename)

        try:
            cv2.imwrite(path, frame)
            print(f"[Photo] Saved: {path}")
            return path
        except Exception as e:
            print(f"[Photo] ERROR saving {path}: {e}")
            return None

    # ============================================================
    #   VIDEO RECORDING (#)
    # ============================================================

    def toggle_video(self, frame_width, frame_height, fps=30):
        """Toggle mp4 recording."""
        if not self.recording:
            # START
            self.video_counter += 1
            vid_name = numbered_file("video", self.video_counter, ext="mp4")
            out_path = os.path.join(self.session_path, vid_name)

            try:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(out_path, fourcc, fps,
                                                    (frame_width, frame_height))
                self.recording = True
                print(f"[Video] Recording STARTED: {out_path}")
                return True
            except Exception as e:
                print(f"[Video] ERROR starting: {e}")
                self.recording = False
                return False

        else:
            # STOP
            try:
                self.recording = False
                if self.video_writer:
                    self.video_writer.release()
                print("[Video] Recording STOPPED.")
                return False
            except Exception as e:
                print(f"[Video] ERROR stopping: {e}")
                return False

    def write_video_frame(self, frame):
        """Write frame to active video, and GIF buffer."""
        # MP4 write
        if self.recording and self.video_writer:
            try:
                self.video_writer.write(frame)
            except Exception as e:
                print(f"[Video] ERROR writing frame: {e}")

        # GIF buffer (&)
        try:
            small = cv2.resize(frame, (320, 180))
            self.gif_buffer.append(small[:, :, ::-1])  # convert BGR→RGB

            # keep GIF at 4-second sliding window
            if len(self.gif_buffer) > self.gif_max:
                self.gif_buffer.pop(0)
        except Exception as e:
            print(f"[GIF] Buffer error: {e}")

    # ============================================================
    #   EXPORT GIF (&)
    # ============================================================

    def export_gif(self):
        """Export last 4 seconds as GIF."""
        if len(self.gif_buffer) < 10:
            print("[GIF] Not enough frames.")
            return None

        gif_dir = ensure_dir(os.path.join(self.session_path, "gif"))
        name = timestamp_name("clip", "gif")
        out_path = os.path.join(gif_dir, name)

        try:
            imageio.mimsave(out_path, self.gif_buffer, fps=20)
            print(f"[GIF] Saved 4-sec GIF: {out_path}")
            return out_path
        except Exception as e:
            print(f"[GIF] ERROR: {e}")
            return None

    # ============================================================
    #   SOCIAL EXPORT ($)
    # ============================================================

    def export_latest(self):
        """Copy newest file to share folder."""
        share_dir = ensure_dir(os.path.join(self.session_path, "share"))

        try:
            files = [os.path.join(self.session_path, f)
                     for f in os.listdir(self.session_path)
                     if os.path.isfile(os.path.join(self.session_path, f))]

            if not files:
                print("[Export] No files to export.")
                return None

            newest = max(files, key=os.path.getmtime)
            export_path = os.path.join(share_dir, os.path.basename(newest))
            shutil.copy(newest, export_path)

            print(f"[Export] Saved: {export_path}")
            return export_path

        except Exception as e:
            print(f"[Export] ERROR: {e}")
            return None

    # ============================================================
    #   SAVE FULL SESSION (CTRL+S or !)
    # ============================================================

    def save_full_session_zip(self):
        """Zip entire session."""
        zip_name = f"{os.path.basename(self.session_path)}.zip"
        zip_path = os.path.join(session_root(), zip_name)

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.session_path):
                    for file in files:
                        fp = os.path.join(root, file)
                        arc = os.path.relpath(fp, self.session_path)
                        zipf.write(fp, arc)

            print(f"[Session] Full session saved: {zip_path}")
            return zip_path

        except Exception as e:
            print(f"[Session] ERROR: {e}")
            return None

    # ============================================================
    #   PRESETS {  }
    # ============================================================

    def save_preset(self, mode, intensity):
        """Save (mode + intensity) to presets.json"""
        try:
            import json

            data = {"mode": mode, "intensity": intensity}

            with open(self.preset_path, "w") as f:
                json.dump(data, f, indent=4)

            print(f"[Preset] Saved preset → {self.preset_path}")

        except Exception as e:
            print(f"[Preset] ERROR saving: {e}")

    def load_preset(self):
        """Load stored preset file"""
        try:
            import json

            if not os.path.exists(self.preset_path):
                print("[Preset] No preset saved.")
                return None

            with open(self.preset_path, "r") as f:
                data = json.load(f)

            print(f"[Preset] Loaded preset: {data}")
            return data

        except Exception as e:
            print(f"[Preset] ERROR loading: {e}")
            return None

    # ============================================================
    #   VOICE COMMANDS (|)
    # ============================================================

    def toggle_voice(self):
        self.voice_enabled = not self.voice_enabled
        print(f"[Voice] Voice Recognition Enabled = {self.voice_enabled}")
        return self.voice_enabled
