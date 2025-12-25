# capture/multicam_capture.py
import cv2
import threading
import time
from typing import List, Dict, Optional


class CameraStream:
    def __init__(self, cam_id: int, name: Optional[str] = None, width: int = 640, height: int = 480):
        self.cam_id = cam_id
        self.name = name or f"cam_{cam_id}"
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.frame = None
        self.timestamp = 0.0
        self.running = False
        self.lock = threading.Lock()

    def _loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            ts = time.time()
            with self.lock:
                self.frame = frame
                self.timestamp = ts

    def start(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()
        self.cap.release()

    def get_latest(self):
        with self.lock:
            return self.frame, self.timestamp


class MultiCamCapture:
    def __init__(self, cam_ids: List[int]):
        self.cams = [CameraStream(c) for c in cam_ids]

    def start(self):
        for cam in self.cams:
            cam.start()

    def stop(self):
        for cam in self.cams:
            cam.stop()

    def get_synced_frames(self, max_time_diff: float = 0.03):
        frames: Dict[str, tuple] = {}
        timestamps = []

        for cam in self.cams:
            frame, ts = cam.get_latest()
            if frame is None:
                return None  # Not ready
            frames[cam.name] = (frame, ts)
            timestamps.append(ts)

        # Optional sync check
        if max(timestamps) - min(timestamps) > max_time_diff:
            # out of sync; you could wait/retry here
            pass

        return frames


if __name__ == "__main__":
    # Simple preview test
    mc = MultiCamCapture([0, 1])  # adjust camera IDs
    mc.start()

    try:
        while True:
            frames = mc.get_synced_frames()
            if frames is None:
                continue

            for name, (frame, ts) in frames.items():
                cv2.imshow(name, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        mc.stop()
        cv2.destroyAllWindows()
