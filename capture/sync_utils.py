# capture/sync_utils.py

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

@dataclass
class FramePacket:
    timestamp: float
    frame: np.ndarray

class FrameBuffer:
    """
    Thread-safe buffer per camera.
    """
    def __init__(self, maxlen: int = 10):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def push(self, packet: FramePacket):
        with self.lock:
            self.buffer.append(packet)

    def latest(self) -> Optional[FramePacket]:
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[-1]

    def get_closest(self, ts: float, max_diff: float = 0.03) -> Optional[FramePacket]:
        with self.lock:
            best = None
            best_diff = max_diff
            for p in self.buffer:
                diff = abs(p.timestamp - ts)
                if diff < best_diff:
                    best = p
                    best_diff = diff
            return best

class MultiCamSync:
    """
    Keeps multiple FrameBuffers and can retrieve time-synced frames across cameras.
    """
    def __init__(self, cam_ids):
        self.buffers: Dict[int, FrameBuffer] = {cid: FrameBuffer() for cid in cam_ids}

    def push_frame(self, cam_id: int, frame: np.ndarray, timestamp: Optional[float] = None):
        if timestamp is None:
            timestamp = time.time()
        self.buffers[cam_id].push(FramePacket(timestamp=timestamp, frame=frame))

    def get_synced_frames(self, ref_cam_id: int, max_diff: float = 0.03) -> Optional[Dict[int, np.ndarray]]:
        ref_packet = self.buffers[ref_cam_id].latest()
        if ref_packet is None:
            return None

        ts_ref = ref_packet.timestamp
        out = {ref_cam_id: ref_packet.frame}

        for cid, buf in self.buffers.items():
            if cid == ref_cam_id:
                continue
            pkt = buf.get_closest(ts_ref, max_diff=max_diff)
            if pkt is None:
                return None
            out[cid] = pkt.frame

        return out
