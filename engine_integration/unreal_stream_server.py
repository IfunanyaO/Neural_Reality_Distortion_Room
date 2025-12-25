# engine_integration/unreal_stream_server.py

import socket
import struct
import cv2
import threading
import numpy as np

class FrameStreamServer:
    """
    Very simple TCP server that streams JPEG frames to a single client.
    Unreal will connect, read each frame, and update a texture.
    """

    def __init__(self, host="127.0.0.1", port=5000, jpeg_quality=80):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.client_socket = None
        self.server_socket = None
        self.lock = threading.Lock()

    def start(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"[FrameStreamServer] Listening on {self.host}:{self.port}...")
        self.client_socket, addr = self.server_socket.accept()
        print(f"[FrameStreamServer] Client connected from {addr}")

    def send_frame(self, frame_bgr: np.ndarray):
        if self.client_socket is None:
            return

        # encode as JPEG
        ret, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ret:
            return

        data = buf.tobytes()
        length = len(data)

        with self.lock:
            # send length (4 bytes) then frame data
            self.client_socket.sendall(struct.pack("!I", length))
            self.client_socket.sendall(data)

    def stop(self):
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()


if __name__ == "__main__":
    # test: stream webcam frames
    import time
    cap = cv2.VideoCapture(0)
    server = FrameStreamServer()
    server.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            server.send_frame(frame)
            time.sleep(0.03)  # ~30 fps
    finally:
        cap.release()
        server.stop()
