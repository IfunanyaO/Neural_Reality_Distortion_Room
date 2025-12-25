# interaction/audio_controller.py

import pyaudio
import numpy as np
import threading
from .control_state import ControlState

class AudioController:
    def __init__(self, control_state: ControlState, device_index=None):
        self.control_state = control_state
        self.running = False
        self.device_index = device_index

    def start(self):
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False

    def _loop(self):
        CHUNK = 1024
        RATE = 44100

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=RATE,
                        input=True,
                        input_device_index=self.device_index,
                        frames_per_buffer=CHUNK)

        while self.running:
            data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.float32)
            energy = float(np.sqrt(np.mean(data**2)))

            # map to 0–1
            energy_norm = min(1.0, energy * 20.0)

            self.control_state.update(audio_energy=energy_norm,
                                      distortion_intensity=energy_norm)

        stream.stop_stream()
        stream.close()
        p.terminate()
