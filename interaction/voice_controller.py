# interaction/voice_controller.py

import threading
from .control_state import ControlState

try:
    import speech_recognition as sr
    HAS_SR = True
except ImportError:
    HAS_SR = False


class VoiceController:
    def __init__(self, control_state: ControlState):
        self.control_state = control_state
        self.running = False

    def start(self):
        if not HAS_SR:
            print("[VoiceController] speech_recognition not installed; skipping.")
            return
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False

    def _loop(self):
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)

        while self.running:
            try:
                with mic as source:
                    print("[VoiceController] Listening...")
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                cmd = recognizer.recognize_google(audio).lower()
                print("[VoiceController] Heard:", cmd)
                self._handle_command(cmd)
            except Exception as e:
                print("[VoiceController] Error:", e)

    def _handle_command(self, cmd: str):
        # very simple mapping, you can extend
        if "chrome" in cmd or "metal" in cmd:
            self.control_state.update(style_prompt="melting chrome surreal XR room")
        elif "glass" in cmd or "crystal" in cmd:
            self.control_state.update(style_prompt="holographic glass, crystal dream room")
        elif "calm" in cmd or "soft" in cmd:
            self.control_state.update(
                style_prompt="soft watercolor dream, pastel ethereal ambience",
                distortion_intensity=0.2
            )
        elif "intense" in cmd or "crazy" in cmd:
            self.control_state.update(distortion_intensity=0.9)
