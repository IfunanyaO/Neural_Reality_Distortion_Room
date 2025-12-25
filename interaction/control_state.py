# interaction/control_state.py

from dataclasses import dataclass, field
from typing import Dict
import threading

@dataclass
class EffectState:
    # Core visual controls
    distortion_intensity: float = 0.3       # 0–1
    shader_mode: str = "swirl"              # "swirl", "ripple", "chromatic", ...
    style_prompt: str = "surreal melting chrome XR room"
    enable_sd: bool = True
    sd_frequency: float = 0.5      # run SD twice per second max (very stable)
    sd_strength: float = 0.55      # best value for SSD-1B


    # Audio-reactive & motion
    audio_energy: float = 0.0               # 0–1
    motion_energy: float = 0.0              # 0–1 (pose-based)

    # Extra arbitrary params
    extra: Dict[str, float] = field(default_factory=dict)

class ControlState:
    def __init__(self):
        self._state = EffectState()
        self._lock = threading.Lock()

    def get_state(self) -> EffectState:
        with self._lock:
            return EffectState(**self._state.__dict__)

    def update(self, **kwargs):
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._state, k):
                    setattr(self._state, k, v)
                else:
                    self._state.extra[k] = v
