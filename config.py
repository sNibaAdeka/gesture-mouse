from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    camera_index: int = 0
    model_path: Path = Path(__file__).parent / "models" / "hand_landmarker.task"
    width: int = 480
    height: int = 270
    detection_confidence: float = 0.60
    tracking_confidence: float = 0.60
    frame_margin: float = 0.10
    cursor_smoothing: float = 0.38
    pinch_ratio: float = 0.39
    release_ratio: float = 0.64
    click_cooldown_seconds: float = 0.28
    click_min_seconds: float = 0.04
    drag_seconds: float = 0.45
    keyboard_tap_velocity: float = 0.65
    keyboard_tap_cooldown_seconds: float = 0.20
