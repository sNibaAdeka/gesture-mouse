from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import cv2
import mediapipe as mp


@dataclass
class HandPose:
    landmarks: list
    handedness: str
    pointer_active: bool
    index_extended: bool
    middle_extended: bool
    index_pinch_ratio: float
    middle_pinch_ratio: float
    right_click_pose: bool
    scroll_active: bool
    scroll_y: float


@dataclass
class HandState:
    hands: list[HandPose]

    @property
    def primary(self) -> HandPose | None:
        return self.hands[0] if self.hands else None

    @property
    def palms_joined(self) -> bool:
        if len(self.hands) != 2:
            return False
        first, second = self.hands
        first_size = self._distance(first.landmarks[0], first.landmarks[9])
        second_size = self._distance(second.landmarks[0], second.landmarks[9])
        gap = self._distance(first.landmarks[9], second.landmarks[9])
        return gap / max((first_size + second_size) / 2, 1e-5) < 1.45

    @staticmethod
    def _distance(first, second) -> float:
        return math.hypot(first.x - second.x, first.y - second.y)


class HandTracker:
    def __init__(self, model_path: Path, detection_confidence: float, tracking_confidence: float) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Не найдена модель рук: {model_path}")
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr, timestamp_ms: int) -> HandState:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        hands: list[HandPose] = []
        for index, landmarks in enumerate(result.hand_landmarks):
            labels = result.handedness[index] if index < len(result.handedness) else []
            name = labels[0].category_name if labels else f"hand-{index}"
            hands.append(self._features(list(landmarks), name))
        return HandState(hands)

    def draw(self, frame_bgr, state: HandState) -> None:
        connections = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20))
        for hand in state.hands:
            colour = (230, 130, 255) if hand.handedness == "Left" else (90, 255, 100)
            for start, end in connections:
                first, second = hand.landmarks[start], hand.landmarks[end]
                cv2.line(frame_bgr, (int(first.x * frame_bgr.shape[1]), int(first.y * frame_bgr.shape[0])), (int(second.x * frame_bgr.shape[1]), int(second.y * frame_bgr.shape[0])), colour, 2)
            tip = hand.landmarks[8]
            cv2.circle(frame_bgr, (int(tip.x * frame_bgr.shape[1]), int(tip.y * frame_bgr.shape[0])), 8, colour, 2)

    def close(self) -> None:
        self._landmarker.close()

    def _features(self, landmarks: list, handedness: str) -> HandPose:
        palm = max(self._distance(landmarks[0], landmarks[9]), 1e-5)
        index_pinch_ratio = self._distance(landmarks[4], landmarks[8]) / palm
        middle_pinch_ratio = self._distance(landmarks[4], landmarks[12]) / palm
        index_extended = landmarks[8].y < landmarks[6].y
        middle_extended = landmarks[12].y < landmarks[10].y
        return HandPose(
            landmarks=landmarks,
            handedness=handedness,
            pointer_active=index_extended and not middle_extended,
            index_extended=index_extended,
            middle_extended=middle_extended,
            index_pinch_ratio=index_pinch_ratio,
            middle_pinch_ratio=middle_pinch_ratio,
            right_click_pose=index_extended and middle_extended,
            scroll_active=index_extended and middle_extended and index_pinch_ratio > 0.78 and middle_pinch_ratio > 0.78,
            scroll_y=(landmarks[8].y + landmarks[12].y) / 2,
        )

    @staticmethod
    def _distance(first, second) -> float:
        return math.hypot(first.x - second.x, first.y - second.y)
