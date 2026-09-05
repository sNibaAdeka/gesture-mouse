from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pyautogui


@dataclass(frozen=True)
class Key:
    label: str
    value: str
    x: int
    y: int
    width: int
    height: int


class AirKeyboard:
    def __init__(self, width: int, height: int, tap_velocity: float, cooldown: float) -> None:
        self.width = width
        self.height = height
        self.tap_velocity = tap_velocity
        self.cooldown = cooldown
        self.keys = self._make_keys()
        self._previous_y: dict[str, tuple[float, float]] = {}
        self._armed: dict[str, bool] = {}
        self._last_tap = 0.0
        self.last_key = "—"

    def update(self, hands, now: float) -> str:
        pressed: list[str] = []
        for hand in hands:
            if not hand.index_extended:
                continue
            name = hand.handedness
            tip = hand.landmarks[8]
            key = self.key_at(tip.x, tip.y)
            previous = self._previous_y.get(name)
            velocity = 0.0 if previous is None else (tip.y - previous[0]) / max(now - previous[1], 1e-4)
            self._previous_y[name] = (tip.y, now)
            if velocity < -0.16:
                self._armed[name] = True
            if key and self._armed.get(name, True) and velocity > self.tap_velocity and now - self._last_tap > self.cooldown:
                self._press(key)
                self._armed[name] = False
                self._last_tap = now
                pressed.append(key.label)
        return f"НАЖАТО: {' '.join(pressed)}" if pressed else f"AR‑КЛАВИАТУРА · {self.last_key}"

    def reset(self) -> None:
        self._previous_y.clear()
        self._armed.clear()

    def draw_overlay(self, frame, hands, active: bool) -> None:
        if not active:
            cv2.putText(frame, "Соедини две ладони на 0.4 сек — включить AR-клавиатуру", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 235, 236), 1)
            return
        overlay = frame.copy()
        deck = np.array([(35, 225), (self.width - 35, 225), (self.width - 8, self.height - 8), (8, self.height - 8)], dtype=np.int32)
        cv2.fillConvexPoly(overlay, deck, (14, 47, 62))
        cv2.polylines(overlay, [deck], True, (114, 244, 236), 2)
        cv2.putText(overlay, "AR AIR KEYBOARD", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (111, 249, 240), 2)
        cv2.putText(overlay, "Два указательных пальца: наведи и сделай короткий удар вниз", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (214, 246, 245), 1)
        hovered = {key.value for hand in hands if hand.index_extended for key in [self.key_at(hand.landmarks[8].x, hand.landmarks[8].y)] if key}
        for key in self.keys:
            self._draw_key(overlay, key, key.value in hovered)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)
        for hand in hands:
            if not hand.index_extended:
                continue
            key = self.key_at(hand.landmarks[8].x, hand.landmarks[8].y)
            if key:
                colour = (232, 146, 255) if hand.handedness == "Left" else (118, 255, 164)
                cv2.putText(frame, f"{hand.handedness}: {key.label}", (20, 92 if hand.handedness == "Left" else 116), cv2.FONT_HERSHEY_SIMPLEX, 0.52, colour, 1)

    def key_at(self, x: float, y: float) -> Key | None:
        px, py = int(x * self.width), int(y * self.height)
        for key in self.keys:
            if key.x <= px <= key.x + key.width and key.y <= py <= key.y + key.height:
                return key
        return None

    def _press(self, key: Key) -> None:
        if key.value == "BACKSPACE":
            pyautogui.press("backspace", _pause=False)
        elif key.value == "ENTER":
            pyautogui.press("enter", _pause=False)
        elif key.value == "SPACE":
            pyautogui.press("space", _pause=False)
        else:
            pyautogui.write(key.value, _pause=False)
        self.last_key = key.label

    def _make_keys(self) -> list[Key]:
        layout = [("QWERTYUIOP", 255), ("ASDFGHJKL", 323), ("ZXCVBNM", 391)]
        keys: list[Key] = []
        key_width, key_height, gap = 74, 56, 8
        for letters, y in layout:
            start = (self.width - (len(letters) * key_width + (len(letters) - 1) * gap)) // 2
            for index, letter in enumerate(letters):
                keys.append(Key(letter, letter.lower(), start + index * (key_width + gap), y, key_width, key_height))
        keys.extend([
            Key("⌫", "BACKSPACE", 184, 463, 100, key_height),
            Key("SPACE", "SPACE", 300, 463, 360, key_height),
            Key("↵", "ENTER", 676, 463, 100, key_height),
        ])
        return keys

    @staticmethod
    def _draw_key(canvas, key: Key, hovered: bool) -> None:
        top = (69, 202, 194) if hovered else (38, 103, 127)
        edge = (25, 83, 102) if hovered else (11, 42, 56)
        cv2.rectangle(canvas, (key.x + 4, key.y + 7), (key.x + key.width + 4, key.y + key.height + 7), edge, -1)
        cv2.rectangle(canvas, (key.x, key.y), (key.x + key.width, key.y + key.height), top, -1)
        cv2.rectangle(canvas, (key.x, key.y), (key.x + key.width, key.y + key.height), (139, 247, 240), 1)
        size = 0.58 if len(key.label) <= 2 else 0.46
        text_width = cv2.getTextSize(key.label, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0]
        cv2.putText(canvas, key.label, (key.x + (key.width - text_width) // 2, key.y + 35), cv2.FONT_HERSHEY_SIMPLEX, size, (245, 255, 255), 1)
