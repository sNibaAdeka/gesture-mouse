from __future__ import annotations

import time

import cv2

from air_keyboard import AirKeyboard
from config import Config
from controller import MouseController
from hand_tracker import HandTracker


WINDOW_NAME = "Gesture Desk"
WINDOW_WIDTH = 960
WINDOW_HEIGHT = 540


def main() -> None:
    config = Config()
    camera = cv2.VideoCapture(config.camera_index)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    camera.set(cv2.CAP_PROP_FPS, 30)
    if not camera.isOpened():
        raise RuntimeError("Камера не найдена. Проверь разрешение Camera для Terminal/Python.")

    tracker = HandTracker(config.model_path, config.detection_confidence, config.tracking_confidence)
    mouse = MouseController(config.frame_margin, config.cursor_smoothing, config.pinch_ratio, config.release_ratio, config.click_cooldown_seconds, config.click_min_seconds, config.drag_seconds)
    keyboard = AirKeyboard(WINDOW_WIDTH, WINDOW_HEIGHT, config.keyboard_tap_velocity, config.keyboard_tap_cooldown_seconds)
    last_timestamp = 0
    join_started_at: float | None = None
    join_latched = False
    last_toggle_at = 0.0
    keyboard_active = False
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 720, 405)

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            timestamp_ms = max(last_timestamp + 1, int(time.monotonic() * 1000))
            last_timestamp = timestamp_ms
            state = tracker.detect(frame, timestamp_ms)
            tracker.draw(frame, state)
            now = time.monotonic()

            if state.palms_joined and not join_latched:
                join_started_at = now if join_started_at is None else join_started_at
                if now - join_started_at >= 0.40 and now - last_toggle_at >= 1.0:
                    keyboard_active = not keyboard_active
                    keyboard.reset()
                    last_toggle_at = now
                    join_latched = True
            else:
                if not state.palms_joined:
                    join_started_at = None
                    join_latched = False

            output = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT), interpolation=cv2.INTER_LINEAR)
            if keyboard_active:
                mouse.update(None)
                action = keyboard.update(state.hands, now)
            else:
                keyboard.reset()
                action = mouse.update(state.primary)
            keyboard.draw_overlay(output, state.hands, keyboard_active)

            if state.palms_joined:
                cv2.putText(output, "ЛАДОНИ СОЕДИНЕНЫ: удержи 0.4 сек", (20, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (116, 245, 239), 2)
            mode = "AR-KEYBOARD" if keyboard_active else "MOUSE"
            colour = (99, 255, 188) if keyboard_active else (99, 215, 255)
            cv2.putText(output, f"{mode} | {action}", (20, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.54, colour, 1)
            cv2.putText(output, "Q / Esc: выход", (20, 204), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (216, 235, 236), 1)
            cv2.imshow(WINDOW_NAME, output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        mouse.update(None)
        tracker.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
