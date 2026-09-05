from __future__ import annotations

import time

import pyautogui


class MouseController:
    def __init__(self, margin: float, smoothing: float, pinch_ratio: float, release_ratio: float, cooldown: float, click_min_seconds: float, drag_seconds: float) -> None:
        pyautogui.PAUSE = 0
        pyautogui.MINIMUM_DURATION = 0
        pyautogui.DARWIN_CATCH_UP_TIME = 0
        self._screen_width, self._screen_height = pyautogui.size()
        self._margin = margin
        self._smoothing = smoothing
        self._pinch_ratio = pinch_ratio
        self._release_ratio = release_ratio
        self._cooldown = cooldown
        self._click_min_seconds = click_min_seconds
        self._drag_seconds = drag_seconds
        self._x: float | None = None
        self._y: float | None = None
        self._click_candidate = False
        self._click_started = 0.0
        self._dragging = False
        self._right_candidate = False
        self._right_started = 0.0
        self._last_click = 0.0
        self._last_scroll_y: float | None = None
        self._last_pointer_at = 0.0

    def update(self, hand) -> str:
        if hand is None:
            self._cancel_pending()
            self._last_scroll_y = None
            return "Пауза: рука не видна"

        now = time.monotonic()
        if hand.scroll_active:
            self._cancel_pending()
            self._scroll(hand.scroll_y)
            return "ПРОКРУТКА: два пальца"
        self._last_scroll_y = None

        if hand.pointer_active:
            self._last_pointer_at = now
            self._move_cursor(hand.landmarks[8].x, hand.landmarks[8].y)
            return self._left_gesture(hand, now)

        if self._click_candidate:
            return self._left_gesture(hand, now)
        if hand.index_pinch_ratio < self._pinch_ratio and now - self._last_pointer_at < 0.35:
            return self._left_gesture(hand, now)

        self._cancel_left_if_needed()
        if hand.right_click_pose:
            return self._right_gesture(hand, now)
        self._right_candidate = False
        return "Пауза: покажи только указательный палец"

    def _left_gesture(self, hand, now: float) -> str:
        if hand.index_pinch_ratio < self._pinch_ratio:
            if not self._click_candidate:
                self._click_candidate = True
                self._click_started = now
                return "PINCH: удерживай или отпусти для клика"
            if not self._dragging and now - self._click_started >= self._drag_seconds:
                pyautogui.mouseDown(_pause=False)
                self._dragging = True
            return "ПЕРЕТАСКИВАНИЕ" if self._dragging else "PINCH: готовится клик"

        if self._click_candidate and hand.index_pinch_ratio > self._release_ratio:
            held_for = now - self._click_started
            was_dragging = self._dragging
            self._cancel_left_if_needed()
            if not was_dragging and self._click_min_seconds <= held_for < self._drag_seconds and now - self._last_click > self._cooldown:
                pyautogui.click(_pause=False)
                self._last_click = now
                return "ЛЕВЫЙ КЛИК"
        return "Курсор: один указательный палец"

    def _right_gesture(self, hand, now: float) -> str:
        if hand.middle_pinch_ratio < self._pinch_ratio:
            if not self._right_candidate:
                self._right_candidate = True
                self._right_started = now
                return "ПРАВЫЙ КЛИК: отпусти пальцы"
            return "ПРАВЫЙ КЛИК: отпусти пальцы"
        if self._right_candidate and hand.middle_pinch_ratio > self._release_ratio:
            held_for = now - self._right_started
            self._right_candidate = False
            if held_for >= self._click_min_seconds:
                pyautogui.rightClick(_pause=False)
                return "ПРАВЫЙ КЛИК"
        return "Укажи двумя пальцами для правого клика"

    def _move_cursor(self, x: float, y: float) -> None:
        target_x = self._map(x, self._screen_width)
        target_y = self._map(y, self._screen_height)
        self._x = target_x if self._x is None else self._x + (target_x - self._x) * self._smoothing
        self._y = target_y if self._y is None else self._y + (target_y - self._y) * self._smoothing
        pyautogui.moveTo(int(self._x), int(self._y), duration=0, _pause=False)

    def _scroll(self, current_y: float) -> None:
        if self._last_scroll_y is not None:
            delta = self._last_scroll_y - current_y
            if abs(delta) > 0.009:
                pyautogui.scroll(int(delta * 85), _pause=False)
        self._last_scroll_y = current_y

    def _cancel_left_if_needed(self) -> None:
        if self._dragging:
            pyautogui.mouseUp(_pause=False)
        self._dragging = False
        self._click_candidate = False

    def _cancel_pending(self) -> None:
        self._cancel_left_if_needed()
        self._right_candidate = False

    def _map(self, value: float, screen_size: int) -> float:
        clipped = min(max(value, self._margin), 1.0 - self._margin)
        normalised = (clipped - self._margin) / (1.0 - 2.0 * self._margin)
        return 1 + normalised * (screen_size - 3)
