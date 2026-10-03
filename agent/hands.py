"""Hands: translate window coords to screen and drive mouse/keyboard."""

from __future__ import annotations

import sys
import time
from typing import Any, Callable

from agent.window import WindowRect, to_screen_xy


def _focus_blender_window(rect: WindowRect) -> None:
    """
    Bring Blender to the foreground WITHOUT clicking the viewport.
    Viewport clicks select/deselect meshes and break Add-menu workflows.
    """
    if sys.platform == "win32" and rect.hwnd:
        import ctypes

        user32 = ctypes.windll.user32
        hwnd = int(rect.hwnd)
        try:
            fg = user32.GetForegroundWindow()
            fg_tid = user32.GetWindowThreadProcessId(fg, None)
            target_tid = user32.GetWindowThreadProcessId(hwnd, None)
            attached = False
            if fg_tid and target_tid and fg_tid != target_tid:
                attached = bool(user32.AttachThreadInput(fg_tid, target_tid, True))
            # SW_RESTORE also un-maximizes; only restore when Blender is minimized.
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            user32.SetForegroundWindow(hwnd)
            user32.BringWindowToTop(hwnd)
            if attached:
                user32.AttachThreadInput(fg_tid, target_tid, False)
        except Exception:
            pass
        time.sleep(0.2)
        return
    time.sleep(0.1)


def _viewport_aim_xy(rect: WindowRect, aim_norm: tuple[float, float] | None = None) -> tuple[int, int]:
    """Screen point inside the 3D viewport (Blender routes keys to the area under the cursor).

    aim_norm comes from the addon's live viewport rect when available; otherwise a
    default-layout guess.
    """
    nx, ny = (0.45, 0.52)
    if aim_norm and len(aim_norm) == 2:
        nx = float(max(0.02, min(0.98, aim_norm[0])))
        ny = float(max(0.02, min(0.98, aim_norm[1])))
    x = int(rect.left + rect.width * nx)
    y = int(rect.top + rect.height * ny)
    return x, y


# Keys that must hit the 3D View (N-panel / Outliner steal focus if cursor is elsewhere).
_VIEWPORT_KEYS = frozenset(
    {
        "tab",
        "g",
        "r",
        "s",
        "e",
        "x",
        "delete",
        "i",
        "k",
        "j",
        "v",
        "f",
        "l",
        "h",
        "p",
        "n",
        "o",
        "b",
        "c",
        "z",
        "w",
        "y",
        "u",
        "m",
        "q",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
        "numpad1",
        "numpad2",
        "numpad3",
        "numpad4",
        "numpad5",
        "numpad6",
        "numpad7",
        "numpad8",
        "numpad9",
        "numpad0",
        "period",
        "comma",
        "slash",
        "backslash",
        "equal",
        "minus",
        "space",
        "enter",
        "return",
        "esc",
        "escape",
        "backspace",
    }
)


def _needs_viewport_aim(keys: list[str]) -> bool:
    for raw in keys:
        label = str(raw).strip().lower().replace("key.", "")
        if label in _VIEWPORT_KEYS or (len(label) == 1 and label.isalnum()):
            return True
    return False


def _clamp_screen_xy(rect: WindowRect, x: float, y: float) -> tuple[int, int]:
    sx, sy = to_screen_xy(rect, x, y)
    sx = max(rect.left, min(rect.right - 1, sx))
    sy = max(rect.top, min(rect.bottom - 1, sy))
    return sx, sy


def _drag_duration(x1: float, y1: float, x2: float, y2: float, explicit: float = 0.0) -> float:
    if explicit > 0:
        return float(max(0.03, min(2.0, explicit)))
    dist = ((float(x2) - float(x1)) ** 2 + (float(y2) - float(y1)) ** 2) ** 0.5
    return float(max(0.04, min(0.28, dist / 2200.0)))


def _as_points(raw: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(raw, list):
        return points
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            try:
                points.append((float(item[0]), float(item[1])))
            except (TypeError, ValueError):
                continue
    return points


_UNMAPPED_KEYS: set[str] = set()


def _pynput_key(label: str) -> Any:
    from pynput.keyboard import Key, KeyCode

    key = str(label).strip().lower().replace("key.", "")
    named = {
        "enter": Key.enter,
        "return": Key.enter,
        "esc": Key.esc,
        "escape": Key.esc,
        "tab": Key.tab,
        "space": Key.space,
        "backspace": Key.backspace,
        "delete": Key.delete,
        "del": Key.delete,
        "insert": Key.insert,
        "home": Key.home,
        "end": Key.end,
        "page_up": Key.page_up,
        "pageup": Key.page_up,
        "page_down": Key.page_down,
        "pagedown": Key.page_down,
        "up": Key.up,
        "down": Key.down,
        "left": Key.left,
        "right": Key.right,
        "shift": Key.shift,
        "ctrl": Key.ctrl,
        "control": Key.ctrl,
        "alt": Key.alt,
        "win": Key.cmd,
        "cmd": Key.cmd,
        # Punctuation the recorder/knowledge base spell out by name.
        "minus": "-",
        "period": ".",
        "comma": ",",
        "slash": "/",
        "backslash": "\\",
        "equal": "=",
        "plus": "+",
        "semicolon": ";",
        "quote": "'",
        "grave": "`",
        "bracketleft": "[",
        "bracketright": "]",
    }
    for i in range(1, 13):
        named[f"f{i}"] = getattr(Key, f"f{i}")
    if key in named:
        return named[key]
    if len(key) == 1:
        return key
    # Numpad keys carry Blender view hotkeys; main-row digits mean something else.
    if key.startswith("numpad") and key[6:].isdigit() and len(key) == 7:
        return KeyCode.from_vk(0x60 + int(key[6]))
    if key in {"numpad_period", "numpaddecimal", "numpad."}:
        return KeyCode.from_vk(0x6E)
    if key not in _UNMAPPED_KEYS:
        _UNMAPPED_KEYS.add(key)
        print(f"[hands] unmapped key label {key!r} — ignored", flush=True)
    return None


class Hands:
    def __init__(self, on_stop: Callable[[], None] | None = None):
        self._on_stop = on_stop
        self._esc_listener = None
        self._armed = False
        self._esc_armed_at = 0.0
        self._focused_hwnd = 0
        self.aim_norm: tuple[float, float] | None = None
        # The agent presses Esc itself (cancel modal slide etc.); the global
        # "Esc stops the agent" listener must ignore those synthetic presses.
        self._own_esc_until = 0.0
        # After F3 the search popup is open; moving the mouse into the viewport
        # for Enter/Esc would dismiss it.
        self._in_search = False

    def reset_run(self) -> None:
        self._focused_hwnd = 0
        self._in_search = False

    def _aim_viewport_for_keys(self, rect: WindowRect, *, force: bool = True) -> None:
        """Move cursor into the 3D View so Tab/E/G/etc. are not eaten by the N-panel."""
        import pyautogui

        if not force:
            return
        x, y = _viewport_aim_xy(rect, self.aim_norm)
        pyautogui.moveTo(x, y, duration=0)
        time.sleep(0.05)

    def arm_safety(self) -> None:
        if self._armed:
            return
        import pyautogui
        from pynput import keyboard

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
        if hasattr(pyautogui, "MINIMUM_DURATION"):
            pyautogui.MINIMUM_DURATION = 0
        if hasattr(pyautogui, "MINIMUM_SLEEP"):
            pyautogui.MINIMUM_SLEEP = 0
        self._esc_armed_at = time.time()

        def on_press(key: Any) -> None:
            try:
                if key != keyboard.Key.esc:
                    return
                if time.time() - self._esc_armed_at < 1.0:
                    return
                # Ignore Esc the agent itself just sent (loop cut / modal cancel).
                if time.time() < self._own_esc_until:
                    return
                if self._on_stop:
                    self._on_stop()
            except Exception:
                pass

        self._esc_listener = keyboard.Listener(on_press=on_press)
        self._esc_listener.start()
        self._armed = True

    def disarm(self) -> None:
        if self._esc_listener is not None:
            try:
                self._esc_listener.stop()
            except Exception:
                pass
            self._esc_listener = None
        self._armed = False
        self._focused_hwnd = 0

    def _ensure_focus(self, rect: WindowRect) -> bool:
        """Bring Blender forward if it is not already; returns True when it had to."""
        hwnd = int(rect.hwnd or 0)
        if hwnd and hwnd == self._focused_hwnd:
            if sys.platform != "win32":
                return False
            try:
                import ctypes

                if int(ctypes.windll.user32.GetForegroundWindow()) == hwnd:
                    return False
            except Exception:
                return False
            # User alt-tabbed away mid-run: re-focus instead of typing into another app.
        _focus_blender_window(rect)
        self._focused_hwnd = hwnd
        return True

    def _mark_own_esc(self, keys: list[Any]) -> None:
        labels = {str(k).strip().lower().replace("key.", "") for k in keys}
        if labels & {"esc", "escape"}:
            self._own_esc_until = time.time() + 1.0

    def _press_hotkey(self, keys: list[str]) -> None:
        from pynput.keyboard import Controller, Key

        self._mark_own_esc(keys)
        kb = Controller()
        resolved = [_pynput_key(k) for k in keys]
        resolved = [k for k in resolved if k is not None]
        if not resolved:
            return
        if len(resolved) == 1:
            kb.press(resolved[0])
            kb.release(resolved[0])
            return
        modifiers = [k for k in resolved if k in {Key.shift, Key.ctrl, Key.alt, Key.cmd}]
        normal = [k for k in resolved if k not in modifiers]
        pressed: list[Any] = []
        try:
            for mod in modifiers:
                kb.press(mod)
                pressed.append(mod)
            for key in normal:
                kb.press(key)
                kb.release(key)
        finally:
            # Never leave Ctrl/Shift/Alt held system-wide if a press raised.
            for mod in reversed(pressed):
                try:
                    kb.release(mod)
                except Exception:
                    pass

    def _press_key(self, key_label: str) -> None:
        from pynput.keyboard import Controller

        self._mark_own_esc([key_label])
        key = _pynput_key(key_label)
        if key is None:
            return
        kb = Controller()
        kb.press(key)
        kb.release(key)

    def _type_text(self, text: str) -> None:
        from pynput.keyboard import Controller

        if not text:
            return
        Controller().type(text)

    def _drag_along(
        self,
        rect: WindowRect,
        points: list[tuple[float, float]],
        duration: float,
    ) -> None:
        import pyautogui

        screen = [_clamp_screen_xy(rect, x, y) for x, y in points]
        if len(screen) < 2:
            return
        pyautogui.moveTo(screen[0][0], screen[0][1], duration=0)
        pyautogui.mouseDown(button="left")
        try:
            n = max(1, len(screen) - 1)
            step = max(0.0, float(duration) / float(n))
            t0 = time.perf_counter()
            for i, (x, y) in enumerate(screen[1:], start=1):
                pyautogui.moveTo(x, y, duration=0)
                remaining = (t0 + step * i) - time.perf_counter()
                if remaining > 0.001:
                    time.sleep(remaining)
        finally:
            pyautogui.mouseUp(button="left")

    def execute(self, action: dict[str, Any], rect: WindowRect) -> None:
        import pyautogui

        kind = str(action.get("action") or "wait")
        if kind == "stop":
            return
        if kind == "blender_apply":
            cb = getattr(self, "on_blender_apply", None)
            if cb:
                try:
                    cb(action.get("apply") if isinstance(action.get("apply"), dict) else {})
                except Exception:
                    pass
            deadline = time.time() + 0.7
            stop_event = getattr(self, "stop_event", None)
            while time.time() < deadline:
                if stop_event is not None and stop_event.is_set():
                    return
                time.sleep(0.05)
            return
        if kind == "wait":
            # Interruptible: Stop/Esc must not wait out a long recorded pause.
            try:
                from agent.policy import clamp_wait_seconds

                wait_sec = clamp_wait_seconds(action.get("seconds"), 0.5)
            except Exception:
                wait_sec = 0.5
            deadline = time.time() + wait_sec
            stop_event = getattr(self, "stop_event", None)
            while True:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return
                if stop_event is not None and stop_event.is_set():
                    return
                time.sleep(min(0.1, remaining))

        try:
            self._ensure_focus(rect)

            if kind == "type":
                # Typing usually follows F3 search; do not yank cursor into viewport mid-search.
                self._type_text(str(action.get("text") or ""))
                return

            if kind == "hotkey":
                keys = [str(k) for k in (action.get("keys") or []) if str(k).strip()]
                if keys:
                    if _needs_viewport_aim(keys):
                        self._aim_viewport_for_keys(rect, force=True)
                    self._in_search = False
                    self._press_hotkey(keys)
                return

            if kind == "key":
                keys = [str(k) for k in (action.get("keys") or []) if str(k).strip()]
                if keys:
                    label = keys[0].strip().lower().replace("key.", "")
                    if label == "f3":
                        # Search popup opens under the cursor: aim first, then keep still.
                        self._aim_viewport_for_keys(rect, force=True)
                        self._in_search = True
                    elif self._in_search and label in {"enter", "return", "esc", "escape"}:
                        # Moving the mouse now would close the F3 popup before Enter lands.
                        self._in_search = False
                    elif _needs_viewport_aim(keys):
                        self._aim_viewport_for_keys(rect, force=True)
                        self._in_search = False
                    self._press_key(keys[0])
                return

            self._in_search = False
            x = float(action.get("x") or 0)
            y = float(action.get("y") or 0)
            sx, sy = _clamp_screen_xy(rect, x, y)

            if kind == "click":
                pyautogui.click(sx, sy)
                return
            if kind == "drag":
                x2 = float(action.get("x2") or x)
                y2 = float(action.get("y2") or y)
                ex, ey = _clamp_screen_xy(rect, x2, y2)
                duration = _drag_duration(
                    sx, sy, ex, ey, float(action.get("duration") or 0.0)
                )
                points = _as_points(action.get("points"))
                if len(points) >= 2:
                    self._drag_along(rect, points, duration)
                    return
                pyautogui.moveTo(sx, sy, duration=0)
                pyautogui.dragTo(ex, ey, duration=duration, button="left")
                return
        except pyautogui.FailSafeException:
            if self._on_stop:
                self._on_stop()
            raise RuntimeError("PyAutoGUI failsafe: mouse hit a screen corner — agent stopped") from None
