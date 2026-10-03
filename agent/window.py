"""Find the Blender window and capture its client area."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass

import numpy as np

_tls = threading.local()
_dpi_done = False

# Blender's Win32 window class (GHOST). Matching on it keeps a browser tab titled
# "Blender tutorial" from being picked as the target.
_BLENDER_WINDOW_CLASS = "GHOST_WindowClass"


def ensure_dpi_awareness() -> None:
    """Make coordinates consistent across pyautogui/mss/ctypes on scaled displays.

    Must run before the first pyautogui/mss import; those set awareness as a side
    effect and would otherwise decide it for us (differently per import order).
    """
    global _dpi_done
    if _dpi_done or sys.platform != "win32":
        return
    _dpi_done = True
    try:
        import ctypes

        user32 = ctypes.windll.user32
        try:
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == -4
            if not user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                raise OSError("SetProcessDpiAwarenessContext failed")
        except (AttributeError, OSError):
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                user32.SetProcessDPIAware()
    except Exception:
        pass


@dataclass
class WindowRect:
    left: int
    top: int
    width: int
    height: int
    hwnd: int = 0
    title: str = ""

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


def _windows_find_blender(title_substr: str = "Blender") -> WindowRect | None:
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    results: list[WindowRect] = []

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)

    def _callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if title_substr.lower() not in title.lower():
            return True
        cls_buf = ctypes.create_unicode_buffer(128)
        user32.GetClassNameW(hwnd, cls_buf, 128)
        if cls_buf.value != _BLENDER_WINDOW_CLASS:
            # Not a Blender window (e.g. browser tab about Blender).
            return True
        rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return True
        point = wintypes.POINT(0, 0)
        user32.ClientToScreen(hwnd, ctypes.byref(point))
        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width < 100 or height < 100:
            return True
        results.append(
            WindowRect(
                left=int(point.x),
                top=int(point.y),
                width=width,
                height=height,
                hwnd=int(hwnd),
                title=title,
            )
        )
        return True

    user32.EnumWindows(EnumWindowsProc(_callback), 0)
    if not results:
        return None
    # Prefer the largest client area (main Blender window).
    return max(results, key=lambda r: r.width * r.height)


def find_blender_window(title_substr: str = "Blender") -> WindowRect | None:
    if sys.platform == "win32":
        ensure_dpi_awareness()
        return _windows_find_blender(title_substr)
    # Fallback: full primary monitor (non-Windows).
    try:
        import mss

        with mss.mss() as sct:
            mon = sct.monitors[1]
            return WindowRect(
                left=int(mon["left"]),
                top=int(mon["top"]),
                width=int(mon["width"]),
                height=int(mon["height"]),
                title="primary",
            )
    except Exception:
        return None


def _mss():
    """Reuse one mss handle per thread — constructing it every frame tanks FPS."""
    import mss

    sct = getattr(_tls, "sct", None)
    if sct is None:
        sct = mss.mss()
        _tls.sct = sct
    return sct


def capture_window(rect: WindowRect | None = None) -> tuple[np.ndarray, WindowRect]:
    """Return BGR uint8 image and the rect used."""
    target = rect or find_blender_window()
    if target is None:
        raise RuntimeError("Blender window not found. Open Blender and try again.")

    monitor = {
        "left": target.left,
        "top": target.top,
        "width": target.width,
        "height": target.height,
    }
    try:
        sct = _mss()
        shot = sct.grab(monitor)
    except Exception:
        _tls.sct = None
        sct = _mss()
        shot = sct.grab(monitor)
    frame = np.array(shot, dtype=np.uint8)
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise RuntimeError("Window capture returned an unexpected image shape")
    bgr = np.ascontiguousarray(frame[:, :, :3])
    return bgr, target


def to_screen_xy(rect: WindowRect, x: float, y: float) -> tuple[int, int]:
    return int(rect.left + x), int(rect.top + y)


def downsample_for_vlm(bgr: np.ndarray, max_side: int = 672) -> np.ndarray:
    """Downscale for Ollama vision (keeps aspect). Returns BGR."""
    import cv2

    h, w = bgr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale >= 0.999:
        return bgr
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def encode_jpeg_b64(bgr: np.ndarray, quality: int = 80) -> str:
    import base64

    import cv2

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def encode_png_bytes(bgr: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    return bytes(buf)
