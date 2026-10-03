"""Teacher module: record Blender window + mouse/keyboard for imitation.

Supports two capture modes:
- video (default): continuous OpenCV VideoWriter + timestamped events
- screenshots: legacy PNG dumps every interval_ms (previous-shot linking)
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from agent.paths import (
    ensure_dataset,
    episodes_path,
    last_session_path,
    screenshots_dir,
    videos_dir,
)
from agent.window import WindowRect, capture_window, encode_png_bytes, find_blender_window

CAPTURE_MODES = ("video", "screenshots")
# Forgotten Record sessions must not capture forever. Six hours is far longer
# than a teaching take, so a normal Record → Stop session is unchanged.
MAX_RECORD_SECONDS = 6 * 60 * 60
MAX_DRAG_SECONDS = 2.0


def record_duration_exceeded(started_at: float | None, now: float) -> bool:
    if started_at is None:
        return False
    try:
        elapsed = float(now) - float(started_at)
    except (TypeError, ValueError):
        return False
    if elapsed != elapsed:  # NaN
        return False
    return elapsed >= MAX_RECORD_SECONDS


def bounded_drag_duration(raw: Any) -> float | None:
    """Clamp a recorded drag to the same 2s ceiling replay already uses."""
    try:
        duration = float(raw)
    except (TypeError, ValueError):
        return None
    if duration != duration or duration <= 0.0:  # NaN or non-positive
        return None
    return min(MAX_DRAG_SECONDS, duration)


def _normalize_capture_mode(raw: Any) -> str:
    mode = str(raw or "video").strip().lower()
    if mode in {"screenshot", "png", "images", "image"}:
        return "screenshots"
    if mode not in CAPTURE_MODES:
        return "video"
    return mode


def _fps_from_interval_ms(interval_ms: float) -> float:
    interval = max(16.0, float(interval_ms))
    fps = 1000.0 / interval
    return float(max(5.0, min(60.0, fps)))


def _thin_points(
    points: list[tuple[int, int]],
    *,
    max_points: int = 48,
) -> list[list[int]]:
    """Keep first/last and drop tiny jitter so replay stays smooth without huge JSON."""
    if not points:
        return []
    cleaned: list[tuple[int, int]] = [points[0]]
    for x, y in points[1:]:
        px, py = cleaned[-1]
        if abs(int(x) - px) + abs(int(y) - py) >= 2:
            cleaned.append((int(x), int(y)))
    if len(cleaned) == 1:
        return [[cleaned[0][0], cleaned[0][1]]]
    if len(cleaned) <= max_points:
        return [[x, y] for x, y in cleaned]
    out: list[list[int]] = [[cleaned[0][0], cleaned[0][1]]]
    last_i = len(cleaned) - 1
    for i in range(1, max_points - 1):
        idx = int(round(i * last_i / float(max_points - 1)))
        pt = [cleaned[idx][0], cleaned[idx][1]]
        if pt != out[-1]:
            out.append(pt)
    end = [cleaned[-1][0], cleaned[-1][1]]
    if end != out[-1]:
        out.append(end)
    return out


_MODIFIER_ALIASES = {
    "ctrl": "ctrl",
    "control": "ctrl",
    "ctrl_l": "ctrl",
    "ctrl_r": "ctrl",
    "alt": "alt",
    "alt_l": "alt",
    "alt_r": "alt",
    "alt_gr": "alt",
    "shift": "shift",
    "shift_l": "shift",
    "shift_r": "shift",
    "cmd": "cmd",
    "win": "cmd",
    "cmd_l": "cmd",
    "cmd_r": "cmd",
}
_MODIFIER_ORDER = ("ctrl", "alt", "shift", "cmd")


def _normalize_key_label(raw: Any) -> str:
    label = str(raw or "").strip()
    if not label:
        return ""
    if label.startswith("Key."):
        label = label[4:]
    if len(label) >= 2 and label[0] == label[-1] and label[0] in {"'", '"'}:
        label = label[1:-1]
    label = label.strip().lower()
    if label in {"return"}:
        return "enter"
    if label in {"escape"}:
        return "esc"
    return label


def _canonical_modifier(label: str) -> str | None:
    return _MODIFIER_ALIASES.get(_normalize_key_label(label))


def _is_modifier(label: str) -> bool:
    return _canonical_modifier(label) is not None


def _resolve_pynput_key(key: Any) -> str:
    """Turn a pynput key into a stable label (e, tab, f3, ctrl, …)."""
    name = getattr(key, "name", None)
    if isinstance(name, str) and name:
        return _normalize_key_label(name)

    char = getattr(key, "char", None)
    vk = getattr(key, "vk", None)
    # Numpad first: pynput reports numpad 1 as char "1", but in Blender numpad
    # digits are view hotkeys while main-row digits switch select mode / collections.
    if isinstance(vk, int) and 0x60 <= vk <= 0x69:
        return f"numpad{vk - 0x60}"
    if isinstance(vk, int) and vk == 0x6E:
        return "numpad_period"
    if isinstance(char, str) and char:
        code = ord(char)
        # Ctrl+letter arrives as a control character; recover via virtual-key.
        if code < 32 and isinstance(vk, int) and 65 <= vk <= 90:
            return chr(vk).lower()
        if char.isprintable():
            return _normalize_key_label(char)

    if isinstance(vk, int) and 65 <= vk <= 90:
        return chr(vk).lower()
    if isinstance(vk, int) and 112 <= vk <= 123:  # F1–F12
        return f"f{vk - 111}"

    return _normalize_key_label(str(key))


def _ordered_hotkey(mods: set[str], key: str) -> list[str]:
    keys = [m for m in _MODIFIER_ORDER if m in mods]
    keys.append(key)
    return keys


def summarize_recorded_inputs(events: list[dict[str, Any]], *, limit: int = 12) -> str:
    """Short human summary of clicks/keys for status text."""
    clicks = 0
    drags = 0
    for ev in events:
        if not isinstance(ev, dict):
            continue
        et = ev.get("type")
        if et == "click" and bool(ev.get("pressed", True)):
            clicks += 1
        elif et == "drag":
            drags += 1
    key_labels: list[str] = []
    for action in events_to_actions(events):
        kind = str(action.get("action") or "")
        if kind not in {"key", "hotkey"}:
            continue
        raw = action.get("keys") or []
        if isinstance(raw, str):
            raw = [raw]
        parts = [_normalize_key_label(k) for k in raw if _normalize_key_label(k)]
        if not parts:
            continue
        key_labels.append("+".join(parts) if len(parts) > 1 else parts[0])
    bits: list[str] = []
    if clicks:
        bits.append(f"{clicks} click{'s' if clicks != 1 else ''}")
    if drags:
        bits.append(f"{drags} drag{'s' if drags != 1 else ''}")
    if key_labels:
        shown = key_labels[:limit]
        extra = len(key_labels) - len(shown)
        joined = ", ".join(shown)
        if extra > 0:
            joined = f"{joined}, +{extra} more"
        bits.append(f"{len(key_labels)} key{'s' if len(key_labels) != 1 else ''}: {joined}")
    return "; ".join(bits) if bits else "no clicks/keys"


def _to_window_xy(rect: WindowRect | None, x: int, y: int) -> tuple[int, int, str]:
    if rect is None:
        return int(x), int(y), "screen"
    return int(x) - int(rect.left), int(y) - int(rect.top), "window"


def extract_video_frame(
    video_path: Path | str,
    *,
    frame_idx: int | None = None,
    t_rel: float | None = None,
    fps: float | None = None,
) -> Any | None:
    """Return a BGR frame from a recorded video, or None on failure."""
    import cv2
    import numpy as np

    path = Path(video_path)
    if not path.is_file():
        return None
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        video_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or float(fps or 15.0)
        if frame_idx is None:
            if t_rel is None:
                frame_idx = 0
            else:
                frame_idx = max(0, int(round(float(t_rel) * video_fps)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return np.asarray(frame)
    finally:
        cap.release()


def events_to_actions(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert recorded input events into Hands-compatible action steps."""
    useful = [
        e
        for e in events
        if isinstance(e, dict) and e.get("type") in {"click", "drag", "key", "hotkey"}
    ]
    if not useful:
        return []

    actions: list[dict[str, Any]] = []
    i = 0
    prev_t: float | None = None

    def _maybe_wait(t: float | None) -> None:
        nonlocal prev_t
        if t is None or prev_t is None:
            if t is not None:
                prev_t = t
            return
        delay = max(0.0, float(t) - float(prev_t))
        if delay >= 0.05:
            actions.append(
                {
                    "action": "wait",
                    "seconds": min(2.0, delay),
                    "reason": "recorded pause",
                }
            )
        prev_t = t

    while i < len(useful):
        ev = useful[i]
        et = ev.get("type")
        t = float(ev.get("t") or 0.0) or None

        if et == "drag":
            _maybe_wait(t)
            drag: dict[str, Any] = {
                "action": "drag",
                "x": float(ev.get("x1") or 0),
                "y": float(ev.get("y1") or 0),
                "x2": float(ev.get("x2") or 0),
                "y2": float(ev.get("y2") or 0),
                "reason": "recorded drag",
            }
            duration = bounded_drag_duration(ev.get("duration") or 0.0)
            if duration is not None:
                drag["duration"] = duration
            pts = ev.get("points")
            if isinstance(pts, list) and len(pts) >= 2:
                drag["points"] = pts
            actions.append(drag)
            i += 1
            continue

        if et == "hotkey":
            raw = ev.get("keys") or []
            if isinstance(raw, str):
                raw = [raw]
            keys = [_normalize_key_label(k) for k in raw if _normalize_key_label(k)]
            keys = [k for k in keys if k]
            if keys:
                _maybe_wait(t)
                actions.append(
                    {
                        "action": "hotkey" if len(keys) > 1 else "key",
                        "keys": keys,
                        "reason": f"recorded {'+'.join(keys)}",
                    }
                )
            i += 1
            continue

        if et == "key":
            key = _normalize_key_label(ev.get("key"))
            if not key or _is_modifier(key):
                # Bare modifiers are tracked only for chord look-back below.
                i += 1
                continue
            chord_mods: set[str] = set()
            j = i - 1
            while j >= 0:
                prev = useful[j]
                if prev.get("type") != "key":
                    break
                prev_key = _normalize_key_label(prev.get("key"))
                prev_mod = _canonical_modifier(prev_key)
                if not prev_mod:
                    break
                prev_t_val = float(prev.get("t") or 0.0)
                if t is not None and abs(float(t) - prev_t_val) > 0.6:
                    break
                chord_mods.add(prev_mod)
                j -= 1
            _maybe_wait(t)
            if chord_mods:
                keys = _ordered_hotkey(chord_mods, key)
                actions.append(
                    {
                        "action": "hotkey",
                        "keys": keys,
                        "reason": f"recorded {'+'.join(keys)}",
                    }
                )
            else:
                actions.append(
                    {
                        "action": "key",
                        "keys": [key],
                        "reason": f"recorded key {key}",
                    }
                )
            i += 1
            continue

        if et == "click":
            pressed = bool(ev.get("pressed", True))
            # Prefer a press event; skip bare releases.
            if not pressed:
                i += 1
                continue
            x = float(ev.get("x") or 0)
            y = float(ev.get("y") or 0)
            j = i + 1
            # Press → drag → release: keep only the drag.
            if j < len(useful) and useful[j].get("type") == "drag":
                i = j
                continue
            # Press → matching release: one click.
            if j < len(useful):
                nxt = useful[j]
                if (
                    nxt.get("type") == "click"
                    and not bool(nxt.get("pressed", True))
                    and abs(float(nxt.get("x") or 0) - x) <= 3
                    and abs(float(nxt.get("y") or 0) - y) <= 3
                ):
                    i = j  # advance to release; loop will +1
            _maybe_wait(t)
            actions.append(
                {
                    "action": "click",
                    "x": x,
                    "y": y,
                    "reason": "recorded click",
                }
            )
            i += 1
            continue

        i += 1

    if actions and actions[-1].get("action") != "stop":
        actions.append({"action": "stop", "reason": "end of recorded session"})
    return actions


class Recorder:
    def __init__(
        self,
        dataset_root: Path | None = None,
        interval_ms: float = 80.0,
        capture_mode: str = "video",
    ):
        self.root = ensure_dataset(dataset_root)
        self.interval_ms = float(interval_ms)
        self.capture_mode = _normalize_capture_mode(capture_mode)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._listener = None
        self._key_listener = None
        self._lock = threading.Lock()
        self._pending: list[dict[str, Any]] = []
        self._session_events: list[dict[str, Any]] = []
        self._last_shot_id: str | None = None
        self._last_rect: WindowRect | None = None
        self._drag_start: tuple[int, int] | None = None
        self._drag_start_local: tuple[int, int, str] | None = None
        self._drag_points: list[tuple[int, int]] = []
        self._drag_t0: float | None = None
        self._last_move_t: float = 0.0
        self._held_modifiers: set[str] = set()
        self._last_key_event: tuple[str, float] | None = None
        self._running = False
        self._shots = 0
        self._frames = 0
        self._events = 0
        self._error = ""
        self._session_id: str | None = None
        self._session_started_at: float | None = None
        self._video_rel: str | None = None
        self._video_path: Path | None = None
        self._writer = None
        self._writer_size: tuple[int, int] | None = None
        self._fps = 15.0
        self._requested_mode = self.capture_mode
        self._session_goal = ""
        self._session_saved = False

    @property
    def running(self) -> bool:
        return self._running

    def status(self) -> dict[str, Any]:
        meta = self._read_last_session_meta()
        event_count = int(meta.get("event_count") or 0) if meta else 0
        return {
            "recording": self._running,
            "shots": self._shots,
            "frames": self._frames,
            "events": self._events,
            "dataset": str(self.root),
            "error": self._error,
            "session_id": self._session_id or (meta.get("session_id") if meta else ""),
            "last_session_events": event_count,
            "replay_available": event_count > 0,
            "capture_mode": self.capture_mode,
            "video": self._video_rel or (meta.get("video") if meta else ""),
            "fps": self._fps,
            "last_session_goal": (meta.get("goal") if meta else "") or self._session_goal,
        }

    def start(
        self,
        interval_ms: float | None = None,
        capture_mode: str | None = None,
        goal: str = "",
    ) -> None:
        if self._running:
            return
        if interval_ms is not None:
            self.interval_ms = float(interval_ms)
        if capture_mode is not None:
            self.capture_mode = _normalize_capture_mode(capture_mode)
        self._requested_mode = self.capture_mode
        self._stop.clear()
        self._error = ""
        self._shots = 0
        self._frames = 0
        self._events = 0
        self._pending = []
        self._session_events = []
        self._last_shot_id = None
        self._last_rect = None
        self._video_rel = None
        self._video_path = None
        self._writer = None
        self._writer_size = None
        self._fps = _fps_from_interval_ms(self.interval_ms)
        self._session_id = uuid.uuid4().hex[:12]
        self._session_started_at = time.time()
        self._session_goal = str(goal or "").strip()
        self._session_saved = False
        self._held_modifiers = set()
        self._last_key_event = None
        self._drag_start = None
        self._drag_start_local = None
        self._drag_points = []
        self._drag_t0 = None
        self._last_move_t = 0.0
        self._running = True
        self._start_input_listener()
        self._thread = threading.Thread(target=self._loop, name="agent-recorder", daemon=True)
        self._thread.start()

    def stop(self, goal: str = "") -> None:
        self._stop.set()
        self._running = False
        # Stop goal overrides the goal locked at start (if provided).
        if str(goal or "").strip():
            self._session_goal = str(goal).strip()
        self._stop_input_listeners()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                # Capture thread is stuck in grab()/write(); releasing the writer
                # underneath it can crash natively. Give it more time, then let
                # the thread's own exit path release it.
                self._thread.join(timeout=5.0)
            still_alive = self._thread.is_alive()
            self._thread = None
        else:
            still_alive = False
        # A stuck grab/write must keep the writer; releasing it can crash natively.
        self._close_session_files(release_writer=not still_alive)

    def _stop_input_listeners(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
            self._listener = None
        if self._key_listener is not None:
            try:
                self._key_listener.stop()
            except Exception:
                pass
            self._key_listener = None
        self._held_modifiers = set()
        self._last_key_event = None

    def _close_session_files(self, *, release_writer: bool) -> None:
        with self._lock:
            if self._session_saved:
                return
            self._session_saved = True
        if release_writer:
            self._release_writer()
        if self.capture_mode == "screenshots":
            self._flush_pending()
        else:
            self._flush_video_session()
        self._write_last_session()

    def _expire_recording(self) -> None:
        """Stop a forgotten session at MAX_RECORD_SECONDS without joining this thread."""
        self._error = "Recording stopped at the 6 hour safety limit."
        self._stop.set()
        self._running = False
        self._stop_input_listeners()
        self._close_session_files(release_writer=True)

    def _release_writer(self) -> None:
        with self._lock:
            writer = self._writer
            self._writer = None
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

    def _write_last_session(self) -> None:
        with self._lock:
            events = list(self._session_events)
            session_id = self._session_id
            started = self._session_started_at
            video_rel = self._video_rel
            frames = self._frames
            fps = self._fps
            mode = self.capture_mode
            goal = self._session_goal
        if not session_id:
            return
        payload = {
            "session_id": session_id,
            "goal": goal or "",
            "started_at": started,
            "ended_at": time.time(),
            "event_count": len(events),
            "events": events,
            "capture_mode": mode,
            "video": video_rel or "",
            "fps": fps,
            "frame_count": frames,
        }
        path = last_session_path(self.root)
        # Atomic replace: /status polls read this file while we write it.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def _read_last_session_meta(self) -> dict[str, Any] | None:
        path = last_session_path(self.root)
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def last_session_actions(self) -> list[dict[str, Any]]:
        """Return Hands actions for the most recent Record→Stop session."""
        meta = self._read_last_session_meta()
        if not meta:
            return []
        events = meta.get("events")
        if not isinstance(events, list):
            events = []
        # Prefer events stored on the session file; fall back to episodes by session_id.
        if not events:
            events = self._load_session_events_from_episodes(str(meta.get("session_id") or ""))
        # Only replay window-local mouse events (or key-only sequences).
        filtered: list[dict[str, Any]] = []
        for ev in events:
            if not isinstance(ev, dict):
                continue
            et = ev.get("type")
            if et in {"key", "hotkey"}:
                filtered.append(ev)
                continue
            if et in {"click", "drag"} and ev.get("coord_space") == "window":
                filtered.append(ev)
        return events_to_actions(filtered)

    def last_session_events_raw(self) -> list[dict[str, Any]]:
        """Raw events from the last saved session (for status summaries)."""
        meta = self._read_last_session_meta()
        if not meta:
            return []
        events = meta.get("events")
        return list(events) if isinstance(events, list) else []

    def _load_session_events_from_episodes(self, session_id: str) -> list[dict[str, Any]]:
        if not session_id:
            return []
        path = episodes_path(self.root)
        if not path.is_file():
            return []
        out: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, dict) or row.get("session_id") != session_id:
                    continue
                for ev in row.get("events") or []:
                    if isinstance(ev, dict):
                        out.append(ev)
        out.sort(key=lambda e: float(e.get("t") or 0.0))
        return out

    def _timing(self) -> tuple[float, float, int]:
        """Return (wall_t, t_rel, frame_idx)."""
        now = time.time()
        started = self._session_started_at or now
        t_rel = max(0.0, now - started)
        with self._lock:
            frame_idx = int(self._frames)
            fps = float(self._fps) or 15.0
        if self.capture_mode == "video":
            # Prefer wall-clock mapping so events stay synced even if encode lags.
            frame_idx = max(0, int(round(t_rel * fps)))
        return now, t_rel, frame_idx

    def _start_input_listener(self) -> None:
        from pynput import keyboard, mouse

        def on_click(x: int, y: int, button: Any, pressed: bool) -> None:
            if not self._running:
                return
            name = getattr(button, "name", str(button))
            rect = self._last_rect or find_blender_window()
            lx, ly, space = _to_window_xy(rect, int(x), int(y))
            now, t_rel, frame_idx = self._timing()
            if pressed:
                self._drag_start = (int(x), int(y))
                self._drag_start_local = (lx, ly, space)
                self._drag_points = [(lx, ly)]
                self._drag_t0 = now
                self._last_move_t = now
                self._push_event(
                    {
                        "type": "click",
                        "x": lx,
                        "y": ly,
                        "button": name,
                        "pressed": True,
                        "coord_space": space,
                        "t": now,
                        "t_rel": t_rel,
                        "frame_idx": frame_idx,
                    }
                )
            else:
                start = self._drag_start
                start_local = self._drag_start_local
                path = list(self._drag_points)
                t0 = self._drag_t0
                self._drag_start = None
                self._drag_start_local = None
                self._drag_points = []
                self._drag_t0 = None
                if start and (abs(start[0] - x) > 3 or abs(start[1] - y) > 3):
                    sx, sy, sspace = start_local or (start[0], start[1], "screen")
                    path.append((lx, ly))
                    drag_ev: dict[str, Any] = {
                        "type": "drag",
                        "x1": sx,
                        "y1": sy,
                        "x2": lx,
                        "y2": ly,
                        "button": name,
                        "coord_space": sspace if sspace == space else space,
                        "t": now,
                        "t_rel": t_rel,
                        "frame_idx": frame_idx,
                        "duration": bounded_drag_duration(now - float(t0 or now)) or 0.0,
                    }
                    thinned = _thin_points(path)
                    if len(thinned) >= 2:
                        drag_ev["points"] = thinned
                    self._push_event(drag_ev)
                self._push_event(
                    {
                        "type": "click",
                        "x": lx,
                        "y": ly,
                        "button": name,
                        "pressed": False,
                        "coord_space": space,
                        "t": now,
                        "t_rel": t_rel,
                        "frame_idx": frame_idx,
                    }
                )

        def on_move(x: int, y: int) -> None:
            if not self._running or self._drag_start is None:
                return
            now = time.time()
            if now - self._last_move_t < 0.012:
                return
            self._last_move_t = now
            rect = self._last_rect or find_blender_window()
            lx, ly, _space = _to_window_xy(rect, int(x), int(y))
            if self._drag_points:
                px, py = self._drag_points[-1]
                if abs(lx - px) + abs(ly - py) < 2:
                    return
            if len(self._drag_points) < 240:
                self._drag_points.append((lx, ly))

        def on_press(key: Any) -> None:
            if not self._running:
                return
            label = _resolve_pynput_key(key)
            if not label:
                return
            mod = _canonical_modifier(label)
            if mod:
                self._held_modifiers.add(mod)
                return
            now, t_rel, frame_idx = self._timing()
            # Debounce OS auto-repeat for the same key.
            last = self._last_key_event
            chord = _ordered_hotkey(self._held_modifiers, label)
            sig = "+".join(chord)
            if last and last[0] == sig and (now - last[1]) < 0.08:
                return
            self._last_key_event = (sig, now)
            if len(chord) > 1:
                self._push_event(
                    {
                        "type": "hotkey",
                        "keys": chord,
                        "t": now,
                        "t_rel": t_rel,
                        "frame_idx": frame_idx,
                    }
                )
            else:
                self._push_event(
                    {
                        "type": "key",
                        "key": label,
                        "t": now,
                        "t_rel": t_rel,
                        "frame_idx": frame_idx,
                    }
                )

        def on_release(key: Any) -> None:
            if not self._running:
                return
            label = _resolve_pynput_key(key)
            mod = _canonical_modifier(label)
            if mod:
                self._held_modifiers.discard(mod)

        self._held_modifiers = set()
        self._last_key_event = None
        self._listener = mouse.Listener(on_click=on_click, on_move=on_move)
        self._key_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.start()
        self._key_listener.start()

    def _push_event(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._pending.append(event)
            self._session_events.append(event)
            self._events += 1

    def _flush_pending(self, screenshot_rel: str | None = None) -> None:
        with self._lock:
            events = list(self._pending)
            self._pending = []
            session_id = self._session_id
        if not events and screenshot_rel is None:
            return
        row = {
            "t": time.time(),
            "session_id": session_id,
            "capture_mode": "screenshots",
            "screenshot": screenshot_rel or self._last_shot_id,
            "events": events,
        }
        with episodes_path(self.root).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _flush_video_session(self) -> None:
        with self._lock:
            events = list(self._pending)
            self._pending = []
            session_id = self._session_id
            video_rel = self._video_rel
            frames = self._frames
            fps = self._fps
        if not events and not video_rel:
            return
        row = {
            "t": time.time(),
            "session_id": session_id,
            "capture_mode": "video",
            "video": video_rel or "",
            "fps": fps,
            "frame_count": frames,
            "events": events,
        }
        with episodes_path(self.root).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _open_writer(self, width: int, height: int) -> bool:
        import cv2

        w = max(2, int(width) - (int(width) % 2))
        h = max(2, int(height) - (int(height) % 2))
        session_id = self._session_id or uuid.uuid4().hex[:12]
        candidates = [
            (f"{session_id}.mp4", "mp4v"),
            (f"{session_id}.avi", "XVID"),
        ]
        for name, fourcc_name in candidates:
            path = videos_dir(self.root) / name
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(str(path), fourcc, float(self._fps), (w, h))
            if writer is not None and writer.isOpened():
                self._writer = writer
                self._video_path = path
                self._video_rel = name
                self._writer_size = (w, h)
                return True
            try:
                writer.release()
            except Exception:
                pass
            try:
                if path.is_file():
                    path.unlink()
            except OSError:
                pass
        return False

    def _ensure_video_writer(self, bgr: Any, rect: WindowRect) -> bool:
        if self._writer is not None:
            return True
        h, w = int(bgr.shape[0]), int(bgr.shape[1])
        if w < 2 or h < 2:
            w, h = int(rect.width), int(rect.height)
        ok = self._open_writer(w, h)
        if not ok:
            self._error = (
                f"Video writer failed (requested {self._requested_mode}); "
                "falling back to screenshots"
            )
            self.capture_mode = "screenshots"
            return False
        return True

    def _write_video_frame(self, bgr: Any) -> None:
        import cv2

        with self._lock:
            writer = self._writer
        if writer is None:
            return
        size = getattr(self, "_writer_size", None)
        frame = bgr
        if size is not None:
            tw, th = size
            if frame.shape[1] != tw or frame.shape[0] != th:
                frame = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        with self._lock:
            self._frames += 1

    def _loop(self) -> None:
        if self.capture_mode == "video":
            self._loop_video()
        else:
            self._loop_screenshots()

    def _loop_video(self) -> None:
        interval = 1.0 / float(self._fps or 15.0)
        while not self._stop.is_set():
            if record_duration_exceeded(self._session_started_at, time.time()):
                self._expire_recording()
                return
            tick = time.time()
            try:
                # Mode may flip to screenshots if writer init fails.
                if self.capture_mode != "video":
                    break
                rect = find_blender_window()
                bgr, rect = capture_window(rect)
                self._last_rect = rect
                if not self._ensure_video_writer(bgr, rect):
                    break
                self._write_video_frame(bgr)
            except Exception as exc:
                self._error = str(exc)
            elapsed = time.time() - tick
            self._stop.wait(max(0.001, interval - elapsed))
        if self.capture_mode == "screenshots" and not self._stop.is_set():
            # Fall back mid-session without dropping the rest of the recording.
            self._release_writer()
            self._loop_screenshots()

    def _loop_screenshots(self) -> None:
        interval = max(0.05, self.interval_ms / 1000.0)
        while not self._stop.is_set():
            if record_duration_exceeded(self._session_started_at, time.time()):
                self._expire_recording()
                return
            try:
                rect = find_blender_window()
                bgr, rect = capture_window(rect)
                self._last_rect = rect
                shot_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.png"
                dest = screenshots_dir(self.root) / shot_id
                dest.write_bytes(encode_png_bytes(bgr))
                # Link buffered events to the *previous* screenshot (actions after that shot).
                prev = self._last_shot_id
                if prev is not None:
                    self._flush_pending(screenshot_rel=prev)
                else:
                    # Still drain any early events onto this first frame.
                    self._flush_pending(screenshot_rel=shot_id)
                self._last_shot_id = shot_id
                self._shots += 1
            except Exception as exc:
                self._error = str(exc)
            self._stop.wait(interval)
        # Final flush
        if self._last_shot_id:
            self._flush_pending(screenshot_rel=self._last_shot_id)

    def similar_demos(self, goal: str, limit: int = 2) -> list[dict[str, Any]]:
        """Load a few recent recorded episodes for few-shot (goal used as soft filter)."""
        path = episodes_path(self.root)
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("events"):
                    rows.append(row)
        # Prefer recent rows with clicks/drags.
        rows = rows[-50:]
        scored = []
        g = (goal or "").lower()
        for row in rows:
            events = row.get("events") or []
            useful = [e for e in events if e.get("type") in {"click", "drag", "key", "hotkey"}]
            if not useful:
                continue
            score = float(row.get("t") or 0.0)
            if g and any(g[:12] in str(e).lower() for e in useful):
                score += 1e9
            demo: dict[str, Any] = {"events": useful[:8]}
            video_name = str(row.get("video") or "")
            if video_name:
                demo["video"] = video_name
                first = useful[0]
                frame = extract_video_frame(
                    videos_dir(self.root) / video_name,
                    frame_idx=int(first.get("frame_idx")) if first.get("frame_idx") is not None else None,
                    t_rel=float(first.get("t_rel")) if first.get("t_rel") is not None else None,
                    fps=float(row.get("fps") or 15.0),
                )
                if frame is not None:
                    demo["frame_bgr"] = frame
                    demo["frame_idx"] = first.get("frame_idx")
                    demo["t_rel"] = first.get("t_rel")
            else:
                demo["screenshot"] = row.get("screenshot")
            scored.append((score, demo))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in scored[:limit]]
