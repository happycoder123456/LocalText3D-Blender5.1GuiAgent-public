"""Live Blender scene context posted by the addon (mode, selection, mesh stats,
operator history). Gives the agent ground truth instead of pixel guesses."""

from __future__ import annotations

import threading
import time
from typing import Any

from agent.knowledge import is_noise_op, normalize_idname, params_from_props

MAX_SNAPSHOTS = 4000
MAX_OPS = 2000
MAX_UI_RECTS = 48
MAX_OPS_PER_UPDATE = 64
MAX_STR = 256


def normalize_mode(raw: Any) -> str:
    mode = str(raw or "").upper()
    if not mode:
        return ""
    if mode.startswith("EDIT"):
        return "EDIT"
    if mode in {"OBJECT", "SCULPT", "POSE"} or mode.startswith("PAINT"):
        return "OBJECT" if mode == "OBJECT" else mode
    return mode


def _to_int(v: Any) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


def _clip_str(value: Any, limit: int = MAX_STR) -> str:
    return str(value or "")[:limit]


def _clip_props(props: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in list(props.items())[:24]:
        name = str(key)[:64]
        if isinstance(value, str):
            out[name] = value[:MAX_STR]
        elif isinstance(value, (int, float, bool)) or value is None:
            out[name] = value
        elif isinstance(value, list):
            out[name] = value[:16]
        elif isinstance(value, dict):
            out[name] = _clip_props(value)
        else:
            out[name] = str(value)[:MAX_STR]
    return out


class ContextTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: list[dict[str, Any]] = []
        self._ops: list[dict[str, Any]] = []
        self._session_start: float | None = None

    # ------------------------------------------------------------------ input
    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        snap = {
            "t": now,
            "mode": normalize_mode(payload.get("mode")),
            "raw_mode": _clip_str(payload.get("mode"), 32),
            "active": _clip_str(payload.get("active")),
            "active_type": _clip_str(payload.get("active_type"), 32),
            "objects": _to_int(payload.get("objects")),
            "selected_objects": _to_int(payload.get("selected_objects")),
            "verts": _to_int(payload.get("verts")),
            "edges": _to_int(payload.get("edges")),
            "faces": _to_int(payload.get("faces")),
            "sel_verts": _to_int(payload.get("sel_verts")),
            "sel_edges": _to_int(payload.get("sel_edges")),
            "sel_faces": _to_int(payload.get("sel_faces")),
            "select_mode": _clip_str(payload.get("select_mode"), 16).upper(),
            "workspace": _clip_str(payload.get("workspace")),
            "material": _clip_str(payload.get("material"), 64),
            "material_color": list(payload.get("material_color") or [])[:3],
            "viewport": payload.get("viewport") if isinstance(payload.get("viewport"), dict) else None,
            "ui_rects": [
                item for item in (payload.get("ui_rects") or []) if isinstance(item, dict)
            ][:MAX_UI_RECTS]
            if isinstance(payload.get("ui_rects"), list)
            else [],
            "window": payload.get("window") if isinstance(payload.get("window"), dict) else None,
        }
        new_ops: list[dict[str, Any]] = []
        for raw in list(payload.get("ops") or [])[:MAX_OPS_PER_UPDATE]:
            if not isinstance(raw, dict):
                continue
            idn = normalize_idname(str(raw.get("op") or raw.get("idname") or ""))
            if not idn:
                continue
            props = raw.get("props") if isinstance(raw.get("props"), dict) else {}
            new_ops.append(
                {
                    "t": now,
                    "op": idn[:128],
                    "name": _clip_str(raw.get("name"), 64),
                    "props": _clip_props(props),
                    "params": params_from_props(idn, props),
                    "noise": is_noise_op(idn),
                    "mode": snap["mode"],
                }
            )
        with self._lock:
            self._snapshots.append(snap)
            if len(self._snapshots) > MAX_SNAPSHOTS:
                del self._snapshots[: len(self._snapshots) - MAX_SNAPSHOTS]
            self._ops.extend(new_ops)
            if len(self._ops) > MAX_OPS:
                del self._ops[: len(self._ops) - MAX_OPS]
        return snap

    def mark_session_start(self) -> None:
        with self._lock:
            self._session_start = time.time()

    # ------------------------------------------------------------------ query
    def latest(self, max_age: float | None = None) -> dict[str, Any] | None:
        with self._lock:
            if not self._snapshots:
                return None
            snap = self._snapshots[-1]
        if max_age is not None and time.time() - float(snap.get("t") or 0.0) > max_age:
            return None
        return dict(snap)

    def available(self, max_age: float = 3.0) -> bool:
        return self.latest(max_age=max_age) is not None

    def snapshot_at(self, t: float) -> dict[str, Any] | None:
        """Latest snapshot taken at or before time t."""
        with self._lock:
            best = None
            for snap in self._snapshots:
                if float(snap.get("t") or 0.0) <= t:
                    best = snap
                else:
                    break
        return dict(best) if best else None

    def wait_for_fresh(self, after_t: float, timeout: float = 0.8) -> dict[str, Any] | None:
        """Block until a snapshot newer than after_t arrives (or timeout)."""
        deadline = time.time() + max(0.0, timeout)
        while True:
            snap = self.latest()
            if snap and float(snap.get("t") or 0.0) > after_t:
                return snap
            if time.time() >= deadline:
                return snap
            time.sleep(0.04)

    def ops_between(self, t0: float, t1: float | None = None, *, include_noise: bool = False) -> list[dict[str, Any]]:
        t1 = t1 if t1 is not None else float("inf")
        with self._lock:
            rows = [dict(o) for o in self._ops if t0 <= float(o.get("t") or 0.0) <= t1]
        if not include_noise:
            rows = [o for o in rows if not o.get("noise")]
        return rows

    def session_ops(self) -> list[dict[str, Any]]:
        start = self._session_start
        if start is None:
            return []
        return self.ops_between(start)

    def session_snapshots(self) -> list[dict[str, Any]]:
        start = self._session_start
        if start is None:
            return []
        with self._lock:
            return [dict(s) for s in self._snapshots if float(s.get("t") or 0.0) >= start - 1.0]

    def viewport_center_norm(self) -> tuple[float, float] | None:
        snap = self.latest(max_age=5.0)
        if not snap:
            return None
        vp = snap.get("viewport")
        win = snap.get("window")
        if not isinstance(vp, dict) or not isinstance(win, dict):
            return None
        try:
            w = float(win.get("w") or 0)
            h = float(win.get("h") or 0)
            if w <= 0 or h <= 0:
                return None
            cx = (float(vp.get("x") or 0) + float(vp.get("w") or 0) / 2.0) / w
            cy = (float(vp.get("y") or 0) + float(vp.get("h") or 0) / 2.0) / h
        except (TypeError, ValueError):
            return None
        return (min(0.98, max(0.02, cx)), min(0.98, max(0.02, cy)))

    def point_region(self, x: float, y: float, snap: dict[str, Any] | None = None) -> str:
        """Classify a window-local point: 'viewport' | 'ui' | 'unknown'."""
        snap = snap or self.latest(max_age=30.0)
        if not snap:
            return "unknown"
        vp = snap.get("viewport")
        if isinstance(vp, dict):
            try:
                if (
                    float(vp.get("x") or 0) <= x <= float(vp.get("x") or 0) + float(vp.get("w") or 0)
                    and float(vp.get("y") or 0) <= y <= float(vp.get("y") or 0) + float(vp.get("h") or 0)
                ):
                    return "viewport"
            except (TypeError, ValueError):
                pass
            return "ui"
        return "unknown"

    def summary(self) -> str:
        snap = self.latest(max_age=10.0)
        if not snap:
            return ""
        mode = snap.get("mode") or "?"
        active = snap.get("active") or "(no active)"
        if mode == "EDIT":
            return (
                f"{mode} · {active} · {snap.get('sel_faces', 0)}/{snap.get('faces', 0)} faces sel "
                f"· {snap.get('select_mode') or ''}".strip()
            )
        return f"{mode} · {active} · {snap.get('selected_objects', 0)}/{snap.get('objects', 0)} objects sel"

    def clear(self) -> None:
        with self._lock:
            self._snapshots.clear()
            self._ops.clear()
            self._session_start = None
