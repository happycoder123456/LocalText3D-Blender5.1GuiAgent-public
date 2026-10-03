"""Turn recordings into semantic skills and semantic skills back into actions.

A semantic skill is a list of ops:
    {"op": "mesh.extrude_region_move", "params": {"distance": 1.2}, "pointer": {...}|None}
plus start/end mode and a plain-language summary. Because ops carry
parameters, the same skill can be re-run with different values, in a different
mode, on a different window size — it is a technique, not a replay."""

from __future__ import annotations

import copy
import re
from typing import Any

from agent import knowledge as kb

_NUM_RE = re.compile(r"^-?\d*\.?\d+$")
_DIGIT_KEYS = set("0123456789.-")
# Recorder key labels that stand for printable characters inside F3 search text.
_SEARCH_NAMED_KEYS = {
    "space": " ",
    "minus": "-",
    "period": ".",
    "comma": ",",
    "slash": "/",
    "underscore": "_",
    "plus": "+",
    "equal": "=",
}


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _pointer_from_action(action: dict[str, Any], w: int, h: int) -> dict[str, Any]:
    return {
        "x_norm": _clamp01(float(action.get("x") or 0) / float(max(1, w))),
        "y_norm": _clamp01(float(action.get("y") or 0) / float(max(1, h))),
    }


def _looks_like_viewport(xn: float, yn: float) -> bool:
    return 0.05 <= xn <= 0.80 and 0.07 <= yn <= 0.95


def new_op(op: str, params: dict[str, Any] | None = None, pointer: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {"op": op, "params": dict(params or {})}
    if pointer:
        row["pointer"] = dict(pointer)
    row.update(extra)
    return row


# --------------------------------------------------------------------------- inference


def infer_ops_from_actions(
    actions: list[dict[str, Any]],
    *,
    start_mode: str = "OBJECT",
    window_w: int = 0,
    window_h: int = 0,
) -> list[dict[str, Any]]:
    """Hotkey-only inference (used when no Blender operator history is available)."""
    if window_w <= 0 or window_h <= 0:
        from agent.skills import estimate_window_size

        window_w, window_h = estimate_window_size([a for a in actions if isinstance(a, dict)])
    mode = start_mode if start_mode in {"EDIT", "OBJECT"} else "OBJECT"
    ops: list[dict[str, Any]] = []
    i = 0
    rows = [a for a in actions if isinstance(a, dict)]
    while i < len(rows):
        a = rows[i]
        kind = str(a.get("action") or "")
        if kind in {"wait", "stop", ""}:
            i += 1
            continue
        if kind in {"click", "drag"}:
            if a.get("x_norm") is not None:
                ptr = {"x_norm": float(a["x_norm"]), "y_norm": float(a["y_norm"])}
            else:
                ptr = _pointer_from_action(a, window_w, window_h)
            if _looks_like_viewport(ptr["x_norm"], ptr["y_norm"]):
                if kind == "drag":
                    ptr["drag"] = True
                    ptr["x2_norm"] = float(a.get("x2_norm") if a.get("x2_norm") is not None else float(a.get("x2") or 0) / max(1, window_w))
                    ptr["y2_norm"] = float(a.get("y2_norm") if a.get("y2_norm") is not None else float(a.get("y2") or 0) / max(1, window_h))
                ops.append(new_op("view3d.select", pointer=ptr, mode=mode))
            i += 1
            continue
        if kind == "type":
            i += 1
            continue
        keys = [str(k).lower() for k in (a.get("keys") or [])]
        if not keys:
            i += 1
            continue
        chord = "+".join(keys)
        if chord == "f3":
            text_parts: list[str] = []
            j = i + 1
            while j < len(rows):
                nxt = rows[j]
                nk = str(nxt.get("action") or "")
                nkeys = [str(k).lower() for k in (nxt.get("keys") or [])]
                if nk == "type":
                    text_parts.append(str(nxt.get("text") or ""))
                elif nk == "key" and nkeys == ["enter"]:
                    j += 1
                    break
                elif nk == "key" and nkeys in (["esc"], ["escape"]):
                    text_parts = []  # search cancelled
                    j += 1
                    break
                elif nk in {"wait"}:
                    pass
                elif nk == "key" and len(nkeys) == 1 and nkeys[0] in _SEARCH_NAMED_KEYS:
                    # The recorder emits one key per character and names
                    # punctuation ("space", "minus"); "Add Cube" must not stop at "add".
                    text_parts.append(_SEARCH_NAMED_KEYS[nkeys[0]])
                elif nk == "key" and len(nkeys) == 1 and len(nkeys[0]) == 1:
                    text_parts.append(nkeys[0])
                elif nk == "key" and nkeys == ["backspace"]:
                    if text_parts:
                        text_parts.pop()
                elif nk == "click":
                    j += 1  # clicked a result
                    break
                else:
                    break
                j += 1
            text = "".join(text_parts).strip()
            op = kb.op_for_search_text(text) if text else None
            if op:
                ops.append(new_op(op, mode=mode, via="search"))
                spec = kb.get(op)
                if spec and spec.mode == "ANY" and spec.effect.get("mode") == "toggle":
                    mode = "OBJECT" if mode == "EDIT" else "EDIT"
            elif text:
                ops.append(new_op("search", {"text": text}, mode=mode))
            i = j
            continue
        op = kb.op_for_hotkey(keys, mode)
        if op is None:
            if chord in {"enter", "esc", "escape"}:
                i += 1
                continue
            ops.append(new_op("key", {"keys": keys}, mode=mode))
            i += 1
            continue
        spec = kb.get(op)
        params: dict[str, Any] = {}
        if op == "mesh.select_mode":
            params["type"] = {"1": "VERT", "2": "EDGE", "3": "FACE"}.get(chord, "FACE")
        elif op == "object.subdivision_set":
            params["level"] = int(chord[-1])
        elif op in {"mesh.select_all", "object.select_all"}:
            params["action"] = "DESELECT" if chord == "alt+a" else "SELECT"
        j = i + 1
        loopcut = op == "mesh.loopcut_slide"
        if spec and (spec.typed or spec.confirm or loopcut):
            # Gather axis + typed number until Enter / click. Loop cut is
            # modal too: digits = number of cuts, click confirms, Esc/RMB ends slide.
            axis = ""
            number = ""
            confirmed = False
            while j < len(rows):
                nxt = rows[j]
                nk = str(nxt.get("action") or "")
                nkeys = [str(k).lower() for k in (nxt.get("keys") or [])]
                if nk == "type":
                    number += str(nxt.get("text") or "")
                elif nk == "key" and len(nkeys) == 1 and nkeys[0] in {"x", "y", "z"} and not number and not loopcut:
                    axis = nkeys[0]
                elif nk == "key" and len(nkeys) == 1 and nkeys[0] in _DIGIT_KEYS:
                    number += nkeys[0]
                elif nk == "key" and nkeys == ["minus"]:
                    number += "-"
                elif nk == "key" and nkeys == ["period"]:
                    number += "."
                elif nk == "key" and nkeys == ["enter"]:
                    j += 1
                    break
                elif nk == "click":
                    j += 1  # mouse confirm; distance came from the pointer
                    if loopcut and not confirmed:
                        confirmed = True
                        # Slide phase follows; swallow the Esc/right-click that ends it.
                        while j < len(rows):
                            nn = rows[j]
                            nnk = str(nn.get("action") or "")
                            nnkeys = [str(k).lower() for k in (nn.get("keys") or [])]
                            if nnk == "wait":
                                j += 1
                                continue
                            if nnk == "key" and nnkeys in (["esc"], ["escape"], ["enter"]):
                                j += 1
                            elif nnk == "click" and str(nn.get("button") or "left") == "right":
                                j += 1
                            break
                    break
                elif nk == "wait":
                    pass
                else:
                    break
                j += 1
            if loopcut:
                if number and number.isdigit():
                    params["number_cuts"] = max(1, min(20, int(number)))
            elif spec.typed and number and _NUM_RE.match(number):
                params[spec.typed] = float(number)
            if spec.axis_param and axis:
                params[spec.axis_param] = axis
        ops.append(new_op(op, params, mode=mode))
        if spec and spec.effect.get("mode") == "toggle":
            mode = "OBJECT" if mode == "EDIT" else "EDIT"
        i = max(j, i + 1)
    return ops


def _nearest_click(
    events: list[dict[str, Any]], t: float, *, before: float = 1.6, after: float = 0.35
) -> dict[str, Any] | None:
    best = None
    best_d = None
    for ev in events:
        if ev.get("type") != "click" or not bool(ev.get("pressed", True)):
            continue
        if ev.get("coord_space") != "window":
            continue
        et = float(ev.get("t") or 0.0)
        if t - before <= et <= t + after:
            d = abs(t - et)
            if best_d is None or d < best_d:
                best, best_d = ev, d
    return best


def ops_from_operator_history(
    op_events: list[dict[str, Any]],
    input_events: list[dict[str, Any]],
    *,
    tracker: Any = None,
    window_w: int = 0,
    window_h: int = 0,
) -> list[dict[str, Any]]:
    """Ground truth path: Blender's own operator history + pointer positions."""
    ops: list[dict[str, Any]] = []
    used_click_ts: set[float] = set()
    w = max(1, int(window_w))
    h = max(1, int(window_h))
    for ev in op_events:
        idn = str(ev.get("op") or "")
        if not idn or ev.get("noise"):
            continue
        spec = kb.get(idn)
        params = dict(ev.get("params") or {})
        pointer = None
        if spec and spec.pointer:
            click = _nearest_click(input_events, float(ev.get("t") or 0.0))
            if click:
                used_click_ts.add(float(click.get("t") or 0.0))
                pointer = _pointer_from_action(click, w, h)
        row = new_op(idn, params, pointer=pointer, mode=str(ev.get("mode") or ""), t=float(ev.get("t") or 0.0))
        if spec is None:
            row["unknown"] = True
        ops.append(row)
    # Viewport clicks that did not register an operator are selections.
    for ev in input_events:
        if ev.get("type") != "click" or not bool(ev.get("pressed", True)) or ev.get("coord_space") != "window":
            continue
        t = float(ev.get("t") or 0.0)
        if t in used_click_ts:
            continue
        x, y = float(ev.get("x") or 0), float(ev.get("y") or 0)
        snap = tracker.snapshot_at(t) if tracker is not None else None
        region = tracker.point_region(x, y, snap) if tracker is not None else "unknown"
        if region == "ui":
            continue
        ptr = _pointer_from_action(ev, w, h)
        if region == "unknown" and not _looks_like_viewport(ptr["x_norm"], ptr["y_norm"]):
            continue
        ops.append(new_op("view3d.select", pointer=ptr, mode=str((snap or {}).get("mode") or ""), t=t))
    ops.sort(key=lambda o: float(o.get("t") or 0.0))
    # Selections immediately followed by an op that consumed the click are duplicates.
    cleaned: list[dict[str, Any]] = []
    for row in ops:
        row = dict(row)
        row.pop("t", None)
        cleaned.append(row)
    return cleaned


def strip_trailing_ui(ops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop trailing selection clicks (e.g. pressing Stop Recording in the N-panel)."""
    out = list(ops)
    while out and out[-1].get("op") == "view3d.select":
        out.pop()
    return out


# --------------------------------------------------------------------------- summaries


def summarize_ops(ops: list[dict[str, Any]], limit: int = 8) -> str:
    bits: list[str] = []
    for row in ops:
        op = str(row.get("op") or "")
        if op == "view3d.select":
            bits.append("click-select")
        elif op == "search":
            bits.append(f"search '{row.get('params', {}).get('text', '')}'")
        elif op == "key":
            bits.append("+".join(row.get("params", {}).get("keys") or []))
        else:
            bits.append(kb.describe_op(op, row.get("params") or {}))
    if len(bits) > limit:
        bits = bits[: limit - 1] + [f"… +{len(bits) - limit + 1} more"]
    return " → ".join(bits)


def collect_params(ops: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in ops:
        spec = kb.get(str(row.get("op") or ""))
        if not spec:
            continue
        for name, val in (row.get("params") or {}).items():
            if name in spec.params:
                out[f"{spec.op.split('.')[-1]}.{name}"] = val
    return out


def make_skill_card(
    goal: str,
    ops: list[dict[str, Any]],
    *,
    start_mode: str = "",
    end_mode: str = "",
    effects: dict[str, Any] | None = None,
    origin: str = "recorded",
    grounded: bool = False,
) -> dict[str, Any]:
    ops = [copy.deepcopy(o) for o in ops if isinstance(o, dict) and o.get("op")]
    return {
        "goal": str(goal or "").strip(),
        "ops": ops,
        "start_mode": start_mode,
        "end_mode": end_mode,
        "effects": dict(effects or {}),
        "summary": summarize_ops(ops),
        "params": collect_params(ops),
        "origin": origin,
        "grounded": bool(grounded),
    }


def abstract_recording(
    goal: str,
    actions: list[dict[str, Any]],
    *,
    tracker: Any = None,
    input_events: list[dict[str, Any]] | None = None,
    window_w: int = 0,
    window_h: int = 0,
) -> dict[str, Any]:
    """Build a semantic skill card from a Record→Stop session."""
    op_events = tracker.session_ops() if tracker is not None else []
    snaps = tracker.session_snapshots() if tracker is not None else []
    start_mode = str(snaps[0].get("mode") or "") if snaps else ""
    end_mode = str(snaps[-1].get("mode") or "") if snaps else ""
    has_rects = any(isinstance(s.get("viewport"), dict) for s in snaps)
    if (window_w <= 0 or window_h <= 0) and snaps:
        # Blender's own window size beats a guess from click extents.
        for s in reversed(snaps):
            win = s.get("window")
            if isinstance(win, dict) and int(win.get("w") or 0) > 0:
                window_w, window_h = int(win["w"]), int(win.get("h") or 0)
                break
    effects: dict[str, Any] = {}
    if len(snaps) >= 2:
        a, b = snaps[0], snaps[-1]
        for key in ("faces", "verts", "objects"):
            effects[f"{key}_delta"] = int(b.get(key) or 0) - int(a.get(key) or 0)
    grounded = bool(op_events)
    if grounded:
        ops = ops_from_operator_history(
            op_events,
            list(input_events or []),
            tracker=tracker,
            window_w=window_w,
            window_h=window_h,
        )
    else:
        ops = infer_ops_from_actions(
            actions,
            start_mode=start_mode or "OBJECT",
            window_w=window_w,
            window_h=window_h,
        )
    if not (grounded and has_rects):
        # Without exact region rects, the last click is almost always Stop Recording.
        ops = strip_trailing_ui(ops)
    return make_skill_card(
        goal,
        ops,
        start_mode=start_mode,
        end_mode=end_mode,
        effects=effects,
        origin="recorded",
        grounded=grounded,
    )


# --------------------------------------------------------------------------- execution


def _toggle_step(reason: str) -> dict[str, Any]:
    return {"action": "key", "keys": ["tab"], "reason": reason}


def ops_to_steps(
    ops: list[dict[str, Any]],
    *,
    start_mode: str = "",
    variant_for: Any = None,
    default_pointer: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Expand semantic ops into primitives, inserting Tab where a mode is required.

    Each primitive is tagged with sem_i (index of its op) and sem_last.
    variant_for(op) -> 0/1 chooses hotkey vs. F3 search per op (learned preference).
    """
    mode = start_mode if start_mode in {"EDIT", "OBJECT"} else ""
    steps: list[dict[str, Any]] = []
    for i, row in enumerate(ops):
        op = str(row.get("op") or "")
        params = dict(row.get("params") or {})
        pointer = row.get("pointer") if isinstance(row.get("pointer"), dict) else None
        prims: list[dict[str, Any]] = []
        spec = kb.get(op)
        if op == "key":
            prims = [{"action": "hotkey" if len(params.get("keys") or []) > 1 else "key", "keys": list(params.get("keys") or []), "reason": "recorded key"}]
        elif op == "search":
            prims = kb._search_steps(str(params.get("text") or ""), "recorded search")
        elif spec is None:
            continue
        else:
            if op == "object.editmode_toggle":
                target = str(params.get("target") or "").upper()
                if target in {"EDIT", "OBJECT"} and mode == target:
                    steps.append({"action": "wait", "seconds": 0.05, "reason": f"already in {target} mode", "sem_i": i, "sem_last": False, "settle_only": True})
                    continue
                if target in {"EDIT", "OBJECT"}:
                    prims = kb.expand_op(op, {}, variant=0)
                    mode = target
                    for p in prims:
                        p.update({"sem_i": i, "sem_last": False, "from_teacher": True, "variant": 0})
                        p.setdefault("stats", {"ok": 0, "fail": 0, "last_reward": 0.0})
                    steps.extend(prims)
                    if prims:
                        steps[-1]["sem_last"] = True
                    steps.append({"action": "wait", "seconds": 0.3, "reason": "mode switch", "sem_i": i, "sem_last": False, "settle_only": True})
                    continue
            need = spec.mode
            if need in {"EDIT", "OBJECT"} and mode and mode != need:
                steps.append({**_toggle_step(f"enter {need.title()} Mode for {spec.label}"), "sem_i": i, "sem_last": False, "auto_mode": True})
                steps.append({"action": "wait", "seconds": 0.3, "reason": "mode switch", "sem_i": i, "sem_last": False})
                mode = need
            elif need in {"EDIT", "OBJECT"} and not mode:
                mode = need  # assume the user is already where the op works
            variant = int(variant_for(op)) if callable(variant_for) else 0
            if op == "view3d.select" and pointer is None:
                pointer = dict(default_pointer or {"x_norm": 0.45, "y_norm": 0.52})
            if spec.pointer and pointer is None and default_pointer:
                pointer = dict(default_pointer)
            prims = kb.expand_op(op, params, variant=variant, pointer=pointer)
            if spec.effect.get("mode") == "toggle":
                mode = "OBJECT" if mode == "EDIT" else "EDIT" if mode else ""
        for p in prims:
            p = dict(p)
            p.setdefault("stats", {"ok": 0, "fail": 0, "last_reward": 0.0})
            p["sem_i"] = i
            p["sem_last"] = False
            p["from_teacher"] = True
            p["variant"] = int(variant_for(op)) if (callable(variant_for) and spec is not None) else 0
            if p.get("action") in {"click", "drag"} and "x_norm" in p:
                p.setdefault("intent", "viewport_point")
            steps.append(p)
        if prims:
            steps[-1]["sem_last"] = True
        # Give Blender time to register the operator before we judge it.
        steps.append({"action": "wait", "seconds": 0.25, "reason": "let Blender apply", "sem_i": i, "sem_last": False, "settle_only": True})
    return steps

