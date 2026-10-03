"""Extract canonical skill parameters from Blender operator properties.

Blender's operator history (wm.operators) records exactly what ran and with
which values; this maps those raw property dicts onto the small parameter
vocabulary used by the knowledge base (distance, factor, axis, thickness…)."""

from __future__ import annotations

import math
import re
from typing import Any

_IDNAME_RE = re.compile(r"^([A-Z0-9_]+)_OT_(\w+)$")


def normalize_idname(op: str) -> str:
    raw = str(op or "").strip()
    m = _IDNAME_RE.match(raw)
    if m:
        return f"{m.group(1).lower()}.{m.group(2)}"
    return raw.lower()


def _vec(v: Any) -> list[float]:
    if isinstance(v, (list, tuple)):
        try:
            return [float(x) for x in v]
        except (TypeError, ValueError):
            return []
    return []


def _dominant_axis(vec: list[float], *, around: float = 0.0) -> int:
    return max(range(len(vec)), key=lambda i: abs(vec[i] - around))


def params_from_props(op: str, props: dict[str, Any] | None) -> dict[str, Any]:
    props = props or {}
    idn = normalize_idname(op)
    out: dict[str, Any] = {}

    if idn in {"mesh.extrude_region_move", "mesh.extrude_faces_move", "mesh.extrude_edges_move"}:
        tr = props.get("TRANSFORM_OT_translate") or props.get("TRANSFORM_OT_shrink_fatten") or {}
        vec = _vec(tr.get("value")) if isinstance(tr, dict) else []
        if vec:
            along_normal = len(vec) >= 3 and abs(vec[2]) >= max(abs(vec[0]), abs(vec[1])) * 0.5
            dist = vec[2] if along_normal else math.sqrt(sum(x * x for x in vec))
            out["distance"] = round(float(dist), 4)
        return out
    if idn == "mesh.inset":
        if "thickness" in props:
            out["thickness"] = round(float(props["thickness"]), 4)
        return out
    if idn == "mesh.bevel":
        if "offset" in props:
            out["offset"] = round(float(props["offset"]), 4)
        if "segments" in props:
            out["segments"] = int(props["segments"])
        return out
    if idn == "mesh.loopcut_slide":
        lc = props.get("MESH_OT_loopcut") if isinstance(props.get("MESH_OT_loopcut"), dict) else props
        if isinstance(lc, dict) and "number_cuts" in lc:
            out["number_cuts"] = int(lc["number_cuts"])
        return out
    if idn == "transform.translate":
        vec = _vec(props.get("value"))
        if vec:
            i = _dominant_axis(vec)
            out["distance"] = round(float(vec[i]), 4)
            out["axis"] = "xyz"[i] if i < 3 else "z"
        return out
    if idn == "transform.resize":
        vec = _vec(props.get("value"))
        if vec:
            if all(abs(v - vec[0]) < 1e-4 for v in vec):
                out["factor"] = round(float(vec[0]), 4)
                out["axis"] = ""
            else:
                i = _dominant_axis(vec, around=1.0)
                out["factor"] = round(float(vec[i]), 4)
                out["axis"] = "xyz"[i] if i < 3 else ""
        return out
    if idn == "transform.rotate":
        if "value" in props:
            try:
                out["angle"] = round(math.degrees(float(props["value"])), 2)
            except (TypeError, ValueError):
                pass
        axis = str(props.get("orient_axis") or "").lower()
        if axis in {"x", "y", "z"}:
            out["axis"] = axis
        return out
    if idn == "mesh.select_mode":
        if "type" in props:
            out["type"] = str(props["type"]).upper()
        return out
    if idn in {"mesh.select_all", "object.select_all"}:
        if "action" in props:
            out["action"] = str(props["action"]).upper()
        return out
    if idn == "object.subdivision_set":
        if "level" in props:
            out["level"] = int(props["level"])
        return out
    if idn == "object.modifier_add":
        if "type" in props:
            out["type"] = str(props["type"]).upper()
        return out
    if idn in {"object.duplicate_move", "mesh.duplicate_move"}:
        tr = props.get("TRANSFORM_OT_translate") or {}
        vec = _vec(tr.get("value")) if isinstance(tr, dict) else []
        if vec and any(abs(v) > 1e-6 for v in vec):
            i = _dominant_axis(vec)
            out["distance"] = round(float(vec[i]), 4)
            out["axis"] = "xyz"[i]
        return out
    if idn.startswith("mesh.primitive_"):
        for key in ("size", "radius", "depth"):
            if key in props:
                try:
                    out[key] = round(float(props[key]), 4)
                except (TypeError, ValueError):
                    pass
        return out
    return out
