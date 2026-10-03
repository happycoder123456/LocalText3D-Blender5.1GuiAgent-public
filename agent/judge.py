"""Judge one semantic operation using Blender's own state (not pixels).

Strongest evidence: Blender's operator history shows the op ran. Then expected
effects (face/vert/object counts, mode, select mode). None = no opinion, in
which case the caller falls back to the visual critic."""

from __future__ import annotations

from typing import Any

from agent import knowledge as kb


def same_family(a: str, b: str) -> bool:
    """mesh.extrude_region_move ~ mesh.extrude_region ~ mesh.extrude_faces_move."""

    def _root(idn: str) -> str:
        idn = str(idn or "")
        for suffix in ("_move", "_slide", "_region", "_faces", "_edges", "_verts"):
            if idn.endswith(suffix):
                idn = idn[: -len(suffix)]
        return idn

    return bool(a) and bool(b) and _root(a) == _root(b)


def _count_delta(before: dict[str, Any], after: dict[str, Any], key: str) -> int:
    return int(after.get(key) or 0) - int(before.get(key) or 0)


def judge_op(
    op: str,
    params: dict[str, Any],
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    observed_ops: list[dict[str, Any]],
) -> tuple[float | None, str]:
    spec = kb.get(op)
    if before is None or after is None:
        return None, "no Blender context"
    seen = {str(o.get("op") or "") for o in observed_ops if not o.get("noise")}
    if spec is None:
        # Unknown to the knowledge base: only credit it when some real (non-noise)
        # operator ran; view/UI ops must not count as success.
        return (0.6, "ran") if seen else (None, "unknown op")

    if op == "view3d.select":
        if after.get("mode") == "EDIT":
            changed = any(after.get(k) != before.get(k) for k in ("sel_faces", "sel_verts", "sel_edges"))
            has_sel = int(after.get("sel_faces") or 0) + int(after.get("sel_verts") or 0) > 0
            if has_sel:
                return (0.6 if changed else 0.3), "selection present"
            return -0.4, "nothing selected after click"
        return (0.5 if int(after.get("selected_objects") or 0) > 0 else -0.3), "object selection"

    if op in seen or any(same_family(op, s) for s in seen):
        note = f"Blender ran {spec.label}"
        for key, want in spec.effect.items():
            if key in {"faces", "verts", "objects"}:
                delta = _count_delta(before, after, key)
                if want == "+" and delta <= 0:
                    return 0.2, f"{note} but {key} did not increase"
                if want == "-" and delta >= 0:
                    return 0.2, f"{note} but {key} did not decrease"
        return 0.9, note

    # Operator not observed: fall back to expected effects.
    if spec.effect.get("mode") == "toggle":
        if after.get("mode") != before.get("mode"):
            return 0.9, f"mode is now {after.get('mode')}"
        return -0.6, "mode did not change"
    for key, want in spec.effect.items():
        if key in {"faces", "verts", "objects"}:
            delta = _count_delta(before, after, key)
            if want == "+" and delta > 0:
                return 0.7, f"{key} +{delta}"
            if want == "-" and delta < 0:
                return 0.7, f"{key} {delta}"
            return -0.5, f"{spec.label} had no effect on {key}"
        if key == "select_mode":
            want_mode = str(params.get("type") or "FACE").upper()
            if after.get("select_mode") == want_mode:
                return 0.8, f"select mode {want_mode}"
            return -0.5, f"select mode still {after.get('select_mode')}"
    if op == "material.set":
        got = after.get("material_color") or []
        if after.get("material") and _color_close(list(got), params.get("color") or "red"):
            return 0.9, f"material {after.get('material')}"
        if after.get("material"):
            return 0.5, "material assigned"
        return -0.5, "no material on the object"
    return -0.4, f"{spec.label} not registered by Blender"


def _color_close(got: list[Any], want: Any) -> bool:
    from agent.knowledge import parse_color

    try:
        a = parse_color(got)
        b = parse_color(want)
    except Exception:
        return False
    dist = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
    return dist < 0.35
