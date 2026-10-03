"""Report live Blender state to the GUI Agent sidecar.

While you Record or the agent Runs, a timer posts: mode, active object,
selection counts, mesh stats, the 3D viewport rectangle and — most important —
the operators Blender actually executed (with their parameters). That is how a
recording becomes "extrude by 1.2", not "press E and click at 1067,782".
"""

from __future__ import annotations

import json
import time

import bpy

from . import agent_client

_INTERVAL = 0.2
_LINGER_SEC = 2.5  # keep posting a little after Stop so the last operator lands

_last_sigs: list[str] = []
_timer_registered = False
_wanted_until = 0.0
_fail_count = 0
_url = ""
_last_error = ""


# ----------------------------------------------------------------------------- state


def _props_dict(props, depth: int = 0) -> dict:
    out: dict = {}
    if props is None or depth > 2:
        return out
    try:
        rna_props = props.bl_rna.properties
    except Exception:
        return out
    for prop in rna_props:
        pid = prop.identifier
        if pid == "rna_type":
            continue
        try:
            if prop.type != "POINTER" and not props.is_property_set(pid):
                continue
            val = getattr(props, pid)
        except Exception:
            continue
        try:
            if prop.type == "POINTER":
                nested = _props_dict(val, depth + 1)
                if nested:
                    out[pid] = nested
            elif prop.type == "ENUM":
                out[pid] = sorted(str(v) for v in val) if isinstance(val, set) else str(val)
            elif getattr(prop, "is_array", False) or (hasattr(val, "__len__") and not isinstance(val, str)):
                out[pid] = [round(float(x), 4) for x in list(val)[:16]]
            elif isinstance(val, bool):
                out[pid] = bool(val)
            elif isinstance(val, float):
                out[pid] = round(val, 4)
            elif isinstance(val, int):
                out[pid] = int(val)
            elif isinstance(val, str):
                out[pid] = val[:64]
        except Exception:
            continue
        if len(out) >= 24:
            break
    return out


def _op_rows(wm) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    sigs: list[str] = []
    try:
        ops = list(wm.operators)
    except Exception:
        return rows, sigs
    for op in ops[-40:]:
        try:
            props = _props_dict(op.properties)
            idname = str(op.bl_idname)
            name = str(op.name)
        except Exception:
            continue
        sigs.append(idname + "|" + json.dumps(props, sort_keys=True, default=str))
        rows.append({"op": idname, "name": name, "props": props})
    return rows, sigs


def _new_ops(wm) -> list[dict]:
    """Operators executed since the previous tick (wm.operators is a sliding window)."""
    global _last_sigs
    rows, sigs = _op_rows(wm)
    prev = _last_sigs
    _last_sigs = sigs
    if not prev:
        return []
    # Find the longest suffix of the previous window that is a prefix of the new
    # one; everything after it is new. d == len(prev) would always match the
    # empty list and replay the whole window, so stop before it.
    for d in range(0, len(prev)):
        k = len(prev) - d
        if k <= len(sigs) and prev[d:] == sigs[:k]:
            return rows[k:]
    # History was replaced (undo/file load): report only the newest entry.
    return rows[-1:]


def reset_op_baseline() -> None:
    """Forget older operators so a new recording starts clean."""
    global _last_sigs
    try:
        _rows, sigs = _op_rows(bpy.context.window_manager)
        _last_sigs = sigs
    except Exception:
        _last_sigs = []


def _select_mode(scene) -> str:
    try:
        v, e, f = scene.tool_settings.mesh_select_mode
    except Exception:
        return ""
    if f and not v and not e:
        return "FACE"
    if e and not v:
        return "EDGE"
    return "VERT" if v else "FACE"


def _mesh_stats(obj) -> dict:
    out: dict = {}
    if obj is None or obj.type != "MESH":
        return out
    me = obj.data
    try:
        if obj.mode == "EDIT":
            import bmesh

            bm = bmesh.from_edit_mesh(me)
            out["verts"], out["edges"], out["faces"] = len(bm.verts), len(bm.edges), len(bm.faces)
        else:
            out["verts"], out["edges"], out["faces"] = len(me.vertices), len(me.edges), len(me.polygons)
        out["sel_verts"] = int(me.total_vert_sel)
        out["sel_edges"] = int(me.total_edge_sel)
        out["sel_faces"] = int(me.total_face_sel)
    except Exception:
        pass
    return out


def _regions(win) -> tuple[dict | None, dict | None, list[dict]]:
    """Window size, 3D viewport rect and UI rects in top-left window coords."""
    if win is None:
        return None, None, []
    window = {"w": int(win.width), "h": int(win.height)}
    best_area = None
    for area in win.screen.areas:
        if area.type != "VIEW_3D":
            continue
        if best_area is None or area.width * area.height > best_area.width * best_area.height:
            best_area = area
    if best_area is None:
        return window, None, []
    viewport = None
    ui: list[dict] = []
    for region in best_area.regions:
        if region.width <= 0 or region.height <= 0:
            continue
        rect = {
            "x": int(region.x),
            "y": int(win.height - (region.y + region.height)),
            "w": int(region.width),
            "h": int(region.height),
            "type": str(region.type),
        }
        if region.type == "WINDOW":
            viewport = rect
        elif region.type in {"UI", "TOOLS", "HEADER", "TOOL_HEADER", "ASSET_SHELF", "HUD"}:
            ui.append(rect)
    return window, viewport, ui


def gather_context() -> dict:
    ctx = bpy.context
    wm = ctx.window_manager
    scene = ctx.scene
    view_layer = ctx.view_layer
    obj = view_layer.objects.active if view_layer else None
    win = ctx.window
    if win is None:
        try:
            win = wm.windows[0]
        except Exception:
            win = None
    data: dict = {
        "t": time.time(),
        "mode": str(obj.mode) if obj is not None else "OBJECT",
        "active": obj.name if obj is not None else "",
        "active_type": str(obj.type) if obj is not None else "",
        "select_mode": _select_mode(scene),
    }
    try:
        ws = getattr(win, "workspace", None) if win is not None else None
        if ws is not None:
            data["workspace"] = str(getattr(ws, "name", "") or "")
    except Exception:
        pass
    try:
        objs = view_layer.objects
        data["objects"] = len(objs)
        data["selected_objects"] = len([o for o in objs if o.select_get(view_layer=view_layer)])
    except Exception:
        pass
    data.update(_mesh_stats(obj))
    window, viewport, ui = _regions(win)
    if window:
        data["window"] = window
    if viewport:
        data["viewport"] = viewport
    if ui:
        data["ui_rects"] = ui
    data["ops"] = _new_ops(wm)
    data.update(_material_state(obj))
    return data


def _material_state(obj) -> dict:
    out = {"material": "", "material_color": []}
    if obj is None:
        return out
    try:
        mat = getattr(obj, "active_material", None)
        if mat is None:
            return out
        out["material"] = str(mat.name or "")[:64]
        color = None
        if getattr(mat, "use_nodes", False) and mat.node_tree:
            for node in mat.node_tree.nodes:
                if getattr(node, "type", "") == "BSDF_PRINCIPLED":
                    sock = node.inputs.get("Base Color")
                    if sock is not None:
                        color = list(sock.default_value)[:3]
                    break
        if color is None:
            color = list(getattr(mat, "diffuse_color", (0, 0, 0, 1)))[:3]
        out["material_color"] = [round(float(c), 4) for c in color]
    except Exception:
        return out
    return out


def _apply_cmds(cmds) -> None:
    if not isinstance(cmds, list) or not cmds:
        return
    for cmd in cmds:
        if not isinstance(cmd, dict):
            continue
        op = str(cmd.get("op") or "")
        if op == "material.set":
            _assign_material(cmd)
        elif op == "workspace.set":
            _set_workspace(cmd)
        elif op == "object.mode_set":
            _set_object_mode(cmd)


def _set_workspace(cmd: dict) -> None:
    want = str(cmd.get("name") or "").strip()
    if not want:
        return
    ctx = bpy.context
    win = ctx.window
    if win is None:
        try:
            win = ctx.window_manager.windows[0]
        except Exception:
            return
    # Exact match first, then case-insensitive / fuzzy (Blender workspace names vary slightly).
    ws = bpy.data.workspaces.get(want)
    if ws is None:
        low = want.lower()
        for candidate in bpy.data.workspaces:
            name = str(candidate.name or "")
            if name.lower() == low or low in name.lower() or name.lower() in low:
                ws = candidate
                break
    if ws is None:
        return
    try:
        win.workspace = ws
    except Exception:
        pass


def _set_object_mode(cmd: dict) -> None:
    mode = str(cmd.get("mode") or "OBJECT").strip().upper() or "OBJECT"
    aliases = {
        "EDIT_MESH": "EDIT",
        "EDITMODE": "EDIT",
        "OBJECTMODE": "OBJECT",
        "PAINT_TEXTURE": "TEXTURE_PAINT",
        "TEXTUREPAINT": "TEXTURE_PAINT",
        "VERTEXPAINT": "VERTEX_PAINT",
        "WEIGHTPAINT": "WEIGHT_PAINT",
    }
    mode = aliases.get(mode, mode)
    try:
        if bpy.context.object is None:
            return
        current = str(bpy.context.object.mode or "")
        if current == mode or (mode == "EDIT" and current.startswith("EDIT")):
            return
        bpy.ops.object.mode_set(mode=mode)
    except Exception:
        pass


def _assign_material(cmd: dict) -> None:
    raw = cmd.get("color") or [0.86, 0.12, 0.12]
    try:
        r, g, b = [max(0.0, min(1.0, float(x))) for x in list(raw)[:3]]
    except (TypeError, ValueError):
        r, g, b = 0.86, 0.12, 0.12
    name = str(cmd.get("name") or "AgentColor").strip()[:48] or "AgentColor"
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    principled = None
    if tree is not None:
        for node in tree.nodes:
            if getattr(node, "type", "") == "BSDF_PRINCIPLED":
                principled = node
                break
    if principled is not None:
        sock = principled.inputs.get("Base Color")
        if sock is not None:
            sock.default_value = (r, g, b, 1.0)
    try:
        mat.diffuse_color = (r, g, b, 1.0)
    except Exception:
        pass
    ctx = bpy.context
    view = ctx.view_layer
    try:
        if ctx.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass
    meshes = []
    try:
        meshes = [o for o in view.objects if o.type == "MESH"]
    except Exception:
        return
    selected = [o for o in meshes if o.select_get(view_layer=view)]
    targets = selected or meshes
    for obj in targets:
        try:
            obj.color = (r, g, b, 1.0)
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)
            obj.active_material = mat
        except Exception:
            continue
    try:
        screen = ctx.screen
        if screen is not None:
            for area in screen.areas:
                if area.type != "VIEW_3D":
                    continue
                for space in area.spaces:
                    if space.type == "VIEW_3D" and getattr(space.shading, "type", "") == "SOLID":
                        space.shading.color_type = "MATERIAL"
    except Exception:
        pass


# ----------------------------------------------------------------------------- timer


_last_post_cost = 0.0


def _tick():
    global _timer_registered, _fail_count, _last_error, _last_post_cost
    try:
        scene = bpy.context.scene
        props = getattr(scene, "localtext3d_agent", None) if scene else None
        if props is None:
            _timer_registered = False
            return None
        active = bool(props.is_recording or props.is_running or props.is_learning_video)
        if not active and time.time() > _wanted_until:
            _timer_registered = False
            return None
        payload = gather_context()
        try:
            t0 = time.time()
            resp = agent_client.post_context(_url or props.sidecar_url, payload)
            _last_post_cost = time.time() - t0
            _fail_count = 0
            if isinstance(resp, dict):
                _apply_cmds(resp.get("apply") or [])
        except agent_client.AgentClientError as exc:
            _fail_count += 1
            if getattr(exc, "status", None) == 404:
                _last_error = "Sidecar is an old version — restart Start GUI Agent.bat"
                props.status_detail = _last_error
                _timer_registered = False
                return None
            if _fail_count > 10:
                _timer_registered = False
                return None
        # Posting blocks Blender's main thread; back off when the sidecar is slow
        # instead of stacking 0.8 s stalls every 0.2 s.
        if _last_post_cost > 0.3:
            return max(_INTERVAL, min(1.0, _last_post_cost * 2.0))
        return _INTERVAL
    except Exception:
        _timer_registered = False
        return None


def ensure_running(url: str, *, linger: float = _LINGER_SEC) -> None:
    """Start (or extend) the context reporter."""
    global _timer_registered, _wanted_until, _url, _fail_count
    _url = url
    _fail_count = 0
    _wanted_until = max(_wanted_until, time.time() + linger)
    # The module flag resets on Reload Scripts while the old timer keeps running;
    # ask Blender instead of trusting the flag.
    try:
        if bpy.app.timers.is_registered(_tick):
            _timer_registered = True
            return
    except Exception:
        pass
    if _timer_registered:
        return
    _timer_registered = True
    bpy.app.timers.register(_tick, first_interval=0.05)


def stop() -> None:
    """Unregister the context reporter (addon disable / Reload Scripts)."""
    global _timer_registered, _wanted_until
    _timer_registered = False
    _wanted_until = 0.0
    try:
        if bpy.app.timers.is_registered(_tick):
            bpy.app.timers.unregister(_tick)
    except Exception:
        pass


def post_now(url: str) -> bool:
    """One synchronous context post (used right before Run so the planner sees the scene)."""
    try:
        agent_client.post_context(url, gather_context(), timeout=1.5)
        return True
    except agent_client.AgentClientError:
        return False
    except Exception:
        return False


def linger(seconds: float = _LINGER_SEC) -> None:
    global _wanted_until
    _wanted_until = max(_wanted_until, time.time() + seconds)
