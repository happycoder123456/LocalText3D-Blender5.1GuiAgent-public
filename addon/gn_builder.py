"""Build Blender Geometry Node trees from validated JSON specs."""

from __future__ import annotations

import re
from typing import Any

import bpy

from .gn_spec import NODE_TYPE_MAP, GnSpecError


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())[:56].strip("_")
    return cleaned or "GN_Generated"


def _find_socket(node, name: str, is_output: bool, *, allow_first: bool = False):
    """Socket by name/identifier (exact, then substring).

    Only falls back to the first socket when `allow_first` is set and the name is
    empty/generic; a misspelled key must not silently overwrite socket[0].
    """
    sockets = node.outputs if is_output else node.inputs
    target = name.strip().lower()
    if target:
        for sock in sockets:
            if sock.name.lower() == target or sock.identifier.lower() == target:
                return sock
        for sock in sockets:
            if target in sock.name.lower() or target in sock.identifier.lower():
                return sock
    if (allow_first or not target) and len(sockets):
        # Empty or generic ("geometry"/"value") socket names mean "the main one".
        if not target or target in {"geometry", "mesh", "value", "result", "out", "in"}:
            return sockets[0]
    return None


def _prop_id_for_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _set_node_property(node, key: str, value: Any) -> bool:
    """Node *settings* (Operation, Domain, Data Type, Mode…) are RNA props, not sockets."""
    prop_id = _prop_id_for_key(key)
    if not prop_id:
        return False
    prop = node.bl_rna.properties.get(prop_id)
    if prop is None:
        return False
    try:
        if prop.type == "ENUM":
            if not isinstance(value, str):
                return False
            want = value.strip().upper().replace(" ", "_")
            for item in prop.enum_items:
                if item.identifier.upper() == want or item.name.upper().replace(" ", "_") == want:
                    setattr(node, prop_id, item.identifier)
                    return True
            return False
        if prop.type in {"BOOLEAN", "INT", "FLOAT", "STRING"} and not prop.is_readonly:
            setattr(node, prop_id, value)
            return True
    except (TypeError, AttributeError, ValueError):
        return False
    return False


def _set_input_value(node, key: str, value: Any) -> None:
    # 1) Node settings such as {"Operation": "MULTIPLY"} / {"Domain": "FACE"}.
    if _set_node_property(node, key, value):
        return
    # 2) Otherwise an input socket default.
    sock = _find_socket(node, key, is_output=False)
    if sock is None:
        return
    try:
        if isinstance(value, (bool, int, float)):
            sock.default_value = value
        elif isinstance(value, (list, tuple)):
            sock.default_value = tuple(float(v) for v in value)
        elif isinstance(value, str):
            for item in getattr(sock, "enum_items", []) or []:
                if item.identifier.upper() == value.upper() or item.name.upper() == value.upper():
                    sock.default_value = item.identifier
                    return
            if hasattr(sock, "default_value"):
                try:
                    sock.default_value = value
                except TypeError:
                    pass
    except (TypeError, AttributeError, ValueError):
        pass


def _create_node(nodes, node_type: str):
    candidates = NODE_TYPE_MAP.get(node_type, [])
    for bl_idname in candidates:
        try:
            return nodes.new(type=bl_idname)
        except RuntimeError:
            continue
    raise GnSpecError(f"Blender does not have a node for type '{node_type}'")


def _ensure_io_nodes(node_tree):
    nodes = node_tree.nodes
    group_in = None
    group_out = None
    for node in nodes:
        if node.bl_idname == "NodeGroupInput":
            group_in = node
        elif node.bl_idname == "NodeGroupOutput":
            group_out = node
    if group_in is None:
        group_in = nodes.new(type="NodeGroupInput")
        group_in.location = (-400, 0)
    if group_out is None:
        group_out = nodes.new(type="NodeGroupOutput")
        group_out.location = (400, 0)
    return group_in, group_out


def _add_group_inputs(node_tree, spec_inputs: list[dict], group_in) -> dict[str, Any]:
    interface = node_tree.interface
    socket_map: dict[str, Any] = {}
    for item in spec_inputs:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        socket_type = str(item.get("type") or "FLOAT").upper()
        bl_socket = "NodeSocketFloat"
        if socket_type == "INT":
            bl_socket = "NodeSocketInt"
        elif socket_type == "VECTOR":
            bl_socket = "NodeSocketVector"
        elif socket_type == "BOOL":
            bl_socket = "NodeSocketBool"
        iface_item = None
        try:
            iface_item = interface.new_socket(name=name, in_out="INPUT", socket_type=bl_socket)
        except RuntimeError:
            pass
        # The modifier panel shows the INTERFACE item's default/min/max, not the
        # Group Input node's output socket value.
        if iface_item is not None:
            for src, dst in (("default", "default_value"), ("min", "min_value"), ("max", "max_value")):
                val = item.get(src)
                if val is None or not hasattr(iface_item, dst):
                    continue
                try:
                    if socket_type == "VECTOR" and isinstance(val, (list, tuple)):
                        setattr(iface_item, dst, tuple(float(v) for v in val))
                    elif socket_type == "BOOL":
                        setattr(iface_item, dst, bool(val))
                    elif socket_type == "INT":
                        setattr(iface_item, dst, int(val))
                    else:
                        setattr(iface_item, dst, float(val))
                except (TypeError, AttributeError, ValueError):
                    pass
        sock = _find_socket(group_in, name, is_output=True)
        if sock is not None:
            socket_map[name] = sock
    return socket_map


def build_node_group(spec: dict[str, Any], *, unique_suffix: str = "") -> bpy.types.NodeTree:
    base_name = _sanitize_name(str(spec.get("name") or "GN_Generated"))
    group_name = f"{base_name}{unique_suffix}"
    if group_name in bpy.data.node_groups:
        index = 1
        while f"{group_name}.{index:03d}" in bpy.data.node_groups:
            index += 1
        group_name = f"{group_name}.{index:03d}"

    node_tree = bpy.data.node_groups.new(group_name, "GeometryNodeTree")
    nodes = node_tree.nodes
    links = node_tree.links
    nodes.clear()

    group_in, group_out = _ensure_io_nodes(node_tree)
    spec_inputs = spec.get("inputs") or []
    if isinstance(spec_inputs, list):
        _add_group_inputs(node_tree, spec_inputs, group_in)

    id_map: dict[str, Any] = {"input": group_in, "output": group_out}
    x_offset = -200
    y_offset = 0

    for node_spec in spec.get("nodes") or []:
        node_id = str(node_spec.get("id") or "").strip()
        node_type = str(node_spec.get("type") or "").strip()
        node = _create_node(nodes, node_type)
        node.label = node_id
        node.name = node_id
        node.location = (x_offset, y_offset)
        y_offset -= 180
        if y_offset < -720:
            y_offset = 0
            x_offset += 220
        inputs = node_spec.get("inputs") or {}
        if isinstance(inputs, dict):
            for key, value in inputs.items():
                _set_input_value(node, str(key), value)
        id_map[node_id] = node

    for link_spec in spec.get("links") or []:
        src_id = str(link_spec.get("from") or "").strip()
        dst_id = str(link_spec.get("to") or "").strip()
        src_sock_name = str(link_spec.get("from_socket") or "").strip()
        dst_sock_name = str(link_spec.get("to_socket") or "").strip()
        src_node = id_map.get(src_id)
        dst_node = id_map.get(dst_id)
        if src_node is None or dst_node is None:
            raise GnSpecError(f"Link references missing node: {src_id} -> {dst_id}")
        out_sock = _find_socket(src_node, src_sock_name, is_output=True, allow_first=True)
        in_sock = _find_socket(dst_node, dst_sock_name, is_output=False, allow_first=True)
        if out_sock is None or in_sock is None:
            raise GnSpecError(f"Could not resolve sockets: {src_id}.{src_sock_name} -> {dst_id}.{dst_sock_name}")
        try:
            links.new(out_sock, in_sock)
        except RuntimeError as exc:
            raise GnSpecError(f"Failed to link {src_id} -> {dst_id}: {exc}") from exc

    return node_tree


def _ensure_object_mode(context) -> None:
    """select_all/mode-dependent ops fail from Edit/Sculpt mode (timer callbacks land anywhere)."""
    if getattr(context, "mode", "OBJECT") != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            pass


def apply_to_object(context, spec: dict[str, Any], object_name: str, meta: dict) -> bpy.types.Object:
    _ensure_object_mode(context)
    node_tree = build_node_group(spec)
    mesh = bpy.data.meshes.new(object_name)
    obj = bpy.data.objects.new(object_name, mesh)
    context.collection.objects.link(obj)

    modifier = obj.modifiers.new(name="GeometryNodes", type="NODES")
    modifier.node_group = node_tree

    obj["localtext3d_prompt"] = meta.get("prompt") or ""
    obj["localtext3d_mode"] = "geometry_nodes"
    obj["localtext3d_seed"] = int(meta.get("seed") or 0)
    obj["localtext3d_model"] = meta.get("model") or ""
    instructions = meta.get("instructions") or ""
    if instructions:
        obj["localtext3d_instructions"] = instructions
    instructions_path = meta.get("instructions_path") or ""
    if instructions_path:
        obj["localtext3d_instructions_path"] = instructions_path

    cursor = context.scene.cursor.location
    obj.location = cursor.copy()

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    context.view_layer.objects.active = obj
    if context.area and context.area.type == "VIEW_3D":
        try:
            bpy.ops.view3d.view_selected()
        except Exception:
            pass
    return obj
