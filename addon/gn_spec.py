"""Geometry Nodes JSON spec validation and node whitelist."""

from __future__ import annotations

import json
import re
from typing import Any

# Short type name -> Blender node bl_idname candidates (first match wins in builder).
NODE_TYPE_MAP: dict[str, list[str]] = {
    "MeshGrid": ["GeometryNodeMeshGrid"],
    "MeshLine": ["GeometryNodeMeshLine"],
    "MeshCube": ["GeometryNodeMeshCube"],
    "CurveLine": ["GeometryNodeCurveLine"],
    "CurveCircle": ["GeometryNodeCurveCircle"],
    "InstanceOnPoints": ["GeometryNodeInstanceOnPoints"],
    "RealizeInstances": ["GeometryNodeRealizeInstances"],
    "ScaleInstances": ["GeometryNodeScaleInstances"],
    "RotateInstances": ["GeometryNodeRotateInstances"],
    "ExtrudeMesh": ["GeometryNodeExtrudeMesh"],
    "Transform": ["GeometryNodeTransform"],
    "DeleteGeometry": ["GeometryNodeDeleteGeometry"],
    "JoinGeometry": ["GeometryNodeJoinGeometry"],
    "MergeByDistance": ["GeometryNodeMergeByDistance"],
    "FillCurve": ["GeometryNodeFillCurve"],
    "SetPosition": ["GeometryNodeSetPosition"],
    "StoreNamedAttribute": ["GeometryNodeStoreNamedAttribute"],
    "RandomValue": ["FunctionNodeRandomValue"],
    "Index": ["GeometryNodeInputIndex"],
    "Position": ["GeometryNodeInputPosition"],
    "Normal": ["GeometryNodeInputNormal"],
    "Math": ["FunctionNodeMath", "ShaderNodeMath"],
    "CombineXYZ": ["FunctionNodeCombineXYZ", "ShaderNodeCombineXYZ"],
    "SeparateXYZ": ["FunctionNodeSeparateXYZ", "ShaderNodeSeparateXYZ"],
    "Compare": ["FunctionNodeCompare", "FunctionNodeCompareFloats"],
    "Switch": ["GeometryNodeSwitch"],
    "FloatCurve": ["ShaderNodeFloatCurve", "FloatCurve"],
    "MapRange": ["FunctionNodeMapRange", "ShaderNodeMapRange"],
}

ALLOWED_NODE_TYPES = frozenset(NODE_TYPE_MAP.keys())
REQUIRED_KEYS = ("name", "nodes", "links", "instructions")
RESERVED_IDS = frozenset({"input", "output"})
MAX_SPEC_CHARS = 100_000
MAX_NODES = 48
MAX_LINKS = 96
MAX_INPUTS = 24
MAX_INSTRUCTIONS = 40
MAX_ID_LEN = 64
MAX_NAME_LEN = 64
MAX_INSTRUCTION_LEN = 500
MAX_INPUT_KEYS = 24
_VERTEX_KEY = re.compile(r"(vertices|count|resolution|subdiv|cuts)", re.I)
_MAX_VERTEX = 256
_MAX_SCALAR = 1000.0


class GnSpecError(Exception):
    pass


def parse_json_text(text: str) -> dict[str, Any]:
    text = text.strip()
    if len(text) > MAX_SPEC_CHARS:
        raise GnSpecError("Geometry Nodes spec is too large")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GnSpecError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise GnSpecError("Root must be a JSON object")
    return data


def _sanitize_node_inputs(inputs: Any) -> dict[str, Any]:
    if not isinstance(inputs, dict):
        return {}
    out: dict[str, Any] = {}
    for key, value in list(inputs.items())[:MAX_INPUT_KEYS]:
        name = str(key).strip()[:MAX_NAME_LEN]
        if not name:
            continue
        out[name] = _sanitize_input_value(name, value)
    return out


def _sanitize_input_value(name: str, value: Any) -> Any:
    cap = _MAX_VERTEX if _VERTEX_KEY.search(name) else int(_MAX_SCALAR)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return max(-cap, min(cap, value))
    if isinstance(value, float):
        bound = float(cap)
        return max(-bound, min(bound, value))
    if isinstance(value, str):
        return value[:MAX_NAME_LEN]
    if isinstance(value, (list, tuple)) and len(value) <= 4:
        cleaned = []
        for item in value:
            try:
                cleaned.append(max(-_MAX_SCALAR, min(_MAX_SCALAR, float(item))))
            except (TypeError, ValueError):
                continue
        return cleaned
    return value


def validate_spec(data: dict[str, Any]) -> dict[str, Any]:
    for key in REQUIRED_KEYS:
        if key not in data:
            raise GnSpecError(f"Missing required key: {key}")

    nodes = data.get("nodes")
    links = data.get("links")
    if not isinstance(nodes, list) or not nodes:
        raise GnSpecError("'nodes' must be a non-empty list")
    if len(nodes) > MAX_NODES:
        raise GnSpecError(f"'nodes' is limited to {MAX_NODES} entries")
    if not isinstance(links, list):
        raise GnSpecError("'links' must be a list")
    if len(links) > MAX_LINKS:
        raise GnSpecError(f"'links' is limited to {MAX_LINKS} entries")

    node_ids: set[str] = set(RESERVED_IDS)
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise GnSpecError(f"Node {index} must be an object")
        node_id = str(node.get("id") or "").strip()
        if not node_id:
            raise GnSpecError(f"Node {index} missing 'id'")
        if len(node_id) > MAX_ID_LEN:
            raise GnSpecError(f"Node id '{node_id[:24]}…' is too long")
        if node_id in node_ids:
            raise GnSpecError(f"Duplicate node id: {node_id}")
        node_ids.add(node_id)
        node_type = str(node.get("type") or "").strip()
        if node_type not in ALLOWED_NODE_TYPES:
            raise GnSpecError(f"Unknown node type '{node_type}' on '{node_id}'")
        node["id"] = node_id
        node["type"] = node_type
        node["inputs"] = _sanitize_node_inputs(node.get("inputs"))

    for index, link in enumerate(links):
        if not isinstance(link, dict):
            raise GnSpecError(f"Link {index} must be an object")
        for side in ("from", "to"):
            ref = str(link.get(side) or "").strip()
            if not ref:
                raise GnSpecError(f"Link {index} missing '{side}'")
            if ref not in node_ids:
                raise GnSpecError(f"Link {index} references unknown node '{ref}'")
        for sock_key in ("from_socket", "to_socket"):
            if not str(link.get(sock_key) or "").strip():
                raise GnSpecError(f"Link {index} missing '{sock_key}'")

    inputs = data.get("inputs")
    if inputs is not None:
        if not isinstance(inputs, list):
            raise GnSpecError("'inputs' must be a list")
        if len(inputs) > MAX_INPUTS:
            raise GnSpecError(f"'inputs' is limited to {MAX_INPUTS} entries")
        for index, item in enumerate(inputs):
            if not isinstance(item, dict):
                raise GnSpecError(f"Input {index} must be an object")
            name = str(item.get("name") or "").strip()
            if not name:
                raise GnSpecError(f"Input {index} missing 'name'")
            if len(name) > MAX_NAME_LEN:
                raise GnSpecError(f"Input {index} name is too long")
            item["name"] = name[:MAX_NAME_LEN]

    instructions = data.get("instructions")
    if not isinstance(instructions, list) or len(instructions) < 2:
        raise GnSpecError("'instructions' must be a list of at least 2 usage steps")
    if len(instructions) > MAX_INSTRUCTIONS:
        raise GnSpecError(f"'instructions' is limited to {MAX_INSTRUCTIONS} steps")
    for index, step in enumerate(instructions):
        text = str(step or "").strip()
        if not text:
            raise GnSpecError(f"Instruction step {index + 1} is empty")
        if len(text) > MAX_INSTRUCTION_LEN:
            raise GnSpecError(f"Instruction step {index + 1} is too long")
        instructions[index] = text[:MAX_INSTRUCTION_LEN]

    name = str(data.get("name") or "").strip()
    if name:
        data["name"] = name[:MAX_NAME_LEN]

    return data


def format_instructions(spec: dict[str, Any]) -> str:
    """Turn spec instructions into numbered panel text."""
    raw = spec.get("instructions")
    if isinstance(raw, list):
        lines = [str(step).strip() for step in raw if str(step).strip()]
        if lines:
            return "\n".join(f"{index + 1}. {line}" for index, line in enumerate(lines))

    description = str(spec.get("description") or "").strip()
    if description:
        return f"1. Select the object and open its Geometry Nodes modifier.\n2. {description}"
    return "1. Select the object.\n2. Open the Geometry Nodes modifier to edit the node tree."


def parse_and_validate(text: str) -> dict[str, Any]:
    return validate_spec(parse_json_text(text))
