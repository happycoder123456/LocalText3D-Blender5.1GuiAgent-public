"""System prompts for Ollama Geometry Nodes generation."""

from __future__ import annotations

from .gn_spec import ALLOWED_NODE_TYPES

_NODE_LIST = ", ".join(sorted(ALLOWED_NODE_TYPES))

SYSTEM_PROMPT = f"""You generate Blender Geometry Nodes as JSON only. No markdown, no explanation.

Allowed node types: {_NODE_LIST}

Reserved node ids: "input" (Group Input), "output" (Group Output).

JSON schema:
{{
  "name": "ShortName",
  "description": "one sentence summary of what this builds",
  "instructions": [
    "Step tailored to THIS specific generator and its exposed inputs",
    "Another step explaining how to tweak a key parameter for this setup",
    "How to open and edit the node group in Blender",
    "Optional creative tip for this procedural system"
  ],
  "inputs": [
    {{"name": "Param", "type": "FLOAT", "default": 1.0, "min": 0.0, "max": 10.0}}
  ],
  "nodes": [
    {{"id": "unique_id", "type": "MeshGrid", "inputs": {{"Size X": 2.0, "Vertices X": 10}}}}
  ],
  "links": [
    {{"from": "grid", "from_socket": "Mesh", "to": "output", "to_socket": "Geometry"}}
  ]
}}

Rules:
- Return ONLY valid JSON matching the schema.
- Use ONLY types from the allowed list.
- Every graph must connect geometry to "output" socket "Geometry".
- Use descriptive exposed inputs for key parameters.
- ids must be unique lowercase snake_case strings.
- links must reference existing node ids plus input/output.
- "instructions" is REQUIRED: 3–6 short steps written for the exact system the user asked for.
- Mention the object's modifier, its exposed input names, and what changing them does for THIS generator.
- Do not give generic Blender tutorials — steps must match the prompt and the inputs you exposed.
"""

EXAMPLE_WALL = """{
  "name": "StoneWall",
  "description": "Procedural grid wall with random missing bricks and depth extrusion",
  "instructions": [
    "Select the generated wall object and find the Geometry Nodes modifier in the Properties panel.",
    "Adjust Wall Width to make the wall longer or shorter along X.",
    "Raise Missing Chance to delete more brick faces; lower it for a fuller wall.",
    "Change Seed (or re-generate with a new seed) to get a different random brick pattern.",
    "Click the modifier's node-group icon to open the tree and fine-tune extrusion or grid density."
  ],
  "inputs": [
    {"name": "Wall Width", "type": "FLOAT", "default": 4.0, "min": 0.5, "max": 20.0},
    {"name": "Missing Chance", "type": "FLOAT", "default": 0.15, "min": 0.0, "max": 1.0}
  ],
  "nodes": [
    {"id": "grid", "type": "MeshGrid", "inputs": {"Size X": 4.0, "Size Y": 1.0, "Vertices X": 16, "Vertices Y": 4}},
    {"id": "rand", "type": "RandomValue", "inputs": {"Data Type": "FLOAT", "Min": 0.0, "Max": 1.0}},
    {"id": "compare", "type": "Compare", "inputs": {"Operation": "GREATER_THAN", "Threshold": 0.15}},
    {"id": "delete", "type": "DeleteGeometry", "inputs": {"Domain": "FACE"}},
    {"id": "extrude", "type": "ExtrudeMesh", "inputs": {"Offset Scale": 0.08}}
  ],
  "links": [
    {"from": "grid", "from_socket": "Mesh", "to": "delete", "to_socket": "Geometry"},
    {"from": "rand", "from_socket": "Value", "to": "compare", "to_socket": "A"},
    {"from": "compare", "from_socket": "Result", "to": "delete", "to_socket": "Selection"},
    {"from": "delete", "from_socket": "Geometry", "to": "extrude", "to_socket": "Mesh"},
    {"from": "extrude", "from_socket": "Mesh", "to": "output", "to_socket": "Geometry"}
  ]
}"""


def build_messages(user_prompt: str, retry_error: str | None = None) -> list[dict]:
    user_text = user_prompt.strip()
    if retry_error:
        user_text = f"{user_text}\n\nPrevious attempt failed validation:\n{retry_error}\nFix and return valid JSON only."
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Example output for a stone wall:\n{EXAMPLE_WALL}"},
        {"role": "assistant", "content": EXAMPLE_WALL},
        {"role": "user", "content": user_text},
    ]
